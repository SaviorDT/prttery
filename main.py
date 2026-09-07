"""Run the active profile configured in param.py."""

from __future__ import annotations

import os
import queue
import threading
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader.cvat import CvatLabelMap
from data_loader.normal import EvalDataset, EvalItem, eval_collate, get_eval_items
from mask_formats import get_mask_format
from param import ACTIVE_RUN, MaskFormat, RunMode
from outputer.mp4 import close_nvenc_writer, open_nvenc_writer, probe_nvenc_available, write_nvenc_frame, write_video
from outputer.overlay import ClassOverlayRenderer

# Cap on how many full-resolution original frames the background reader
# thread may hold in memory at once (backpressure via a bounded queue),
# waiting to be composited with their corresponding GPU-computed mask.
ORIGINAL_IMAGE_QUEUE_SIZE = 500

# Same backpressure idea, but for BGRA frames waiting to be webp-encoded by the writer pool.
WEBP_QUEUE_SIZE = 500
WEBP_WRITER_THREADS = 4

# Models whose output is a per-pixel softmax over multiple classes (not a
# single-channel sigmoid). Drives --mask-format validation (every mode requires
# --model/--mask-format to agree) and which train/test module handles --model.

def _compute_fps(seconds: list[int]) -> float:
    """fps = (# frames with second < max_second) / max_second.

    The last (possibly incomplete) second is excluded from the fps
    calculation so it doesn't skew the average.
    """
    max_second = max(seconds)
    if max_second == 0:
        return float(len(seconds))
    full_second_frames = sum(1 for s in seconds if s < max_second)
    return round(full_second_frames / max_second)


def _read_original_images(items: list[EvalItem], q: "queue.Queue[tuple[EvalItem, np.ndarray | None] | None]") -> None:
    """Background-thread producer: read full-res original frames in order.

    Runs concurrently with GPU inference on the (separately loaded, resized)
    tensor batches so the main loop never has to stall on disk I/O for the
    original image needed at compositing time. ``q``'s bounded size caps how
    far this thread can run ahead, and blocks (backpressure) once full.
    """
    for item in items:
        image = cv2.imread(item.image_path, cv2.IMREAD_COLOR)
        q.put((item, image))


def _webp_writer(q: "queue.Queue[tuple[str, np.ndarray] | None]") -> None:
    """Worker-thread consumer: cv2.imwrite releases the GIL, so several of these run truly in parallel."""
    while True:
        job = q.get()
        if job is None:
            q.task_done()
            return
        out_path, bgra = job
        if not cv2.imwrite(out_path, bgra, [cv2.IMWRITE_WEBP_QUALITY, 95]):
            tqdm.write(f"Failed to write {out_path}, skipping frame")
        q.task_done()


def run_eval(
    dirs: list[str],
    model_class,
    model_path: str,
    output_dir: str,
    mask_format: str,
    batch_size: int = 8,
    mode: str = "eval",
    expected_image_size: tuple[int, int] = (180, 320),
) -> None:
    """Shared pipeline for ``--mode eval`` and ``--mode eval_mul`` -- same
    background reader thread, DataLoader loop, webp writer pool, and
    NVENC/video assembly either way. ``mode`` only switches the per-frame
    compositing step (see the two branches inside the loop below):

    - ``"eval"``: today's alpha-matte behavior, producing a transparent
      foreground/background matte (see the collapse comment below).
    - ``"eval_mul"``: a colored overlay of each pixel's predicted class (via
      ``outputer.overlay.ClassOverlayRenderer``), always fully opaque.
    """
    model = model_class(*expected_image_size)
    model.load(model_path)

    in_height, in_width = model.input_shape[0], model.input_shape[1]
    if (in_height, in_width) != expected_image_size:
        raise ValueError(
            f"checkpoint input size {(in_height, in_width)} does not match profile image_size {expected_image_size}"
        )

    # mask_format, not the wrapper class, is the source of
    # truth for how to interpret a checkpoint's raw output, matching how it's already the
    # source of truth for --mode train/test.
    format_obj = get_mask_format(mask_format)
    is_multiclass_model = mask_format == "cvat_6"

    # A multi-class model's raw output is (B, num_classes, H, W) softmax
    # probabilities. --mode eval's output format stays the same alpha-matte
    # pipeline as the binary models: collapse each pixel to its argmax
    # class, then to a binary foreground/background decision, treating
    # CvatLabelMap.FOREGROUND_CLASSES classes as foreground and everything
    # else (background, and any other class) as background -- see
    # CvatLabelMap.foreground_indices(), the same definition shared by
    # --mode test's --convert-mask-format and train/test's fg metrics
    # (mask_formats.Cvat6Format.background_classes()). Note this makes the
    # collapse a discrete decision at the model's native resolution (not a
    # continuous probability), so the alpha edges below lose some of the
    # smoothing the linear upscale used to give a continuous foreground
    # probability. --mode eval_mul instead keeps every class distinct (see
    # its branch below), so this collapse only applies to --mode eval.
    foreground_class_indices = list(CvatLabelMap().foreground_indices()) if is_multiclass_model else None

    # --mode eval_mul: colored overlay of every predicted class, generic
    # across whatever --mask-format says the checkpoint's raw channels mean
    # (background_classes()/class_names() are both per-format, everything
    # else in ClassOverlayRenderer is format-agnostic). If the format has no
    # names to offer (e.g. 'binary'), warn once up front and run without a
    # legend for the whole run, rather than failing -- coloring itself never
    # depends on having names.
    overlay_renderer = None
    class_names = None
    if mode == "eval_mul":
        overlay_renderer = ClassOverlayRenderer()
        class_names = format_obj.class_names()
        if class_names is None:
            print(f"Warning: --mask-format '{mask_format}' provides no class names; eval_mul will run without a legend")

    items_by_dir = get_eval_items(dirs)

    num_workers = min(4, os.cpu_count() or 1)

    # Probed once for the whole run: hardware capability doesn't change mid-run, and a failed
    # probe means every directory falls back to the CPU (read-webp-then-encode) path uniformly.
    nvenc_ok = probe_nvenc_available()
    print(f"NVENC hardware video encoding: {'available' if nvenc_ok else 'unavailable, falling back to CPU encoding'}")

    webp_queue: "queue.Queue[tuple[str, np.ndarray] | None]" = queue.Queue(maxsize=WEBP_QUEUE_SIZE)
    webp_threads = [threading.Thread(target=_webp_writer, args=(webp_queue,), daemon=True) for _ in range(WEBP_WRITER_THREADS)]
    for t in webp_threads:
        t.start()

    # Finalizing a directory's NVENC encode (stdin.close + wait) runs here so the next
    # directory's GPU inference can start immediately instead of blocking on ffmpeg.
    video_finalize_threads: list[threading.Thread] = []

    for dir_name, items in items_by_dir.items():
        if not items:
            print(f"[{dir_name}] no images found, skipping")
            continue

        # Use only the basename here: RunParams expansion may produce an
        # absolute path, and
        # os.path.join() would otherwise discard output_dir entirely.
        out_dir = os.path.join(output_dir, os.path.basename(dir_name))
        os.makedirs(out_dir, exist_ok=True)

        dataset = EvalDataset(items, image_size=(in_height, in_width))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            collate_fn=eval_collate,
        )

        original_queue: queue.Queue = queue.Queue(maxsize=ORIGINAL_IMAGE_QUEUE_SIZE)
        reader_thread = threading.Thread(target=_read_original_images, args=(items, original_queue), daemon=True)
        reader_thread.start()

        fps = _compute_fps([item.second for item in items])
        video_path = os.path.join(output_dir, f"{os.path.basename(dir_name)}.mp4")
        nvenc_proc = None  # opened lazily below, once the first valid frame reveals frame_size

        frame_paths: list[str] = []
        frame_size: tuple[int, int] | None = None
        for tensors, batch_items, valid_mask in tqdm(loader, desc=dir_name):
            preds = model.eval(tensors) if tensors is not None else None  # (B, 1, in_H, in_W) probabilities in [0, 1]
            if preds is not None:
                # The checkpoint's actual channel count is ground truth; --mask-format is only
                # a human-supplied claim about it. If they disagree, the collapse below would
                # silently read the wrong channel as a foreground probability (e.g. reading a
                # cvat_6 model's raw background-class channel as if it were already a fg
                # probability) instead of failing loudly, so catch the mismatch here.
                num_channels = preds.shape[1]
                expected = "> 1 (cvat_6)" if is_multiclass_model else "1 (binary)"
                if is_multiclass_model and num_channels <= 1:
                    raise RuntimeError(
                        f"--mask-format cvat_6 but '{model_path}' only outputs {num_channels} channel(s) "
                        f"(expected {expected}); wrong --model-path, or --mask-format doesn't match this checkpoint"
                    )
                if not is_multiclass_model and num_channels != 1:
                    raise RuntimeError(
                        f"--mask-format binary but '{model_path}' outputs {num_channels} channels "
                        f"(expected {expected}); wrong --model-path, or --mask-format doesn't match this checkpoint"
                    )
            if mode == "eval" and is_multiclass_model and preds is not None:
                class_idx = get_mask_format("cvat_6").raw_output_to_class_index(preds)  # (B, ...) -> (B, H, W) argmax class
                # (B, H, W) -> (B, 1, H, W) in {0., 1.}: FOREGROUND_CLASSES classes -> 1, else -> 0.
                preds = np.isin(class_idx, foreground_class_indices)[:, np.newaxis, :, :].astype(np.float32)
            # mode == "eval_mul": preds stays the raw (B, C, h, w) probabilities untouched here --
            # each item below resizes its own C channels to full resolution before collapsing, so
            # the collapse happens at the frame's native resolution instead of the model's small
            # input resolution (see the per-item branch).

            valid_idx = 0
            for item, is_valid in zip(batch_items, valid_mask):
                orig_item, image = original_queue.get()
                assert orig_item.image_path == item.image_path, "original-image reader fell out of sync"

                if not is_valid or image is None:
                    tqdm.write(f"Failed to read {item.image_path}, skipping")
                    if is_valid:
                        valid_idx += 1  # tensor decoded fine but its original-image re-read failed
                    continue

                pred = preds[valid_idx]
                valid_idx += 1

                orig_height, orig_width = image.shape[:2]
                if frame_size is None:
                    frame_size = (orig_width, orig_height)
                    if nvenc_ok:
                        nvenc_proc = open_nvenc_writer(video_path, fps, frame_size)

                if mode == "eval_mul":
                    # Resize every raw probability channel to full resolution first (linear
                    # interpolation, same sub-pixel-accurate boundary placement as --mode eval's
                    # single-channel resize below), *then* collapse to a class index -- so the
                    # class boundary is decided at the frame's native resolution, not the
                    # model's small input resolution. Works unmodified for any channel count:
                    # 1 (binary) or 6 (cvat_6) today, whatever a future format uses tomorrow.
                    raw = pred.astype(np.float32).transpose(1, 2, 0)  # (h, w, C)
                    raw_resized = cv2.resize(raw, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
                    if raw_resized.ndim == 2:  # cv2.resize squeezes a single-channel (C=1) result
                        raw_resized = raw_resized[:, :, np.newaxis]
                    raw_resized = raw_resized.transpose(2, 0, 1)[np.newaxis, ...]  # (1, C, H, W)
                    class_idx = format_obj.raw_output_to_class_index(raw_resized)[0]  # (H, W)

                    composited = overlay_renderer.render(
                        image, class_idx, format_obj.background_classes(), class_names
                    )
                    bgra = cv2.cvtColor(composited, cv2.COLOR_BGR2BGRA)
                    bgra[:, :, 3] = 255  # always opaque -- outputer.mp4's black-composite is then a no-op
                else:
                    # Upscale the continuous probability map (not an already-binarized
                    # mask) with linear interpolation for a smooth edge contour, then
                    # threshold so alpha values stay strictly 0/255.
                    prob = cv2.resize(
                        pred[0].astype(np.float32), (orig_width, orig_height), interpolation=cv2.INTER_LINEAR
                    )
                    mask = np.where(prob > 0.5, np.uint8(255), np.uint8(0))

                    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)  # original resolution, not resized
                    bgra[:, :, 3] = mask

                # Lossy WebP (with alpha preserved): far smaller than lossless PNG
                # (roughly an order of magnitude on photographic content) while
                # staying visually clean at this quality level. Handed off to the
                # writer pool instead of encoded inline, so the GPU pipeline isn't
                # stalled waiting on disk + codec time.
                out_path = os.path.join(out_dir, f"{item.second}_{item.frame}.webp")
                webp_queue.put((out_path, bgra))
                frame_paths.append(out_path)

                if nvenc_ok:
                    write_nvenc_frame(nvenc_proc, bgra, frame_size)

        reader_thread.join()

        if not frame_paths or frame_size is None:
            continue

        if nvenc_ok:
            finalize_thread = threading.Thread(target=close_nvenc_writer, args=(nvenc_proc,), daemon=True)
            finalize_thread.start()
            video_finalize_threads.append(finalize_thread)
            print(f"[{dir_name}] {len(frame_paths)} frames queued, encoding {video_path} in background (fps={fps})")
        else:
            # Webp writes are shared with future directories' work too, but nothing for a
            # later directory has been queued yet at this point in the (sequential) loop,
            # so this only waits for frames belonging to dir_name.
            webp_queue.join()
            write_video(frame_paths, video_path, fps, frame_size)
            print(f"[{dir_name}] wrote {len(frame_paths)} frames, video saved to {video_path} (fps={fps})")

    webp_queue.join()
    for _ in webp_threads:
        webp_queue.put(None)
    for t in webp_threads:
        t.join()
    for t in video_finalize_threads:
        t.join()


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string (e.g. "1h 2m 3s")."""
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes or hours:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def main() -> None:
    start_time = time.monotonic()
    params = ACTIVE_RUN
    try:
        params.validate()
    except ValueError as error:
        raise SystemExit(f"Configuration error: {error}") from None

    dirs = [str(path) for path in params.dirs]
    mask_paths = [str(path) for path in params.mask_paths]
    model_path = str(params.model_path)
    output_dir = str(params.output_dir)
    image_size = params.image_size

    if params.mode is RunMode.TRAIN:
        if params.model_class.__name__ == "UNetResNet18Mul":
            from train.multiclass import run
            run(
                dirs=dirs, mask_paths=mask_paths, model_class=params.model_class,
                epochs=params.epochs, batch_size=params.batch_size, lr=params.lr,
                model_path=model_path, val_ratio=params.val_ratio, patience=params.patience,
                lr_mode=params.lr_mode.value, lr_decrease_rate=params.lr_decrease_rate,
                lr_patience=params.lr_patience, split_seed=params.split_seed,
                preprocessor=params.preprocessors, copy_paste_count=params.copy_paste_count,
                copy_paste_seed=params.copy_paste_seed, freeze_encoder=params.freeze_encoder,
                unfreeze_patience=params.unfreeze_patience, encoder_lr_factor=params.encoder_lr_factor,
                image_size=image_size,
            )
        else:
            from train.normal import run
            run(
                dirs=dirs, model_class=params.model_class, epochs=params.epochs,
                batch_size=params.batch_size, lr=params.lr, model_path=model_path,
                val_ratio=params.val_ratio, patience=params.patience, lr_mode=params.lr_mode.value,
                lr_decrease_rate=params.lr_decrease_rate, lr_patience=params.lr_patience,
                split_seed=params.split_seed, preprocessor=params.preprocessors,
                copy_paste_count=params.copy_paste_count, copy_paste_seed=params.copy_paste_seed,
                freeze_encoder=params.freeze_encoder, mask_format=params.mask_format.value,
                mask_paths=mask_paths, unfreeze_patience=params.unfreeze_patience,
                encoder_lr_factor=params.encoder_lr_factor, image_size=image_size,
                loss_name=params.loss.value, early_stop_check=params.early_stop_check.value if params.early_stop_check else None,
                boundary_tolerance_px=params.boundary_tolerance_px,
            )
    elif params.mode is RunMode.TEST:
        if params.mask_format is MaskFormat.CVAT_6:
            from train.multiclass import run_test
        else:
            from train.normal import run_test
        run_test(
            dirs=dirs, mask_paths=mask_paths, model_class=params.model_class,
            model_path=model_path, batch_size=params.batch_size,
            test_mask_format=params.test_mask_format.value,
            convert_mask_format=params.convert_mask_format.value if params.convert_mask_format else None,
            image_size=image_size,
        )
    else:
        run_eval(
            dirs=dirs, model_class=params.model_class, model_path=model_path,
            output_dir=output_dir, mask_format=params.mask_format.value,
            batch_size=params.batch_size, mode=params.mode.value,
            expected_image_size=image_size,
        )

    elapsed = time.monotonic() - start_time
    print(f"Total time: {_format_duration(elapsed)}")


if __name__ == "__main__":
    main()
