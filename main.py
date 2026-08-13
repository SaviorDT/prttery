"""CLI entry point.

    uv run main.py --mode train --dirs foo bar
    uv run main.py --mode eval  --dirs foo bar
"""

from __future__ import annotations

import argparse
import os
import queue
import random
import threading
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader.normal import EvalDataset, EvalItem, eval_collate, get_eval_items
from mask_formats import get_mask_format, to_binary
from models import get_model_class
from outputer.mp4 import close_nvenc_writer, open_nvenc_writer, probe_nvenc_available, write_nvenc_frame, write_video

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
MULTI_CLASS_MODELS = {"unet_resnet18_mul"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UNet background removal")
    parser.add_argument("--mode", choices=["train", "eval", "test"], required=True)
    parser.add_argument("--model", choices=["unet", "unet_resnet18", "unet_resnet18_mul"], default="unet_resnet18_mul")
    parser.add_argument("--dirs", nargs="+", required=True, help="Folder names under ./data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="./result", help="Directory for outputs (model, videos, frames)")
    parser.add_argument("--model-path", default=None, help="Defaults to '{output-dir}/model.onnx'")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr-mode", choices=["default", "decreasing"], default="decreasing")
    parser.add_argument("--lr-decrease-rate", type=float, default=0.5, help="Used only when --lr-mode is 'decreasing'")
    parser.add_argument("--lr-patience", type=int, default=3, help="Used only when --lr-mode is 'decreasing'")
    parser.add_argument(
        "--freeze-encoder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Freeze the model's pretrained encoder (weights + BatchNorm/LayerNorm running stats) "
            "at the start of training, then automatically unfreeze it once val loss goes "
            "--unfreeze-patience epochs without improving (its post-unfreeze lr is set by "
            "--encoder-lr-factor). Only valid for models with a pretrained encoder (unet_resnet18, "
            "unet_resnet18_mul); defaults to on for those, off for 'unet'. Requires validation data. "
            "Pass --no-freeze-encoder to always train directly/unfrozen regardless of --model."
        ),
    )
    parser.add_argument(
        "--unfreeze-patience",
        type=int,
        default=4,
        help="Epochs without val-loss improvement before unfreezing (see --freeze-encoder); must be <= --patience",
    )
    parser.add_argument(
        "--encoder-lr-factor",
        type=float,
        default=0.1,
        help="Encoder's lr after unfreezing, as a fraction of the decoder's lr at that moment (see --freeze-encoder)",
    )
    parser.add_argument(
        "--split-seed",
        default="random",
        help="Integer seed for the train/val split, or 'random' to generate one",
    )
    parser.add_argument(
        "--mask-format",
        choices=["binary", "cvat_5"],
        default="cvat_5",
        help=(
            "What a model's output represents, for every mode (train/test/eval alike) -- "
            "the single source of truth for whether a --model's output needs the multi-class "
            "collapse below. 'binary': single-channel foreground probability; training reads "
            "./data/{dir}_mask/ grayscale mask images (default). 'cvat_5': 5-class per-pixel "
            "softmax; training/testing parse ground truth from --mask-path CVAT 1.1 XML "
            "file(s). Either way, requires (and is required by) --model unet_resnet18_mul."
        ),
    )
    parser.add_argument(
        "--mask-path",
        nargs="+",
        default=None,
        help=(
            "CVAT 1.1 XML annotation file(s); required when --mode is train/test and "
            "--mask-format is 'cvat_5', or when --mode is test and --test-mask-format is 'cvat_5'"
        ),
    )
    parser.add_argument(
        "--test-mask-format",
        choices=["binary", "cvat_5"],
        default=None,
        help=(
            "Format of the --mode test ground-truth masks on disk, if different from "
            "--mask-format (which continues to describe the model's own native format). "
            "Defaults to 'binary' when omitted. Only used with --mode test; if it differs "
            "from --mask-format, --convert-mask-format must also be given."
        ),
    )
    parser.add_argument(
        "--convert-mask-format",
        choices=["binary"],
        default=None,
        help=(
            "Convert both the test-set ground truth and the model's prediction into this format "
            "before computing --mode test metrics, regardless of whether they already match "
            "(when they do match, per-format metrics are reported as usual unless this is set). "
            "Currently only 'binary' is supported. Only used with --mode test."
        ),
    )
    parser.add_argument(
        "--preprocessor",
        nargs="+",
        default=None,
        choices=["copy_paste"],
        help="Train-set data-augmentation preprocessor(s) to apply, in order. Only used with --mode train.",
    )
    parser.add_argument(
        "--copy-paste-count",
        type=int,
        default=40,
        help="Number of synthetic copy-paste samples to add to the train set; required when --preprocessor includes copy_paste",
    )
    parser.add_argument(
        "--copy-paste-seed",
        default="random",
        help="Integer seed for copy_paste's randomness, or 'random' to generate one; only used when --preprocessor includes copy_paste",
    )
    args = parser.parse_args()
    if args.model_path is None:
        args.model_path = os.path.join(args.output_dir, "model.onnx")
    if args.split_seed == "random":
        args.split_seed = random.randint(0, 2**31 - 1)
        print(f"Using random split seed: {args.split_seed}")
    else:
        try:
            args.split_seed = int(args.split_seed)
        except ValueError:
            parser.error("--split-seed must be an integer or 'random'")

    # --model/--mask-format must agree only in --mode train, where --model determines the
    # actual network architecture being built, so it must match what --mask-format says the
    # ground truth/output looks like. --mode eval/test instead load a saved checkpoint from
    # --model-path -- --model plays no architectural role there (it's only used to instantiate
    # a wrapper for .load()/.eval(), identical across every --model choice) -- so --mask-format
    # alone decides which code path handles a checkpoint's output, exactly as eval already did
    # internally; --model keeps a default that's easy to forget to override, so requiring it to
    # also agree there would reject valid eval/test invocations for no reason.
    is_multiclass_model = args.model in MULTI_CLASS_MODELS
    if args.mode == "train":
        if is_multiclass_model and args.mask_format != "cvat_5":
            parser.error(f"--model {args.model} requires --mask-format cvat_5")
        if args.mask_format == "cvat_5" and not is_multiclass_model:
            parser.error(
                f"--mask-format cvat_5 requires a multi-class --model (one of {sorted(MULTI_CLASS_MODELS)}), "
                f"got '{args.model}'"
            )

    # --freeze-encoder only means anything for a model with a pretrained encoder to freeze;
    # None (neither --freeze-encoder nor --no-freeze-encoder given) resolves to "on" for those
    # models and "off" for 'unet', so direct/unfrozen training stays 'unet's default while
    # pretrained models opt into freeze-then-unfreeze transfer learning by default -- an explicit
    # --no-freeze-encoder always falls back to direct training regardless of --model.
    has_pretrained_encoder = get_model_class(args.model).HAS_PRETRAINED_ENCODER
    if args.freeze_encoder is None:
        args.freeze_encoder = has_pretrained_encoder
    elif args.freeze_encoder and not has_pretrained_encoder:
        parser.error(f"--freeze-encoder requires a model with a pretrained encoder, got '{args.model}'")
    if args.freeze_encoder and args.unfreeze_patience > args.patience:
        parser.error(
            f"--unfreeze-patience ({args.unfreeze_patience}) must be <= --patience ({args.patience}); "
            "otherwise early stopping would end training before the encoder ever gets a chance to unfreeze"
        )

    # --test-mask-format / --convert-mask-format only mean anything for --mode test, where
    # ground-truth masks (--test-mask-format) get compared against a model's prediction
    # (native --mask-format), optionally through a common --convert-mask-format.
    if args.test_mask_format is not None and args.mode != "test":
        parser.error("--test-mask-format is only used with --mode test")
    if args.convert_mask_format is not None and args.mode != "test":
        parser.error("--convert-mask-format is only used with --mode test")
    if args.test_mask_format is None:
        args.test_mask_format = "binary"
    if args.mode == "test" and args.test_mask_format != args.mask_format and args.convert_mask_format is None:
        parser.error(
            f"--test-mask-format {args.test_mask_format} differs from --mask-format {args.mask_format}; "
            "specify --convert-mask-format to say what format to compare them in"
        )

    # --mask-path only locates ground-truth CVAT annotations for train/test; eval has no
    # ground truth to read, so it plays no part in --mode eval regardless of --mask-format.
    # Which flag decides "is the ground truth actually cvat_5" differs by mode: --mode train
    # always reads it via --mask-format (there's no --test-mask-format there); --mode test
    # reads ground truth via --test-mask-format specifically (defaulted above to
    # 'binary' when not given), regardless of the model's own native --mask-format.
    if args.mode == "train":
        needs_cvat_path = args.mask_format == "cvat_5"
    elif args.mode == "test":
        needs_cvat_path = args.test_mask_format == "cvat_5"
    else:
        needs_cvat_path = False

    if args.mode == "train":
        if needs_cvat_path and not args.mask_path:
            parser.error("--mask-format cvat_5 requires --mask-path")
        if not needs_cvat_path and args.mask_path:
            parser.error("--mask-path is only used when --mask-format is 'cvat_5'")
    elif args.mode == "test":
        if needs_cvat_path and not args.mask_path:
            parser.error("--test-mask-format cvat_5 requires --mask-path")
        if not needs_cvat_path and args.mask_path:
            parser.error("--mask-path is only used when --test-mask-format is 'cvat_5'")
    elif args.mask_path:
        parser.error("--mask-path is only used with --mode train/test")

    # --preprocessor only augments the train dataset built inside --mode train's
    # run(); it plays no part in test/eval, and --copy-paste-count/--copy-paste-seed
    # only mean anything when --preprocessor actually includes copy_paste.
    use_copy_paste = args.preprocessor is not None and "copy_paste" in args.preprocessor
    if args.preprocessor is not None and args.mode != "train":
        parser.error("--preprocessor is only used with --mode train")
    if use_copy_paste and args.copy_paste_count is None:
        parser.error("--preprocessor copy_paste requires --copy-paste-count")
    if use_copy_paste:
        if args.copy_paste_seed == "random":
            args.copy_paste_seed = random.randint(0, 2**31 - 1)
            print(f"Using random copy-paste seed: {args.copy_paste_seed}")
        else:
            try:
                args.copy_paste_seed = int(args.copy_paste_seed)
            except ValueError:
                parser.error("--copy-paste-seed must be an integer or 'random'")

    return args


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
    dirs: list[str], model_name: str, model_path: str, output_dir: str, mask_format: str, batch_size: int = 8
) -> None:
    model = get_model_class(model_name)()
    model.load(model_path)

    in_height, in_width = model.input_shape[0], model.input_shape[1]

    # A multi-class model's raw output is (B, num_classes, H, W) softmax
    # probabilities. Output format here stays the same alpha-matte pipeline
    # as the binary models: collapse each pixel to its argmax class, then to
    # a binary foreground/background decision (background -> 0, else -> 1),
    # matching the same argmax-based collapse --mode test uses for
    # --convert-mask-format. Note this makes the collapse a discrete decision
    # at the model's native resolution (not a continuous probability), so the
    # alpha edges below lose some of the smoothing the linear upscale used to
    # give a continuous foreground probability.
    #
    # --mask-format (validated against --model in parse_args), not --model, is the source of
    # truth here: it's what determines the collapse below, matching how it's already the
    # source of truth for --mode train/test.
    is_multiclass_model = mask_format == "cvat_5"

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

        # Use only the basename here: dir_name may be an absolute path (e.g.
        # when the shell expands a "*" glob before argparse sees it), and
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
                # cvat_5 model's raw background-class channel as if it were already a fg
                # probability) instead of failing loudly, so catch the mismatch here.
                num_channels = preds.shape[1]
                expected = "> 1 (cvat_5)" if is_multiclass_model else "1 (binary)"
                if is_multiclass_model and num_channels <= 1:
                    raise RuntimeError(
                        f"--mask-format cvat_5 but '{model_path}' only outputs {num_channels} channel(s) "
                        f"(expected {expected}); wrong --model-path, or --mask-format doesn't match this checkpoint"
                    )
                if not is_multiclass_model and num_channels != 1:
                    raise RuntimeError(
                        f"--mask-format binary but '{model_path}' outputs {num_channels} channels "
                        f"(expected {expected}); wrong --model-path, or --mask-format doesn't match this checkpoint"
                    )
            if is_multiclass_model and preds is not None:
                class_idx = get_mask_format("cvat_5").raw_output_to_class_index(preds)  # (B, ...) -> (B, H, W) argmax class
                preds = to_binary(class_idx, "cvat_5")[:, np.newaxis, :, :]  # (B, H, W) -> (B, 1, H, W) in {0., 1.}

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

    args = parse_args()

    if args.mode == "train":
        if args.model in MULTI_CLASS_MODELS:
            from train.multiclass import run

            run(
                dirs=args.dirs,
                mask_paths=args.mask_path,
                model_name=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                model_path=args.model_path,
                val_ratio=args.val_ratio,
                patience=args.patience,
                lr_mode=args.lr_mode,
                lr_decrease_rate=args.lr_decrease_rate,
                lr_patience=args.lr_patience,
                split_seed=args.split_seed,
                preprocessor=args.preprocessor,
                copy_paste_count=args.copy_paste_count,
                copy_paste_seed=args.copy_paste_seed,
                freeze_encoder=args.freeze_encoder,
                unfreeze_patience=args.unfreeze_patience,
                encoder_lr_factor=args.encoder_lr_factor,
            )
        else:
            from train.normal import run

            run(
                dirs=args.dirs,
                model_name=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                model_path=args.model_path,
                val_ratio=args.val_ratio,
                patience=args.patience,
                lr_mode=args.lr_mode,
                lr_decrease_rate=args.lr_decrease_rate,
                lr_patience=args.lr_patience,
                split_seed=args.split_seed,
                preprocessor=args.preprocessor,
                copy_paste_count=args.copy_paste_count,
                copy_paste_seed=args.copy_paste_seed,
                freeze_encoder=args.freeze_encoder,
                unfreeze_patience=args.unfreeze_patience,
                encoder_lr_factor=args.encoder_lr_factor,
            )
    elif args.mode == "test":
        # Driven by --mask-format, not --model, for the same reason the validation above only
        # requires them to agree in --mode train: a checkpoint here comes from --model-path, and
        # --mask-format is what actually says whether it's a multi-class or binary checkpoint.
        if args.mask_format == "cvat_5":
            from train.multiclass import run_test

            run_test(
                dirs=args.dirs,
                mask_paths=args.mask_path,
                model_name=args.model,
                model_path=args.model_path,
                batch_size=args.batch_size,
                test_mask_format=args.test_mask_format,
                convert_mask_format=args.convert_mask_format,
            )
        else:
            from train.normal import run_test

            run_test(
                dirs=args.dirs,
                mask_paths=args.mask_path,
                model_name=args.model,
                model_path=args.model_path,
                batch_size=args.batch_size,
                test_mask_format=args.test_mask_format,
                convert_mask_format=args.convert_mask_format,
            )
    else:
        run_eval(
            dirs=args.dirs,
            model_name=args.model,
            model_path=args.model_path,
            output_dir=args.output_dir,
            mask_format=args.mask_format,
            batch_size=args.batch_size,
        )

    elapsed = time.monotonic() - start_time
    print(f"Total time: {_format_duration(elapsed)}")


if __name__ == "__main__":
    main()
