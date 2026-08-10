"""Video output helpers: encode a sequence of transparent PNG frames into an mp4."""

from __future__ import annotations

import shutil
import subprocess

import cv2
import numpy as np
from tqdm import tqdm


def _composite_on_black(bgra: np.ndarray) -> np.ndarray:
    """mp4 has no alpha channel, so composite the transparent background onto black."""
    bgr = bgra[:, :, :3].copy()
    bgr[bgra[:, :, 3] == 0] = 0
    return bgr


def probe_nvenc_available() -> bool:
    """One-shot check (cache the result yourself) for whether ffmpeg's h264_nvenc actually works here."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return False

    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel", "error",
        "-f", "lavfi",
        "-i", "color=black:size=256x256:rate=1",  # below ~145px, NVENC rejects the frame dimensions
        "-frames:v", "1",
        "-c:v", "h264_nvenc",
        "-f", "null",
        "-",
    ]
    try:
        result = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return False
    return result.returncode == 0


def open_nvenc_writer(video_path: str, fps: float, size: tuple[int, int]) -> subprocess.Popen:
    """Start a persistent ffmpeg process that NVENC-encodes raw BGR frames piped to its stdin.

    Only call this after ``probe_nvenc_available()`` has returned True. Feed frames with
    ``write_nvenc_frame()`` and finish with ``close_nvenc_writer()``.
    """
    width, height = size
    even_width = width - (width % 2)
    even_height = height - (height % 2)

    cmd = [
        shutil.which("ffmpeg"),
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vf", f"crop={even_width}:{even_height}:0:0",
        "-pix_fmt", "yuv420p",
        "-c:v", "h264_nvenc",
        "-preset", "p5",
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", "20",
        "-movflags", "+faststart",
        video_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def write_nvenc_frame(proc: subprocess.Popen, bgra: np.ndarray, size: tuple[int, int]) -> None:
    """Composite one BGRA frame onto black and feed it to a writer opened with ``open_nvenc_writer()``."""
    bgr = _composite_on_black(bgra)
    if (bgr.shape[1], bgr.shape[0]) != size:
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_NEAREST)
    proc.stdin.write(bgr.tobytes())


def close_nvenc_writer(proc: subprocess.Popen) -> None:
    """Finish a writer opened with ``open_nvenc_writer()``; logs a warning if ffmpeg reported failure."""
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        tqdm.write(f"Warning: NVENC ffmpeg process exited with code {proc.returncode}")


def _write_video_ffmpeg(frame_paths: list[str], video_path: str, fps: float, size: tuple[int, int]) -> bool:
    """Encode frames into an H.264 mp4 using the system ffmpeg binary.

    Returns True on success, False if ffmpeg is unavailable or fails (caller
    should fall back to the OpenCV writer in that case).
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return False

    width, height = size
    # libx264 + yuv420p requires even width/height.
    even_width = width - (width % 2)
    even_height = height - (height % 2)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vf", f"crop={even_width}:{even_height}:0:0",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        # "slower" spends more effort searching for redundancy, shrinking the
        # file for the same visual quality (crf keeps the default quality
        # target; nothing here reduces quality, only file size).
        "-preset", "slower",
        "-crf", "20",
        "-movflags", "+faststart",
        video_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for frame_path in frame_paths:
            bgra = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            if bgra is None:
                tqdm.write(f"Failed to read {frame_path}, skipping frame")
                continue
            bgr = _composite_on_black(bgra)
            if (bgr.shape[1], bgr.shape[0]) != size:
                bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_NEAREST)
            proc.stdin.write(bgr.tobytes())
    finally:
        proc.stdin.close()
        proc.wait()

    return proc.returncode == 0


def _write_video_opencv(frame_paths: list[str], video_path: str, fps: float, size: tuple[int, int]) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")
    try:
        for frame_path in frame_paths:
            bgra = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            if bgra is None:
                tqdm.write(f"Failed to read {frame_path}, skipping frame")
                continue
            bgr = _composite_on_black(bgra)
            if (bgr.shape[1], bgr.shape[0]) != size:
                bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_NEAREST)
            writer.write(bgr)
    finally:
        writer.release()


def write_video(frame_paths: list[str], video_path: str, fps: float, size: tuple[int, int]) -> None:
    """Encode `frame_paths` (transparent PNGs) into an mp4 at `video_path`.

    Tries the system ffmpeg binary (H.264/libx264) first, falling back to the
    OpenCV `mp4v` writer if ffmpeg is unavailable or fails.
    """
    if _write_video_ffmpeg(frame_paths, video_path, fps, size):
        return
    tqdm.write("ffmpeg (libx264) unavailable or failed, falling back to OpenCV mp4v encoder")
    _write_video_opencv(frame_paths, video_path, fps, size)
