from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def laplacian_variance_bgr(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_blurry_bgr(image_bgr: np.ndarray, threshold: float = 40.0) -> bool:
    return laplacian_variance_bgr(image_bgr) < threshold


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    stride: int = 10,
    max_frames: int | None = None,
    resize: int | None = None,
) -> List[Path]:
    """Extract frames from a video with a fixed stride.

    This function performs frame extraction only. Face detection/alignment can be
    added upstream or replaced with an external face-cropper such as RetinaFace,
    MTCNN, or MediaPipe depending on the project environment.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    saved: List[Path] = []
    frame_idx = 0
    saved_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            if resize is not None:
                frame = cv2.resize(frame, (resize, resize))
            out_path = output_dir / f"{video_path.stem}_frame_{saved_idx:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved.append(out_path)
            saved_idx += 1
            if max_frames is not None and saved_idx >= max_frames:
                break
        frame_idx += 1

    cap.release()
    return saved


def iter_images(root: str | Path) -> Iterable[Path]:
    root = Path(root)
    for p in root.rglob("*"):
        if p.is_file() and is_image_file(p):
            yield p
