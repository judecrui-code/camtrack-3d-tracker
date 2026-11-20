"""Footage probing, frame-accurate sequential decoding and the display frame cache.

Frame numbering: frame ``n`` is the n-th decoded frame of the source (0-based),
exactly as the decoder delivers it. That numbering is preserved through tracking,
solving and export (Blender gets ``n + frame_offset``).

The original file is never re-encoded. The display cache stores each frame as a
high-quality JPEG (q=95, full resolution) purely for browser display; analysis
always decodes the original file.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, asdict
from fractions import Fraction
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mxf", ".mpg", ".mpeg", ".wmv", ".flv", ".ts", ".mts", ".m2ts"}


@dataclass
class FootageInfo:
    path: str
    filename: str
    width: int
    height: int
    fps: float
    fps_num: int
    fps_den: int
    frame_count: int
    duration: float
    codec: str
    pixel_aspect: float = 1.0

    def to_dict(self):
        return asdict(self)


def _ffprobe(path: str) -> dict:
    from .paths import ffprobe_exe
    exe = ffprobe_exe()
    if not exe:
        return {}
    try:
        kw = {"creationflags": 0x08000000} if os.name == "nt" else {}  # CREATE_NO_WINDOW
        out = subprocess.run(
            [exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name,sample_aspect_ratio,duration:format=duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=60, **kw)
        return json.loads(out.stdout or "{}")
    except Exception:
        return {}


def probe(path: str, count_frames: bool = True) -> FootageInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video (unsupported codec or container)")
    ok, frame = cap.read()
    if not ok or frame is None:
        raise ValueError("Video contains no decodable frames")
    h, w = frame.shape[:2]  # decoded (orientation-corrected) size is authoritative
    cv_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cv_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    meta = _ffprobe(path)
    st = (meta.get("streams") or [{}])[0]
    fps_frac = None
    for key in ("avg_frame_rate", "r_frame_rate"):
        v = st.get(key)
        if v and v != "0/0":
            try:
                fr = Fraction(v)
                if fr > 0:
                    fps_frac = fr
                    break
            except (ValueError, ZeroDivisionError):
                pass
    if fps_frac is None:
        fps_frac = Fraction(cv_fps).limit_denominator(1001) if cv_fps > 0 else Fraction(24, 1)
    sar = st.get("sample_aspect_ratio", "1:1")
    try:
        a, b = sar.split(":")
        pixel_aspect = float(a) / float(b) if float(b) else 1.0
        if pixel_aspect <= 0:
            pixel_aspect = 1.0
    except Exception:
        pixel_aspect = 1.0

    frame_count = cv_count
    if count_frames:
        # exact count by decoding headers (grab doesn't convert pixels)
        n = 1
        while cap.grab():
            n += 1
        frame_count = n
    cap.release()
    fps = float(fps_frac)
    return FootageInfo(
        path=path, filename=os.path.basename(path), width=w, height=h, fps=fps,
        fps_num=fps_frac.numerator, fps_den=fps_frac.denominator, frame_count=frame_count,
        duration=frame_count / fps if fps > 0 else 0.0, codec=st.get("codec_name", "unknown"),
        pixel_aspect=pixel_aspect)


def iter_frames(path: str, start: int = 0, end: Optional[int] = None, step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Sequentially decode frames [start, end] (inclusive) yielding (frame_number, BGR)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video")
    idx = 0
    try:
        while idx < start:
            if not cap.grab():
                return
            idx += 1
        while end is None or idx <= end:
            if (idx - start) % step == 0:
                ok, frame = cap.read()
                if not ok:
                    return
                yield idx, frame
            else:
                if not cap.grab():
                    return
            idx += 1
    finally:
        cap.release()


def processing_size(width: int, height: int, requested_width: int) -> tuple[int, int, float]:
    """Return (w, h, scale) for analysis. Never upscales. scale = proc / original."""
    if requested_width and requested_width > 0:
        target = min(requested_width, width)
    else:
        long_edge = max(width, height)
        target = width if long_edge <= 1280 else int(round(width * 1280 / long_edge))
    scale = target / width
    pw = int(round(width * scale))
    ph = int(round(height * scale))
    return pw, ph, pw / width


def write_image(path: str, img: np.ndarray, params=None) -> None:
    """cv2.imwrite replacement that also works for non-ASCII paths on Windows."""
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img, params or [])
    if not ok:
        raise IOError(f"could not encode image for {path}")
    with open(path, "wb") as fh:
        fh.write(buf.tobytes())


class FrameCache:
    """Full-resolution JPEG cache used for browser display only."""

    def __init__(self, directory: str, quality: int = 95):
        self.dir = directory
        self.quality = quality
        os.makedirs(directory, exist_ok=True)

    def path(self, n: int) -> str:
        return os.path.join(self.dir, f"{n:06d}.jpg")

    def has(self, n: int) -> bool:
        return os.path.exists(self.path(n))

    def build(self, video_path: str, progress: Callable[[float, str], None], cancelled: Callable[[], bool], total: int):
        for n, frame in iter_frames(video_path):
            if cancelled():
                return
            p = self.path(n)
            if not os.path.exists(p):
                write_image(p + ".tmp.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                os.replace(p + ".tmp.jpg", p)
            if n % 5 == 0:
                progress(min(1.0, (n + 1) / max(1, total)), f"Caching frame {n}/{total}")
