"""Analysis settings with sensible automatic defaults.

Every field here is exposed in the UI. Values are validated/clamped in
``AnalysisSettings.from_dict`` so a malformed request can never put the
pipeline into a nonsensical state.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Any, Optional

SOLVE_METHODS = ("auto", "typical", "flat", "tripod")
DISTORTION_MODES = ("none", "k1", "k1k2", "manual")
ENGINES = ("native", "colmap")
QUALITY_PRESETS = ("draft", "standard", "high")


@dataclass
class AnalysisSettings:
    # --- engine ---
    engine: str = "native"              # native sequential SfM, or COLMAP incremental mapper
    # --- footage ---
    frame_start: Optional[int] = None   # inclusive, original frame numbers
    frame_end: Optional[int] = None     # inclusive
    processing_width: int = 0           # 0 = auto (long edge 1280, never upscaled)
    frame_step: int = 1                 # track every Nth frame (1 = every frame, the default); 0 = auto (opt-in: ~30 fps, <=400 frames)
    # --- feature tracking ---
    detection_quality: str = "standard"  # draft / standard / high
    max_points: int = 3000              # max simultaneous tracked features
    min_track_length: int = 8           # frames
    tracking_sensitivity: float = 0.5   # 0..1; higher = more permissive matching / more features
    # --- solve ---
    solve_method: str = "auto"          # auto / typical / flat / tripod
    variable_focal: bool = False        # zoom shots: per-frame focal length
    focal_length_px: Optional[float] = None  # known focal in *original* pixels; None = estimate
    focal_length_mm: Optional[float] = None  # alternative: 35mm-style focal in mm (sensor size = image long edge)
    sensor_width_mm: float = 36.0           # sensor size along the image's long edge (36 = full-frame equivalent)
    distortion: str = "k1"              # none / k1 / k1k2 / manual
    k1: float = 0.0                     # used when distortion == manual
    k2: float = 0.0
    reprojection_threshold: float = 2.0  # px (processing resolution); outlier cutoff
    min_triangulation_angle: float = 1.0  # degrees

    @classmethod
    def from_dict(cls, d: Optional[dict[str, Any]]) -> "AnalysisSettings":
        s = cls()
        if not d:
            return s
        names = {f.name for f in fields(cls)}
        for k, v in d.items():
            if k in names and v is not None and v != "":
                setattr(s, k, v)
            elif k in ("frame_start", "frame_end", "focal_length_px", "focal_length_mm") and v in (None, ""):
                setattr(s, k, None)
        s.engine = s.engine if s.engine in ENGINES else "native"
        s.solve_method = s.solve_method if s.solve_method in SOLVE_METHODS else "auto"
        s.distortion = s.distortion if s.distortion in DISTORTION_MODES else "k1"
        s.detection_quality = s.detection_quality if s.detection_quality in QUALITY_PRESETS else "standard"
        s.processing_width = max(0, int(s.processing_width))
        s.frame_step = max(0, min(10, int(s.frame_step)))
        s.max_points = max(100, min(20000, int(s.max_points)))
        s.min_track_length = max(3, min(500, int(s.min_track_length)))
        s.tracking_sensitivity = float(min(1.0, max(0.0, float(s.tracking_sensitivity))))
        s.reprojection_threshold = float(min(20.0, max(0.3, float(s.reprojection_threshold))))
        s.min_triangulation_angle = float(min(10.0, max(0.1, float(s.min_triangulation_angle))))
        s.sensor_width_mm = float(max(1.0, float(s.sensor_width_mm)))
        s.k1 = float(s.k1)
        s.k2 = float(s.k2)
        s.variable_focal = bool(s.variable_focal)
        if s.focal_length_px is not None:
            s.focal_length_px = float(s.focal_length_px) if float(s.focal_length_px) > 0 else None
        if s.focal_length_mm is not None:
            s.focal_length_mm = float(s.focal_length_mm) if float(s.focal_length_mm) > 0 else None
        if s.frame_start is not None:
            s.frame_start = int(s.frame_start)
        if s.frame_end is not None:
            s.frame_end = int(s.frame_end)
        return s

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


QUALITY_PARAMS = {
    #            shi-tomasi quality, min distance (px @1280), LK window, pyramid levels
    "draft":    dict(quality=0.010, min_dist=14, win=15, levels=3, redetect_every=5),
    "standard": dict(quality=0.005, min_dist=10, win=21, levels=4, redetect_every=3),
    "high":     dict(quality=0.002, min_dist=7, win=25, levels=5, redetect_every=1),
}


AUTO_TARGET_FPS = 30.0
AUTO_MAX_FRAMES = 400


def effective_frame_step(settings: "AnalysisSettings", fps: float, n_frames: int) -> int:
    """Resolve frame_step. 0 (auto) tracks ~30 fps and caps the clip at ~400 tracked frames,
    because solve time grows steeply with the number of frames while camera motion between
    consecutive 60 fps frames is tiny. Explicit values (1..10) are used as given."""
    if settings.frame_step >= 1:
        return settings.frame_step
    step = max(1, int(round(fps / AUTO_TARGET_FPS)))
    while n_frames / step > AUTO_MAX_FRAMES and step < 10:
        step += 1
    return step
