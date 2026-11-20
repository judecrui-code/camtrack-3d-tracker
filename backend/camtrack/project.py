"""Project persistence and the processing pipeline glue (ingest / track / solve)."""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from typing import Optional

import numpy as np

from .config import AnalysisSettings
from .footage import FootageInfo, FrameCache, probe
from .scene.scene import default_scene, estimate_auto_ground
from .solver.io import load_solution, save_solution
from .solver.sfm import Solution, solve as native_solve
from .tracking.tracker import TrackSet, run_tracking


class Project:
    def __init__(self, root: str, pid: str):
        self.id = pid
        self.dir = os.path.join(root, pid)
        self.lock = threading.RLock()
        self.meta: dict = {}
        self._solution: Optional[Solution] = None
        self._tracks: Optional[TrackSet] = None

    # ---- paths ----
    @property
    def video_path(self):
        return os.path.join(self.dir, self.meta["video_file"])

    @property
    def frames_dir(self):
        return os.path.join(self.dir, "frames")

    @property
    def tracks_path(self):
        return os.path.join(self.dir, "tracks.npz")

    @property
    def solution_path(self):
        return os.path.join(self.dir, "solution.npz")

    def save(self):
        with self.lock:
            tmp = os.path.join(self.dir, "project.json.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.meta, fh, indent=1)
            os.replace(tmp, os.path.join(self.dir, "project.json"))

    def load(self):
        with open(os.path.join(self.dir, "project.json"), encoding="utf-8") as fh:
            self.meta = json.load(fh)
        return self

    @property
    def info(self) -> FootageInfo:
        return FootageInfo(**self.meta["info"])

    @property
    def settings(self) -> AnalysisSettings:
        return AnalysisSettings.from_dict(self.meta.get("settings"))

    @property
    def scene(self) -> dict:
        return self.meta.setdefault("scene", default_scene())

    @property
    def solution(self) -> Optional[Solution]:
        if self._solution is None and os.path.exists(self.solution_path):
            self._solution = load_solution(self.solution_path)
        return self._solution

    @property
    def tracks(self) -> Optional[TrackSet]:
        if self._tracks is None and os.path.exists(self.tracks_path):
            self._tracks = TrackSet.load(self.tracks_path)
        return self._tracks

    def summary(self) -> dict:
        m = self.meta
        return dict(id=self.id, name=m.get("name"), info=m.get("info"), settings=m.get("settings"),
                    state=m.get("state"), created=m.get("created"), scene=self.scene,
                    deleted_points=len(m.get("deleted_points", [])),
                    has_tracks=os.path.exists(self.tracks_path), has_solution=os.path.exists(self.solution_path),
                    frames_cached=m.get("frames_cached", False), last_track_stats=m.get("track_stats"))


class ProjectStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.projects: dict[str, Project] = {}
        for pid in sorted(os.listdir(root)):
            if os.path.exists(os.path.join(root, pid, "project.json")):
                try:
                    self.projects[pid] = Project(root, pid).load()
                except Exception:
                    pass

    def create(self, src_path: str, filename: str, move: bool = True) -> Project:
        pid = uuid.uuid4().hex[:10]
        p = Project(self.root, pid)
        os.makedirs(p.dir)
        ext = os.path.splitext(filename)[1].lower() or ".mp4"
        dst = os.path.join(p.dir, "source" + ext)
        (shutil.move if move else shutil.copyfile)(src_path, dst)
        try:
            info = probe(dst)
        except Exception:
            shutil.rmtree(p.dir, ignore_errors=True)
            raise
        info.filename = filename
        info.path = dst
        p.meta = dict(id=pid, name=os.path.splitext(filename)[0], video_file="source" + ext, info=info.to_dict(),
                      settings=AnalysisSettings(frame_start=0, frame_end=info.frame_count - 1).to_dict(),
                      state="imported", created=time.time(), scene=default_scene(), deleted_points=[],
                      frames_cached=False)
        p.save()
        self.projects[pid] = p
        return p

    def get(self, pid: str) -> Optional[Project]:
        return self.projects.get(pid)

    def delete(self, pid: str):
        p = self.projects.pop(pid, None)
        if p:
            shutil.rmtree(p.dir, ignore_errors=True)


# ---------------------------------------------------------------- pipeline

def ingest(project: Project, rep):
    rep.stage("Caching frames for display", 0.0, 1.0)
    cache = FrameCache(project.frames_dir)
    cache.build(project.video_path, rep, rep.cancelled, project.info.frame_count)
    project.meta["frames_cached"] = True
    project.save()


def _run_solver(ts: TrackSet, settings: AnalysisSettings, rep, exclude, project: Project) -> Solution:
    if settings.engine == "colmap":
        from .solver.colmap_engine import solve_colmap
        return solve_colmap(project, ts, settings, rep, exclude)
    return native_solve(ts, settings, rep, exclude)


def analyze(project: Project, settings: AnalysisSettings, rep, retrack: bool = True):
    project.meta["settings"] = settings.to_dict()
    project.save()
    if retrack or project.tracks is None:
        rep.stage("Detecting and tracking features", 0.0, 0.35)
        ts = run_tracking(project.video_path, project.info, settings, rep)
        ts.save(project.tracks_path)
        project._tracks = ts
        project.__dict__.pop("_frame_index", None)  # per-frame 2D track index is rebuilt lazily
        project.meta["deleted_points"] = []
        project.meta["track_stats"] = {k: v for k, v in ts.stats.items() if k != "per_frame_active"}
        project.meta["state"] = "tracked"
        project.save()
        rep.log(f"Tracking: {ts.stats['total_tracks']} tracks, mean length {ts.stats['mean_track_length']:.1f} frames, "
                f"{ts.stats['dynamic_tracks']} dynamic (moving-object) tracks rejected, "
                f"{ts.stats['reacquired']} re-acquired after occlusion")
    ts = project.tracks
    _solve_into(project, ts, settings, rep, lo=0.35 if retrack else 0.0)


class _SubReporter:
    """Maps solver stages into a sub-range of the parent job."""

    def __init__(self, rep, lo, hi):
        self.rep, self.lo, self.hi = rep, lo, hi

    def stage(self, name, a, b):
        self.rep.stage(name, self.lo + (self.hi - self.lo) * a, self.lo + (self.hi - self.lo) * b)

    def __call__(self, frac, msg=""):
        self.rep(frac, msg)

    def log(self, msg):
        self.rep.log(msg)

    def check(self):
        self.rep.check()

    def cancelled(self):
        return self.rep.cancelled()


def _solve_into(project: Project, ts: TrackSet, settings: AnalysisSettings, rep, lo=0.0):
    sub = _SubReporter(rep, lo, 1.0)
    sub.stage("Solving camera", 0.0, 0.05)
    exclude = set(int(i) for i in project.meta.get("deleted_points", []))
    old = project.solution
    sol = _run_solver(ts, settings, sub, exclude, project)
    # carry the user's scene across re-solves via a similarity fit on shared points
    scene = project.scene
    if old is not None and (scene.get("ground") or scene.get("objects") or scene.get("origin")):
        _rebase_scene(scene, old, sol)
    scene["auto_ground"] = estimate_auto_ground(sol)
    save_solution(sol, project.solution_path)
    project._solution = sol
    project.meta["state"] = "solved"
    project.meta["solve_stats"] = sol.stats
    project.meta["solved_deleted"] = sorted(exclude)
    project.save()
    rep.log(f"Solve complete: {sol.stats['n_points']} points, mean reprojection error "
            f"{sol.stats['mean_error_px'] / sol.scale:.3f}px (original resolution)")
    for w in sol.warnings:
        rep.log("WARNING: " + w)


def resolve(project: Project, settings: AnalysisSettings, rep):
    if project.tracks is None:
        raise RuntimeError("No tracking data yet: run Analyze first")
    project.meta["settings"] = settings.to_dict()
    _solve_into(project, project.tracks, settings, rep)


def _rebase_scene(scene: dict, old: Solution, new: Solution):
    common, ia, ib = np.intersect1d(old.point_ids, new.point_ids, return_indices=True)
    if len(common) < 10 or old.mode == "tripod" or new.mode == "tripod":
        scene["ground"] = None
        scene["origin"] = None
        scene["objects"] = []
        return
    A, B = old.points[ia], new.points[ib]
    mu_a, mu_b = A.mean(0), B.mean(0)
    xa, xb = A - mu_a, B - mu_b
    U, D, Vt = np.linalg.svd(xb.T @ xa / len(A))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / ((xa ** 2).sum() / len(A))
    mp = lambda p: (s * R @ (np.asarray(p) - mu_a) + mu_b).tolist()
    mv = lambda v: (R @ np.asarray(v)).tolist()
    if scene.get("ground"):
        g = scene["ground"]
        g["point"], g["normal"], g["x_hint"] = mp(g["point"]), mv(g["normal"]), mv(g["x_hint"])
    if scene.get("origin") is not None:
        scene["origin"] = mp(scene["origin"])
    for o in scene.get("objects", []):
        o["position"], o["normal"], o["x_hint"] = mp(o["position"]), mv(o["normal"]), mv(o["x_hint"])
