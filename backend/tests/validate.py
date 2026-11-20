"""End-to-end validation against synthetic ground truth.

For every scenario: track -> solve -> measure
  * reprojection error of reconstructed points in the footage (px, original resolution)
  * focal length error vs. ground truth
  * camera trajectory error after similarity (Umeyama) alignment, as % of path/scene size
  * camera rotation error (deg)
  * lens distortion recovered vs. truth

Usage: python -m tests.validate FOOTAGE_DIR [scenario ...] [--json out.json]
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camtrack.config import AnalysisSettings  # noqa: E402
from camtrack.footage import probe  # noqa: E402
from camtrack.solver.geometry import pose_center  # noqa: E402
from camtrack.solver.sfm import solve  # noqa: E402
from camtrack.tracking.tracker import run_tracking  # noqa: E402


class CLIReporter:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.lines = []

    def stage(self, name, lo, hi):
        self.log(f"[{name}]")

    def __call__(self, frac, msg=""):
        pass

    def log(self, msg):
        self.lines.append(msg)
        if self.verbose:
            print("   ", msg, flush=True)

    def check(self):
        pass


def umeyama(src, dst):
    """Similarity s,R,t minimising |s R src + t - dst|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (xs ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var
    t = mu_d - s * R @ mu_s
    return s, R, t


def evaluate(sol, gt):
    frames = sol.frames
    Rg = [np.array(gt["poses"][f]["R"]) for f in frames]
    tg = [np.array(gt["poses"][f]["t"]) for f in frames]
    cg = np.array([-R.T @ t for R, t in zip(Rg, tg)])
    Rs = [cv2.Rodrigues(r)[0] for r in sol.rvecs]
    cs = np.array([pose_center(r, t) for r, t in zip(sol.rvecs, sol.tvecs)])
    out = {}
    scale = sol.scale
    out["focal_true_px"] = gt["f"]
    out["focal_solved_px"] = sol.f / scale
    out["focal_error_pct"] = 100 * abs(sol.f / scale - gt["f"]) / gt["f"]
    out["k1_true"], out["k1_solved"] = gt["k1"], sol.k1
    if gt.get("f_per_frame"):
        ft = np.array([gt["f_per_frame"][f] for f in frames])
        fs = (sol.cam_focal / scale) if sol.cam_focal is not None else np.full(len(frames), sol.f / scale)
        out["focal_error_pct"] = float(100 * np.mean(np.abs(fs - ft) / ft))
        out["focal_true_range_px"] = [float(ft.min()), float(ft.max())]
        out["focal_solved_range_px"] = [float(fs.min()), float(fs.max())]
        out["focal_solved_px"] = float(np.median(fs))
    if gt.get("tripod"):
        # compare relative rotations to frame 0
        errs = []
        for i in range(len(frames)):
            Rrel_g = Rg[i] @ Rg[0].T
            Rrel_s = Rs[i] @ Rs[0].T
            d = Rrel_g @ Rrel_s.T
            errs.append(np.degrees(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1))))
        out["rot_error_deg_mean"] = float(np.mean(errs))
        out["rot_error_deg_max"] = float(np.max(errs))
        return out
    # Alignment world_gt = s R world_solved + t. R from camera orientations (robust even for
    # straight-line paths, where a position-only fit leaves the roll about the path undetermined).
    acc = sum(Rgi.T @ Rsi for Rgi, Rsi in zip(Rg, Rs))
    U, _, Vt = np.linalg.svd(acc)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    rc = (R @ cs.T).T
    mc, mg = rc.mean(0), cg.mean(0)
    s = float(np.sum((rc - mc) * (cg - mg)) / max(np.sum((rc - mc) ** 2), 1e-12))
    t = mg - s * mc
    aligned = s * rc + t
    pos_err = np.linalg.norm(aligned - cg, axis=1)
    path = float(np.sum(np.linalg.norm(np.diff(cg, axis=0), axis=1)))
    out["path_length_gt"] = path
    out["pos_error_mean_pct_of_path"] = float(100 * pos_err.mean() / max(path, 1e-9))
    out["pos_error_max_pct_of_path"] = float(100 * pos_err.max() / max(path, 1e-9))
    rerr = []
    for Ri, Rgi in zip(Rs, Rg):
        # world rotation: solved cam-to-world mapped by R
        Rw = Ri @ R.T
        d = Rw @ Rgi.T
        rerr.append(np.degrees(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1))))
    out["rot_error_deg_mean"] = float(np.mean(rerr))
    out["rot_error_deg_max"] = float(np.max(rerr))
    return out


def moving_object_check(sol, gt):
    """Outdoor scenario: a car-like box moves independently (x = -8 + 0.15*frame).
    Count solved points whose observations lie on the car's projected silhouette in
    most of their frames - these would be moving-object contamination."""
    from itertools import product
    corners0 = np.array([[cx, cy, cz] for cx, cy, cz in product([-1.25, 1.25], [-0.6, 0.6], [-0.7, 0.7])]) + [0, 8, 0.7]
    hulls = {}
    for fr in sol.frames:
        R = np.array(gt["poses"][fr]["R"])
        t = np.array(gt["poses"][fr]["t"])
        c = corners0 + [-8 + 0.15 * fr, 0, 0]
        pc = c @ R.T + t
        if (pc[:, 2] <= 0).any():
            continue
        uv = pc[:, :2] / pc[:, 2:3] * gt["f"] + [gt["cx"], gt["cy"]]
        hulls[fr] = cv2.convexHull(uv.astype(np.float32))
    on_car = 0
    for obs in sol.point_obs:
        inside = 0
        for fr, x, y in obs:
            h = hulls.get(fr)
            if h is not None:
                X, Y = (x + 0.5) / sol.scale - 0.5, (y + 0.5) / sol.scale - 0.5
                if cv2.pointPolygonTest(h, (float(X), float(Y)), True) > 3:
                    inside += 1
        if inside > 0.5 * len(obs):
            on_car += 1
    return dict(points_on_moving_object=on_car, frames_with_moving_object=len(hulls))


def run(footage_dir, name, settings=None, verbose=False):
    path = os.path.join(footage_dir, f"{name}.mp4")
    gt = json.load(open(os.path.join(footage_dir, f"{name}.gt.json")))
    info = probe(path)
    s = AnalysisSettings.from_dict(settings or {})
    rep = CLIReporter(verbose)
    t0 = time.time()
    ts = run_tracking(path, info, s, rep)
    t1 = time.time()
    sol = solve(ts, s, rep)
    t2 = time.time()
    ev = evaluate(sol, gt)
    if name == "outdoor":
        ev.update(moving_object_check(sol, gt))
    res = dict(scenario=name, description=gt["description"], mode=sol.mode,
               tracks=ts.stats["total_tracks"], mean_track_len=round(ts.stats["mean_track_length"], 1),
               dynamic_tracks=ts.stats["dynamic_tracks"], reacquired=ts.stats["reacquired"],
               points=sol.stats["n_points"], solved_frames=sol.stats["n_solved_frames"], frames=sol.stats["n_frames"],
               reproj_mean_px=sol.stats["mean_error_px"] / sol.scale, reproj_median_px=sol.stats["median_error_px"] / sol.scale,
               track_time_s=round(t1 - t0, 1), solve_time_s=round(t2 - t1, 1), warnings=sol.warnings, **ev)
    return res, ts, sol


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    out_json = None
    if "--json" in sys.argv:
        out_json = sys.argv[sys.argv.index("--json") + 1]
        args = [a for a in args if a != out_json]
    footage = args[0]
    names = args[1:] or ["forward", "sideways", "orbit", "flat", "outdoor", "indoor", "lowtexture", "tripod", "zoom"]
    results = []
    for n in names:
        print(f"=== {n}", flush=True)
        try:
            r, _, _ = run(footage, n, settings={"variable_focal": True} if n == "zoom" else None, verbose="-v" in sys.argv)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = dict(scenario=n, error=f"{type(e).__name__}: {e}")
        results.append(r)
        print(json.dumps(r, indent=1, default=float), flush=True)
    if out_json:
        json.dump(results, open(out_json, "w"), indent=1, default=float)
