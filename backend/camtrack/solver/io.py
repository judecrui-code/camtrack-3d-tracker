"""Solution (de)serialisation and conversion to original-resolution units."""
from __future__ import annotations

import json

import numpy as np

from .sfm import Solution


def save_solution(sol: Solution, path: str):
    lens = np.array([len(o) for o in sol.point_obs], np.int64)
    flat = np.array([x for obs in sol.point_obs for x in obs], np.float64).reshape(-1, 3)
    meta = dict(mode=sol.mode, engine=sol.engine, proc_w=sol.proc_w, proc_h=sol.proc_h, scale=sol.scale, f=sol.f,
                cx=sol.cx, cy=sol.cy, k1=sol.k1, k2=sol.k2, frames=sol.frames, frame_status=sol.frame_status,
                stats=sol.stats, warnings=sol.warnings)
    np.savez_compressed(path, meta=np.array([json.dumps(meta)]), rvecs=sol.rvecs, tvecs=sol.tvecs,
                        cam_focal=sol.cam_focal if sol.cam_focal is not None else np.zeros(0),
                        point_ids=sol.point_ids, points=sol.points, colors=sol.colors, point_error=sol.point_error,
                        point_nobs=sol.point_nobs, point_angle=sol.point_angle, obs_lens=lens, obs=flat,
                        frame_error=sol.frame_error, frame_npoints=sol.frame_npoints)


def load_solution(path: str) -> Solution:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"][0]))
    obs, o = [], 0
    flat = z["obs"]
    for n in z["obs_lens"]:
        obs.append([(int(a), float(b), float(c)) for a, b, c in flat[o:o + n]])
        o += n
    cf = z["cam_focal"]
    return Solution(mode=meta["mode"], engine=meta["engine"], proc_w=meta["proc_w"], proc_h=meta["proc_h"],
                    scale=meta["scale"], f=meta["f"], cx=meta["cx"], cy=meta["cy"], k1=meta["k1"], k2=meta["k2"],
                    frames=meta["frames"], rvecs=z["rvecs"], tvecs=z["tvecs"], frame_status=meta["frame_status"],
                    cam_focal=cf if len(cf) else None, point_ids=z["point_ids"], points=z["points"],
                    colors=z["colors"], point_error=z["point_error"], point_nobs=z["point_nobs"],
                    point_angle=z["point_angle"], point_obs=obs, frame_error=z["frame_error"],
                    frame_npoints=z["frame_npoints"], stats=meta["stats"], warnings=meta["warnings"])


def to_original_px(sol: Solution, x_proc: float) -> float:
    """Processing-res pixel coordinate -> original-res pixel coordinate (pixel centres at integers)."""
    return (x_proc + 0.5) / sol.scale - 0.5


def intrinsics_original(sol: Solution, width: int, height: int) -> dict:
    s = sol.scale
    out = dict(width=width, height=height, f=sol.f / s, cx=to_original_px(sol, sol.cx), cy=to_original_px(sol, sol.cy),
               k1=sol.k1, k2=sol.k2)
    out["cam_focal"] = (sol.cam_focal / s).tolist() if sol.cam_focal is not None else None
    return out
