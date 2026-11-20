"""Scene definition on top of a camera solve: ground plane, origin, scale and
user-created objects (nulls, solids, 3D anchors, ground-plane objects).

Everything the user places is stored in *raw solve coordinates* (the solver's
normalised frame: OpenCV axes of the first solved camera). The world transform
(raw -> world, Z-up, metres-ish) is derived on demand, so changing the ground
plane/origin/scale never requires re-solving the camera.

World convention (identical to Blender): right-handed, +Z up.
"""
from __future__ import annotations

import uuid
from typing import Optional

import cv2
import numpy as np

from ..solver.geometry import fit_plane, pose_center, ransac_plane

OBJECT_TYPES = ("null", "solid", "anchor", "ground_plane")


def default_scene() -> dict:
    return dict(ground=None, origin=None, scale=dict(mode="camera_height", value=1.6, a=None, b=None),
                objects=[], auto_ground=None)


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def camera_axes_raw(sol, fi: int):
    """Camera centre and axes (right, down, forward) in raw coords for sequence index fi."""
    R = cv2.Rodrigues(np.asarray(sol.rvecs[fi], float))[0]
    c = pose_center(sol.rvecs[fi], sol.tvecs[fi])
    return c, R[0], R[1], R[2]


def estimate_auto_ground(sol) -> Optional[dict]:
    """Find the dominant plane roughly perpendicular to the first camera's up vector."""
    if sol.mode == "tripod" or len(sol.points) < 30:
        return None
    rel = sol.reliable_mask(2.0)
    pts = sol.points[rel] if rel.sum() >= 30 else sol.points
    c0, right, down, fwd = camera_axes_raw(sol, 0)
    up = -down
    extent = np.median(np.linalg.norm(pts - np.median(pts, 0), axis=1))
    thresh = 0.02 * extent
    rng = np.random.default_rng(3)
    best = None
    remaining = np.ones(len(pts), bool)
    for _ in range(4):  # peel off the top planes
        idx = np.where(remaining)[0]
        if len(idx) < 30:
            break
        res = ransac_plane(pts[idx], thresh, 800, rng)
        if res is None:
            break
        c, n, inl = res
        if inl.sum() < max(20, 0.08 * len(pts)):
            break
        if n @ up < 0:
            n = -n
        align = float(n @ up)
        # plane must lie below the camera
        below = float((c0 - c) @ n) > 0
        if align > 0.6 and below:
            score = inl.sum() * align
            if best is None or score > best[0]:
                best = (score, c, n, int(inl.sum()))
        remaining[idx[inl]] = False
    if best is None:
        return None
    _, c, n, cnt = best
    # origin: projection of the point cloud centre (seen in frame 0) onto the plane
    vis = pts[((pts - c0) @ fwd) > 0]
    centre = np.median(vis if len(vis) else pts, 0)
    o = centre - ((centre - c) @ n) * n
    return dict(point=o.tolist(), normal=n.tolist(), frame=int(sol.frames[0]), x_hint=right.tolist(),
                inliers=cnt, source="auto")


def world_transform(sol, scene: dict) -> dict:
    """Return dict(R: 3x3 raw->world rotation, o: raw origin, s: scale, M: 4x4 raw->world)."""
    ground = scene.get("ground") or scene.get("auto_ground")
    c0, right0, down0, fwd0 = camera_axes_raw(sol, 0)
    if ground is None:
        z = -down0
        xh = right0
        o = c0 - z * 1.6 * (10.0 / 10.0)  # camera assumed 1.6 units above an implicit floor
        p_ground = o
    else:
        z = _unit(ground["normal"])
        fi = _frame_index(sol, ground.get("frame"))
        cam = camera_axes_raw(sol, fi)[0]
        if (cam - np.asarray(ground["point"])) @ z < 0:
            z = -z
        xh = np.asarray(ground.get("x_hint") or camera_axes_raw(sol, fi)[1], float)
        p_ground = np.asarray(ground["point"], float)
        o = p_ground
    if scene.get("origin") is not None:
        o = np.asarray(scene["origin"], float)
    x = xh - (xh @ z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [0, 0, 1]) if abs(z[2]) < 0.9 else np.cross(z, [1, 0, 0])
    x = _unit(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    s = _scale(sol, scene, z, p_ground)
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = -s * R @ o
    return dict(R=R, o=o, s=s, M=M, ground_point=p_ground, ground_normal=z)


def _scale(sol, scene, z, p_ground) -> float:
    sc = scene.get("scale") or {}
    mode = sc.get("mode", "camera_height")
    val = float(sc.get("value") or 1.0)
    if sol.mode == "tripod":
        return 1.0 if mode != "factor" else val
    if mode == "factor":
        return val
    if mode == "distance" and sc.get("a") is not None and sc.get("b") is not None:
        ids = list(sol.point_ids)
        try:
            pa = sol.points[ids.index(int(sc["a"]))]
            pb = sol.points[ids.index(int(sc["b"]))]
            d = np.linalg.norm(pa - pb)
            return val / d if d > 1e-9 else 1.0
        except ValueError:
            return 1.0
    # camera height above ground at the first frame
    c0 = camera_axes_raw(sol, 0)[0]
    h = abs((c0 - p_ground) @ z)
    return val / h if h > 1e-6 else 1.0


def _frame_index(sol, frame) -> int:
    if frame is None:
        return 0
    try:
        return sol.frames.index(int(frame))
    except ValueError:
        return int(np.argmin(np.abs(np.asarray(sol.frames) - int(frame))))


def refine_plane_from_points(sol, point_ids, fallback_point, fallback_normal, grow=False):
    """Least-squares plane through the selected reconstructed points (real 3D geometry).

    With ``grow`` the plane is extended by region growing: every reconstructed point
    lying on the seed plane (distance gate) is added and the plane refitted, iterating
    while the normal stays within 12 degrees of the seed. This turns a local
    8-point estimate into a fit over the whole visible surface.
    Returns (point, normal, rms, n_points_used).
    """
    idmap = {int(p): i for i, p in enumerate(sol.point_ids)}
    sel = [idmap[int(i)] for i in point_ids if int(i) in idmap]
    p = np.asarray(fallback_point, float)
    if len(sel) < 3:
        return p, _unit(fallback_normal), None, 0
    c, n, rms = fit_plane(sol.points[sel])
    if np.dot(n, fallback_normal) < 0:
        n = -n
    used = len(sel)
    if grow and len(sol.points) > 20:
        rel = sol.reliable_mask(2.0)
        pts = sol.points[rel] if rel.sum() > 20 else sol.points
        extent = float(np.median(np.linalg.norm(pts - np.median(pts, 0), axis=1)))
        seed_n = n.copy()
        tol = max(3 * rms, 0.006 * extent)
        radius = max(0.6 * extent, 4 * float(np.max(np.linalg.norm(sol.points[sel] - c, axis=1))))
        for _ in range(6):
            d = np.abs((pts - c) @ n)
            near = np.linalg.norm((pts - c) - np.outer((pts - c) @ n, n), axis=1) < radius
            inl = (d < tol) & near
            if inl.sum() < max(10, used):
                break
            c2, n2, rms2 = fit_plane(pts[inl])
            if n2 @ seed_n < 0:
                n2 = -n2
            if np.degrees(np.arccos(np.clip(n2 @ seed_n, -1, 1))) > 12 or rms2 > 0.01 * extent:
                break  # growing into other surfaces: keep the last good plane
            c, n, rms, used = c2, n2, rms2, int(inl.sum())
            tol = max(2.5 * rms, 0.004 * extent)
            radius *= 1.5
    # keep the clicked target position but move it onto the refined plane
    p = p - ((p - c) @ n) * n
    return p, n, rms, used


def make_object(kind: str, point, normal, frame, size, sol, name=None) -> dict:
    if kind not in OBJECT_TYPES:
        raise ValueError(f"unknown object type {kind}")
    fi = _frame_index(sol, frame)
    cam_c, right, down, fwd = camera_axes_raw(sol, fi)
    n = _unit(normal)
    if (cam_c - np.asarray(point)) @ n < 0:
        n = -n
    return dict(id=uuid.uuid4().hex[:8], type=kind, name=name or f"{kind.replace('_', ' ').title()}",
                position=list(map(float, point)), normal=list(map(float, n)), x_hint=list(map(float, right)),
                frame=int(frame) if frame is not None else int(sol.frames[0]), size=float(size))


def object_world(obj: dict, T: dict) -> dict:
    """World-space location, rotation matrix (columns = local axes) and size for an object."""
    R, s, o = T["R"], T["s"], T["o"]
    p = s * R @ (np.asarray(obj["position"]) - o)
    z = R @ _unit(obj["normal"])
    xh = R @ np.asarray(obj["x_hint"])
    x = xh - (xh @ z) * z
    x = _unit(x) if np.linalg.norm(x) > 1e-6 else _unit(np.cross([0, 1, 0], z))
    y = np.cross(z, x)
    if obj["type"] == "null":
        # nulls are world-aligned like After Effects' "Create Null" (position only, upright)
        rot = np.eye(3)
    else:
        rot = np.stack([x, y, z], 1)
    return dict(location=p.tolist(), rotation=rot.tolist(), size=float(obj["size"]), type=obj["type"],
                name=obj["name"], id=obj["id"])
