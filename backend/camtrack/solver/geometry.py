"""Camera model & multi-view geometry helpers.

Camera model (OpenCV convention): world-to-camera ``Xc = R @ Xw + t``;
camera axes x right, y down, z forward. Pinhole + radial distortion::

    x = Xc/Zc, y = Yc/Zc, r2 = x^2+y^2
    d = 1 + k1 r2 + k2 r2^2
    u = f d x + cx,  v = f d y + cy

This is identical to OpenCV's model with (k1, k2, 0, 0) and to Blender/libmv's
"polynomial" model, so distortion coefficients transfer directly.
"""
from __future__ import annotations

import cv2
import numpy as np


def rodrigues_rotate(rvec: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Rotate pts[N,3] by per-row axis-angle rvec[N,3] (vectorised)."""
    theta = np.linalg.norm(rvec, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(theta > 1e-12, rvec / theta, 0.0)
    cos = np.cos(theta)
    sin = np.sin(theta)
    dot = np.sum(k * pts, axis=1, keepdims=True)
    cross = np.cross(k, pts)
    small = theta < 1e-12
    rot = pts * cos + cross * sin + k * dot * (1 - cos)
    # first-order for tiny angles
    rot = np.where(small, pts + np.cross(rvec, pts), rot)
    return rot


def project(rvec, tvec, pts, f, cx, cy, k1=0.0, k2=0.0):
    """Vectorised projection. rvec,tvec,pts: [N,3]; f,k1,k2 scalars or [N]. Returns [N,2], depth[N]."""
    pc = rodrigues_rotate(rvec, pts) + tvec
    z = pc[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    x = pc[:, 0] / zs
    y = pc[:, 1] / zs
    r2 = x * x + y * y
    d = 1 + k1 * r2 + k2 * r2 * r2
    return np.stack([f * d * x + cx, f * d * y + cy], axis=1), z


def K_matrix(f, cx, cy):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float64)


def undistort_normalized(xy: np.ndarray, f, cx, cy, k1=0.0, k2=0.0) -> np.ndarray:
    """Pixel -> undistorted normalised image coordinates [N,2]."""
    if len(xy) == 0:
        return np.zeros((0, 2))
    pts = np.asarray(xy, np.float64).reshape(-1, 1, 2)
    if k1 == 0 and k2 == 0:
        out = (pts.reshape(-1, 2) - [cx, cy]) / f
        return out
    out = cv2.undistortPoints(pts, K_matrix(f, cx, cy), np.array([k1, k2, 0, 0], np.float64), None, None, None,
                                  (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-9))
    return out.reshape(-1, 2)


def pose_center(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    return -R.T @ np.asarray(tvec, np.float64)


def triangulate_multiview(Rs: list, ts: list, xn: np.ndarray):
    """Linear (DLT) triangulation from M views. xn: [M,2] normalised coords."""
    A = []
    for R, t, (x, y) in zip(Rs, ts, xn):
        P = np.hstack([R, t.reshape(3, 1)])
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.asarray(A)
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def ray_angle_deg(c1, c2, X) -> float:
    a = X - c1
    b = X - c2
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1))))


def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R minimising |R a_i - b_i| for unit vectors (rows)."""
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    return Vt.T @ D @ U.T


def ransac_rotation(a: np.ndarray, b: np.ndarray, thresh_rad: float, iters: int = 300, rng=None):
    """Robust rotation between bearing sets a->b (pure-rotation camera)."""
    rng = rng or np.random.default_rng(0)
    n = len(a)
    if n < 3:
        return None, np.zeros(n, bool)
    best_inl = None
    best_cnt = -1
    cos_t = np.cos(thresh_rad)
    for _ in range(iters):
        idx = rng.choice(n, 2, replace=False)
        R = kabsch(a[idx], b[idx])
        cosang = np.sum((a @ R.T) * b, axis=1)
        inl = cosang > cos_t
        c = int(inl.sum())
        if c > best_cnt:
            best_cnt, best_inl = c, inl
    if best_cnt < 3:
        return None, np.zeros(n, bool)
    R = kabsch(a[best_inl], b[best_inl])
    cosang = np.sum((a @ R.T) * b, axis=1)
    inl = cosang > cos_t
    if inl.sum() >= 3:
        R = kabsch(a[inl], b[inl])
    return R, inl


def bearings(xn: np.ndarray) -> np.ndarray:
    v = np.c_[xn, np.ones(len(xn))]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def rot_to_quat(R):
    """Rotation matrix -> (w, x, y, z)."""
    R = np.asarray(R)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_to_rot(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def fit_plane(pts: np.ndarray):
    """Least squares plane. Returns (centroid, unit normal, rms)."""
    c = pts.mean(axis=0)
    _, s, vt = np.linalg.svd(pts - c)
    n = vt[-1]
    rms = s[-1] / np.sqrt(max(1, len(pts)))
    return c, n, float(rms)


def ransac_plane(pts: np.ndarray, thresh: float, iters: int = 500, rng=None):
    rng = rng or np.random.default_rng(1)
    n = len(pts)
    if n < 3:
        return None
    best = None
    best_cnt = 0
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p = pts[idx]
        nrm = np.cross(p[1] - p[0], p[2] - p[0])
        ln = np.linalg.norm(nrm)
        if ln < 1e-12:
            continue
        nrm /= ln
        d = np.abs((pts - p[0]) @ nrm)
        inl = d < thresh
        c = int(inl.sum())
        if c > best_cnt:
            best_cnt, best = c, inl
    if best is None:
        return None
    c, nrm, rms = fit_plane(pts[best])
    return c, nrm, best
