"""Fast unit tests for the geometric core. Run: python -m tests.test_unit  (or pytest)."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camtrack.solver.bundle import BAProblem, bundle_adjust, refine_points  # noqa: E402
from camtrack.solver.geometry import (bearings, project, ransac_rotation, triangulate_multiview,  # noqa: E402
                                      undistort_normalized, rot_to_quat, quat_to_rot)
from camtrack.export.scene_data import FLIP  # noqa: E402
from camtrack.solver.geometry import robust_essential, robust_fundamental, robust_homography  # noqa: E402
from camtrack.config import AnalysisSettings, effective_frame_step  # noqa: E402


def _cams(C=40):
    rv, tv = np.zeros((C, 3)), np.zeros((C, 3))
    for i in range(C):
        rv[i] = [0.002 * i, 0.012 * i, 0.001 * i]
        R = cv2.Rodrigues(rv[i])[0]
        tv[i] = -R @ np.array([0.06 * i, 0.01 * i, 0.02 * i])
    return rv, tv


def test_projection_matches_opencv():
    rng = np.random.default_rng(0)
    X = rng.uniform([-3, -2, 5], [3, 2, 15], (200, 3))
    r, t = np.array([0.1, -0.2, 0.05]), np.array([0.3, -0.1, 0.5])
    K = np.array([[900, 0, 639.5], [0, 900, 359.5], [0, 0, 1.0]])
    dist = np.array([-0.08, 0.02, 0, 0])
    ref, _ = cv2.projectPoints(X, r, t, K, dist)
    ours, _ = project(np.tile(r, (200, 1)), np.tile(t, (200, 1)), X, 900, 639.5, 359.5, -0.08, 0.02)
    assert np.abs(ref.reshape(-1, 2) - ours).max() < 1e-6


def test_undistort_inverts_distortion():
    xy = np.random.default_rng(1).uniform([0, 0], [1280, 720], (300, 2))
    xn = undistort_normalized(xy, 800, 639.5, 359.5, -0.08, 0.01)
    x, y = xn[:, 0], xn[:, 1]
    r2 = x * x + y * y
    d = 1 - 0.08 * r2 + 0.01 * r2 * r2
    back = np.stack([800 * d * x + 639.5, 800 * d * y + 359.5], 1)
    assert np.abs(back - xy).max() < 1e-4


def test_bundle_adjustment_recovers_focal_and_distortion():
    rng = np.random.default_rng(2)
    rv, tv = _cams()
    P = 1500
    X = rng.uniform([-5, -3, 8], [6, 3, 20], (P, 3))
    oc, op = np.meshgrid(np.arange(len(rv)), np.arange(P), indexing="ij")
    oc, op = oc.ravel(), op.ravel()
    uv, z = project(rv[oc], tv[oc], X[op], 1000, 639.5, 359.5, -0.06, 0)
    ok = (uv[:, 0] > 0) & (uv[:, 0] < 1280) & (uv[:, 1] > 0) & (uv[:, 1] < 720) & (rng.random(len(oc)) < 0.6)
    oc, op, uv = oc[ok], op[ok], uv[ok] + rng.normal(0, 0.3, (ok.sum(), 2))
    fixed = np.zeros(len(rv), bool)
    fixed[0] = True
    rv0, tv0 = rv + rng.normal(0, 0.002, rv.shape), tv + rng.normal(0, 0.01, tv.shape)
    rv0[0], tv0[0] = rv[0], tv[0]
    prob = BAProblem(rvecs=rv0, tvecs=tv0, points=X + rng.normal(0, 0.05, X.shape), obs_cam=oc, obs_pt=op, obs_xy=uv,
                     f=1080, cx=639.5, cy=359.5, fixed_cams=fixed, opt_f=True, opt_k1=True)
    r = bundle_adjust(prob, loss_scale=1.0, max_nfev=60)
    assert abs(r["f"] - 1000) < 5, r["f"]
    assert abs(r["k1"] + 0.06) < 0.005, r["k1"]
    assert np.median(r["errors"]) < 0.4


def test_refine_points_and_triangulation():
    rng = np.random.default_rng(3)
    rv, tv = _cams(10)
    X = rng.uniform([-2, -2, 6], [2, 2, 12], (50, 3))
    Rs = [cv2.Rodrigues(r)[0] for r in rv]
    for j in range(5):
        xn = np.array([(R @ X[j] + t)[:2] / (R @ X[j] + t)[2] for R, t in zip(Rs, tv)])
        Xt = triangulate_multiview(Rs, list(tv), xn)
        assert np.linalg.norm(Xt - X[j]) < 1e-6
    oc = np.repeat(np.arange(10), 50)
    op = np.tile(np.arange(50), 10)
    uv, _ = project(rv[oc], tv[oc], X[op], 900, 640, 360)
    Xr = refine_points(rv, tv, X + 0.1, oc, op, uv, 900, 640, 360)
    assert np.abs(Xr - X).max() < 1e-4


def test_rotation_ransac():
    rng = np.random.default_rng(4)
    R = cv2.Rodrigues(np.array([0.1, 0.3, -0.05]))[0]
    a = bearings(rng.uniform(-0.5, 0.5, (200, 2)))
    b = a @ R.T
    b[:20] = bearings(rng.uniform(-0.5, 0.5, (20, 2)))  # outliers
    Re, inl = ransac_rotation(a, b, 1e-3, 200, rng)
    assert np.degrees(np.arccos(np.clip((np.trace(Re @ R.T) - 1) / 2, -1, 1))) < 1e-3
    assert inl[20:].all()


def test_quaternion_roundtrip_and_blender_camera_axes():
    R = cv2.Rodrigues(np.array([0.4, -1.1, 0.3]))[0]
    assert np.allclose(quat_to_rot(rot_to_quat(R)), R)
    # an OpenCV camera looking along +Y world (Z-up), x right: Blender camera must look down its -Z
    Rc = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])  # world->cam
    Rbl = Rc.T @ FLIP
    assert np.allclose(Rbl @ [0, 0, -1], [0, 1, 0])   # viewing direction +Y
    assert np.allclose(Rbl @ [0, 1, 0], [0, 0, 1])    # camera up = world up


def test_robust_estimators_never_raise_on_degenerate_input():
    rng = np.random.default_rng(5)
    base = rng.uniform(0, 700, (200, 2))
    line = np.c_[np.linspace(0, 500, 100), np.linspace(0, 500, 100)]
    nan = np.where(base > 650, np.nan, base)
    cases = [(base, base.copy()), (base, base + 1e-4), (line, line + 3), (np.tile([[100., 100.]], (60, 1)),
             np.tile([[101., 100.]], (60, 1))), (base[:3], base[:3]), (nan, base + 1), (base[:0], base[:0]),
             (np.full((50, 2), np.inf), base[:50])]
    for a, b in cases:
        for fn, thr in ((robust_fundamental, 1.5), (robust_essential, 0.002), (robust_homography, 1.5)):
            M, mask = fn(a, b, thr)           # must not raise
            assert (M is None) == (mask is None)
            if mask is not None:
                assert len(mask) == len(a)


def test_robust_estimators_still_find_correct_models():
    rng = np.random.default_rng(6)
    X = rng.uniform([-2, -2, 5], [2, 2, 12], (300, 3))
    R = cv2.Rodrigues(np.array([0.02, 0.05, 0.01]))[0]
    t = np.array([0.3, 0.0, 0.05])
    xa = X[:, :2] / X[:, 2:]
    Xc = X @ R.T + t
    xb = Xc[:, :2] / Xc[:, 2:]
    xb[:30] += rng.uniform(-0.2, 0.2, (30, 2))            # outliers
    E, mask = robust_essential(xa, xb, 1e-3)
    assert E is not None and mask[30:].mean() > 0.95
    F, mask = robust_fundamental(xa * 900 + 640, xb * 900 + 640, 1.0)
    assert F is not None and mask[30:].mean() > 0.9


def test_auto_frame_step():
    d = AnalysisSettings.from_dict({})
    assert d.frame_step == 1                                # default: every frame, never skipped
    assert effective_frame_step(d, 60, 1101) == 1
    s = AnalysisSettings.from_dict({"frame_step": 0})       # Auto is opt-in
    assert effective_frame_step(s, 30, 120) == 1
    assert effective_frame_step(s, 60, 600) == 2
    assert effective_frame_step(s, 59.94, 1101) == 3
    assert 1101 / effective_frame_step(s, 59.94, 1101) <= 400
    assert effective_frame_step(AnalysisSettings.from_dict({"frame_step": 4}), 60, 1101) == 4


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
