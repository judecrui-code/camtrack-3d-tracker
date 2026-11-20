"""Sparse bundle adjustment: Levenberg-Marquardt with analytic Jacobians and the
Schur complement (points eliminated), robust Huber loss via IRLS.

Jointly refines camera poses, 3D points and optionally focal length, radial
distortion k1/k2, or per-frame focal length (zoom shots, with a smoothness prior).

Pose update is left-multiplicative: R <- exp([dw]x) R, t <- t + dt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import scipy.sparse as sp

from .geometry import project


@dataclass
class BAProblem:
    rvecs: np.ndarray          # [C,3]
    tvecs: np.ndarray          # [C,3]
    points: np.ndarray         # [P,3]
    obs_cam: np.ndarray        # [M]
    obs_pt: np.ndarray         # [M]
    obs_xy: np.ndarray         # [M,2]
    f: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    cam_focal: Optional[np.ndarray] = None  # [C] per-frame focal (variable focal mode)
    fixed_cams: Optional[np.ndarray] = None  # bool [C]
    tripod: bool = False       # translation fixed at zero; points are directions
    opt_f: bool = True
    opt_k1: bool = False
    opt_k2: bool = False
    opt_cam_focal: bool = False
    cam_order: Optional[np.ndarray] = None  # temporal order of cameras for focal smoothness
    focal_smooth_weight: float = 300.0     # residual = w * (f_i - f_j) / f
    cam_times: Optional[np.ndarray] = None  # frame index per camera: smoothness weight / sqrt(frame gap)
    focal_prior: Optional[float] = None    # weak MAP prior on focal (used only when f is estimated)
    focal_prior_weight: float = 0.0        # residual = w * ln(f / focal_prior)


def _skew_batch(v):
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], 1),
                     np.stack([v[:, 2], z, -v[:, 0]], 1),
                     np.stack([-v[:, 1], v[:, 0], z], 1)], 1)


def _rodrigues_batch(rv):
    return np.array([cv2.Rodrigues(r)[0] for r in rv]) if len(rv) else np.zeros((0, 3, 3))


def bundle_adjust(p: BAProblem, loss_scale: float = 1.0, max_nfev: int = 50, ftol: float = 2e-5, verbose: int = 0):
    C, P, M = len(p.rvecs), len(p.points), len(p.obs_cam)
    Rs = _rodrigues_batch(p.rvecs)
    ts = p.tvecs.astype(np.float64).copy()
    X = p.points.astype(np.float64).copy()
    f, k1, k2 = float(p.f), float(p.k1), float(p.k2)
    var_f = p.opt_cam_focal and p.cam_focal is not None
    cf = p.cam_focal.astype(np.float64).copy() if p.cam_focal is not None else None
    f0 = f
    fixed = p.fixed_cams if p.fixed_cams is not None else np.zeros(C, bool)
    ncp = (3 if p.tripod else 6) + (1 if var_f else 0)
    free = np.where(~fixed)[0]
    slot = -np.ones(C, np.int64)
    slot[free] = np.arange(len(free))
    intr = []
    if p.opt_f and not var_f:
        intr.append("f")
    if p.opt_k1:
        intr.append("k1")
    if p.opt_k2:
        intr.append("k2")
    ni = len(intr)
    nc = len(free) * ncp + ni
    oc, op = p.obs_cam, p.obs_pt
    smooth = None
    smooth_w = None
    if var_f and p.cam_order is not None and len(p.cam_order) > 1:
        smooth = np.stack([p.cam_order[:-1], p.cam_order[1:]], 1)
        if p.cam_times is not None:
            gap = np.abs(np.diff(np.asarray(p.cam_times)[p.cam_order])).astype(float)
            smooth_w = p.focal_smooth_weight / np.sqrt(np.maximum(gap, 1.0))
        else:
            smooth_w = np.full(len(smooth), p.focal_smooth_weight)

    def residuals(Rs, ts, X, f, k1, k2, cf):
        Xc = np.einsum("nij,nj->ni", Rs[oc], X[op]) + ts[oc]
        z = Xc[:, 2]
        zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
        x, y = Xc[:, 0] / zs, Xc[:, 1] / zs
        r2 = x * x + y * y
        d = 1 + k1 * r2 + k2 * r2 * r2
        fo = cf[oc] if var_f else f
        u = fo * d * x + p.cx
        v = fo * d * y + p.cy
        r = np.stack([u, v], 1) - p.obs_xy
        r[z <= 1e-6] = 1e3
        return r, Xc, x, y, r2, d, z

    def cost_of(r):
        e = np.linalg.norm(r, axis=1)
        d = loss_scale
        rho = np.where(e <= d, e * e, 2 * d * e - d * d)
        c = rho.sum()
        if smooth is not None:
            c += np.sum((smooth_w * (cf[smooth[:, 0]] - cf[smooth[:, 1]]) / f0) ** 2)
        if use_prior:
            c += (p.focal_prior_weight * np.log(fcur[0] / p.focal_prior)) ** 2
        return c

    use_prior = "f" in intr and p.focal_prior is not None and p.focal_prior_weight > 0
    fcur = [f]
    r, Xc, x, y, r2, d, z = residuals(Rs, ts, X, f, k1, k2, cf)
    e0 = np.linalg.norm(r, axis=1)
    cost = cost_of(r)
    lam = 1e-3
    it = 0
    nfev = 1
    status = 0
    rows2 = np.arange(2 * M)
    while it < max_nfev:
        it += 1
        # ---- analytic Jacobian blocks ----
        e = np.linalg.norm(r, axis=1)
        w = np.where(e <= loss_scale, 1.0, loss_scale / np.maximum(e, 1e-12))  # IRLS Huber weights
        valid = z > 1e-6
        w = w * valid
        sw = np.sqrt(w)
        fo = cf[oc] if var_f else np.full(M, f)
        dp = k1 + 2 * k2 * r2
        # d(u,v)/d(x,y)
        J_uv_xy = np.empty((M, 2, 2))
        J_uv_xy[:, 0, 0] = fo * (d + 2 * x * x * dp)
        J_uv_xy[:, 0, 1] = fo * (2 * x * y * dp)
        J_uv_xy[:, 1, 0] = J_uv_xy[:, 0, 1]
        J_uv_xy[:, 1, 1] = fo * (d + 2 * y * y * dp)
        iz = 1 / np.where(valid, z, 1.0)
        J_xy_Xc = np.zeros((M, 2, 3))
        J_xy_Xc[:, 0, 0] = iz
        J_xy_Xc[:, 0, 2] = -x * iz
        J_xy_Xc[:, 1, 1] = iz
        J_xy_Xc[:, 1, 2] = -y * iz
        J_proj = np.einsum("nij,njk->nik", J_uv_xy, J_xy_Xc)            # [M,2,3] d(uv)/dXc
        J_pt = np.einsum("nij,njk->nik", J_proj, Rs[oc])                 # d/dX
        RX = Xc - ts[oc]
        J_rot = np.einsum("nij,njk->nik", J_proj, -_skew_batch(RX))      # d/d(dw)
        cam_blocks = [J_rot]
        if not p.tripod:
            cam_blocks.append(J_proj)                                     # d/dt
        if var_f:
            cam_blocks.append(np.zeros((M, 2, 1)))
        Jc = np.concatenate(cam_blocks, axis=2)                           # [M,2,ncp]
        if var_f:
            Jc[:, :, -1] = np.stack([d * x, d * y], 1) * f0               # param = f_i / f0
        Ji_list = []
        for name in intr:
            if name == "f":
                Ji_list.append(np.stack([d * x, d * y], 1) * f0)          # param = f / f0
            elif name == "k1":
                Ji_list.append(np.stack([fo * x * r2, fo * y * r2], 1))
            elif name == "k2":
                Ji_list.append(np.stack([fo * x * r2 * r2, fo * y * r2 * r2], 1))
        Ji = np.stack(Ji_list, 2) if ni else np.zeros((M, 2, 0))

        # weight
        Jc = Jc * sw[:, None, None]
        Ji = Ji * sw[:, None, None]
        J_pt = J_pt * sw[:, None, None]
        rw = (r * sw[:, None]).ravel()

        # ---- sparse camera-side Jacobian [2M, nc] and point Jacobian [2M, 3P] ----
        fo_mask = slot[oc] >= 0
        rr = np.repeat(rows2.reshape(M, 2), ncp, axis=1).reshape(M, 2, ncp)
        cc = (slot[oc][:, None, None] * ncp + np.arange(ncp)[None, None, :]).repeat(2, axis=1)
        data, ri, ci = [Jc[fo_mask].ravel()], [rr[fo_mask].ravel()], [cc[fo_mask].ravel()]
        if ni:
            base = len(free) * ncp
            data.append(Ji.ravel())
            ri.append(np.repeat(rows2, ni))
            ci.append(np.tile(base + np.arange(ni), 2 * M))
        Jcs = sp.csr_matrix((np.concatenate(data), (np.concatenate(ri), np.concatenate(ci))), shape=(2 * M, nc))
        prr = np.repeat(rows2.reshape(M, 2), 3, axis=1).reshape(M, 2, 3).ravel()
        pcc = (op[:, None, None] * 3 + np.arange(3)[None, None, :]).repeat(2, axis=1).ravel()
        Jps = sp.csr_matrix((J_pt.ravel(), (prr, pcc)), shape=(2 * M, 3 * P))

        A = (Jcs.T @ Jcs).toarray()
        Wm = (Jcs.T @ Jps).tocsr()
        bc = -(Jcs.T @ rw)
        bp = -(Jps.T @ rw)
        # point blocks V_j = sum J_pt^T J_pt
        V = np.zeros((P, 3, 3))
        np.add.at(V, op, np.einsum("nki,nkj->nij", J_pt, J_pt))
        # focal smoothness prior (camera side, dense)
        if smooth is not None:
            for (a, b), sw_ in zip(smooth, smooth_w):
                ia, ib = slot[a], slot[b]
                fa = cf[a] / f0
                fb = cf[b] / f0
                res = sw_ * (fa - fb)
                ja = ia * ncp + ncp - 1 if ia >= 0 else None
                jb = ib * ncp + ncp - 1 if ib >= 0 else None
                for j1, s1 in ((ja, sw_), (jb, -sw_)):
                    if j1 is None:
                        continue
                    bc[j1] -= s1 * res
                    for j2, s2 in ((ja, sw_), (jb, -sw_)):
                        if j2 is not None:
                            A[j1, j2] += s1 * s2

        if use_prior:
            j = len(free) * ncp + intr.index("f")
            w_ = p.focal_prior_weight
            rp = w_ * np.log(f / p.focal_prior)
            jp = w_ * f0 / f
            A[j, j] += jp * jp
            bc[j] -= jp * rp
        improved = False
        for _ in range(5):
            Ad = A + lam * np.diag(np.maximum(np.diag(A), 1e-9)) + 1e-12 * np.eye(nc)
            Vd = V + lam * np.einsum("ij,ni->nij", np.eye(3), np.maximum(np.einsum("nii->ni", V), 1e-9)) + 1e-9 * np.eye(3)
            try:
                Vinv = np.linalg.inv(Vd)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            Vi = _block_diag_csr(Vinv)
            WVi = Wm @ Vi
            S = Ad - (WVi @ Wm.T).toarray()
            rhs = bc - WVi @ bp
            try:
                dc = np.linalg.solve(S, rhs)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            dpt = (Vi @ (bp - Wm.T @ dc)).reshape(P, 3)
            # apply update
            Rn, tn = Rs.copy(), ts.copy()
            cpar = dc[:len(free) * ncp].reshape(-1, ncp) if len(free) else np.zeros((0, ncp))
            dR = _rodrigues_batch(cpar[:, :3])
            Rn[free] = np.einsum("nij,njk->nik", dR, Rs[free])
            if not p.tripod:
                tn[free] = ts[free] + cpar[:, 3:6]
            cfn = cf.copy() if cf is not None else None
            if var_f:
                cfn[free] = cf[free] + cpar[:, -1] * f0
            Xn = X + dpt
            fn, k1n, k2n = f, k1, k2
            for i, name in enumerate(intr):
                v = dc[len(free) * ncp + i]
                if name == "f":
                    fn = f + v * f0
                elif name == "k1":
                    k1n = k1 + v
                elif name == "k2":
                    k2n = k2 + v
            if p.tripod:
                Xn = Xn / np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-12)
            rn, Xcn, xn, yn, r2n, dn, zn = residuals(Rn, tn, Xn, fn, k1n, k2n, cfn)
            nfev += 1
            fcur[0] = fn
            cn = cost_of(rn)
            if not (cn < cost and fn > 0):
                fcur[0] = f
            if cn < cost and fn > 0:
                rel = (cost - cn) / max(cost, 1e-12)
                Rs, ts, X, f, k1, k2, cf = Rn, tn, Xn, fn, k1n, k2n, cfn
                r, Xc, x, y, r2, d, z = rn, Xcn, xn, yn, r2n, dn, zn
                cost = cn
                lam = max(lam / 3, 1e-9)
                improved = True
                break
            lam *= 8
        if verbose:
            print(f"  BA it {it}: cost {cost:.4f} lam {lam:.1e} f={f:.2f}")
        if not improved:
            status = 2
            break
        if rel < ftol:
            status = 1
            break
    err = np.linalg.norm(r, axis=1)
    if var_f:
        f = float(np.median(cf))
    rvecs = np.array([cv2.Rodrigues(R)[0].ravel() for R in Rs]) if C else np.zeros((0, 3))
    return dict(rvecs=rvecs, tvecs=ts if not p.tripod else np.zeros_like(ts), points=X, f=f, k1=k1, k2=k2,
                cam_focal=cf, errors=err, initial_rmse=float(np.sqrt(np.mean(e0 ** 2))) if M else 0.0,
                rmse=float(np.sqrt(np.mean(err ** 2))) if M else 0.0, nfev=nfev, status=status)


def _block_diag_csr(blocks):
    n = len(blocks)
    r = np.repeat(np.arange(n) * 3, 9) + np.tile(np.repeat(np.arange(3), 3), n)
    c = np.repeat(np.arange(n) * 3, 9) + np.tile(np.tile(np.arange(3), 3), n)
    return sp.csr_matrix((blocks.ravel(), (r, c)), shape=(3 * n, 3 * n))


def reprojection_errors(rvecs, tvecs, points, obs_cam, obs_pt, obs_xy, f, cx, cy, k1, k2, cam_focal=None):
    fo = cam_focal[obs_cam] if cam_focal is not None else f
    uv, z = project(rvecs[obs_cam], tvecs[obs_cam], points[obs_pt], fo, cx, cy, k1, k2)
    e = np.linalg.norm(uv - obs_xy, axis=1)
    e[z <= 0] = np.inf
    return e


def refine_points(rvecs, tvecs, points, obs_cam, obs_pt, obs_xy, f, cx, cy, k1=0.0, k2=0.0, cam_focal=None,
                  iters=6, loss_scale=1.0):
    """Structure-only refinement (cameras fixed), vectorised Gauss-Newton per point."""
    P = len(points)
    if P == 0 or len(obs_cam) == 0:
        return points
    Rs = _rodrigues_batch(rvecs)
    X = points.astype(np.float64).copy()
    fo = cam_focal[obs_cam] if cam_focal is not None else np.full(len(obs_cam), f)
    R_o = Rs[obs_cam]
    t_o = tvecs[obs_cam]
    for _ in range(iters):
        Xc = np.einsum("nij,nj->ni", R_o, X[obs_pt]) + t_o
        z = Xc[:, 2]
        valid = z > 1e-6
        zs = np.where(valid, z, 1.0)
        x, y = Xc[:, 0] / zs, Xc[:, 1] / zs
        r2 = x * x + y * y
        d = 1 + k1 * r2 + k2 * r2 * r2
        dp = k1 + 2 * k2 * r2
        r = np.stack([fo * d * x + cx, fo * d * y + cy], 1) - obs_xy
        e = np.linalg.norm(r, axis=1)
        w = np.where(e <= loss_scale, 1.0, loss_scale / np.maximum(e, 1e-12)) * valid
        Juv = np.empty((len(x), 2, 2))
        Juv[:, 0, 0] = fo * (d + 2 * x * x * dp)
        Juv[:, 0, 1] = fo * 2 * x * y * dp
        Juv[:, 1, 0] = Juv[:, 0, 1]
        Juv[:, 1, 1] = fo * (d + 2 * y * y * dp)
        Jxy = np.zeros((len(x), 2, 3))
        Jxy[:, 0, 0] = 1 / zs
        Jxy[:, 0, 2] = -x / zs
        Jxy[:, 1, 1] = 1 / zs
        Jxy[:, 1, 2] = -y / zs
        J = np.einsum("nij,njk,nkl->nil", Juv, Jxy, R_o) * np.sqrt(w)[:, None, None]
        rw = r * np.sqrt(w)[:, None]
        V = np.zeros((P, 3, 3))
        b = np.zeros((P, 3))
        np.add.at(V, obs_pt, np.einsum("nki,nkj->nij", J, J))
        np.add.at(b, obs_pt, -np.einsum("nki,nk->ni", J, rw))
        V += 1e-6 * np.eye(3) * np.maximum(np.einsum("nii->n", V), 1e-9)[:, None, None]
        try:
            dx = np.linalg.solve(V, b[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        dx[~np.isfinite(dx).all(1)] = 0
        X += dx
        if np.max(np.abs(dx)) < 1e-7:
            break
    return X
