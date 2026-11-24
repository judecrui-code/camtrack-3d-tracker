"""Sequential (video) Structure-from-Motion.

Stages
------
1. Keyframe selection by feature displacement / overlap.
2. Initial pair search: essential-matrix RANSAC (USAC/MAGSAC) on every nearby
   keyframe pair, scored by inliers x triangulation angle. In flat scenes the
   homography decomposition hypotheses are evaluated too.
3. Focal length estimation: every focal hypothesis (and every initialisation
   hypothesis) seeds a short incremental reconstruction with focal-length bundle
   adjustment; the lowest reprojection error wins. Sweep errors are kept to
   judge how well the footage constrains focal length.
4. Incremental keyframe registration (PnP RANSAC + LM refinement), multi-view
   triangulation with angle/cheirality/error gates, periodic global BA and
   outlier rejection.
5. Registration of all remaining frames, triangulation of short tracks, final
   global BA with focal + lens distortion (+ per-frame focal for zooms), iterative
   outlier rejection.

Tripod mode estimates rotation only (robust Kabsch on bearing vectors) and
reconstructs points as directions at infinity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ..config import AnalysisSettings
from ..tracking.tracker import TrackSet
from .bundle import BAProblem, bundle_adjust, reprojection_errors, refine_points
from .geometry import (K_matrix, bearings, pose_center, ransac_rotation, ray_angle_deg, rot_to_quat,
                       quat_to_rot, slerp, triangulate_multiview, undistort_normalized, project, robust_essential, robust_homography)


class SolveFailed(Exception):
    pass


@dataclass
class Solution:
    mode: str
    engine: str
    proc_w: int
    proc_h: int
    scale: float                     # processing / original
    f: float                         # processing pixels
    cx: float
    cy: float
    k1: float
    k2: float
    frames: list                     # original frame numbers (sequence)
    rvecs: np.ndarray                # [N,3] world->camera
    tvecs: np.ndarray                # [N,3]
    frame_status: list               # 'keyframe' | 'solved' | 'interpolated'
    cam_focal: Optional[np.ndarray]  # [N] processing px (variable focal) or None
    point_ids: np.ndarray            # track ids [P]
    points: np.ndarray               # [P,3]
    colors: np.ndarray               # [P,3] uint8
    point_error: np.ndarray          # mean reprojection error px (processing)
    point_nobs: np.ndarray
    point_angle: np.ndarray          # max triangulation angle (deg)
    point_obs: list                  # per point: list of (frame_number, x, y) inlier obs (processing px)
    frame_error: np.ndarray          # mean error per frame (nan if none)
    frame_npoints: np.ndarray
    stats: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def reliable_mask(self, thresh: float) -> np.ndarray:
        ok = (self.point_error < 0.7 * thresh) & (self.point_nobs >= 4)
        if self.mode != "tripod":
            ok &= self.point_angle >= 1.0
        return ok


def _robust_similarity(src, dst, rng, iters=300):
    """RANSAC + Umeyama similarity dst = s R src + t. Returns (s, R, t, n_inliers) or None."""
    n = len(src)
    if n < 4:
        return None
    def fit(a, b):
        ma, mb = a.mean(0), b.mean(0)
        xa, xb = a - ma, b - mb
        U, D, Vt = np.linalg.svd(xb.T @ xa / len(a))
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1
        R = U @ S @ Vt
        var = (xa ** 2).sum() / len(a)
        s = np.trace(np.diag(D) @ S) / max(var, 1e-12)
        return s, R, mb - s * R @ ma
    ext = np.median(np.linalg.norm(dst - np.median(dst, 0), axis=1))
    thr = 0.03 * max(ext, 1e-9)
    best, best_in = None, None
    for _ in range(iters):
        idx = rng.choice(n, 4, replace=False)
        s, R, t = fit(src[idx], dst[idx])
        if not np.isfinite(s) or s <= 0:
            continue
        err = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
        inl = err < thr
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = (s, R, t), inl
    if best_in is None or best_in.sum() < max(6, 0.3 * n):
        return None
    s, R, t = fit(src[best_in], dst[best_in])
    return s, R, t, int(best_in.sum())


class SequentialSfM:
    def __init__(self, ts: TrackSet, settings: AnalysisSettings, report, exclude_ids: Optional[set] = None):
        self.s = settings
        self.report = report
        self.ts = ts
        self.w, self.h = ts.proc_w, ts.proc_h
        self.cx, self.cy = (self.w - 1) / 2.0, (self.h - 1) / 2.0
        self.frames = list(ts.frames)
        self.N = len(self.frames)
        fpos = {f: i for i, f in enumerate(self.frames)}
        min_len = max(3, min(settings.min_track_length, max(3, self.N // 2)))
        self.min_len_requested = min_len
        # Sparse footage (faint texture, heavy noise) cannot produce long tracks. Rather than starve the
        # solver, accept shorter ones until there are at least ~200 observations per frame (never below 4).
        while min_len > 4:
            cand = ts.usable(min_len, exclude_ids)
            if sum(len(t.frames) for t in cand) / max(1, self.N) >= 200:
                break
            min_len -= 1
        self.min_len_used = min_len
        self.tracks = ts.usable(min_len, exclude_ids)
        if len(self.tracks) < 20:
            raise SolveFailed(f"Only {len(self.tracks)} usable tracks (min length {min_len}). "
                              "The footage lacks trackable texture or the clip is too short.")
        ot, of, oxy = [], [], []
        for ti, t in enumerate(self.tracks):
            for fr, xy in zip(t.frames, t.xy):
                ot.append(ti)
                of.append(fpos[fr])
                oxy.append(xy)
        self.obs_t = np.asarray(ot, np.int64)
        self.obs_f = np.asarray(of, np.int64)
        self.obs_xy = np.asarray(oxy, np.float64)
        self.M = len(self.obs_t)
        self.T = len(self.tracks)
        order = np.argsort(self.obs_f, kind="stable")
        self.frame_obs = np.split(order, np.searchsorted(self.obs_f[order], np.arange(1, self.N)))
        order_t = np.argsort(self.obs_t, kind="stable")
        self.track_obs = np.split(order_t, np.searchsorted(self.obs_t[order_t], np.arange(1, self.T)))
        self.frame_map = [dict(zip(self.obs_t[ix].tolist(), ix.tolist())) for ix in self.frame_obs]
        # link tracks (wide-baseline matches) may be the only thing tying the two sides of a tracking
        # dead zone together, so bundle adjustment must always include them
        self.is_link = np.array([getattr(t, "kind", "klt") == "link" for t in self.tracks], bool)
        self.thresh = settings.reprojection_threshold
        self.min_angle = settings.min_triangulation_angle
        self.rng = np.random.default_rng(12345)
        self.log = report.log if hasattr(report, "log") else (lambda m: None)
        self._reset(1.0)
        # Weak MAP prior on focal length: a 60 deg horizontal FOV (~31mm full-frame equivalent).
        # Only matters when the footage does not constrain focal (e.g. planar scene + pure translation);
        # its weight is tiny compared to the reprojection term of any well-conditioned solve.
        # (measured across the LONG edge: on phone portrait video the width is the short side)
        self.long_edge = float(max(self.w, self.h))
        self.focal_prior = self.long_edge / (2 * np.tan(np.radians(30)))
        self.focal_prior_weight = 20.0

    # ------------------------------------------------------------------ utils
    def _check(self):
        if hasattr(self.report, "check"):
            self.report.check()

    def _reset(self, f, k1=0.0, k2=0.0):
        self.f, self.k1, self.k2 = float(f), float(k1), float(k2)
        self.inlier = np.ones(self.M, bool)
        self.pt = np.zeros((self.T, 3))
        self.has_pt = np.zeros(self.T, bool)
        self.rv = np.zeros((self.N, 3))
        self.tv = np.zeros((self.N, 3))
        self.reg = np.zeros(self.N, bool)
        self.cam_focal = None

    def _norm(self, xy, fidx=None):
        f = self.cam_focal[fidx] if (self.cam_focal is not None and fidx is not None) else self.f
        return undistort_normalized(xy, f, self.cx, self.cy, self.k1, self.k2)

    def _K(self, fidx=None):
        f = self.cam_focal[fidx] if (self.cam_focal is not None and fidx is not None) else self.f
        return K_matrix(f, self.cx, self.cy)

    def _dist(self):
        return np.array([self.k1, self.k2, 0, 0], np.float64)

    def _common(self, a, b):
        ma, mb = self.frame_map[a], self.frame_map[b]
        ks = ma.keys() & mb.keys()
        ks = [k for k in ks if self.inlier[ma[k]] and self.inlier[mb[k]]]
        return np.array(ks, np.int64), np.array([ma[k] for k in ks], np.int64), np.array([mb[k] for k in ks], np.int64)

    # ------------------------------------------------------------ keyframes
    def select_keyframes(self) -> list[int]:
        thr = 0.025 * max(self.w, self.h)
        max_gap = max(2, min(15, self.N // 8 if self.N > 16 else 2))
        kfs = [0]
        for i in range(1, self.N):
            last = kfs[-1]
            tr, ia, ib = self._common(last, i)
            n_last = len(self.frame_obs[last])
            disp = np.median(np.linalg.norm(self.obs_xy[ia] - self.obs_xy[ib], axis=1)) if len(tr) else 1e9
            if disp > thr or len(tr) < 0.55 * max(1, n_last) or i - last >= max_gap:
                kfs.append(i)
        if kfs[-1] != self.N - 1:
            kfs.append(self.N - 1)
        # link frames carry the wide-baseline SIFT tracks that bridge dead zones: always keyframes
        links = [i for i in self.ts.stats.get("link_seq_idx", []) if 0 <= i < self.N]
        return sorted(set(kfs) | set(links))

    # -------------------------------------------------------- two view init
    def _triangulate_pair(self, xa, xb, R, t):
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = np.hstack([R, t.reshape(3, 1)])
        X = cv2.triangulatePoints(P1, P2, xa.T, xb.T)
        X = (X[:3] / np.where(np.abs(X[3]) < 1e-12, 1e-12, X[3])).T
        za = X[:, 2]
        zb = (X @ R.T + t)[:, 2]
        ca = np.zeros(3)
        cb = -R.T @ t
        va = X - ca
        vb = X - cb
        cosang = np.sum(va * vb, axis=1) / np.maximum(np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1), 1e-12)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        pa = X[:, :2] / np.maximum(za[:, None], 1e-9)
        Xb = X @ R.T + t
        pb = Xb[:, :2] / np.maximum(zb[:, None], 1e-9)
        err = np.maximum(np.linalg.norm(pa - xa, axis=1), np.linalg.norm(pb - xb, axis=1)) * self.f
        good = (za > 0) & (zb > 0) & (err < self.thresh) & np.isfinite(X).all(1)
        return X, good, ang

    def two_view(self, a, b, use_h=False):
        """Return list of hypotheses dicts for pair (a,b) at current focal."""
        tr, ia, ib = self._common(a, b)
        if len(tr) < 30:
            return [], {}
        xa = self._norm(self.obs_xy[ia])
        xb = self._norm(self.obs_xy[ib])
        th = self.thresh / self.f
        hyps = []
        info = {"common": len(tr)}
        E, mask = robust_essential(xa, xb, th, 0.9999)
        if E is not None:
            try:
                n, R, t, mpose = cv2.recoverPose(E, xa, xb, np.eye(3), mask=mask.copy())
            except cv2.error:
                R = None
            if R is not None:
                t = t.ravel()
                X, good, ang = self._triangulate_pair(xa, xb, R, t)
                good &= (mask.ravel() > 0)
                info["e_inliers"] = int(mask.sum())
                if good.sum() >= 15:
                    hyps.append(dict(kind="E", R=R, t=t, tracks=tr[good], X=X[good], ang=float(np.median(ang[good])),
                                     n=int(good.sum())))
        H, hmask = robust_homography(xa, xb, th)
        info["h_inliers"] = int(hmask.sum()) if hmask is not None else 0
        if use_h and H is not None:
            try:
                nsol, Rs, ts_, ns = cv2.decomposeHomographyMat(H, np.eye(3))
            except cv2.error:
                nsol, Rs, ts_, ns = 0, [], [], []
            for R, t in zip(Rs, ts_):
                t = t.ravel()
                if np.linalg.norm(t) < 1e-9:
                    continue
                t = t / np.linalg.norm(t)
                X, good, ang = self._triangulate_pair(xa, xb, R, t)
                good &= (hmask.ravel() > 0)
                if good.sum() >= 15 and good.mean() > 0.3:
                    hyps.append(dict(kind="H", R=R, t=t, tracks=tr[good], X=X[good], ang=float(np.median(ang[good])),
                                     n=int(good.sum())))
        return hyps, info

    def rotation_model_test(self, kfs, fcands):
        """Fraction of correspondences explained by a pure camera rotation (best focal hypothesis),
        versus by a general essential matrix, on the widest-baseline keyframe pairs."""
        pairs = []
        for i, a in enumerate(kfs):
            for b in kfs[::-1]:
                if b <= a:
                    break
                tr, ia, ib = self._common(a, b)
                if len(tr) >= 40:
                    pairs.append((b - a, a, b))
                    break
        pairs.sort(reverse=True)
        rot_fracs, e_fracs = [], []
        for _, a, b in pairs[:3]:
            tr, ia, ib = self._common(a, b)
            def frac(f):
                xa = undistort_normalized(self.obs_xy[ia], f, self.cx, self.cy, self.k1, self.k2)
                xb = undistort_normalized(self.obs_xy[ib], f, self.cx, self.cy, self.k1, self.k2)
                R, inl = ransac_rotation(bearings(xa), bearings(xb), 1.5 * self.thresh / f, 150, self.rng)
                return float(inl.mean()) if R is not None else 0.0
            if len(fcands) == 1:
                best = frac(fcands[0])
            else:
                # coarse log-spaced focal search (FOV 8..120 deg) then local refinement
                grid = self.long_edge / (2 * np.tan(np.radians(np.geomspace(8, 120, 28)) / 2))
                vals = [frac(f) for f in grid]
                k = int(np.argmax(vals))
                lo, hi = grid[min(k + 1, len(grid) - 1)], grid[max(k - 1, 0)]
                fine = np.geomspace(lo, hi, 12)
                best = max(max(vals), max(frac(f) for f in fine))
            f = fcands[len(fcands) // 2]
            xa = undistort_normalized(self.obs_xy[ia], f, self.cx, self.cy, self.k1, self.k2)
            xb = undistort_normalized(self.obs_xy[ib], f, self.cx, self.cy, self.k1, self.k2)
            E, mask = robust_essential(xa, xb, 1.5 * self.thresh / f)
            ef = float(mask.mean()) if mask is not None else 0.0
            rot_fracs.append(best)
            e_fracs.append(ef)
        if not rot_fracs:
            return 0.0, 1.0
        return float(np.min(rot_fracs)), float(np.max(e_fracs))

    def find_init_pairs(self, kfs, use_h, top=3):
        cands = []
        for ia in range(len(kfs)):
            for ib in range(ia + 1, min(len(kfs), ia + 7)):
                self._check()
                a, b = kfs[ia], kfs[ib]
                hyps, info = self.two_view(a, b, use_h=False)
                for h in hyps:
                    score = h["n"] * min(h["ang"], 6.0)
                    if h["ang"] >= self.min_angle * 0.7:
                        cands.append((score, a, b, h["ang"], info))
        cands.sort(key=lambda c: -c[0])
        # spread: prefer distinct a's
        out, seen = [], set()
        for c in cands:
            if c[1] in seen:
                continue
            seen.add(c[1])
            out.append(c)
            if len(out) >= top:
                break
        return out

    # ------------------------------------------------------------ incremental
    def _set_init(self, a, b, hyp):
        self.rv[a] = 0
        self.tv[a] = 0
        self.rv[b] = cv2.Rodrigues(hyp["R"])[0].ravel()
        self.tv[b] = hyp["t"]
        self.reg[a] = self.reg[b] = True
        self.pt[hyp["tracks"]] = hyp["X"]
        self.has_pt[hyp["tracks"]] = True
        self.ref_frame = a

    def register(self, fi) -> bool:
        ix = self.frame_obs[fi]
        ix = ix[self.inlier[ix] & self.has_pt[self.obs_t[ix]]]
        if len(ix) < 10:
            return False
        obj = self.pt[self.obs_t[ix]].astype(np.float64)
        img = self.obs_xy[ix].astype(np.float64)
        regs = np.where(self.reg)[0]
        nearest = regs[np.argmin(np.abs(regs - fi))]
        if self.cam_focal is not None:
            self.cam_focal[fi] = self.cam_focal[nearest]
        K = self._K(fi)
        r0, t0 = self.rv[nearest].copy().reshape(3, 1), self.tv[nearest].copy().reshape(3, 1)
        # Frames bridged across a tracking dead zone see mostly wide-baseline link points, of which only
        # ~25% may be right (repetitive texture). RANSAC copes if it runs long enough and the inlier-ratio
        # gate is not stricter than the data: require >= 15% (and >= 10 points), not 25%.
        min_inl = max(10, 0.15 * len(ix))
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, self._dist(), iterationsCount=2000 if len(ix) > 300 else 6000,
                                                 reprojectionError=max(2.0, 2 * self.thresh), confidence=0.9999,
                                                 flags=cv2.SOLVEPNP_SQPNP)
        if not ok or inl is None or len(inl) < min_inl:
            # fall back to iterative with neighbour's pose as initial guess
            ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, self._dist(), r0, t0, True, 1000,
                                                     max(2.0, 2 * self.thresh), 0.999, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok or inl is None or len(inl) < min_inl:
                return False
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, self._dist(), rvec, tvec)
        if self.cam_focal is not None and len(inl) >= 12:
            rvec, tvec, self.cam_focal[fi] = self._refine_pose_focal(obj[inl], img[inl], rvec, tvec, self.cam_focal[fi])
        self.rv[fi] = rvec.ravel()
        self.tv[fi] = tvec.ravel()
        self.reg[fi] = True
        # reject inconsistent observations in this frame
        e = self._errors(ix)
        self.inlier[ix[e > 2 * self.thresh]] = False
        return True

    def _refine_pose_focal(self, obj, img, rvec, tvec, f):
        """Single-camera LM over pose + focal (zoom shots), structure fixed."""
        from scipy.optimize import least_squares

        def res(x):
            uv, _ = project(np.tile(x[:3], (len(obj), 1)), np.tile(x[3:6], (len(obj), 1)), obj, x[6] * f,
                            self.cx, self.cy, self.k1, self.k2)
            return (uv - img).ravel()
        x0 = np.r_[np.ravel(rvec), np.ravel(tvec), 1.0]
        r = least_squares(res, x0, loss="huber", f_scale=1.0, max_nfev=50)
        if not r.success or not (0.5 < r.x[6] < 2.0):
            return rvec, tvec, f
        return r.x[:3].reshape(3, 1), r.x[3:6].reshape(3, 1), float(r.x[6] * f)

    def _errors(self, ix):
        f = self.cam_focal if self.cam_focal is not None else None
        return reprojection_errors(self.rv, self.tv, self.pt, self.obs_f[ix], self.obs_t[ix], self.obs_xy[ix],
                                   self.f, self.cx, self.cy, self.k1, self.k2, f)

    def triangulate_tracks(self, track_ids):
        added = 0
        Rcache = {}
        for ti in track_ids:
            if self.has_pt[ti]:
                continue
            ix = self.track_obs[ti]
            ix = ix[self.inlier[ix] & self.reg[self.obs_f[ix]]]
            if len(ix) < 2:
                continue
            fids = self.obs_f[ix]
            Rs, ts, xn, cs = [], [], [], []
            for o, fi in zip(ix, fids):
                if fi not in Rcache:
                    R = cv2.Rodrigues(self.rv[fi])[0]
                    Rcache[fi] = (R, self.tv[fi], -R.T @ self.tv[fi])
                R, t, c = Rcache[fi]
                Rs.append(R)
                ts.append(t)
                cs.append(c)
            xn = np.vstack([self._norm(self.obs_xy[o:o + 1], fi) for o, fi in zip(ix, fids)]) \
                if self.cam_focal is not None else self._norm(self.obs_xy[ix])
            X = triangulate_multiview(Rs, ts, xn)
            if X is None or not np.isfinite(X).all():
                continue
            ang = ray_angle_deg(cs[0], cs[-1], X)
            if ang < self.min_angle:
                continue
            depths = np.array([(R @ X + t)[2] for R, t in zip(Rs, ts)])
            if (depths <= 0).any():
                continue
            self.pt[ti] = X
            self.has_pt[ti] = True
            e = self._errors(ix)
            bad = e > self.thresh
            if (~bad).sum() < 2:
                self.has_pt[ti] = False
                continue
            self.inlier[ix[bad]] = False
            added += 1
        return added

    def tracks_in_frame(self, fi):
        ix = self.frame_obs[fi]
        return np.unique(self.obs_t[ix[self.inlier[ix]]])

    def run_ba(self, frames_mask=None, opt_f=True, opt_k1=False, opt_k2=False, var_focal=False, max_nfev=40):
        self._check()
        regmask = self.reg if frames_mask is None else (self.reg & frames_mask)
        use = self.inlier & regmask[self.obs_f] & self.has_pt[self.obs_t]
        # points need >= 2 observations
        cnt = np.bincount(self.obs_t[use], minlength=self.T)
        use &= cnt[self.obs_t] >= 2
        if use.sum() < 20:
            return None
        cams = np.where(regmask)[0]
        cam_idx = -np.ones(self.N, np.int64)
        cam_idx[cams] = np.arange(len(cams))
        all_pts = np.unique(self.obs_t[use])
        subset = None
        if len(all_pts) > self.ba_point_budget:
            subset = self._ba_subset(use, cams)
            use_ba = use & subset[self.obs_t]
        else:
            use_ba = use
        full_use = use
        use = self._cap_obs(use_ba, 24)
        pts = np.unique(self.obs_t[use])
        pt_idx = -np.ones(self.T, np.int64)
        pt_idx[pts] = np.arange(len(pts))
        fixed = np.zeros(len(cams), bool)
        ref = self.ref_frame if regmask[self.ref_frame] else cams[0]
        fixed[cam_idx[ref]] = True
        cf = None
        if var_focal:
            if self.cam_focal is None:
                self.cam_focal = np.full(self.N, self.f)
            cf = self.cam_focal[cams].copy()
        prob = BAProblem(rvecs=self.rv[cams].copy(), tvecs=self.tv[cams].copy(), points=self.pt[pts].copy(),
                         obs_cam=cam_idx[self.obs_f[use]], obs_pt=pt_idx[self.obs_t[use]], obs_xy=self.obs_xy[use],
                         f=self.f, cx=self.cx, cy=self.cy, k1=self.k1, k2=self.k2, fixed_cams=fixed,
                         opt_f=opt_f and not var_focal, opt_k1=opt_k1, opt_k2=opt_k2,
                         cam_focal=cf, opt_cam_focal=var_focal, cam_order=np.arange(len(cams)) if var_focal else None,
                         cam_times=cams, focal_prior=self.focal_prior, focal_prior_weight=self.focal_prior_weight)
        r = bundle_adjust(prob, loss_scale=max(0.5, self.thresh / 2), max_nfev=max_nfev)
        self.rv[cams], self.tv[cams], self.pt[pts] = r["rvecs"], r["tvecs"], r["points"]
        self.f, self.k1, self.k2 = r["f"], r["k1"], r["k2"]
        if var_focal:
            self.cam_focal[cams] = r["cam_focal"]
            self.f = float(np.median(self.cam_focal[cams]))
        if subset is not None:
            # refine every other point with the cameras held fixed
            rest = full_use & ~subset[self.obs_t]
            rpts = np.unique(self.obs_t[rest])
            ridx = -np.ones(self.T, np.int64)
            ridx[rpts] = np.arange(len(rpts))
            self.pt[rpts] = refine_points(self.rv, self.tv, self.pt[rpts], self.obs_f[rest], ridx[self.obs_t[rest]],
                                          self.obs_xy[rest], self.f, self.cx, self.cy, self.k1, self.k2,
                                          self.cam_focal, loss_scale=max(0.5, self.thresh / 2))
        return r

    ba_point_budget = 2500

    def _cap_obs(self, use, cap):
        """Keep at most ~cap evenly spaced observations per point (BA cost grows with obs-per-point squared)."""
        ix = np.where(use)[0]
        if len(ix) == 0:
            return use
        order = np.lexsort((self.obs_f[ix], self.obs_t[ix]))
        ix = ix[order]
        tr = self.obs_t[ix]
        start = np.r_[0, np.where(tr[1:] != tr[:-1])[0] + 1]
        n = np.diff(np.r_[start, len(ix)])
        nn = np.repeat(n, n)
        rank = np.arange(len(ix)) - np.repeat(start, n)
        slot = np.floor(rank * cap / np.maximum(nn, 1)).astype(np.int64)
        keep = (nn <= cap) | np.r_[True, (slot[1:] != slot[:-1]) | (tr[1:] != tr[:-1])] | (rank == nn - 1)
        out = np.zeros_like(use)
        out[ix[keep]] = True
        return out

    def _ba_subset(self, use, cams):
        """Pick well-distributed, long-lived points: per frame, the longest track in each cell of a 4x4 grid (2 rounds)."""
        cnt = np.bincount(self.obs_t[use], minlength=self.T)
        sel = np.zeros(self.T, bool)
        G = 6
        gx = np.clip((self.obs_xy[:, 0] / self.w * G).astype(int), 0, G - 1)
        gy = np.clip((self.obs_xy[:, 1] / self.h * G).astype(int), 0, G - 1)
        cell = gy * G + gx
        rounds = int(np.clip(np.ceil(self.ba_point_budget / max(1, len(cams)) / (G * G)), 1, 8))
        for fi in cams:
            ix = self.frame_obs[fi]
            ix = ix[use[ix]]
            if len(ix) == 0:
                continue
            for rnd in range(rounds):
                tr = self.obs_t[ix]
                score = cnt[tr] - 1e6 * sel[tr]
                order = np.lexsort((-score, cell[ix]))
                cs = cell[ix][order]
                first = np.r_[True, cs[1:] != cs[:-1]]
                pick = tr[order][first]
                pick = pick[~sel[pick]]
                sel[pick] = True
        sel |= self.is_link & (cnt >= 2)
        return sel

    def filter_outliers(self, factor=1.0):
        use = self.inlier & self.reg[self.obs_f] & self.has_pt[self.obs_t]
        ix = np.where(use)[0]
        if len(ix) == 0:
            return 0
        e = self._errors(ix)
        bad = ix[~(e <= self.thresh * factor)]
        self.inlier[bad] = False
        use = self.inlier & self.reg[self.obs_f] & self.has_pt[self.obs_t]
        cnt = np.bincount(self.obs_t[use], minlength=self.T)
        drop = self.has_pt & (cnt < 2)
        self.has_pt[drop] = False
        return len(bad)

    def incremental(self, kfs, a, b, hyp, f, window=None, opt_focal=True):
        """Register keyframes outward from the initial pair. Returns (#registered, rmse)."""
        self._reset(f, *(self._fixed_dist()))
        self._set_init(a, b, hyp)
        ia, ib = kfs.index(a), kfs.index(b)
        between = kfs[ia + 1:ib]
        after = kfs[ib + 1:]
        before = kfs[:ia][::-1]
        if window is not None:
            after = after[:window]
            before = before[:max(2, window // 3)]
        order = list(between)
        for i in range(max(len(after), len(before))):
            if i < len(after):
                order.append(after[i])
            if i < len(before):
                order.append(before[i])
        targets = set(order) | {a, b}
        var = self.s.variable_focal and opt_focal
        if var:
            self.cam_focal = np.full(self.N, self.f)
        self.run_ba(opt_f=False, max_nfev=20)
        self.filter_outliers()
        # Best-next-frame registration (as in COLMAP): always add the unregistered keyframe that
        # currently sees the most reconstructed 3D points, triangulate, repeat. Unlike a fixed outward
        # order, one hard frame cannot strand everything behind it. A failed frame is retried only once
        # it sees clearly more 3D points than at its failed attempt.
        done = 0
        pending = set(order)
        tried_at: dict[int, int] = {}
        while pending:
            self._check()
            use = self.inlier & self.has_pt[self.obs_t]
            cnt = np.bincount(self.obs_f[use], minlength=self.N)
            cand = [fi for fi in pending if cnt[fi] >= 10 and cnt[fi] > 1.2 * tried_at.get(fi, -10) + 2]
            if not cand:
                break
            fi = max(cand, key=lambda x: (cnt[x], -abs(x - a)))
            if not self.register(fi):
                tried_at[fi] = int(cnt[fi])
                continue
            pending.discard(fi)
            self.triangulate_tracks(self.tracks_in_frame(fi))
            done += 1
            if done % 5 == 0 or done <= 2:
                self.run_ba(opt_f=opt_focal and done >= 3 and not var, var_focal=var and done >= 3, max_nfev=15)
                self.filter_outliers()
            if window is None and hasattr(self.report, "__call__"):
                self.report(0.15 + 0.45 * done / max(1, len(order)), f"Registered keyframe {self.frames[fi]} ({done}/{len(order)})")
        self.run_ba(opt_f=opt_focal and not var, var_focal=var, max_nfev=40)
        self.filter_outliers()
        r = self.run_ba(opt_f=opt_focal and not var, var_focal=var, max_nfev=30)
        nreg = int(sum(self.reg[k] for k in targets))
        rmse = r["rmse"] if r else float("inf")
        return nreg, len(targets), rmse

    def _fixed_dist(self):
        if self.s.distortion == "manual":
            return self.s.k1, self.s.k2
        return 0.0, 0.0

    # ------------------------------------------------------------ main driver
    def focal_candidates(self):
        if self.s.focal_length_px:
            return [self.s.focal_length_px * self.ts.scale], False
        if self.s.focal_length_mm:
            # 35mm-style focal length: the sensor size refers to the image's long edge
            return [self.s.focal_length_mm / self.s.sensor_width_mm * self.long_edge], False
        fovs = [20, 35, 50, 65, 80, 100]
        return [self.long_edge / (2 * np.tan(np.radians(fv) / 2)) for fv in fovs], True

    def solve(self) -> Solution:
        rep = self.report
        kfs = self.select_keyframes()
        self.log(f"{self.T} usable tracks, {self.M} observations, {self.N} frames, {len(kfs)} keyframes")
        if self.min_len_used < self.min_len_requested:
            self.log(f"Sparse footage: accepting tracks of {self.min_len_used}+ frames "
                     f"(requested minimum {self.min_len_requested}) to keep enough observations per frame")
        fcands, estimate_f = self.focal_candidates()
        f_nominal = fcands[len(fcands) // 2] if estimate_f else fcands[0]
        self._reset(f_nominal, *self._fixed_dist())

        mode = self.s.solve_method
        # parallax analysis for auto / degenerate detection
        pairs = self.find_init_pairs(kfs, use_h=False, top=3)
        best_angle = max([p[3] for p in pairs], default=0.0)
        h_ratio = 0.0
        if pairs:
            info = pairs[0][4]
            h_ratio = info.get("h_inliers", 0) / max(1, info.get("e_inliers", 1))
        self.log(f"Parallax analysis: best median triangulation angle {best_angle:.2f} deg, H/E inlier ratio {h_ratio:.2f}")
        rot_frac, e_frac = self.rotation_model_test(kfs, fcands)
        self.log(f"Rotation-only model explains {100 * rot_frac:.1f}% of matches over wide keyframe pairs "
                 f"(general motion model: {100 * e_frac:.1f}%)")
        if mode == "auto":
            if best_angle < 0.8 or not pairs or (rot_frac >= 0.9 and rot_frac >= 0.97 * e_frac):
                mode = "tripod"
                self.log("Insufficient parallax for 3D reconstruction -> using tripod (rotation-only) solve")
            else:
                mode = "flat" if h_ratio > 0.95 else "typical"
        self.mode = mode
        if mode == "tripod":
            return self.solve_tripod(kfs, fcands, estimate_f)
        if not pairs:
            raise SolveFailed("No keyframe pair has enough parallax for a 3D solve. Try 'Tripod' mode "
                              "if the camera only rotates, or a longer frame range.")

        # ---- focal & initialisation hypothesis search ----
        rep.stage("Estimating camera intrinsics", 0.05, 0.15) if hasattr(rep, "stage") else None
        trials = []
        use_h = mode == "flat" or h_ratio > 0.8
        fc = fcands if estimate_f else fcands
        total = len(fc) * len(pairs[:1])
        k = 0
        for f in fc:
            for (_, a, b, _, _) in pairs[:1]:
                self._check()
                k += 1
                self._reset(f, *self._fixed_dist())
                hyps, _ = self.two_view(a, b, use_h=use_h)
                for hyp in hyps:
                    try:
                        nreg, ntar, rmse = self.incremental(kfs, a, b, hyp, f, window=8, opt_focal=estimate_f)
                    except np.linalg.LinAlgError:
                        continue
                    score = rmse + 5.0 * (1 - nreg / max(1, ntar)) + 0.1 * abs(np.log(self.f / self.focal_prior))
                    trials.append(dict(f0=f, f=self.f, a=a, b=b, hyp=hyp, rmse=rmse, score=score, nreg=nreg, ntar=ntar))
                    self.log(f"  trial f0={f:.0f}px ({hyp['kind']}, kf {self.frames[a]}-{self.frames[b]}): "
                             f"refined f={self.f:.0f}px rmse={rmse:.3f}px registered {nreg}/{ntar}")
                rep(k / total, f"Focal hypothesis {k}/{total}")
        if not trials:
            raise SolveFailed("Every initialisation hypothesis failed. The footage may lack parallax "
                              "(try Tripod) or reliable features.")
        trials.sort(key=lambda t: t["score"])
        best = trials[0]
        self.focal_sweep = [(t["f0"], t["f"], t["rmse"]) for t in trials]
        if estimate_f and not self.s.variable_focal:
            # (skipped for zoom shots: a single frozen lens cannot describe a lens that changes over the clip)
            best = self._confirm_focal(trials, kfs, use_h, best)
        self.log(f"Selected initialisation: f={best['f']:.1f}px ({best['hyp']['kind']}), rmse {best['rmse']:.3f}px")

        # ---- full keyframe reconstruction ----
        rep.stage("Reconstructing keyframes", 0.15, 0.6) if hasattr(rep, "stage") else None
        f_start = best["f"] if estimate_f else fcands[0]
        self._reset(f_start, *self._fixed_dist())
        hyps, _ = self.two_view(best["a"], best["b"], use_h=use_h)
        hyp = min(hyps, key=lambda h: 0 if h["kind"] == best["hyp"]["kind"] and np.allclose(h["R"], best["hyp"]["R"], atol=1e-3) else 1) if hyps else best["hyp"]
        # Focal stays fixed at the hypothesis-search result while the reconstruction grows: refining it on a
        # partial, weakly-conditioned reconstruction (e.g. walking forward) lets it drift and breaks the chain
        # on long clips. It is refined once all frames are in (final_refinement), where the whole clip
        # constrains it. Zoom shots (variable focal) still estimate per-frame focal incrementally.
        nreg, ntar, rmse = self.incremental(kfs, best["a"], best["b"], hyp, f_start, window=None,
                                            opt_focal=estimate_f and self.s.variable_focal)
        self.log(f"Keyframes registered {nreg}/{ntar}, rmse {rmse:.3f}px, f={self.f:.1f}px")
        self.keyframes = kfs

        # ---- all frames ----
        rep.stage("Registering all frames", 0.6, 0.75) if hasattr(rep, "stage") else None
        self.register_remaining(kfs)
        self.recover_segments(kfs)
        rep.stage("Global bundle adjustment", 0.75, 0.97) if hasattr(rep, "stage") else None
        self.final_refinement(estimate_f)
        return self.build_solution(mode, kfs, estimate_f)

    # ------------------------------------------------------------ segments
    def _snapshot(self):
        return dict(rv=self.rv.copy(), tv=self.tv.copy(), reg=self.reg.copy(), pt=self.pt.copy(),
                    has_pt=self.has_pt.copy(), inlier=self.inlier.copy(), f=self.f, k1=self.k1, k2=self.k2,
                    cam_focal=None if self.cam_focal is None else self.cam_focal.copy(),
                    ref_frame=getattr(self, "ref_frame", 0))

    def _restore(self, sn):
        for k, v in sn.items():
            setattr(self, k, v)

    def recover_segments(self, kfs):
        """Re-initialise on long runs of frames that could not be chained to the main solve
        (e.g. the camera saw entirely new scenery after a dead zone), reconstruct them
        independently, then join them to the main reconstruction:
          * by a robust similarity on 3D points both reconstructions share (measured), or
          * if they share nothing, by motion continuity across the gap (estimated, flagged)."""
        self.segments = []
        min_run = max(12, self.N // 40)
        for _ in range(4):  # a few independent segments at most
            unreg = ~self.reg
            runs, i = [], 0
            while i < self.N:
                if unreg[i]:
                    j = i
                    while j + 1 < self.N and unreg[j + 1]:
                        j += 1
                    runs.append((i, j))
                    i = j + 1
                else:
                    i += 1
            # only runs that actually contain trackable frames
            cand = []
            for r0, r1 in runs:
                trackable = [fi for fi in range(r0, r1 + 1) if len(self.frame_obs[fi]) >= 40]
                if len(trackable) >= min_run:
                    cand.append((len(trackable), r0, r1))
            if not cand:
                return
            _, r0, r1 = max(cand)
            # restrict to the trackable part of the run (drop the dead frames at its edges)
            idx = [fi for fi in range(r0, r1 + 1) if len(self.frame_obs[fi]) >= 40]
            r0, r1 = idx[0], idx[-1]
            if not self._reconstruct_segment(kfs, r0, r1):
                # mark as tried so we do not loop on it
                self.segments.append(dict(frames=(self.frames[r0], self.frames[r1]), mode="failed"))
                return

    def _reconstruct_segment(self, kfs, r0, r1) -> bool:
        main = self._snapshot()
        seg_kfs = [k for k in kfs if r0 <= k <= r1]
        if len(seg_kfs) < 4:
            seg_kfs = sorted(set(seg_kfs) | set(range(r0, r1 + 1, max(1, (r1 - r0) // 12))))
        self.log(f"Frames {self.frames[r0]}-{self.frames[r1]} could not be chained to the main solve; "
                 f"reconstructing them as a separate segment")
        pairs = self.find_init_pairs(seg_kfs, use_h=False, top=2)
        best = None
        for (_, a, b, _, _) in pairs:
            self._restore(main)
            self.inlier = main["inlier"].copy()
            hyps, _ = self.two_view(a, b, use_h=True)
            for hyp in sorted(hyps, key=lambda h: -h["n"])[:3]:
                try:
                    nreg, ntar, rmse = self.incremental(seg_kfs, a, b, hyp, main["f"], window=None, opt_focal=False)
                except np.linalg.LinAlgError:
                    continue
                if nreg >= max(3, 0.6 * ntar) and rmse < 2 * self.thresh and (best is None or nreg > best[0]):
                    for fi in range(r0, r1 + 1):
                        if not self.reg[fi] and self.register(fi):
                            self.triangulate_tracks(self.tracks_in_frame(fi))
                    best = (nreg, self._snapshot())
            if best is not None:
                break
        self._restore(main)
        if best is None:
            self.log("  segment could not be initialised")
            return False
        seg = best[1]
        frames_seg = np.where(seg["reg"] & (np.arange(self.N) >= r0) & (np.arange(self.N) <= r1))[0]
        common = np.where(main["has_pt"] & seg["has_pt"])[0]
        sim = None
        if len(common) >= 8:
            sim = _robust_similarity(seg["pt"][common], main["pt"][common], self.rng)
        pj = None if sim is not None else self._pose_join(seg, main, frames_seg)
        if sim is not None:
            s_, R, t, n_in = sim
            mode = "measured"
            self.log(f"  joined to the main solve through {n_in} shared 3D points")
        elif pj is not None:
            s_, R, t, n_fr = pj
            mode = "measured-frames"
            self.log(f"  joined to the main solve by locating {n_fr} frame(s) against the other side's 3D points")
        else:
            s_, R, t = self._continuity_alignment(seg, frames_seg, r0, r1)
            mode = "estimated"
            self.log("  no shared 3D points: joined by motion continuity (estimated, not measured)")
        # write the transformed segment into the main reconstruction
        for fi in frames_seg:
            Rs = cv2.Rodrigues(seg["rv"][fi])[0]
            Rn = Rs @ R.T
            self.rv[fi] = cv2.Rodrigues(Rn)[0].ravel()
            self.tv[fi] = -Rn @ t + s_ * seg["tv"][fi]
            self.reg[fi] = True
        new_pts = np.where(seg["has_pt"] & ~self.has_pt)[0]
        self.pt[new_pts] = s_ * seg["pt"][new_pts] @ R.T + t
        self.has_pt[new_pts] = True
        seg_obs = np.isin(self.obs_f, frames_seg)
        self.inlier[seg_obs] = seg["inlier"][seg_obs]
        self.segments.append(dict(frames=(self.frames[r0], self.frames[r1]), mode=mode,
                                  shared_points=int(len(common)), shared_frames=int(pj[3]) if pj is not None else 0,
                                  registered=int(len(frames_seg))))
        return True

    def _pose_join(self, seg, main, frames_seg):
        """Join through shared *observations*: frames of either reconstruction that see enough of the
        other reconstruction's 3D points are located in it by PnP, giving camera poses in both coordinate
        systems; the similarity between them is solved from those poses."""
        K, dist = self._K(), self._dist()
        pairs = []  # (R_a, c_a, R_b, c_b): pose in segment coords, pose in main coords

        def locate(fi, pt, has, inl):
            ix = self.frame_obs[fi]
            ix = ix[inl[ix] & has[self.obs_t[ix]]]
            if len(ix) < 12:
                return None
            try:
                ok, rv, tv, iv = cv2.solvePnPRansac(pt[self.obs_t[ix]].astype(np.float64), self.obs_xy[ix].astype(np.float64),
                                                    K, dist, iterationsCount=1000, reprojectionError=max(2.0, 2 * self.thresh),
                                                    confidence=0.999, flags=cv2.SOLVEPNP_SQPNP)
            except cv2.error:
                return None
            if not ok or iv is None or len(iv) < 10:
                return None
            R = cv2.Rodrigues(rv)[0]
            return R, -R.T @ tv.ravel()

        # segment frames located against main's points
        for fi in frames_seg:
            m = locate(fi, main["pt"], main["has_pt"], main["inlier"])
            if m is not None:
                Rs = cv2.Rodrigues(seg["rv"][fi])[0]
                pairs.append((Rs, pose_center(seg["rv"][fi], seg["tv"][fi]), m[0], m[1]))
        # main frames located against the segment's points
        for fi in np.where(main["reg"])[0]:
            m = locate(fi, seg["pt"], seg["has_pt"], seg["inlier"])
            if m is not None:
                Rm = cv2.Rodrigues(main["rv"][fi])[0]
                pairs.append((m[0], m[1], Rm, pose_center(main["rv"][fi], main["tv"][fi])))
        if not pairs:
            return None
        # rotation: R_main_cam = R_seg_cam @ R^T  =>  R = R_main_cam^T R_seg_cam (chordal mean)
        acc = sum(Rb.T @ Ra for Ra, ca, Rb, cb in pairs)
        U, _, Vt = np.linalg.svd(acc)
        R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        ca = np.array([p[1] for p in pairs])
        cb = np.array([p[3] for p in pairs])
        if len(pairs) >= 2 and np.linalg.norm(ca - ca.mean(0), axis=1).max() > 1e-6:
            rc = ca @ R.T
            s_ = float(np.sum((rc - rc.mean(0)) * (cb - cb.mean(0))) / max(np.sum((rc - rc.mean(0)) ** 2), 1e-12))
            if s_ <= 0:
                return None
        else:
            return None
        t = cb.mean(0) - s_ * (ca @ R.T).mean(0)
        return s_, R, t, len(pairs)

    def _continuity_alignment(self, seg, frames_seg, r0, r1):
        """Similarity placing the segment so the camera path is continuous across the gap."""
        main_reg = np.where(self.reg)[0]
        before = main_reg[main_reg < r0]
        after = main_reg[main_reg > r1]
        first, last = frames_seg[0], frames_seg[-1]
        if len(before):
            anchor, seg_f, gap = before[-1], first, first - before[-1]
            prev = before[-min(len(before), 6)]
        else:
            anchor, seg_f, gap = after[0], last, last - after[0]
            prev = after[min(len(after) - 1, 5)]
        Rm = cv2.Rodrigues(self.rv[anchor])[0]
        Rs = cv2.Rodrigues(seg["rv"][seg_f])[0]
        R = Rm.T @ Rs  # camera orientation continuous across the gap
        cm, cp = pose_center(self.rv[anchor], self.tv[anchor]), pose_center(self.rv[prev], self.tv[prev])
        vel = (cm - cp) / max(1, abs(anchor - prev))
        target = cm + vel * gap
        # scale: match the median depth of points seen at the joining frames
        def med_depth(rv, tv, pt, has, fi):
            ix = self.frame_obs[fi]
            tr = self.obs_t[ix]
            tr = tr[has[tr]]
            if len(tr) < 5:
                return None
            Rr = cv2.Rodrigues(rv[fi])[0]
            z = (pt[tr] @ Rr.T + tv[fi])[:, 2]
            z = z[z > 0]
            return float(np.median(z)) if len(z) else None
        dm = med_depth(self.rv, self.tv, self.pt, self.has_pt, anchor)
        ds = med_depth(seg["rv"], seg["tv"], seg["pt"], seg["has_pt"], seg_f)
        s_ = dm / ds if dm and ds else 1.0
        cs = pose_center(seg["rv"][seg_f], seg["tv"][seg_f])
        t = target - s_ * R @ cs
        return s_, R, t

    def _confirm_focal(self, trials, kfs, use_h, best):
        """The short trials above cannot tell focal lengths apart when the opening frames are mostly
        rotation (a pan or tilt). Re-test the leading *distinct* candidates over a much longer stretch with
        the lens frozen at each candidate: the correct lens keeps registering new frames, a wrong one
        stalls or accumulates error. Candidates are ranked by how much of the stretch they register, then
        by reprojection error."""
        top = []
        for t in trials:
            if all(abs(np.log(t["f"] / g["f"])) > 0.06 for g in top):
                top.append(t)
            if len(top) == 4:
                break
        if len(top) < 2:
            return best
        results = []
        for t in top:
            self._check()
            self._reset(t["f"], *self._fixed_dist())
            hyps, _ = self.two_view(t["a"], t["b"], use_h=use_h)
            if not hyps:
                continue
            hyp = min(hyps, key=lambda h: 0 if h["kind"] == t["hyp"]["kind"] else 1)
            try:
                nreg, ntar, rmse = self.incremental(kfs, t["a"], t["b"], hyp, t["f"], window=40, opt_focal=False)
            except np.linalg.LinAlgError:
                continue
            results.append((nreg / max(1, ntar), rmse, t))
            self.log(f"  long test f={t['f']:.0f}px: registered {nreg}/{ntar} keyframes, rmse {rmse:.3f}px")
        if not results:
            return best
        top_frac = max(r[0] for r in results)
        # Among candidates that register (nearly) as much as the best, take the lowest error. When the
        # scene cannot reveal the lens at all (a flat floor: every candidate registers everything with
        # near-identical error) the weak prior breaks the tie, exactly as in the short trials. It must not
        # be dropped here, or the winner is decided by noise (787 px chosen for a true 1000 px on the
        # synthetic flat clip).
        ok = [r for r in results if r[0] >= top_frac - 0.05]
        chosen = min(ok, key=lambda r: r[1] + 0.1 * abs(np.log(r[2]["f"] / self.focal_prior)))[2]
        if chosen is not best:
            self.log(f"Long-stretch test changed the lens estimate: {best['f']:.0f}px -> {chosen['f']:.0f}px")
        return chosen

    def register_remaining(self, kfs):
        kset = set(kfs)
        todo = [i for i in range(self.N) if not self.reg[i]]
        for n, fi in enumerate(todo):
            self._check()
            if self.register(fi):
                self.triangulate_tracks(self.tracks_in_frame(fi))
            if n % 10 == 0:
                self.report(n / max(1, len(todo)), f"Registering frame {self.frames[fi]}")
        # tracks that only live between keyframes
        self.triangulate_tracks(np.where(~self.has_pt)[0])

    def final_refinement(self, estimate_f):
        d = self.s.distortion
        opt_k1 = d in ("k1", "k1k2")
        opt_k2 = d == "k1k2"
        steps = [(estimate_f, False, False, 1.6), (estimate_f, opt_k1, opt_k2, 1.2), (estimate_f, opt_k1, opt_k2, 1.0)]
        var = self.s.variable_focal and self.cam_focal is not None
        for i, (of, k1, k2, fac) in enumerate(steps):
            self._check()
            r = self.run_ba(opt_f=of and not var, opt_k1=k1, opt_k2=k2, var_focal=var, max_nfev=60)
            nbad = self.filter_outliers(fac)
            self.report((i + 1) / (len(steps) + 1), f"BA pass {i + 1}: rmse {r['rmse'] if r else float('nan'):.3f}px, removed {nbad} outlier obs")
            self.log(f"Global BA pass {i + 1}: rmse {r['rmse']:.3f}px f={self.f:.1f} k1={self.k1:.4f} k2={self.k2:.4f}; {nbad} outliers removed")
            # newly registrable frames / tracks after refinement
            for fi in np.where(~self.reg)[0]:
                if self.register(fi):
                    self.triangulate_tracks(self.tracks_in_frame(fi))
        if self.s.variable_focal:
            r = self.run_ba(opt_f=False, opt_k1=opt_k1, opt_k2=opt_k2, var_focal=True, max_nfev=60)
            self.filter_outliers(1.0)
            self.log(f"Variable focal BA: rmse {r['rmse']:.3f}px, focal range {np.nanmin(self.cam_focal):.0f}-{np.nanmax(self.cam_focal):.0f}px")
        r = self.run_ba(opt_f=estimate_f and not self.s.variable_focal, opt_k1=opt_k1, opt_k2=opt_k2,
                        var_focal=self.s.variable_focal, max_nfev=40)
        self.filter_outliers(1.0)

    # ------------------------------------------------------------ tripod
    def solve_tripod(self, kfs, fcands, estimate_f) -> Solution:
        rep = self.report
        if hasattr(rep, "stage"):
            rep.stage("Tripod solve: estimating focal", 0.05, 0.3)
        best = None
        sweep = []
        for i, f in enumerate(fcands):
            self._check()
            self._reset(f, *self._fixed_dist())
            r = self._tripod_reconstruct(kfs, opt_f=estimate_f)
            if r is None:
                continue
            sweep.append((f, self.f, r))
            self.log(f"  tripod trial f0={f:.0f}px -> f={self.f:.0f}px rmse={r:.3f}px")
            if best is None or r < best[1]:
                best = (self.f, r)
            rep((i + 1) / len(fcands), f"Focal hypothesis {i + 1}/{len(fcands)}")
        if best is None:
            raise SolveFailed("Tripod solve failed: not enough features shared between frames.")
        self.focal_sweep = [(a, b, c) for a, b, c in sweep]
        if hasattr(rep, "stage"):
            rep.stage("Tripod solve: all frames", 0.3, 0.97)
        self._reset(best[0], *self._fixed_dist())
        self._tripod_reconstruct(list(range(self.N)), opt_f=estimate_f, final=True)
        self.keyframes = kfs
        return self.build_solution("tripod", kfs, estimate_f)

    def _tripod_reconstruct(self, frames, opt_f=True, final=False):
        th = np.radians(1e-9) + self.thresh / self.f
        self.ref_frame = frames[0]
        self.reg[frames[0]] = True
        R_prev = np.eye(3)
        prev = frames[0]
        for fi in frames[1:]:
            self._check()
            tr, ia, ib = self._common(prev, fi)
            if len(tr) < 8:
                # try against the reconstructed directions instead
                continue
            ba = bearings(self._norm(self.obs_xy[ia]))
            bb = bearings(self._norm(self.obs_xy[ib]))
            Rrel, inl = ransac_rotation(ba, bb, 2 * th, 200, self.rng)
            if Rrel is None or inl.sum() < 8:
                continue
            self.inlier[ia[~inl]] &= True  # keep; final BA decides
            R = Rrel @ R_prev
            self.rv[fi] = cv2.Rodrigues(R)[0].ravel()
            self.reg[fi] = True
            R_prev, prev = R, fi
        # directions
        ix = np.where(self.reg[self.obs_f])[0]
        cnt = np.bincount(self.obs_t[ix], minlength=self.T)
        b = bearings(self._norm(self.obs_xy[ix]))
        Rs = np.array([cv2.Rodrigues(r)[0] for r in self.rv])
        d = np.einsum("nji,nj->ni", Rs[self.obs_f[ix]], b)
        acc = np.zeros((self.T, 3))
        np.add.at(acc, self.obs_t[ix], d)
        ok = cnt >= 2
        self.pt[ok] = acc[ok] / np.maximum(np.linalg.norm(acc[ok], axis=1, keepdims=True), 1e-12)
        self.has_pt[ok] = True
        fmask = np.zeros(self.N, bool)
        fmask[frames] = True
        r = None
        dist = self.s.distortion
        for k, fac in enumerate([2.0, 1.2, 1.0]):
            r = self._tripod_ba(fmask, opt_f=opt_f, opt_k1=final and dist in ("k1", "k1k2"),
                                opt_k2=final and dist == "k1k2", max_nfev=40 if final else 20)
            if r is None:
                return None
            self.filter_outliers(fac)
            if final:
                self.report(0.3 + 0.2 * k, f"Tripod BA pass {k + 1}: rmse {r['rmse']:.3f}px")
        if final and self.s.variable_focal:
            r = self._tripod_ba(fmask, opt_f=False, opt_k1=dist in ("k1", "k1k2"), opt_k2=dist == "k1k2", var=True)
            self.filter_outliers(1.0)
        if final:
            # frames that failed relative rotation: absolute orientation against directions
            for fi in np.where(~self.reg)[0]:
                ix = self.frame_obs[fi]
                ix = ix[self.inlier[ix] & self.has_pt[self.obs_t[ix]]]
                if len(ix) < 6:
                    continue
                b = bearings(self._norm(self.obs_xy[ix]))
                R, inl = ransac_rotation(self.pt[self.obs_t[ix]], b, 2 * th, 200, self.rng)
                if R is not None and inl.sum() >= 6:
                    self.rv[fi] = cv2.Rodrigues(R)[0].ravel()
                    self.reg[fi] = True
            r = self._tripod_ba(self.reg.copy(), opt_f=opt_f and not self.s.variable_focal,
                                opt_k1=dist in ("k1", "k1k2"), opt_k2=dist == "k1k2", var=self.s.variable_focal)
            self.filter_outliers(1.0)
        return r["rmse"] if r else None

    def _tripod_ba(self, fmask, opt_f, opt_k1, opt_k2, max_nfev=40, var=False):
        regmask = self.reg & fmask
        use = self.inlier & regmask[self.obs_f] & self.has_pt[self.obs_t]
        cnt = np.bincount(self.obs_t[use], minlength=self.T)
        use &= cnt[self.obs_t] >= 2
        if use.sum() < 20:
            return None
        cams = np.where(regmask)[0]
        cam_idx = -np.ones(self.N, np.int64)
        cam_idx[cams] = np.arange(len(cams))
        full_use = use
        subset = None
        if len(np.unique(self.obs_t[use])) > self.ba_point_budget:
            subset = self._ba_subset(use, cams)
            use = use & subset[self.obs_t]
        use = self._cap_obs(use, 24)
        pts = np.unique(self.obs_t[use])
        pt_idx = -np.ones(self.T, np.int64)
        pt_idx[pts] = np.arange(len(pts))
        fixed = np.zeros(len(cams), bool)
        fixed[0] = True
        cf = None
        if var:
            if self.cam_focal is None:
                self.cam_focal = np.full(self.N, self.f)
            cf = self.cam_focal[cams].copy()
        prob = BAProblem(rvecs=self.rv[cams].copy(), tvecs=np.zeros((len(cams), 3)), points=self.pt[pts].copy(),
                         obs_cam=cam_idx[self.obs_f[use]], obs_pt=pt_idx[self.obs_t[use]], obs_xy=self.obs_xy[use],
                         f=self.f, cx=self.cx, cy=self.cy, k1=self.k1, k2=self.k2, fixed_cams=fixed, tripod=True,
                         opt_f=opt_f and not var, opt_k1=opt_k1, opt_k2=opt_k2, cam_focal=cf, opt_cam_focal=var,
                         cam_order=np.arange(len(cams)) if var else None,
                         focal_prior=self.focal_prior, focal_prior_weight=self.focal_prior_weight)
        r = bundle_adjust(prob, loss_scale=max(0.5, self.thresh / 2), max_nfev=max_nfev)
        self.rv[cams], self.pt[pts] = r["rvecs"], r["points"]
        self.f, self.k1, self.k2 = r["f"], r["k1"], r["k2"]
        if var:
            self.cam_focal[cams] = r["cam_focal"]
            self.f = float(np.median(self.cam_focal[cams]))
        # re-estimate every point direction with the refined rotations (closed form, vectorised)
        ix = np.where(full_use)[0]
        b = bearings(self._norm_obs(ix))
        Rs = np.array([cv2.Rodrigues(r)[0] for r in self.rv])
        d = np.einsum("nji,nj->ni", Rs[self.obs_f[ix]], b)
        acc = np.zeros((self.T, 3))
        np.add.at(acc, self.obs_t[ix], d)
        upd = np.unique(self.obs_t[ix])
        if subset is not None:
            upd = upd[~subset[upd]]
        nrm = np.linalg.norm(acc[upd], axis=1, keepdims=True)
        self.pt[upd] = acc[upd] / np.maximum(nrm, 1e-12)
        return r

    def _norm_obs(self, ix):
        if self.cam_focal is None:
            return self._norm(self.obs_xy[ix])
        out = np.zeros((len(ix), 2))
        for fi in np.unique(self.obs_f[ix]):
            m = self.obs_f[ix] == fi
            out[m] = self._norm(self.obs_xy[ix[m]], fi)
        return out

    # ------------------------------------------------------------ output
    def build_solution(self, mode, kfs, estimate_f) -> Solution:
        status = []
        kset = set(kfs)
        for i in range(self.N):
            status.append(("keyframe" if i in kset else "solved") if self.reg[i] else "interpolated")
        n_unreg = int((~self.reg).sum())
        if self.reg.sum() < 2:
            raise SolveFailed("Fewer than two frames could be solved.")
        rv, tv = self.rv.copy(), self.tv.copy()
        # interpolate unsolved frames between solved neighbours (flagged)
        regs = np.where(self.reg)[0]
        for i in np.where(~self.reg)[0]:
            lo = regs[regs < i]
            hi = regs[regs > i]
            if len(lo) and len(hi):
                a, b = lo[-1], hi[0]
                w = (i - a) / (b - a)
                qa = rot_to_quat(cv2.Rodrigues(rv[a])[0])
                qb = rot_to_quat(cv2.Rodrigues(rv[b])[0])
                # smoothstep keeps angular velocity continuous-ish at both ends of the gap
                R = quat_to_rot(slerp(qa, qb, w * w * (3 - 2 * w)) if b - a > 2 else slerp(qa, qb, w))
                ca, cb = pose_center(rv[a], tv[a]), pose_center(rv[b], tv[b])
                # cubic Hermite on position using the solved velocities either side of the gap
                a0 = lo[-2] if len(lo) > 1 else a
                b1 = hi[1] if len(hi) > 1 else b
                va = (ca - pose_center(rv[a0], tv[a0])) / max(1, a - a0) * (b - a)
                vb = (pose_center(rv[b1], tv[b1]) - cb) / max(1, b1 - b) * (b - a)
                h00, h10, h01, h11 = 2*w**3 - 3*w**2 + 1, w**3 - 2*w**2 + w, -2*w**3 + 3*w**2, w**3 - w**2
                c = h00 * ca + h10 * va + h01 * cb + h11 * vb
            else:
                a = lo[-1] if len(lo) else hi[0]
                R = cv2.Rodrigues(rv[a])[0]
                c = pose_center(rv[a], tv[a])
            rv[i] = cv2.Rodrigues(R)[0].ravel()
            tv[i] = -R @ c
        cam_focal = self.cam_focal.copy() if self.cam_focal is not None else None

        # ---- gather points ----
        use = self.inlier & self.reg[self.obs_f] & self.has_pt[self.obs_t]
        cnt = np.bincount(self.obs_t[use], minlength=self.T)
        keep = np.where(self.has_pt & (cnt >= 2))[0]
        ix_all = np.where(use)[0]
        err_all = self._errors(ix_all)
        per_obs_err = np.full(self.M, np.nan)
        per_obs_err[ix_all] = err_all
        pts = self.pt[keep].copy()
        if mode == "tripod":
            pass  # directions; scaled to a sphere at normalisation
        colors, perr, pnobs, pang, pobs = [], [], [], [], []
        centers = np.array([pose_center(rv[i], tv[i]) for i in range(self.N)])
        for ti in keep:
            ix = self.track_obs[ti]
            ix = ix[use[ix]]
            e = per_obs_err[ix]
            perr.append(float(np.mean(e)))
            pnobs.append(len(ix))
            fr = self.obs_f[ix]
            if mode == "tripod":
                pang.append(0.0)
            else:
                pang.append(ray_angle_deg(centers[fr[0]], centers[fr[-1]], self.pt[ti]))
            pobs.append([(self.frames[f], float(x), float(y)) for f, (x, y) in zip(fr, self.obs_xy[ix])])
            colors.append(self.tracks[ti].color)
        frame_err = np.full(self.N, np.nan)
        frame_np = np.zeros(self.N, np.int64)
        for fi in range(self.N):
            ix = self.frame_obs[fi]
            ix = ix[use[ix]]
            if len(ix):
                frame_err[fi] = float(np.mean(per_obs_err[ix]))
            frame_np[fi] = len(ix)

        # ---- normalise: first camera at origin, median scene depth = 10 units ----
        ref = int(regs[0])
        R0 = cv2.Rodrigues(rv[ref])[0]
        c0 = pose_center(rv[ref], tv[ref])
        if mode == "tripod":
            s = 1.0
            pts_n = (pts @ R0.T)
            pts_n = 10.0 * pts_n / np.linalg.norm(pts_n, axis=1, keepdims=True)
        else:
            depths = np.array([(cv2.Rodrigues(rv[fr0])[0] @ pts.T + tv[fr0][:, None])[2] for fr0 in regs[::max(1, len(regs) // 20)]])
            md = float(np.median(depths[depths > 0])) if (depths > 0).any() else 1.0
            s = 10.0 / md
            pts_n = s * ((pts - c0) @ R0.T)
        rv_n, tv_n = np.zeros_like(rv), np.zeros_like(tv)
        for i in range(self.N):
            R = cv2.Rodrigues(rv[i])[0]
            c = pose_center(rv[i], tv[i])
            Rn = R @ R0.T
            cn = s * (R0 @ (c - c0)) if mode != "tripod" else np.zeros(3)
            rv_n[i] = cv2.Rodrigues(Rn)[0].ravel()
            tv_n[i] = -Rn @ cn

        all_e = err_all[np.isfinite(err_all)]
        stats = dict(
            mean_error_px=float(np.mean(all_e)) if len(all_e) else float("nan"),
            median_error_px=float(np.median(all_e)) if len(all_e) else float("nan"),
            rms_error_px=float(np.sqrt(np.mean(all_e ** 2))) if len(all_e) else float("nan"),
            p95_error_px=float(np.percentile(all_e, 95)) if len(all_e) else float("nan"),
            n_points=int(len(keep)), n_observations=int(len(all_e)), n_frames=self.N,
            n_solved_frames=int(self.reg.sum()), n_interpolated_frames=n_unreg,
            n_keyframes=len(kfs), n_usable_tracks=self.T,
            focal_px=float(self.f), k1=float(self.k1), k2=float(self.k2),
            focal_estimated=bool(estimate_f),
            focal_prior_px=float(self.focal_prior) if estimate_f else None,
            median_triangulation_angle=float(np.median(pang)) if pang and mode != "tripod" else 0.0,
            focal_sweep=[[float(a), float(b), float(c)] for a, b, c in getattr(self, "focal_sweep", [])],
        )
        warnings = self._warnings(stats, frame_np, mode, estimate_f)
        sol = Solution(mode=mode, engine="native", proc_w=self.w, proc_h=self.h, scale=self.ts.scale,
                       f=float(self.f), cx=self.cx, cy=self.cy, k1=float(self.k1), k2=float(self.k2),
                       frames=list(self.frames), rvecs=rv_n, tvecs=tv_n, frame_status=status, cam_focal=cam_focal,
                       point_ids=np.array([self.tracks[t].id for t in keep], np.int64), points=pts_n,
                       colors=np.array(colors, np.uint8).reshape(-1, 3), point_error=np.array(perr),
                       point_nobs=np.array(pnobs, np.int64), point_angle=np.array(pang), point_obs=pobs,
                       frame_error=frame_err, frame_npoints=frame_np, stats=stats, warnings=warnings)
        return sol

    def _warnings(self, st, frame_np, mode, estimate_f):
        w = []
        if st["mean_error_px"] > 1.0:
            w.append(f"High average reprojection error ({st['mean_error_px']:.2f}px). The solve is not reliable; "
                     "check for moving objects, rolling shutter, or try a different solve method.")
        elif st["mean_error_px"] > 0.5:
            w.append(f"Moderate reprojection error ({st['mean_error_px']:.2f}px). Usable but not a tight solve.")
        for sg in getattr(self, "segments", []):
            if sg["mode"] == "estimated":
                w.append(f"Frames {sg['frames'][0]}-{sg['frames'][1]} were solved as a separate segment that shares no "
                         "features with the rest of the clip; its placement relative to the rest is ESTIMATED from "
                         "camera-motion continuity, not measured. Expect a jump or scale change there.")
            elif sg["mode"] == "measured":
                w.append(f"Frames {sg['frames'][0]}-{sg['frames'][1]} were re-initialised as a separate segment and "
                         f"joined to the main solve through {sg['shared_points']} shared 3D points.")
            elif sg["mode"] == "measured-frames":
                w.append(f"Frames {sg['frames'][0]}-{sg['frames'][1]} were re-initialised as a separate segment and "
                         f"joined to the main solve through {sg['shared_frames']} frame(s) located in both; the join "
                         "is measured but weakly constrained, so check that stretch.")
        if st["n_interpolated_frames"] > 0:
            ranges, i = [], 0
            unreg = ~self.reg
            while i < self.N:
                if unreg[i]:
                    j = i
                    while j + 1 < self.N and unreg[j + 1]:
                        j += 1
                    ranges.append(f"{self.frames[i]}" if i == j else f"{self.frames[i]}-{self.frames[j]}")
                    i = j + 1
                else:
                    i += 1
            shown = ", ".join(ranges[:8]) + (" ..." if len(ranges) > 8 else "")
            w.append(f"{st['n_interpolated_frames']} frame(s) had nothing trackable and were interpolated "
                     f"(smooth curve between solved neighbours): frames {shown}.")
        low = int(np.sum(frame_np < 15))
        if low:
            w.append(f"{low} frame(s) have fewer than 15 solved points; the camera there is weakly constrained.")
        if mode != "tripod" and st["median_triangulation_angle"] < 2.0:
            w.append(f"Low parallax (median triangulation angle {st['median_triangulation_angle']:.2f} deg). "
                     "Depth and focal length are poorly constrained.")
        if mode == "tripod":
            w.append("Tripod solve: camera rotation only. Points have no real depth (placed on a sphere); "
                     "a ground plane cannot be measured from this footage.")
        if estimate_f:
            sweep = getattr(self, "focal_sweep", [])
            fs = np.array([s[1] for s in sweep if np.isfinite(s[2])])
            es = np.array([s[2] for s in sweep if np.isfinite(s[2])])
            if len(es) > 2:
                best = es.min()
                alt = fs[(es < best * 1.10 + 0.02)]
                spread = (alt.max() - alt.min()) / max(1e-9, np.median(alt)) if len(alt) else 0
                if spread > 0.15:
                    w.append(f"Focal length is weakly constrained by this footage (hypotheses within 10% error span "
                             f"{alt.min():.0f}-{alt.max():.0f}px). Enter the known focal length if available.")
        if st["n_points"] < 60:
            w.append(f"Only {st['n_points']} points were reconstructed; the footage has few reliable features.")
        if abs(st["k1"]) > 0.3:
            w.append(f"Large estimated distortion k1={st['k1']:.3f}; verify the lens model.")
        return w


def solve(ts: TrackSet, settings: AnalysisSettings, report, exclude_ids=None) -> Solution:
    sfm = SequentialSfM(ts, settings, report, exclude_ids)
    return sfm.solve()
