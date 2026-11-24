"""Multi-frame feature tracking.

Pipeline per frame
------------------
1. Pyramidal Lucas-Kanade optical flow (OpenCV) for all active tracks.
2. Forward-backward consistency check (track back to the previous frame).
3. Geometric verification: RANSAC fundamental matrix (USAC/MAGSAC) between the
   current frame and a frame ``gap`` frames back. Tracks that are repeatedly
   epipolar outliers are classified as *dynamic* (independently moving objects)
   and their image region is masked against re-detection for a while.
4. Descriptor verification: ORB descriptors are recomputed periodically and
   compared to the last verified descriptor (Hamming distance) to catch
   tracker drift / feature swaps.
5. Re-detection of Shi-Tomasi corners in empty regions. New detections are
   matched (ORB descriptor + predicted position) against recently lost tracks so
   features that were briefly occluded continue under their original id.

All coordinates are in *processing-resolution* pixels (pixel centres at
integer coordinates, OpenCV convention).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ..config import AnalysisSettings, QUALITY_PARAMS
from ..footage import iter_frames, processing_size
from ..solver.geometry import robust_fundamental


@dataclass(eq=False)   # identity comparison: Tracks are removed from lists by the thousand
class Track:
    id: int
    frames: list = field(default_factory=list)
    xy: list = field(default_factory=list)
    color: tuple = (200, 200, 200)
    desc: Optional[np.ndarray] = None
    checks: int = 0
    outliers: int = 0
    consecutive_outliers: int = 0
    dynamic: bool = False
    reacquired: int = 0
    end_reason: str = ""
    anchor_k: int = -1            # sequence index of the anchor (reference appearance) frame
    anchor_xy: tuple = (0.0, 0.0)
    kind: str = "klt"             # "klt" (optical-flow track) or "link" (wide-baseline SIFT link track)


class TrackSet:
    """Result of tracking. Serialisable to/from npz."""

    def __init__(self, tracks: list[Track], frames: list[int], proc_w: int, proc_h: int, scale: float, stats: dict):
        self.tracks = tracks
        self.frames = frames
        self.proc_w, self.proc_h, self.scale = proc_w, proc_h, scale
        self.stats = stats

    # --- persistence ---
    def save(self, path: str):
        lens = np.array([len(t.frames) for t in self.tracks], np.int32)
        np.savez_compressed(
            path,
            ids=np.array([t.id for t in self.tracks], np.int64),
            lens=lens,
            frames=np.concatenate([np.asarray(t.frames, np.int32) for t in self.tracks]) if self.tracks else np.zeros(0, np.int32),
            xy=np.concatenate([np.asarray(t.xy, np.float32).reshape(-1, 2) for t in self.tracks]) if self.tracks else np.zeros((0, 2), np.float32),
            colors=np.array([t.color for t in self.tracks], np.uint8).reshape(-1, 3),
            dynamic=np.array([t.dynamic for t in self.tracks], bool),
            reacq=np.array([t.reacquired for t in self.tracks], np.int32),
            link=np.array([t.kind == "link" for t in self.tracks], bool),
            seq_frames=np.asarray(self.frames, np.int32),
            meta=np.array([self.proc_w, self.proc_h, self.scale], np.float64),
            stats=np.array([repr(self.stats)]),
        )

    @classmethod
    def load(cls, path: str) -> "TrackSet":
        # NpzFile decompresses an array on *every* item access, so read each array exactly once
        # (indexing z["frames"] inside the per-track loop made loading O(tracks x file size)).
        with np.load(path, allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        lens, ids, frames, xy = a["lens"], a["ids"], a["frames"], a["xy"]
        colors, dyn, reacq = a["colors"], a["dynamic"], a["reacq"]
        link = a.get("link")
        offs = np.concatenate([[0], np.cumsum(lens)])
        tracks = []
        for i in range(len(lens)):
            o, e = offs[i], offs[i + 1]
            t = Track(id=int(ids[i]))
            t.frames = frames[o:e].tolist()
            t.xy = [tuple(p) for p in xy[o:e].tolist()]
            t.color = tuple(int(c) for c in colors[i])
            t.dynamic = bool(dyn[i])
            t.reacquired = int(reacq[i])
            if link is not None and bool(link[i]):
                t.kind = "link"
            tracks.append(t)
        pw, ph, s = a["meta"]
        import ast
        stats = ast.literal_eval(str(a["stats"][0]))
        return cls(tracks, a["seq_frames"].tolist(), int(pw), int(ph), float(s), stats)

    def usable(self, min_len: int, exclude: Optional[set] = None) -> list[Track]:
        exclude = exclude or set()
        # link tracks only exist on link frames (~3/s), so they are judged by observation count
        return [t for t in self.tracks if not t.dynamic and t.id not in exclude and
                (len(t.frames) >= 3 if t.kind == "link" else len(t.frames) >= min_len)]


class FeatureTracker:
    def __init__(self, settings: AnalysisSettings, proc_w: int, proc_h: int):
        self.s = settings
        q = QUALITY_PARAMS[settings.detection_quality]
        res = max(proc_w, proc_h) / 1280.0
        sens = settings.tracking_sensitivity
        self.w, self.h = proc_w, proc_h
        self.quality = q["quality"] * (1.6 - 1.2 * sens)
        self.min_dist = max(4, int(round(q["min_dist"] * res)))
        win = int(q["win"] * max(0.6, res)) | 1
        self.lk = dict(winSize=(win, win), maxLevel=q["levels"],
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
        self.redetect_every = q["redetect_every"]
        self.fb_thresh = (0.6 + 1.2 * sens) * max(0.75, res)
        self.f_thresh = (0.8 + 1.2 * sens) * max(0.75, res)
        self.hamming_drift = int(48 + 40 * sens)       # of 256 bits
        self.hamming_reacq = int(36 + 24 * sens)
        self.margin = 8
        self.orb = cv2.ORB_create(nfeatures=1, edgeThreshold=15, patchSize=31)
        self.desc_every = 4
        self.reacq_window = 40
        self.max_points = settings.max_points
        self.anchor_max_age = 30
        self.anchor_policy = "end"
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # Fraction of recent good-frame corner strength below which a corner is noise. Low on purpose: plain
        # painted/epoxy floors have only faint texture, and a stricter floor discarded it (142 -> 324 usable
        # tracks on the garage clip when relaxed); the forward-backward and anchor checks reject real noise.
        # Every lost track is a re-acquisition candidate, however short. Skipping fragments under 5 frames
        # made re-acquisition 11x cheaper on its own but broke solves on noisy / blurred clips (a dead-zone
        # clip went from 0.16% to 7% path error): stitched short fragments are what keep such footage solvable.
        # The speed comes from the KD-tree search, not from the filter.
        self.reacq_min_len = 1
        self.strength_floor = 0.06
        # Contrast-limited adaptive histogram equalisation before detection and tracking: brings out the faint
        # texture of plain surfaces (about 2x the usable tracks in the same stretch)
        self.use_clahe = True
        self.anchor_noise_factor = 3.0
        self._strength_hist: list[float] = []
        # tight appearance check: a track whose position no longer matches its reference template to within
        # ~0.4 px has started to slide; it is ended (anchor_policy) rather than allowed to drift
        self.anchor_tol = 0.4 * max(0.75, res)

    # ------------------------------------------------------------------
    def _descriptors(self, gray: np.ndarray, pts: np.ndarray):
        """ORB descriptors at given points. Returns (desc[N,32] or None rows, valid mask)."""
        n = len(pts)
        out = np.zeros((n, 32), np.uint8)
        valid = np.zeros(n, bool)
        if n == 0:
            return out, valid
        kps = [cv2.KeyPoint(float(x), float(y), 31, -1, 0, 0, i) for i, (x, y) in enumerate(pts)]
        kps2, desc = self.orb.compute(gray, kps)
        if desc is not None:
            for kp, d in zip(kps2, desc):
                out[kp.class_id] = d
                valid[kp.class_id] = True
        return out, valid

    @staticmethod
    def _hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.unpackbits(np.bitwise_xor(a, b), axis=-1).sum(-1)

    def run(self, video_path: str, start: int, end: int, step: int, scale: float, report, linker=None) -> TrackSet:
        tracks: list[Track] = []
        active: list[Track] = []           # tracks alive in previous frame
        lost: list[Track] = []             # candidates for re-acquisition
        next_id = 0
        prev_gray = None
        seq_frames: list[int] = []
        history: dict[int, dict[int, tuple]] = {}   # frame -> {track_id: (x,y)} for gap checks
        dynamic_zones: list[tuple] = []    # (x, y, radius, expires_at_frame_index)
        anchors: dict[int, np.ndarray] = {}  # sequence index -> grey image referenced by tracks
        stats = dict(detected=0, fb_rejected=0, epipolar_rejected=0, descriptor_rejected=0,
                     dynamic_tracks=0, reacquired=0, left_frame=0, per_frame_active=[])
        total = (end - start) // step + 1
        k = -1
        for n, frame in iter_frames(video_path, start, end, step):
            k += 1
            report(k / max(1, total), f"Tracking frame {n} ({len(active)} active tracks)")
            if scale != 1.0:
                frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.use_clahe:
                gray = self._clahe.apply(gray)
            seq_frames.append(n)
            if linker is not None:
                linker.maybe_add(k, n, gray, frame, force=(n + step > end))

            # ---- 1+2: optical flow with forward-backward check ----
            if prev_gray is not None and active:
                p0 = np.array([t.xy[-1] for t in active], np.float32).reshape(-1, 1, 2)
                p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **self.lk)
                p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **self.lk)
                fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
                p1 = p1.reshape(-1, 2)
                inside = ((p1[:, 0] >= self.margin) & (p1[:, 0] < self.w - self.margin) &
                          (p1[:, 1] >= self.margin) & (p1[:, 1] < self.h - self.margin))
                good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < self.fb_thresh) & inside
                survivors = []
                for t, ok, p, ins in zip(active, good, p1, inside):
                    if ok:
                        t.frames.append(n)
                        t.xy.append((float(p[0]), float(p[1])))
                        survivors.append(t)
                    else:
                        t.end_reason = "left_frame" if not ins else "flow"
                        if ins:
                            stats["fb_rejected"] += 1
                            lost.append(t)
                        else:
                            stats["left_frame"] += 1
                active = survivors

                # ---- 2b: drift removal - refine against each track's anchor appearance ----
                drop = self._anchor_refine(active, anchors, gray, k, stats)
                if drop:
                    for i in drop:
                        active[i].end_reason = "anchor"
                    stats["anchor_ended"] = stats.get("anchor_ended", 0) + len(drop)
                    active = [t for i, t in enumerate(active) if i not in drop]

                # ---- 3: epipolar verification against an earlier frame ----
                gap_frame = seq_frames[max(0, len(seq_frames) - 1 - 5)]
                if gap_frame != n and gap_frame in history:
                    ref = history[gap_frame]
                    idx = [i for i, t in enumerate(active) if t.id in ref]
                    if len(idx) >= 30:
                        a = np.array([ref[active[i].id] for i in idx], np.float64)
                        b = np.array([active[i].xy[-1] for i in idx], np.float64)
                        disp = np.median(np.linalg.norm(a - b, axis=1))
                        if disp > 0.5:  # need some motion for F to mean anything
                            F, mask = robust_fundamental(a, b, self.f_thresh)
                            if F is not None and F.shape == (3, 3):
                                # Sampson distance for every correspondence
                                ah = np.c_[a, np.ones(len(a))]
                                bh = np.c_[b, np.ones(len(b))]
                                Fa = ah @ F.T
                                Ftb = bh @ F
                                num = np.sum(bh * Fa, axis=1) ** 2
                                den = Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Ftb[:, 0] ** 2 + Ftb[:, 1] ** 2
                                sd = np.sqrt(num / np.maximum(den, 1e-12))
                                kill = set()
                                for j, i in enumerate(idx):
                                    t = active[i]
                                    t.checks += 1
                                    if sd[j] > self.f_thresh:
                                        t.outliers += 1
                                        t.consecutive_outliers += 1
                                    else:
                                        t.consecutive_outliers = 0
                                    if sd[j] > 4 * self.f_thresh or t.consecutive_outliers >= 3 or \
                                            (t.checks >= 6 and t.outliers / t.checks > 0.35):
                                        kill.add(i)
                                        if t.consecutive_outliers >= 2 or t.outliers >= 3:
                                            t.dynamic = True
                                if kill:
                                    newactive = []
                                    for i, t in enumerate(active):
                                        if i in kill:
                                            t.end_reason = "epipolar"
                                            stats["epipolar_rejected"] += 1
                                            if t.dynamic:
                                                stats["dynamic_tracks"] += 1
                                                x, y = t.xy[-1]
                                                dynamic_zones.append((x, y, 3 * self.min_dist, k + 30))
                                        else:
                                            newactive.append(t)
                                    active = newactive

                # ---- 4: descriptor drift check ----
                if k % self.desc_every == 0 and active:
                    pts = np.array([t.xy[-1] for t in active], np.float32)
                    desc, valid = self._descriptors(gray, pts)
                    keep = []
                    for t, d, v in zip(active, desc, valid):
                        if not v:
                            keep.append(t)
                            continue
                        if t.desc is not None and self._hamming(t.desc, d) > self.hamming_drift:
                            t.end_reason = "descriptor"
                            stats["descriptor_rejected"] += 1
                            # drop the last few (possibly drifted) samples
                            cut = min(len(t.frames) - 1, self.desc_every)
                            del t.frames[-cut:]
                            del t.xy[-cut:]
                            continue
                        t.desc = d
                        keep.append(t)
                    active = keep

            # expire dynamic zones / old lost tracks
            dynamic_zones = [z for z in dynamic_zones if z[3] > k]
            lost = [t for t in lost if n - t.frames[-1] <= self.reacq_window * step and not t.dynamic]

            # ---- 5: re-detection & re-acquisition ----
            if (k % self.redetect_every == 0 or len(active) < 0.5 * self.max_points) and len(active) < self.max_points:
                mask = np.full((self.h, self.w), 255, np.uint8)
                cv2.rectangle(mask, (0, 0), (self.w - 1, self.h - 1), 0, self.margin * 2)
                for t in active:
                    cv2.circle(mask, (int(t.xy[-1][0]), int(t.xy[-1][1])), self.min_dist, 0, -1)
                for (x, y, r, _) in dynamic_zones:
                    cv2.circle(mask, (int(x), int(y)), int(r), 0, -1)
                need = self.max_points - len(active)
                corners = cv2.goodFeaturesToTrack(gray, maxCorners=need, qualityLevel=self.quality,
                                                  minDistance=self.min_dist, mask=mask, blockSize=7,
                                                  useHarrisDetector=False)
                if corners is not None and len(corners):
                    corners = corners.reshape(-1, 2)
                    corners = self._strength_filter(gray, corners, stats)
                if corners is not None and len(corners):
                    cv2.cornerSubPix(gray, corners.reshape(-1, 1, 2), (5, 5), (-1, -1),
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
                    desc, valid = self._descriptors(gray, corners)
                    reused = self._reacquire(corners, desc, valid, lost, active, history, n)
                    for i, (c, d, v) in enumerate(zip(corners, desc, valid)):
                        if not v:
                            continue
                        if i in reused:
                            t = reused[i]
                            t.reacquired += 1
                            stats["reacquired"] += 1
                            lost.remove(t)
                        else:
                            t = Track(id=next_id)
                            next_id += 1
                            tracks.append(t)
                            px = frame[int(round(c[1])), int(round(c[0]))]
                            t.color = (int(px[2]), int(px[1]), int(px[0]))
                            stats["detected"] += 1
                        t.frames.append(n)
                        t.xy.append((float(c[0]), float(c[1])))
                        t.desc = d
                        t.anchor_k = k
                        t.anchor_xy = (float(c[0]), float(c[1]))
                        active.append(t)

            # keep grey images only for frames that are still someone's anchor
            used = {t.anchor_k for t in active}
            if k in used:
                anchors[k] = gray
            for ak in list(anchors.keys()):
                if ak not in used:
                    del anchors[ak]
            history[n] = {t.id: t.xy[-1] for t in active}
            old = seq_frames[max(0, len(seq_frames) - self.reacq_window - 2)]
            for f in list(history.keys()):
                if f < old:
                    del history[f]
            stats["per_frame_active"].append(len(active))
            prev_gray = gray

        report(1.0, "Tracking complete")
        lengths = [len(t.frames) for t in tracks]
        stats["total_tracks"] = len(tracks)
        stats["mean_track_length"] = float(np.mean(lengths)) if lengths else 0.0
        stats["frames"] = len(seq_frames)
        return TrackSet(tracks, seq_frames, self.w, self.h, scale, stats)

    def _strength_filter(self, gray, corners, stats):
        """Absolute corner-strength floor. goodFeaturesToTrack's threshold is relative to the
        strongest corner *in the same frame*, so a blurred / dark frame (whip pan, covered lens)
        happily yields thousands of noise 'corners' that die immediately. Compare against the
        typical strength in recent good frames instead."""
        eig = cv2.cornerMinEigenVal(gray, 7, 3)
        ij = np.clip(np.round(corners).astype(int), 0, [self.w - 1, self.h - 1])
        strength = eig[ij[:, 1], ij[:, 0]]
        self._strength_hist.append(float(np.percentile(strength, 90)))
        if len(self._strength_hist) > 200:
            self._strength_hist.pop(0)
        ref = float(np.percentile(self._strength_hist, 75))
        keep = strength >= self.strength_floor * ref
        if (~keep).any():
            stats["weak_corners_rejected"] = stats.get("weak_corners_rejected", 0) + int((~keep).sum())
        return corners[keep]

    def _anchor_refine(self, active, anchors, gray, k, stats):
        """Re-align each track against its anchor frame (template) to cancel the
        drift that chained frame-to-frame optical flow accumulates."""
        groups: dict[int, list[int]] = {}
        drop: set[int] = set()
        for i, t in enumerate(active):
            if 0 <= t.anchor_k < k and t.anchor_k in anchors:
                groups.setdefault(t.anchor_k, []).append(i)
            elif t.anchor_k != k:
                t.anchor_k, t.anchor_xy = k, t.xy[-1]
        results = []
        for ak, idxs in groups.items():
            a_pts = np.array([active[i].anchor_xy for i in idxs], np.float32).reshape(-1, 1, 2)
            guess = np.array([active[i].xy[-1] for i in idxs], np.float32).reshape(-1, 1, 2)
            ref, st, err = cv2.calcOpticalFlowPyrLK(anchors[ak], gray, a_pts, guess.copy(),
                                                    winSize=self.lk["winSize"], maxLevel=2,
                                                    criteria=self.lk["criteria"], flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            d = np.linalg.norm((ref - guess).reshape(-1, 2), axis=1)
            results.append((ak, idxs, ref.reshape(-1, 2), st.ravel(), d))
        # Tolerance adapts to this frame's localisation noise: on clean texture tracks agree with their
        # template to ~0.1 px and anything past 0.4 px is sliding; on faint, noisy texture the typical
        # disagreement is larger, and ending every track that exceeds a fixed 0.4 px would leave too few
        # long tracks. Use max(base, 3 x median disagreement).
        all_d = np.concatenate([r[4][r[3] == 1] for r in results]) if results else np.zeros(0)
        tol = max(self.anchor_tol, self.anchor_noise_factor * float(np.median(all_d))) if len(all_d) >= 20 else self.anchor_tol
        for ak, idxs, ref, st, d in results:
            for j, i in enumerate(idxs):
                t = active[i]
                if st[j] == 1 and d[j] < tol:
                    t.xy[-1] = (float(ref[j, 0]), float(ref[j, 1]))
                    stats["anchor_refined"] = stats.get("anchor_refined", 0) + 1
                    if k - ak >= self.anchor_max_age:
                        t.anchor_k, t.anchor_xy = k, t.xy[-1]
                else:
                    stats["anchor_reset"] = stats.get("anchor_reset", 0) + 1
                    if self.anchor_policy == "end":
                        # the feature no longer matches its reference appearance: frame-to-frame flow has
                        # slid off it. End the track (a fresh one will be detected) instead of re-anchoring,
                        # which would lock the accumulated drift into the template.
                        drop.add(i)
                    else:
                        t.anchor_k, t.anchor_xy = k, t.xy[-1]
        return drop

    def _reacquire(self, corners, desc, valid, lost, active, history, n):
        """Match new detections to recently lost tracks. Returns {corner_index: Track}.

        Lost tracks are grouped by the frame they were lost in, so the motion of the surviving
        neighbours is computed once per group (KD-tree), and candidate corners come from a radius
        query instead of a scan of every corner. Tracks that lived fewer than 5 frames are not worth
        re-acquiring (and on faint texture there are tens of thousands of them)."""
        if not lost or len(corners) == 0:
            return {}
        from scipy.spatial import cKDTree
        vidx = np.where(valid)[0]
        if len(vidx) == 0:
            return {}
        ctree = cKDTree(corners[vidx])
        cur = {t.id: t.xy[-1] for t in active}
        by_frame: dict[int, list] = {}
        for t in lost:
            if t.desc is not None and len(t.frames) >= self.reacq_min_len:
                by_frame.setdefault(t.frames[-1], []).append(t)
        out, taken = {}, set()
        for f_l, group in by_frame.items():
            ref = history.get(f_l)
            if ref is None:
                continue
            ids = [i for i in ref if i in cur]
            if len(ids) < 6:
                continue
            a = np.array([ref[i] for i in ids])
            b = np.array([cur[i] for i in ids])
            lasts = np.array([t.xy[-1] for t in group])
            _, nn = cKDTree(a).query(lasts, k=min(8, len(a)))
            nn = nn.reshape(len(group), -1)
            preds = lasts + np.median(b[nn] - a[nn], axis=1)   # predicted position of each lost track
            radius = 6.0 + 0.5 * (n - f_l)
            for t, pred in zip(group, preds):
                cand = ctree.query_ball_point(pred, radius)
                if not cand:
                    continue
                cand = vidx[np.asarray(cand)]
                hd = self._hamming(desc[cand], t.desc)
                order = np.argsort(hd)
                best = cand[order[0]]
                if hd[order[0]] > self.hamming_reacq or best in taken:
                    continue
                if len(order) > 1 and hd[order[1]] < hd[order[0]] * 1.15 + 2:
                    continue  # ambiguous
                out[int(best)] = t
                taken.add(best)
        return out


def run_tracking(video_path: str, info, settings: AnalysisSettings, report) -> TrackSet:
    pw, ph, scale = processing_size(info.width, info.height, settings.processing_width)
    start = settings.frame_start if settings.frame_start is not None else 0
    end = settings.frame_end if settings.frame_end is not None else info.frame_count - 1
    start = max(0, min(start, info.frame_count - 1))
    end = max(start, min(end, info.frame_count - 1))
    from ..config import effective_frame_step
    step = effective_frame_step(settings, info.fps, end - start + 1)
    if step != settings.frame_step and hasattr(report, "log"):
        report.log(f"Auto frame sampling: tracking every {step} frame(s) "
                   f"({(end - start) // step + 1} of {end - start + 1} frames; {info.fps / step:.1f} fps effective)")
    tr = FeatureTracker(settings, pw, ph)
    from .linking import LinkFeatureCollector, build_link_tracks, connectivity_gaps, merge_ranges, weak_stretches
    eff_fps = max(1.0, info.fps / step)
    linker = LinkFeatureCollector(every=max(1, int(round(eff_fps / 3))), proc_w=pw, proc_h=ph)
    ts = tr.run(video_path, start, end, step, scale, report, linker=linker)
    ts.stats["frame_step"] = step
    # ---- wide-baseline linking (bridges dead zones, limits drift on long clips) ----
    if hasattr(report, "log"):
        report.log(f"Collected {len(linker.frames)} wide-baseline link frames (SIFT)")
    window = int(round(4.0 * eff_fps))
    breaks = connectivity_gaps(ts.frames, ts.tracks)
    weak = weak_stretches(ts.frames, ts.tracks, ph)
    gaps = merge_ranges(breaks + weak, pad=step)
    link_tracks, lstats = [], {"link_tracks": 0, "verified_matches": 0, "pairs_linked": 0, "pairs_tested": 0}
    link_seq = []
    if gaps:
        # Link tracks are only used to bridge places where optical-flow tracking actually broke; clips that
        # track continuously are solved exactly as before (wide-baseline matches carry more outliers).
        res = max(pw, ph) / 1280.0
        link_tracks, lstats = build_link_tracks(linker, window, 1.5 * max(0.75, res), report, ratio=0.75, mode="star")
        span = lambda t: any(min(t.frames) <= a and max(t.frames) >= b for a, b in gaps)
        link_tracks = [t for t in link_tracks if span(t)]
        lstats["link_tracks"] = len(link_tracks)
        fpos = {n: i for i, n in enumerate(ts.frames)}
        near = lambda n: any(fpos.get(a, 0) - window <= fpos[n] <= fpos.get(b, 0) + window for a, b in gaps)
        link_seq = [k for k, n in zip(linker.seq_idx, linker.frames) if near(n)]
    ts.tracks.extend(link_tracks)
    ts.stats.update({f"link_{k}" if not k.startswith("link") else k: v for k, v in lstats.items()})
    ts.stats["link_seq_idx"] = link_seq
    ts.stats["dead_zones"] = breaks
    ts.stats["weak_stretches"] = weak
    if hasattr(report, "log") and gaps:
        report.log(f"Link tracks bridging tracking breaks: {lstats['link_tracks']} "
                   f"(from {lstats['verified_matches']} verified matches, {lstats['pairs_linked']}/{lstats['pairs_tested']} frame pairs)")
        for a, b in gaps:
            kind = "broke" if any(x <= a and b <= y + step for x, y in breaks) else "is too sparse or lopsided"
            report.log(f"Tracking {kind} in frames {a}-{b}; bridging it with wide-baseline link tracks")
    return ts
