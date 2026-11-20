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


@dataclass
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
            seq_frames=np.asarray(self.frames, np.int32),
            meta=np.array([self.proc_w, self.proc_h, self.scale], np.float64),
            stats=np.array([repr(self.stats)]),
        )

    @classmethod
    def load(cls, path: str) -> "TrackSet":
        z = np.load(path, allow_pickle=False)
        tracks = []
        o = 0
        for i, n in enumerate(z["lens"]):
            t = Track(id=int(z["ids"][i]))
            t.frames = z["frames"][o:o + n].tolist()
            t.xy = z["xy"][o:o + n].tolist()
            t.color = tuple(int(c) for c in z["colors"][i])
            t.dynamic = bool(z["dynamic"][i])
            t.reacquired = int(z["reacq"][i])
            tracks.append(t)
            o += n
        pw, ph, s = z["meta"]
        import ast
        stats = ast.literal_eval(str(z["stats"][0]))
        return cls(tracks, z["seq_frames"].tolist(), int(pw), int(ph), float(s), stats)

    def usable(self, min_len: int, exclude: Optional[set] = None) -> list[Track]:
        exclude = exclude or set()
        return [t for t in self.tracks if not t.dynamic and len(t.frames) >= min_len and t.id not in exclude]


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
        self.anchor_tol = 1.0 * max(0.75, res)

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

    def run(self, video_path: str, start: int, end: int, step: int, scale: float, report) -> TrackSet:
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
            seq_frames.append(n)

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
                self._anchor_refine(active, anchors, gray, k, stats)

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
                            F, mask = cv2.findFundamentalMat(a, b, cv2.USAC_MAGSAC, self.f_thresh, 0.999, 2000)
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

    def _anchor_refine(self, active, anchors, gray, k, stats):
        """Re-align each track against its anchor frame (template) to cancel the
        drift that chained frame-to-frame optical flow accumulates."""
        groups: dict[int, list[int]] = {}
        for i, t in enumerate(active):
            if 0 <= t.anchor_k < k and t.anchor_k in anchors:
                groups.setdefault(t.anchor_k, []).append(i)
            elif t.anchor_k != k:
                t.anchor_k, t.anchor_xy = k, t.xy[-1]
        for ak, idxs in groups.items():
            a_pts = np.array([active[i].anchor_xy for i in idxs], np.float32).reshape(-1, 1, 2)
            guess = np.array([active[i].xy[-1] for i in idxs], np.float32).reshape(-1, 1, 2)
            ref, st, err = cv2.calcOpticalFlowPyrLK(anchors[ak], gray, a_pts, guess.copy(),
                                                    winSize=self.lk["winSize"], maxLevel=2,
                                                    criteria=self.lk["criteria"], flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            d = np.linalg.norm((ref - guess).reshape(-1, 2), axis=1)
            ref = ref.reshape(-1, 2)
            for j, i in enumerate(idxs):
                t = active[i]
                if st[j, 0] == 1 and d[j] < self.anchor_tol:
                    t.xy[-1] = (float(ref[j, 0]), float(ref[j, 1]))
                    stats["anchor_refined"] = stats.get("anchor_refined", 0) + 1
                    if k - ak >= self.anchor_max_age:
                        t.anchor_k, t.anchor_xy = k, t.xy[-1]
                else:
                    # appearance changed too much: start a new anchor here
                    t.anchor_k, t.anchor_xy = k, t.xy[-1]
                    stats["anchor_reset"] = stats.get("anchor_reset", 0) + 1

    def _reacquire(self, corners, desc, valid, lost, active, history, n):
        """Match new detections to recently lost tracks. Returns {corner_index: Track}."""
        if not lost:
            return {}
        cur = {t.id: t.xy[-1] for t in active}
        out = {}
        taken = set()
        for t in lost:
            f_l = t.frames[-1]
            ref = history.get(f_l)
            if ref is None or t.desc is None:
                continue
            # predict position from motion of nearby tracks that survived since f_l
            common = [(ref[i], cur[i]) for i in ref.keys() & cur.keys()]
            if len(common) < 6:
                continue
            a = np.array([c[0] for c in common])
            b = np.array([c[1] for c in common])
            last = np.array(t.xy[-1])
            near = np.argsort(np.linalg.norm(a - last, axis=1))[:8]
            pred = last + np.median(b[near] - a[near], axis=0)
            radius = 6.0 + 0.5 * (n - f_l)
            d = np.linalg.norm(corners - pred, axis=1)
            cand = np.where((d < radius) & valid)[0]
            if len(cand) == 0:
                continue
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
    tr = FeatureTracker(settings, pw, ph)
    return tr.run(video_path, start, end, settings.frame_step, scale, report)
