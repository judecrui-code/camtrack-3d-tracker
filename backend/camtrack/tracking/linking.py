"""Wide-baseline feature linking (gap bridging and drift control).

Frame-to-frame optical flow (KLT) cannot survive a stretch of footage with no
trackable detail: heavy motion blur, a whip pan, the lens briefly covered, a
blank wall. Every track dies there, so the frames after the dead zone share no
features with the frames before it and a sequential solve breaks in two.

This module adds a second, independent kind of correspondence:

1. SIFT features (scale- and rotation-invariant) are extracted on "link frames",
   about three per second, during the tracking pass.
2. Every link frame is matched against every other link frame within a window
   of a few seconds, *including pairs on opposite sides of a dead zone*. Matches
   use Lowe's ratio test plus a mutual-consistency check, then a robust
   fundamental-matrix check (which also rejects independently moving objects).
3. Verified matches are chained with union-find into multi-view "link tracks".
   Components that contain two different keypoints in the same frame are
   ambiguous and discarded.

The link tracks are ordinary tracks for the solver (observations only at link
frames), so frames after a gap can be registered against 3D points triangulated
before it, and the long-baseline constraints also reduce drift on long clips.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..solver.geometry import robust_fundamental

LINK_ID_OFFSET = 10_000_000


class LinkFeatureCollector:
    """Collects SIFT features on link frames while the tracker decodes the video."""

    def __init__(self, every: int, proc_w: int, proc_h: int, max_features: int = 1500):
        self.every = max(1, int(every))
        self.sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.02)
        self.frames: list[int] = []          # original frame numbers
        self.seq_idx: list[int] = []         # index into the tracked sequence
        self.kps: list[np.ndarray] = []      # [N,2] positions (processing px)
        self.desc: list[np.ndarray] = []     # [N,128] float32 (RootSIFT)
        self.colors: list[np.ndarray] = []   # [N,3] uint8 RGB
        res = max(proc_w, proc_h) / 1280.0
        self.border = int(10 * max(0.75, res))
        self.w, self.h = proc_w, proc_h

    def maybe_add(self, k: int, n: int, gray: np.ndarray, bgr: np.ndarray, force: bool = False):
        if k % self.every and not force:
            return
        kps, desc = self.sift.detectAndCompute(gray, None)
        if desc is None or len(kps) == 0:
            pts = np.zeros((0, 2), np.float32)
            desc = np.zeros((0, 128), np.float32)
        else:
            pts = np.array([kp.pt for kp in kps], np.float32)
            keep = ((pts[:, 0] >= self.border) & (pts[:, 0] < self.w - self.border) &
                    (pts[:, 1] >= self.border) & (pts[:, 1] < self.h - self.border))
            pts, desc = pts[keep], desc[keep].astype(np.float32)
            # RootSIFT: L1-normalise then sqrt (Hellinger kernel) - markedly better matching
            desc /= np.maximum(desc.sum(axis=1, keepdims=True), 1e-7)
            desc = np.sqrt(desc)
        ij = np.clip(np.round(pts).astype(int), 0, [self.w - 1, self.h - 1])
        col = bgr[ij[:, 1], ij[:, 0]][:, ::-1] if len(ij) else np.zeros((0, 3), np.uint8)
        self.frames.append(n)
        self.seq_idx.append(k)
        self.kps.append(pts)
        self.desc.append(desc)
        self.colors.append(col.astype(np.uint8))


def _match(da: np.ndarray, db: np.ndarray, ratio: float) -> np.ndarray:
    """Mutual nearest neighbours passing Lowe's ratio test. Returns [M,2] index pairs."""
    if len(da) < 8 or len(db) < 8:
        return np.zeros((0, 2), np.int64)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    ab = bf.knnMatch(da, db, k=2)
    ba = bf.knnMatch(db, da, k=1)
    back = {m[0].queryIdx: m[0].trainIdx for m in ba if m}
    out = []
    for m in ab:
        if len(m) < 2:
            continue
        a, b = m
        if a.distance < ratio * b.distance and back.get(a.trainIdx) == a.queryIdx:
            out.append((a.queryIdx, a.trainIdx))
    return np.array(out, np.int64).reshape(-1, 2)


def build_link_tracks(col: LinkFeatureCollector, window_frames: int, f_thresh: float, report=None,
                      ratio: float = 0.8, min_inliers: int = 15, mode: str = "star"):
    """Match link frames pairwise within `window_frames` (sequence frames) and chain the
    verified matches into tracks. Returns (list of (frames, xy, color)), stats)."""
    from .tracker import Track  # local import to avoid a cycle

    L = len(col.frames)
    offsets = np.cumsum([0] + [len(k) for k in col.kps])
    parent = np.arange(offsets[-1])

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    pairs = [(i, j) for i in range(L) for j in range(i + 1, L)
             if col.seq_idx[j] - col.seq_idx[i] <= window_frames]
    stats = dict(link_frames=L, pairs_tested=len(pairs), pairs_linked=0, verified_matches=0)
    pair_matches = {}
    for p, (i, j) in enumerate(pairs):
        if report is not None and p % 25 == 0:
            report(p / max(1, len(pairs)), f"Linking frames {col.frames[i]} and {col.frames[j]}")
        m = _match(col.desc[i], col.desc[j], ratio)
        if len(m) < min_inliers:
            continue
        a, b = col.kps[i][m[:, 0]], col.kps[j][m[:, 1]]
        F, mask = robust_fundamental(a, b, f_thresh)
        if F is None:
            continue
        inl = mask.ravel().astype(bool)
        if inl.sum() < min_inliers:
            continue
        stats["pairs_linked"] += 1
        stats["verified_matches"] += int(inl.sum())
        if mode == "star":
            pair_matches[(i, j)] = m[inl]
            continue
        for qa, qb in m[inl]:
            ra, rb = find(offsets[i] + qa), find(offsets[j] + qb)
            if ra != rb:
                parent[rb] = ra
    if mode == "star":
        tracks = _star_tracks(col, pair_matches, L, Track)
        stats["link_tracks"] = len(tracks)
        stats["ambiguous_rejected"] = 0
        return tracks, stats

    # gather components
    link_of = np.searchsorted(offsets, np.arange(offsets[-1]), side="right") - 1
    roots = np.array([find(x) for x in range(offsets[-1])], np.int64) if offsets[-1] else np.zeros(0, np.int64)
    order = np.argsort(roots, kind="stable")
    tracks = []
    rejected_ambiguous = 0
    if len(order):
        r_sorted = roots[order]
        starts = np.r_[0, np.where(np.diff(r_sorted))[0] + 1, len(order)]
        for s0, s1 in zip(starts[:-1], starts[1:]):
            nodes = order[s0:s1]
            if len(nodes) < 3:
                continue
            links = link_of[nodes]
            if len(np.unique(links)) != len(links):
                rejected_ambiguous += 1
                continue
            srt = np.argsort(links)
            nodes, links = nodes[srt], links[srt]
            t = Track(id=LINK_ID_OFFSET + len(tracks))
            t.frames = [col.frames[l] for l in links]
            t.xy = [tuple(map(float, col.kps[l][x - offsets[l]])) for l, x in zip(links, nodes)]
            c = col.colors[links[0]][nodes[0] - offsets[links[0]]]
            t.color = (int(c[0]), int(c[1]), int(c[2]))
            t.kind = "link"
            tracks.append(t)
    stats["link_tracks"] = len(tracks)
    stats["ambiguous_rejected"] = rejected_ambiguous
    return tracks, stats


def _star_tracks(col, pair_matches, L, Track):
    """Direct-match tracks: a keypoint in an anchor link frame plus its *directly* verified matches in
    other link frames. No transitive chaining, so one false match cannot fuse unrelated features.
    Where pairs between two members were also matched, they must agree (cycle check)."""
    fwd = {}  # (i, qi) -> {j: qj}
    for (i, j), m in pair_matches.items():
        for qa, qb in m:
            fwd.setdefault((i, int(qa)), {})[j] = int(qb)
            fwd.setdefault((j, int(qb)), {})[i] = int(qa)
    used = set()
    tracks = []
    for i in range(L):
        for qi in range(len(col.kps[i])):
            if (i, qi) in used:
                continue
            nb = fwd.get((i, qi))
            if not nb or len(nb) < 2:
                continue
            members = {i: qi, **nb}
            # cycle check: for member pairs that were matched directly, the match must agree
            bad = set()
            items = list(nb.items())
            for a in range(len(items)):
                ja, qa = items[a]
                for b in range(a + 1, len(items)):
                    jb, qb = items[b]
                    direct = fwd.get((ja, qa), {})
                    if jb in direct and direct[jb] != qb:
                        bad.update((ja, jb))
            for j in bad:
                members.pop(j, None)
            if len(members) < 3:
                continue
            for j, q in members.items():
                used.add((j, q))
            order = sorted(members)
            t = Track(id=LINK_ID_OFFSET + len(tracks))
            t.frames = [col.frames[j] for j in order]
            t.xy = [tuple(map(float, col.kps[j][members[j]])) for j in order]
            c = col.colors[i][qi]
            t.color = (int(c[0]), int(c[1]), int(c[2]))
            t.kind = "link"
            tracks.append(t)
    return tracks


def connectivity_gaps(frames: list[int], tracks, min_shared: int = 12) -> list[tuple[int, int]]:
    """Ranges of consecutive frames where fewer than `min_shared` tracks continue from one
    frame to the next (dead zones), as (first_frame, last_frame) original numbers."""
    idx = {f: i for i, f in enumerate(frames)}
    cont = np.zeros(len(frames), np.int64)  # tracks continuing from frame i to i+1
    for t in tracks:
        ii = [idx[f] for f in t.frames if f in idx]
        for a, b in zip(ii[:-1], ii[1:]):
            if b == a + 1:
                cont[a] += 1
    bad = cont[:-1] < min_shared
    gaps, i = [], 0
    while i < len(bad):
        if bad[i]:
            j = i
            while j + 1 < len(bad) and bad[j + 1]:
                j += 1
            gaps.append((frames[i], frames[j + 1]))
            i = j + 1
        else:
            i += 1
    return gaps


def weak_stretches(frames: list[int], tracks, proc_h: int, min_tracks: int = 150, min_cover: float = 0.35,
                   min_len: int = 8, join: int = 6, min_frames: int = 4) -> list[tuple[int, int]]:
    """Stretches where the footage is trackable but *poorly constrained*: very few usable tracks, or tracks
    bunched into a thin strip of the image (e.g. a plain floor with only a pillar and a painted line along
    one edge). Those frames cannot be placed reliably from their neighbours, so wide-baseline links should
    bridge them just like a true dead zone. Returns (first_frame, last_frame) ranges."""
    idx = {f: i for i, f in enumerate(frames)}
    n = np.zeros(len(frames), np.int64)
    ys = [[] for _ in frames]
    for t in tracks:
        if getattr(t, "dynamic", False) or len(t.frames) < min_len or getattr(t, "kind", "klt") == "link":
            continue
        for f, (x, y) in zip(t.frames, t.xy):
            i = idx.get(f)
            if i is not None:
                n[i] += 1
                ys[i].append(y)
    weak = np.zeros(len(frames), bool)
    for i, v in enumerate(ys):
        if n[i] < min_tracks:
            weak[i] = True
        else:
            lo, hi = np.percentile(v, [10, 90])
            weak[i] = (hi - lo) / proc_h < min_cover
    out, i = [], 0
    while i < len(weak):
        if weak[i]:
            j = i
            gap = 0
            k = i
            while k + 1 < len(weak) and gap <= join:        # tolerate short strong blips inside a weak stretch
                k += 1
                if weak[k]:
                    j, gap = k, 0
                else:
                    gap += 1
            if j - i + 1 >= min_frames:
                out.append((frames[i], frames[j]))
            i = j + 1
        else:
            i += 1
    return out


def merge_ranges(ranges: list[tuple[int, int]], pad: int = 0) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + pad:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]
