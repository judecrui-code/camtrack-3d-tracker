"""Synthetic test-footage generator with exact ground truth.

Renders textured planar geometry (ground, walls, boxes, a sky dome, independently
moving objects) through a perspective camera with optional radial distortion,
sensor noise and H.264 compression. Writes ``<name>.mp4`` and ``<name>.gt.json``
containing intrinsics and per-frame world-to-camera poses (OpenCV convention).

Scenarios cover: forward motion, sideways motion, orbiting a stationary object,
mostly-flat ground, outdoor, indoor (with lens distortion), low texture, tripod pan.

Usage: python -m tests.synth OUT_DIR [scenario ...]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import cv2
import numpy as np

W, H = 1280, 720


# ---------------------------------------------------------------- textures
def texture(seed, size=1024, contrast=1.0, base=(120, 110, 100), detail=1.0):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.float32)
    for octave, amp in [(8, 0.5), (32, 0.3), (128, 0.2), (256, 0.12)]:
        n = rng.random((octave, octave, 3)).astype(np.float32)
        img += amp * cv2.resize(n, (size, size), interpolation=cv2.INTER_CUBIC)
    img = (img - img.mean()) * contrast
    out = np.clip(np.array(base, np.float32)[None, None] + 160 * img, 0, 255).astype(np.uint8)
    nshapes = int(220 * detail)
    for _ in range(nshapes):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        c = tuple(int(b + (v - 128) * 0.8 * contrast) for b, v in zip(base, c))
        c = tuple(max(0, min(255, v)) for v in c)
        x, y = rng.integers(0, size, 2)
        r = int(rng.integers(4, 30))
        if rng.random() < 0.5:
            cv2.circle(out, (int(x), int(y)), r, c, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(out, (int(x), int(y)), (int(x + r * 1.5), int(y + r)), c, -1)
    return out


def sky_texture(seed):
    t = np.zeros((256, 256, 3), np.uint8)
    for y in range(256):
        t[y] = (235 - y * 0.25, 200 - y * 0.3, 150 - y * 0.2)
    rng = np.random.default_rng(seed)
    n = cv2.resize(rng.random((6, 6)).astype(np.float32), (256, 256), interpolation=cv2.INTER_CUBIC)
    t = np.clip(t.astype(np.float32) + 12 * n[..., None], 0, 255).astype(np.uint8)
    return t


# ---------------------------------------------------------------- geometry
class Quad:
    def __init__(self, corners, tex, subdiv=8, uv_repeat=1.0, moving=None):
        self.corners = np.asarray(corners, np.float64)  # 4x3: p00, p10, p11, p01
        self.tex = tex
        self.subdiv = subdiv
        self.rep = uv_repeat
        self.moving = moving  # function(frame)->offset vector

    def sub_quads(self, frame):
        off = self.moving(frame) if self.moving else np.zeros(3)
        p00, p10, p11, p01 = self.corners + off
        n = self.subdiv
        th, tw = self.tex.shape[:2]
        for i in range(n):
            for j in range(n):
                u0, u1, v0, v1 = i / n, (i + 1) / n, j / n, (j + 1) / n

                def P(u, v):
                    return (1 - u) * (1 - v) * p00 + u * (1 - v) * p10 + u * v * p11 + (1 - u) * v * p01
                pts = np.array([P(u0, v0), P(u1, v0), P(u1, v1), P(u0, v1)])
                # texture coords (repeat wraps via modulo on a tile basis)
                uv = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]) * self.rep
                yield pts, uv


def box(center, size, tex, moving=None, subdiv=3):
    cx, cy, cz = center
    sx, sy, sz = np.asarray(size) / 2
    v = lambda x, y, z: (cx + x, cy + y, cz + z)
    faces = [
        [v(-sx, -sy, -sz), v(sx, -sy, -sz), v(sx, -sy, sz), v(-sx, -sy, sz)],
        [v(sx, sy, -sz), v(-sx, sy, -sz), v(-sx, sy, sz), v(sx, sy, sz)],
        [v(-sx, sy, -sz), v(-sx, -sy, -sz), v(-sx, -sy, sz), v(-sx, sy, sz)],
        [v(sx, -sy, -sz), v(sx, sy, -sz), v(sx, sy, sz), v(sx, -sy, sz)],
        [v(-sx, -sy, sz), v(sx, -sy, sz), v(sx, sy, sz), v(-sx, sy, sz)],
    ]
    return [Quad(f, tex, subdiv, 1.0, moving) for f in faces]


def look_at(cam, target, up=(0, 0, 1)):
    """World (Z-up) -> OpenCV camera R,t."""
    cam = np.asarray(cam, float)
    z = np.asarray(target, float) - cam
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ cam


class Renderer:
    def __init__(self, f, w=W, h=H, k1=0.0, k2=0.0, noise=2.0, ss=1):
        self.f, self.w, self.h, self.k1, self.k2, self.noise = f, w, h, k1, k2, noise
        self.ss = ss
        self.K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])
        if k1 or k2:
            # map: for every distorted output pixel, where to sample in the undistorted render
            margin = 1.25
            self.uw, self.uh = int(w * margin), int(h * margin)
            self.Ku = np.array([[f, 0, (self.uw - 1) / 2], [0, f, (self.uh - 1) / 2], [0, 0, 1]])
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
            pts = np.stack([gx.ravel(), gy.ravel()], 1).reshape(-1, 1, 2)
            und = cv2.undistortPoints(pts, self.K, np.array([k1, k2, 0, 0]), None, None, None,
                                          (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-10)).reshape(-1, 2)
            self.mapx = (und[:, 0] * f + self.Ku[0, 2]).reshape(h, w).astype(np.float32)
            self.mapy = (und[:, 1] * f + self.Ku[1, 2]).reshape(h, w).astype(np.float32)
        else:
            self.uw, self.uh, self.Ku = w, h, self.K

    def render(self, quads, R, t, frame, rng, bg=(90, 100, 110)):
        canvas = np.zeros((self.uh, self.uw, 3), np.uint8)
        canvas[:] = bg
        items = []
        def add(pts, uv, tex, level):
            pc = pts @ R.T + t
            behind = pc[:, 2] < 0.05
            if behind.all():
                return
            if behind.any():
                if level >= 5:
                    return
                # split into 4 and recurse (near-plane handling)
                m = lambda a, b: (a + b) / 2
                c = pts.mean(0)
                cu = uv.mean(0)
                e = [m(pts[i], pts[(i + 1) % 4]) for i in range(4)]
                eu = [m(uv[i], uv[(i + 1) % 4]) for i in range(4)]
                add(np.array([pts[0], e[0], c, e[3]]), np.array([uv[0], eu[0], cu, eu[3]]), tex, level + 1)
                add(np.array([e[0], pts[1], e[1], c]), np.array([eu[0], uv[1], eu[1], cu]), tex, level + 1)
                add(np.array([c, e[1], pts[2], e[2]]), np.array([cu, eu[1], uv[2], eu[2]]), tex, level + 1)
                add(np.array([e[3], c, e[2], pts[3]]), np.array([eu[3], cu, eu[2], uv[3]]), tex, level + 1)
                return
            depth = float(np.linalg.norm(pc.mean(0)))
            items.append((depth, pc, uv, tex))

        for q in quads:
            for pts, uv in q.sub_quads(frame):
                add(pts, uv, q.tex, 0)
        items.sort(key=lambda x: -x[0])
        for depth, pc, uv, tex in items:
            pix = (pc[:, :2] / pc[:, 2:3]) * self.f + self.Ku[:2, 2]
            x0, y0 = np.floor(pix.min(0)).astype(int)
            x1, y1 = np.ceil(pix.max(0)).astype(int) + 1
            x0c, y0c = max(0, x0), max(0, y0)
            x1c, y1c = min(self.uw, x1), min(self.uh, y1)
            if x1c <= x0c or y1c <= y0c:
                continue
            th, tw = tex.shape[:2]
            # texture-space source quad (tile with repeat)
            uvs = uv.copy()
            base = np.floor(uvs.min(0))
            uvs -= base
            span = np.ceil(uvs.max(0)).astype(int)
            if (span > 1).any():
                tile = np.tile(tex, (max(1, span[1]), max(1, span[0]), 1))
            else:
                tile = tex
            src = np.stack([uvs[:, 0] * tw, uvs[:, 1] * th], 1).astype(np.float32)
            dst = (pix - [x0c, y0c]).astype(np.float32)
            Hm = cv2.getPerspectiveTransform(src, dst)
            ww, hh = x1c - x0c, y1c - y0c
            warped = cv2.warpPerspective(tile, Hm, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = np.zeros((hh, ww), np.uint8)
            cv2.fillConvexPoly(mask, np.round(dst * 16).astype(np.int32), 255, cv2.LINE_AA, 4)
            roi = canvas[y0c:y1c, x0c:x1c]
            m = (mask.astype(np.float32) / 255.0)[..., None]
            roi[:] = (warped * m + roi * (1 - m)).astype(np.uint8)
        if self.k1 or self.k2:
            canvas = cv2.remap(canvas, self.mapx, self.mapy, cv2.INTER_LINEAR)
        img = canvas.astype(np.float32)
        img = cv2.GaussianBlur(img, (0, 0), 0.6)
        img += rng.normal(0, self.noise, img.shape).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)


def encode(frames_dir, out, fps=30):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                    os.path.join(frames_dir, "%05d.png"), "-c:v", "libx264", "-crf", "14", "-preset", "medium",
                    "-pix_fmt", "yuv420p", out], check=True)


# ---------------------------------------------------------------- scenarios
def ground(tex, extent=40, z=0.0, subdiv=16, rep=8):
    e = extent
    return Quad([(-e, -e, z), (e, -e, z), (e, e, z), (-e, e, z)], tex, subdiv, rep)


def wall(p0, p1, height, tex, subdiv=8, rep=3, z0=0.0):
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    return Quad([(*p0, z0), (*p1, z0), (*p1, z0 + height), (*p0, z0 + height)], tex, subdiv, rep)


def sky(R=120):
    t = sky_texture(5)
    # four far vertical planes form a "dome"; low texture by design
    q = []
    for a in range(4):
        ang0, ang1 = a * np.pi / 2, (a + 1) * np.pi / 2
        p0 = (R * np.cos(ang0), R * np.sin(ang0))
        p1 = (R * np.cos(ang1), R * np.sin(ang1))
        q.append(wall(p0, p1, 80, t, subdiv=4, rep=1, z0=-5))
    return q


def scenario(name, n=120):
    """Return (quads, poses[list of (R,t)], f, k1, k2, noise, description)."""
    T = {i: texture(100 + i) for i in range(10)}
    f = 1100.0
    k1 = k2 = 0.0
    noise = 2.0
    quads = []
    poses = []
    desc = ""
    if name == "forward":
        desc = "Camera dolly forward down a street (ground + building walls + boxes)."
        quads += [ground(texture(1, base=(110, 105, 100)))]
        quads += [wall((-6, -10), (-6, 60), 8, T[2], rep=6), wall((6, 60), (6, -10), 8, T[3], rep=6)]
        for i, y in enumerate([8, 15, 22, 30]):
            quads += box((-3 + 1.5 * (i % 2) * 2, y, 0.6), (1.2, 1.2, 1.2), T[4 + i % 3])
        quads += sky()
        for i in range(n):
            u = i / (n - 1)
            cam = (0.3 * np.sin(u * 3), -2 + 14 * u, 1.6 + 0.1 * np.sin(u * 5))
            poses.append(look_at(cam, (0.5 * np.sin(u * 2), cam[1] + 10, 1.2)))
    elif name == "sideways":
        desc = "Camera trucks sideways past boxes in front of a textured wall."
        quads += [ground(texture(11, base=(100, 110, 100)))]
        quads += [wall((-20, 12), (20, 12), 7, T[1], rep=5)]
        for i, x in enumerate([-6, -3, 0, 3, 6]):
            quads += box((x, 5 + (i % 3), 0.75), (1.5, 1.5, 1.5), T[2 + i % 4])
        quads += sky()
        f = 1000.0
        for i in range(n):
            u = i / (n - 1)
            cam = (-5 + 10 * u, -4, 1.7)
            poses.append(look_at(cam, (cam[0] * 0.6, 6, 1.0)))
    elif name == "orbit":
        desc = "Camera orbits 90 degrees around a stationary stack of boxes."
        quads += [ground(texture(21, base=(120, 115, 95)))]
        quads += box((0, 0, 0.75), (1.5, 1.5, 1.5), T[3]) + box((0.2, 0.1, 2.0), (1.0, 1.0, 1.0), T[4])
        quads += box((2.5, -1.5, 0.4), (0.8, 0.8, 0.8), T[5]) + box((-2, 2, 0.5), (1, 1, 1), T[6])
        quads += sky()
        f = 1200.0
        for i in range(n):
            a = np.radians(-30 + 90 * i / (n - 1))
            cam = (7 * np.cos(a), 7 * np.sin(a), 3.0)
            poses.append(look_at(cam, (0, 0, 0.9)))
    elif name == "flat":
        desc = "Camera moves over flat textured ground only (planar scene)."
        quads += [ground(texture(31, base=(115, 100, 90)), extent=60, rep=12)]
        f = 1000.0
        for i in range(n):
            u = i / (n - 1)
            cam = (-3 + 6 * u, -3 + 2 * u, 3.0)
            poses.append(look_at(cam, (cam[0] + 1, cam[1] + 5, 0)))
    elif name == "outdoor":
        desc = "Outdoor: ground, distant buildings, sky, a car-like box moving independently."
        quads += [ground(texture(41, base=(95, 105, 90)), extent=80, rep=14)]
        quads += [wall((-30, 40), (30, 40), 15, T[7], rep=6), wall((30, 40), (40, 0), 12, T[8], rep=5)]
        quads += box((-4, 12, 1.5), (3, 3, 3), T[2]) + box((5, 16, 2), (2, 2, 4), T[5])
        mover = lambda fr: np.array([-8 + 0.15 * fr, 0, 0])
        quads += box((0, 8, 0.7), (2.5, 1.2, 1.4), texture(99, base=(160, 40, 40)), moving=mover)
        quads += sky(150)
        f = 950.0
        noise = 3.0
        for i in range(n):
            u = i / (n - 1)
            cam = (-4 + 6 * u, -6 + 3 * u, 1.8 + 0.3 * u)
            poses.append(look_at(cam, (2 * u - 1, 14, 1.5)))
    elif name == "indoor":
        desc = "Indoor room (floor, walls, ceiling, furniture) with barrel distortion k1=-0.08."
        quads += [Quad([(-5, -5, 0), (5, -5, 0), (5, 5, 0), (-5, 5, 0)], T[0], 12, 3)]
        quads += [Quad([(-5, 5, 3), (5, 5, 3), (5, -5, 3), (-5, -5, 3)], texture(51, base=(180, 180, 175), contrast=0.6), 8, 2)]
        quads += [wall((-5, 5), (5, 5), 3, T[1], rep=3), wall((5, 5), (5, -5), 3, T[2], rep=3),
                  wall((5, -5), (-5, -5), 3, T[3], rep=3), wall((-5, -5), (-5, 5), 3, T[4], rep=3)]
        quads += box((1.5, 2.5, 0.4), (2, 1, 0.8), T[5]) + box((-2.5, 1, 0.5), (1, 1, 1), T[6])
        f = 800.0
        k1 = -0.08
        for i in range(n):
            u = i / (n - 1)
            cam = (-2 + 3 * u, -3 + 1.5 * u, 1.6)
            poses.append(look_at(cam, (2 * np.sin(u * 2.5), 4, 1.2)))
    elif name == "lowtexture":
        desc = "Low-texture: faint, low-contrast surfaces with sparse detail and extra noise."
        lt = lambda s, b: texture(s, contrast=0.15, base=b, detail=0.15)
        quads += [ground(lt(61, (128, 128, 128)), rep=6)]
        quads += [wall((-15, 10), (15, 10), 6, lt(62, (150, 140, 130)), rep=3)]
        quads += box((0, 5, 0.75), (1.5, 1.5, 1.5), lt(63, (110, 110, 140)))
        noise = 4.0
        for i in range(n):
            u = i / (n - 1)
            cam = (-3 + 6 * u, -5, 1.6)
            poses.append(look_at(cam, (cam[0] * 0.5, 6, 1.0)))
    elif name == "tripod":
        desc = "Tripod: fixed camera position panning 40 degrees and tilting."
        quads += [ground(texture(71, base=(110, 105, 95)))]
        quads += [wall((-30, 20), (30, 20), 10, T[1], rep=8), wall((30, 20), (30, -20), 10, T[2], rep=6)]
        quads += box((3, 10, 1), (2, 2, 2), T[3]) + box((-4, 12, 1), (2, 2, 2), T[4])
        quads += sky()
        f = 1300.0
        cam = (0, 0, 1.7)
        for i in range(n):
            u = i / (n - 1)
            a = np.radians(-20 + 40 * u)
            tilt = 0.15 * np.sin(u * np.pi)
            poses.append(look_at(cam, (cam[0] + 10 * np.sin(a), 10 * np.cos(a), 1.7 - 1.5 + tilt * 10)))
    elif name == "zoom":
        desc = "Zoom: camera trucks past boxes while zooming 900px -> 1400px (variable focal length)."
        quads += [ground(texture(81, base=(105, 110, 100)))]
        quads += [wall((-20, 14), (20, 14), 8, T[1], rep=5)]
        for i, x in enumerate([-5, -2, 1, 4]):
            quads += box((x, 6 + (i % 2) * 2, 0.75), (1.5, 1.5, 1.5), T[2 + i % 4])
        f = [900.0 + 500.0 * (0.5 - 0.5 * np.cos(np.pi * i / (n - 1))) for i in range(n)]
        for i in range(n):
            u = i / (n - 1)
            cam = (-3 + 5 * u, -5, 1.8)
            poses.append(look_at(cam, (cam[0] * 0.5, 7, 1.0)))
    elif name in ("longwalk", "longwalk_nogap"):
        desc = ("Long handheld walk (20 s, 600 frames): down a street, round a corner, with a 2 s "
                "motion-blur dead zone (frames 300-359) where nothing can be tracked." if name == "longwalk"
                else "Long handheld walk (20 s, 600 frames) without a dead zone (control).")
        # large NON-repeating textures: exact tiling makes wide-baseline matching ambiguous by construction
        # (a feature and its copy one tile away are indistinguishable), which real scenes rarely are
        big = lambda sd, b: texture(sd, size=4096, base=b, detail=6.0)
        quads += [ground(big(91, (110, 105, 98)), extent=90, subdiv=20, rep=1)]
        # street along +y (x in [-7, 7]) turning right into a corridor along +x (y in [14, 27])
        quads += [wall((-7, -25), (-7, 27), 9, big(92, (120, 100, 90)), subdiv=12, rep=1),
                  wall((7, 14), (7, -25), 9, big(94, (110, 115, 100)), subdiv=10, rep=1),
                  wall((7, 14), (45, 14), 8, big(95, (125, 105, 110)), subdiv=10, rep=1),
                  wall((45, 27), (-7, 27), 12, big(97, (115, 115, 105)), subdiv=12, rep=1),
                  wall((45, 14), (45, 27), 10, big(96, (95, 105, 125)), subdiv=4, rep=1)]
        rng_b = np.random.default_rng(11)
        for k in range(10):
            y = -15 + 3.0 * k
            x = (-5.0 if k % 2 else 5.0) + rng_b.uniform(-0.6, 0.6)
            quads += box((x, y, 0.5), (1.0, 1.0, 1.0), T[k % 10])
        for k in range(7):
            quads += box((10 + 4.0 * k, 16.0 if k % 2 else 25.0, 0.5), (1.0, 1.0, 1.0), T[(k + 3) % 10])
        quads += sky(160)
        f = 1000.0
        # path: straight along +y, quarter turn (r=6) to +x, straight along +x; uniform speed
        L1, R_, L3 = 24.0, 6.0, 14.0
        La = R_ * np.pi / 2
        total = L1 + La + L3
        def at(sl):
            if sl <= L1:
                return np.array([0.0, -10 + sl]), np.array([0.0, 1.0])
            if sl <= L1 + La:
                a = (sl - L1) / R_
                return np.array([R_ - R_ * np.cos(a), -10 + L1 + R_ * np.sin(a)]), np.array([np.sin(a), np.cos(a)])
            d = sl - L1 - La
            return np.array([R_ + d, -10 + L1 + R_]), np.array([1.0, 0.0])
        for i in range(n):
            u = i / (n - 1)
            p, tan = at(u * total)
            bob = 0.04 * np.sin(2 * np.pi * u * 34)           # footsteps
            sway = 0.10 * np.sin(2 * np.pi * u * 3.0)          # looking around
            yaw = np.arctan2(tan[0], tan[1]) + sway
            cam = (p[0], p[1], 1.65 + bob)
            look = (p[0] + 10 * np.sin(yaw), p[1] + 10 * np.cos(yaw), 1.2 + 0.4 * np.sin(2 * np.pi * u * 2))
            poses.append(look_at(cam, look))
    else:
        raise ValueError(name)
    return quads, poses, f, k1, k2, noise, desc


# frames where tracking is destroyed (heavy motion blur + underexposure), per scenario
DEAD_ZONES = {"longwalk": (300, 360)}


def dead_zone(img, strength=1.0):
    k = np.zeros((1, 71), np.float32)
    k[0, :] = 1.0 / 71
    out = cv2.filter2D(img, -1, k)
    out = cv2.GaussianBlur(out, (0, 0), 6)
    return np.clip(out.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)


_W = {}


def _init_worker(name, n):
    quads, poses, f, k1, k2, noise, desc = scenario(name, n)
    flist = f if isinstance(f, list) else None
    _W.update(quads=quads, poses=poses, flist=flist,
              r=Renderer(flist[0] if flist else f, k1=k1, k2=k2, noise=noise), name=name)


def _render_one(args):
    i, out_path = args
    r, flist = _W["r"], _W["flist"]
    if flist:
        r.f = flist[i]
        r.K[0, 0] = r.K[1, 1] = r.Ku[0, 0] = r.Ku[1, 1] = flist[i]
    R, t = _W["poses"][i]
    img = r.render(_W["quads"], R, t, i, np.random.default_rng(1000 + i))
    dz = DEAD_ZONES.get(_W["name"])
    if dz and dz[0] <= i < dz[1]:
        img = dead_zone(img)
    cv2.imwrite(out_path, img)
    return i


SCENARIOS = ["forward", "sideways", "orbit", "flat", "outdoor", "indoor", "lowtexture", "tripod", "zoom"]


def generate(name, out_dir, n=120, fps=30):
    import tempfile
    quads, poses, f, k1, k2, noise, desc = scenario(name, n)
    flist = f if isinstance(f, list) else None
    r = Renderer(flist[0] if flist else f, k1=k1, k2=k2, noise=noise)
    rng = np.random.default_rng(7)
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        if n > 150:  # long clips: render in parallel
            import multiprocessing as mp
            ctx = mp.get_context("fork")
            with ctx.Pool(os.cpu_count() or 4, initializer=_init_worker, initargs=(name, n)) as pool:
                for k, _ in enumerate(pool.imap_unordered(_render_one, [(i, os.path.join(td, f"{i:05d}.png"))
                                                                        for i in range(n)], chunksize=4)):
                    if k % 50 == 0:
                        print(f"  {k}/{n}", flush=True)
        else:
            for i, (R, t) in enumerate(poses):
                if flist:  # zoom: per-frame focal length (no distortion in this scenario)
                    r.f = flist[i]
                    r.K[0, 0] = r.K[1, 1] = r.Ku[0, 0] = r.Ku[1, 1] = flist[i]
                img = r.render(quads, R, t, i, rng)
                dz = DEAD_ZONES.get(name)
                if dz and dz[0] <= i < dz[1]:
                    img = dead_zone(img)
                cv2.imwrite(os.path.join(td, f"{i:05d}.png"), img)
        mp4 = os.path.join(out_dir, f"{name}.mp4")
        encode(td, mp4, fps)
    gt = dict(name=name, description=desc, width=W, height=H, fps=fps, frames=n, f=flist[0] if flist else f,
              f_per_frame=flist, cx=(W - 1) / 2, cy=(H - 1) / 2,
              k1=k1, k2=k2, poses=[dict(R=R.tolist(), t=t.tolist()) for R, t in poses],
              tripod=name == "tripod", dead_zone=DEAD_ZONES.get(name))
    with open(os.path.join(out_dir, f"{name}.gt.json"), "w") as fh:
        json.dump(gt, fh)
    return mp4


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "test_footage"
    names = sys.argv[2:] or SCENARIOS
    for nm in names:
        print("rendering", nm, flush=True)
        print(generate(nm, out, n=600 if nm.startswith("longwalk") else 120))
