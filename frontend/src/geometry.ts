/**
 * Camera projection + plane fitting on the reconstructed 3D points.
 *
 * Camera model matches the backend (OpenCV): Xc = R X + t, x=Xc/Zc,
 * d = 1 + k1 r^2 + k2 r^4, u = f d x + cx. Units: original-resolution pixels,
 * pixel centres at integer coordinates.
 */
import type { SolutionData, TargetState } from "./types";

export type V3 = [number, number, number];

export const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a: V3) => Math.hypot(a[0], a[1], a[2]);
export const unit = (a: V3): V3 => {
  const n = norm(a);
  return n > 1e-12 ? scale(a, 1 / n) : a;
};

export class FrameCamera {
  R: number[]; // row-major 3x3 world->cam
  t: V3;
  f: number;
  cx: number;
  cy: number;
  k1: number;
  k2: number;
  center: V3;

  constructor(sol: SolutionData, seqIndex: number) {
    const i = seqIndex;
    this.R = sol.R.slice(i * 9, i * 9 + 9);
    this.t = [sol.t[i * 3], sol.t[i * 3 + 1], sol.t[i * 3 + 2]];
    const intr = sol.intrinsics;
    this.f = intr.cam_focal ? intr.cam_focal[i] : intr.f;
    this.cx = intr.cx;
    this.cy = intr.cy;
    this.k1 = intr.k1;
    this.k2 = intr.k2;
    const R = this.R;
    const t = this.t;
    // C = -R^T t
    this.center = [
      -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
      -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
      -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
    ];
  }

  toCam(X: V3): V3 {
    const R = this.R;
    return [
      R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + this.t[0],
      R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + this.t[1],
      R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + this.t[2],
    ];
  }

  /** Returns [u, v, depth] or null if behind the camera. */
  project(X: V3): [number, number, number] | null {
    const c = this.toCam(X);
    if (c[2] <= 1e-6) return null;
    const x = c[0] / c[2];
    const y = c[1] / c[2];
    const r2 = x * x + y * y;
    const d = 1 + this.k1 * r2 + this.k2 * r2 * r2;
    return [this.f * d * x + this.cx, this.f * d * y + this.cy, c[2]];
  }

  /** World-space ray through pixel (u,v) (distortion removed iteratively). */
  ray(u: number, v: number): { origin: V3; dir: V3 } {
    const xd = (u - this.cx) / this.f;
    const yd = (v - this.cy) / this.f;
    let x = xd;
    let y = yd;
    for (let k = 0; k < 20; k++) {
      const r2 = x * x + y * y;
      const d = 1 + this.k1 * r2 + this.k2 * r2 * r2;
      x = xd / d;
      y = yd / d;
    }
    const R = this.R;
    const dc: V3 = [x, y, 1];
    const dir: V3 = [
      R[0] * dc[0] + R[3] * dc[1] + R[6] * dc[2],
      R[1] * dc[0] + R[4] * dc[1] + R[7] * dc[2],
      R[2] * dc[0] + R[5] * dc[1] + R[8] * dc[2],
    ];
    return { origin: this.center, dir: unit(dir) };
  }

  axis(row: number): V3 {
    return [this.R[row * 3], this.R[row * 3 + 1], this.R[row * 3 + 2]];
  }
}

/** Smallest eigenvector of a symmetric 3x3 matrix via Jacobi rotations. */
function smallestEigvec(A: number[][]): { vec: V3; val: number } {
  const a = A.map((r) => r.slice());
  const V = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0;
    for (let p = 0; p < 3; p++) for (let q = p + 1; q < 3; q++) off += a[p][q] * a[p][q];
    if (off < 1e-20) break;
    for (let p = 0; p < 3; p++)
      for (let q = p + 1; q < 3; q++) {
        if (Math.abs(a[p][q]) < 1e-30) continue;
        const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 3; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 3; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 3; k++) {
          const vkp = V[k][p];
          const vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
  }
  let mi = 0;
  for (let i = 1; i < 3; i++) if (a[i][i] < a[mi][mi]) mi = i;
  return { vec: unit([V[0][mi], V[1][mi], V[2][mi]]), val: a[mi][mi] };
}

/** Least-squares plane through 3D points. */
export function fitPlane(pts: V3[]): { centroid: V3; normal: V3; rms: number } {
  const n = pts.length;
  const c: V3 = [0, 0, 0];
  for (const p of pts) {
    c[0] += p[0] / n;
    c[1] += p[1] / n;
    c[2] += p[2] / n;
  }
  const C = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (const p of pts) {
    const d = sub(p, c);
    for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) C[i][k] += d[i] * d[k];
  }
  const { vec, val } = smallestEigvec(C);
  return { centroid: c, normal: vec, rms: Math.sqrt(Math.max(0, val) / n) };
}

export function rayPlane(origin: V3, dir: V3, p0: V3, n: V3): V3 | null {
  const den = dot(n, dir);
  if (Math.abs(den) < 1e-9) return null;
  const s = dot(n, sub(p0, origin)) / den;
  if (s <= 0) return null;
  return add(origin, scale(dir, s));
}

export interface Projected {
  idx: number; // index into solution points
  u: number;
  v: number;
  depth: number;
}

function inTriangle(px: number, py: number, a: Projected, b: Projected, c: Projected) {
  const d1 = (px - b.u) * (a.v - b.v) - (a.u - b.u) * (py - b.v);
  const d2 = (px - c.u) * (b.v - c.v) - (b.u - c.u) * (py - c.v);
  const d3 = (px - a.u) * (c.v - a.v) - (c.u - a.u) * (py - a.v);
  const neg = d1 < 0 || d2 < 0 || d3 < 0;
  const pos = d1 > 0 || d2 > 0 || d3 > 0;
  return !(neg && pos);
}

/**
 * After Effects-style target: find neighbouring solved points around the cursor,
 * fit a local plane to their *3D* positions, intersect the viewing ray with it.
 */
export function computeTarget(
  sol: SolutionData,
  cam: FrameCamera,
  projected: Projected[],
  mu: number,
  mv: number,
  frame: number,
  sizeFactor: number,
  searchRadiusPx: number,
): TargetState | null {
  if (sol.mode === "tripod") return null;
  const near = projected
    .map((p) => ({ p, d: Math.hypot(p.u - mu, p.v - mv) }))
    .filter((x) => x.d < searchRadiusPx)
    .sort((a, b) => a.d - b.d)
    .slice(0, 10)
    .map((x) => x.p);
  if (near.length < 3) return null;
  // triangle enclosing the cursor with the smallest perimeter among the nearest candidates
  let tri: Projected[] | null = null;
  let best = Infinity;
  const m = Math.min(near.length, 8);
  for (let a = 0; a < m; a++)
    for (let b = a + 1; b < m; b++)
      for (let c = b + 1; c < m; c++) {
        const A = near[a],
          B = near[b],
          C = near[c];
        if (!inTriangle(mu, mv, A, B, C)) continue;
        const area = Math.abs((B.u - A.u) * (C.v - A.v) - (C.u - A.u) * (B.v - A.v)) / 2;
        if (area < 4) continue;
        const per = Math.hypot(A.u - B.u, A.v - B.v) + Math.hypot(B.u - C.u, B.v - C.v) + Math.hypot(C.u - A.u, C.v - A.v);
        if (per < best) {
          best = per;
          tri = [A, B, C];
        }
      }
  if (!tri) tri = near.slice(0, 3);
  const xyz = sol.points.xyz;
  const P = (i: number): V3 => [xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]];
  // fit uses the triangle + up to 5 further neighbours (robust to noise in a single point)
  const used = Array.from(new Set([...tri.map((p) => p.idx), ...near.slice(0, 8).map((p) => p.idx)]));
  let fit = fitPlane(used.map(P));
  // one round of outlier removal: drop points far off the plane (e.g. mixing two surfaces)
  if (used.length > 4) {
    const res = used.map((i) => Math.abs(dot(sub(P(i), fit.centroid), fit.normal)));
    const med = [...res].sort((a, b) => a - b)[Math.floor(res.length / 2)];
    const keep = used.filter((_, k) => res[k] <= Math.max(3 * med, 1e-9) || tri!.some((t) => t.idx === used[k]));
    if (keep.length >= 3 && keep.length < used.length) {
      used.length = 0;
      used.push(...keep);
      fit = fitPlane(keep.map(P));
    }
  }
  let n = fit.normal;
  // orient the normal towards the camera
  if (dot(sub(cam.center, fit.centroid), n) < 0) n = scale(n, -1);
  const { origin, dir } = cam.ray(mu, mv);
  const hit = rayPlane(origin, dir, fit.centroid, n);
  if (!hit) return null;
  const dist = norm(sub(hit, cam.center));
  return {
    center: hit,
    normal: n,
    pointIdx: used,
    triangle: tri.map((p) => p.idx),
    rms: fit.rms,
    frame,
    radius: dist * 0.06 * sizeFactor,
  };
}

/** Orthonormal in-plane basis for drawing, x aligned with the camera's right vector. */
export function planeBasis(n: V3, right: V3): [V3, V3] {
  let x = sub(right, scale(n, dot(right, n)));
  if (norm(x) < 1e-6) x = cross(n, [0, 0, 1]);
  x = unit(x);
  const y = cross(n, x);
  return [x, y];
}

/** raw -> world (Z-up) using 4x4 matrix rows. */
export function toWorld(M: number[][], p: V3): V3 {
  return [
    M[0][0] * p[0] + M[0][1] * p[1] + M[0][2] * p[2] + M[0][3],
    M[1][0] * p[0] + M[1][1] * p[1] + M[1][2] * p[2] + M[1][3],
    M[2][0] * p[0] + M[2][1] * p[1] + M[2][2] * p[2] + M[2][3],
  ];
}

export function worldDir(M: number[][], v: V3): V3 {
  return unit([
    M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
    M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
    M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2],
  ]);
}

/** Inverse of a similarity 4x4 (s*R | t). */
export function invertSimilarity(M: number[][]): number[][] {
  const s = Math.cbrt(
    M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
      M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
      M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]),
  );
  const Rt = [0, 1, 2].map((i) => [0, 1, 2].map((k) => M[k][i] / (s * s)));
  const t = [M[0][3], M[1][3], M[2][3]];
  const ti = [0, 1, 2].map((i) => -(Rt[i][0] * t[0] + Rt[i][1] * t[1] + Rt[i][2] * t[2]));
  return [
    [...Rt[0], ti[0]],
    [...Rt[1], ti[1]],
    [...Rt[2], ti[2]],
    [0, 0, 0, 1],
  ];
}

export function seqIndexOf(sol: SolutionData, frame: number): number {
  const f0 = sol.frames[0];
  const step = sol.frames.length > 1 ? sol.frames[1] - sol.frames[0] : 1;
  const i = Math.round((frame - f0) / step);
  if (i >= 0 && i < sol.frames.length && sol.frames[i] === frame) return i;
  return -1;
}
