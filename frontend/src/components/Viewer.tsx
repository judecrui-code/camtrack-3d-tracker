/**
 * Footage viewer: frame display, solved-point overlay, selection, and the
 * After Effects-style interactive ground-plane target.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import {
  FrameCamera,
  Projected,
  V3,
  add,
  computeTarget,
  dot,
  planeBasis,
  rayPlane,
  scale,
  seqIndexOf,
  sub,
  invertSimilarity,
  unit,
} from "../geometry";
import type { DisplaySettings, ProjectSummary, SolutionData, TargetState } from "../types";

interface Props {
  project: ProjectSummary;
  solution: SolutionData | null;
  frame: number;
  display: DisplaySettings;
  selection: Set<number>;
  setSelection: (s: Set<number>) => void;
  target: TargetState | null;
  setTarget: (t: TargetState | null) => void;
  lockedTarget: TargetState | null;
  setLockedTarget: (t: TargetState | null) => void;
  onContextAction: (action: string, t: TargetState) => void;
  onDeleteSelected: () => void;
  imageCache: Map<number, HTMLImageElement>;
  onHoverInfo: (s: string) => void;
}

interface ViewXf {
  zoom: number; // 0 = fit
  panX: number;
  panY: number;
}

const MENU = [
  { id: "ground_origin", label: "Set Ground Plane and Origin" },
  { id: "ground", label: "Set Ground Plane" },
  { id: "origin", label: "Set Origin" },
  { sep: true },
  { id: "null", label: "Create Null" },
  { id: "solid", label: "Create Solid" },
  { id: "anchor", label: "Create 3D Anchor" },
  { id: "ground_plane", label: "Create Ground Plane Object" },
  { sep: true },
  { id: "select", label: "Select Target Points" },
  { id: "delete", label: "Delete Selected Points" },
] as const;

function hash01(i: number) {
  let x = (i * 2654435761) >>> 0;
  x ^= x >>> 13;
  x = Math.imul(x, 0x5bd1e995) >>> 0;
  x ^= x >>> 15;
  return (x >>> 0) / 4294967296;
}

export default function Viewer(p: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 800, h: 450 });
  const [xf, setXf] = useState<ViewXf>({ zoom: 1, panX: 0, panY: 0 });
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [mouse, setMouse] = useState<{ u: number; v: number } | null>(null);
  const [menu, setMenu] = useState<{ x: number; y: number; t: TargetState } | null>(null);
  const [box, setBox] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const [raw2d, setRaw2d] = useState<[number, number, number, number, number][]>([]);
  const drag = useRef<{ mode: "pan" | "target" | "box"; sx: number; sy: number; px: number; py: number } | null>(null);
  const W = p.project.info.width;
  const H = p.project.info.height;

  // ---- layout ----
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const fit = Math.min(size.w / W, size.h / H) * 0.96;
  const sc = fit * xf.zoom;
  const ox = (size.w - W * sc) / 2 + xf.panX;
  const oy = (size.h - H * sc) / 2 + xf.panY;
  const toScreen = useCallback((u: number, v: number) => [ox + (u + 0.5) * sc, oy + (v + 0.5) * sc] as const, [ox, oy, sc]);
  const toImage = useCallback((x: number, y: number) => [(x - ox) / sc - 0.5, (y - oy) / sc - 0.5] as const, [ox, oy, sc]);

  // ---- frame image ----
  useEffect(() => {
    let alive = true;
    const cached = p.imageCache.get(p.frame);
    if (cached && cached.complete && cached.naturalWidth) {
      setImg(cached);
      return;
    }
    const im = new Image();
    im.decoding = "async";
    im.src = api.frameUrl(p.project.id, p.frame);
    im.onload = () => {
      p.imageCache.set(p.frame, im);
      if (alive) setImg(im);
    };
    return () => {
      alive = false;
    };
  }, [p.frame, p.project.id]);

  // ---- raw 2D tracks (before/after solve) ----
  useEffect(() => {
    if (!p.display.showRaw2D || !p.project.has_tracks) {
      setRaw2d([]);
      return;
    }
    let alive = true;
    api.tracks2d(p.project.id, p.frame).then((r) => alive && setRaw2d(r.tracks)).catch(() => {});
    return () => {
      alive = false;
    };
  }, [p.frame, p.display.showRaw2D, p.project.has_tracks, p.project.id]);

  // ---- camera for this frame & projected points ----
  const seq = p.solution ? seqIndexOf(p.solution, p.frame) : -1;
  const cam = useMemo(() => (p.solution && seq >= 0 ? new FrameCamera(p.solution, seq) : null), [p.solution, seq]);
  const projected: Projected[] = useMemo(() => {
    if (!p.solution || !cam) return [];
    const pts = p.solution.points;
    const out: Projected[] = [];
    const n = pts.ids.length;
    for (let i = 0; i < n; i++) {
      if (pts.deleted[i]) continue;
      if (pts.first[i] > p.frame || pts.last[i] < p.frame) continue;
      const pr = cam.project([pts.xyz[3 * i], pts.xyz[3 * i + 1], pts.xyz[3 * i + 2]]);
      if (!pr || pr[0] < -0.5 || pr[1] < -0.5 || pr[0] > W - 0.5 || pr[1] > H - 0.5) continue;
      out.push({ idx: i, u: pr[0], v: pr[1], depth: pr[2] });
    }
    return out;
  }, [p.solution, cam, p.frame, W, H]);
  const shown = useMemo(() => {
    const pts = p.solution?.points;
    if (!pts) return [];
    return projected.filter(
      (q) => (p.display.showUnreliable || pts.reliable[q.idx] || p.selection.has(pts.ids[q.idx])) && hash01(pts.ids[q.idx]) <= p.display.density,
    );
  }, [projected, p.display.density, p.display.showUnreliable, p.solution, p.selection]);

  // ---- hover target ----
  useEffect(() => {
    if (!mouse || !cam || !p.solution || drag.current || menu) return;
    if (mouse.u < 0 || mouse.v < 0 || mouse.u > W - 1 || mouse.v > H - 1) {
      p.setTarget(null);
      return;
    }
    const t = computeTarget(p.solution, cam, shown, mouse.u, mouse.v, p.frame, p.display.targetSize, Math.max(80, 160 / sc));
    p.setTarget(t);
  }, [mouse, cam, shown, p.display.targetSize]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- drawing ----
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = size.w * dpr;
    cv.height = size.h * dpr;
    const g = cv.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = "#0b0c0e";
    g.fillRect(0, 0, size.w, size.h);
    if (img) {
      g.imageSmoothingQuality = "high";
      g.drawImage(img, ox, oy, W * sc, H * sc);
    }
    g.strokeStyle = "#333";
    g.strokeRect(ox - 0.5, oy - 0.5, W * sc + 1, H * sc + 1);
    const sol = p.solution;
    g.save();
    g.beginPath();
    g.rect(ox, oy, W * sc, H * sc);
    g.clip();
    // raw 2D tracks
    if (raw2d.length) {
      for (const [, u, v, , dyn] of raw2d) {
        const [x, y] = toScreen(u, v);
        g.fillStyle = dyn ? "rgba(255,60,200,0.9)" : "rgba(80,200,255,0.75)";
        g.fillRect(x - 1.5, y - 1.5, 3, 3);
      }
    }
    if (sol && cam) {
      const pts = sol.points;
      // depth range for marker size
      let dmin = Infinity,
        dmax = 0;
      for (const q of shown) {
        dmin = Math.min(dmin, q.depth);
        dmax = Math.max(dmax, q.depth);
      }
      const ps = p.display.pointSize;
      for (const q of shown) {
        const [x, y] = toScreen(q.u, q.v);
        const id = pts.ids[q.idx];
        const sel = p.selection.has(id);
        const rel = pts.reliable[q.idx];
        const nd = dmax > dmin ? (q.depth - dmin) / (dmax - dmin) : 0.5;
        const r = ps * (1.3 - 0.6 * nd);
        if (sel) {
          g.strokeStyle = "#fff";
          g.lineWidth = 1;
          g.beginPath();
          g.arc(x, y, r + 2.5, 0, Math.PI * 2);
          g.stroke();
          g.fillStyle = "#00e5ff";
          g.beginPath();
          g.arc(x, y, r, 0, Math.PI * 2);
          g.fill();
        } else if (!rel) {
          g.strokeStyle = "rgba(255,90,60,0.95)";
          g.lineWidth = 1.2;
          g.beginPath();
          g.moveTo(x - r, y - r);
          g.lineTo(x + r, y + r);
          g.moveTo(x + r, y - r);
          g.lineTo(x - r, y + r);
          g.stroke();
        } else {
          // AE-like coloured cross markers, hue by depth (near = warm)
          const hue = 50 + 130 * nd;
          g.strokeStyle = `hsla(${hue},95%,58%,0.95)`;
          g.lineWidth = 1.4;
          g.beginPath();
          g.moveTo(x - r, y);
          g.lineTo(x + r, y);
          g.moveTo(x, y - r);
          g.lineTo(x, y + r);
          g.stroke();
        }
      }
      // scene overlay (origin axes, ground grid, objects) projected through the solved camera
      if (p.display.showOverlay) drawSceneOverlay(g, sol, cam, toScreen);
      // target
      const t = p.lockedTarget && p.lockedTarget.frame === p.frame ? p.lockedTarget : null;
      const hov = p.target && !drag.current?.mode.startsWith("pan") ? p.target : null;
      if (hov && hov !== t) drawTarget(g, sol, cam, hov, toScreen, false);
      if (t) drawTarget(g, sol, cam, t, toScreen, true);
    }
    g.restore();
    if (box) {
      g.strokeStyle = "#00e5ff";
      g.setLineDash([4, 3]);
      g.strokeRect(Math.min(box.x0, box.x1), Math.min(box.y0, box.y1), Math.abs(box.x1 - box.x0), Math.abs(box.y1 - box.y0));
      g.setLineDash([]);
    }
    if (!sol && img && !raw2d.length) {
      g.fillStyle = "rgba(0,0,0,0.55)";
      g.fillRect(ox + 12, oy + 12, 280, 26);
      g.fillStyle = "#ddd";
      g.font = "12px system-ui";
      g.fillText("Not analysed yet — press Analyze to track & solve", ox + 20, oy + 29);
    } else if (sol && !cam) {
      g.fillStyle = "rgba(0,0,0,0.55)";
      g.fillRect(ox + 12, oy + 12, 380, 26);
      g.fillStyle = "#ffb74d";
      g.font = "12px system-ui";
      g.fillText("Frame not solved (outside the analysed range or skipped by frame sampling)", ox + 20, oy + 29);
    }
  }, [img, size, ox, oy, sc, shown, p.selection, p.target, p.lockedTarget, p.display, box, raw2d, cam, p.solution]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- interaction ----
  const evPos = (e: React.MouseEvent) => {
    const r = canvasRef.current!.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top] as const;
  };
  const nearestPoint = (x: number, y: number, maxPx = 8) => {
    let best: Projected | null = null;
    let bd = maxPx;
    for (const q of shown) {
      const [sx, sy] = toScreen(q.u, q.v);
      const d = Math.hypot(sx - x, sy - y);
      if (d < bd) {
        bd = d;
        best = q;
      }
    }
    return best;
  };

  const onMouseDown = (e: React.MouseEvent) => {
    setMenu(null);
    const [x, y] = evPos(e);
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      drag.current = { mode: "pan", sx: x, sy: y, px: xf.panX, py: xf.panY };
      e.preventDefault();
      return;
    }
    if (e.button !== 0) return;
    if (e.shiftKey || e.ctrlKey || e.metaKey) {
      // point picking / box select
      const q = nearestPoint(x, y);
      if (q && p.solution) {
        const id = p.solution.points.ids[q.idx];
        const s = new Set(p.selection);
        s.has(id) ? s.delete(id) : s.add(id);
        p.setSelection(s);
        return;
      }
      drag.current = { mode: "box", sx: x, sy: y, px: 0, py: 0 };
      setBox({ x0: x, y0: y, x1: x, y1: y });
      return;
    }
    // drag an existing locked target along its plane
    const lt = p.lockedTarget;
    if (lt && cam && lt.frame === p.frame) {
      const c = cam.project(lt.center);
      if (c) {
        const [sx, sy] = toScreen(c[0], c[1]);
        if (Math.hypot(sx - x, sy - y) < 22) {
          drag.current = { mode: "target", sx: x, sy: y, px: 0, py: 0 };
          return;
        }
      }
    }
    // click: lock the current hover target and select its points
    if (p.target && p.solution) {
      p.setLockedTarget(p.target);
      p.setSelection(new Set(p.target.pointIdx.map((i) => p.solution!.points.ids[i])));
    } else {
      p.setLockedTarget(null);
      const q = nearestPoint(x, y);
      p.setSelection(q && p.solution ? new Set([p.solution.points.ids[q.idx]]) : new Set());
    }
  };

  const onMouseMove = (e: React.MouseEvent) => {
    const [x, y] = evPos(e);
    const [u, v] = toImage(x, y);
    p.onHoverInfo(`x ${u.toFixed(1)}  y ${v.toFixed(1)}`);
    const d = drag.current;
    if (d?.mode === "pan") {
      setXf({ ...xf, panX: d.px + x - d.sx, panY: d.py + y - d.sy });
      return;
    }
    if (d?.mode === "box") {
      setBox((b) => (b ? { ...b, x1: x, y1: y } : b));
      return;
    }
    if (d?.mode === "target" && p.lockedTarget && cam) {
      // slide the target along its own plane under the cursor
      const { origin, dir } = cam.ray(u, v);
      const hit = rayPlane(origin, dir, p.lockedTarget.center, p.lockedTarget.normal);
      if (hit) p.setLockedTarget({ ...p.lockedTarget, center: hit });
      return;
    }
    setMouse({ u, v });
  };

  const onMouseUp = (e: React.MouseEvent) => {
    const d = drag.current;
    drag.current = null;
    if (d?.mode === "box" && box && p.solution) {
      const x0 = Math.min(box.x0, box.x1),
        x1 = Math.max(box.x0, box.x1);
      const y0 = Math.min(box.y0, box.y1),
        y1 = Math.max(box.y0, box.y1);
      const s = new Set(p.selection);
      for (const q of shown) {
        const [sx, sy] = toScreen(q.u, q.v);
        if (sx >= x0 && sx <= x1 && sy >= y0 && sy <= y1) s.add(p.solution.points.ids[q.idx]);
      }
      p.setSelection(s);
      setBox(null);
    }
  };

  const onContext = (e: React.MouseEvent) => {
    e.preventDefault();
    const t = p.lockedTarget && p.lockedTarget.frame === p.frame ? p.lockedTarget : p.target;
    if (!t) return;
    if (!p.lockedTarget) {
      p.setLockedTarget(t);
      if (p.solution) p.setSelection(new Set(t.pointIdx.map((i) => p.solution!.points.ids[i])));
    }
    const r = wrapRef.current!.getBoundingClientRect();
    setMenu({ x: Math.min(e.clientX - r.left, size.w - 240), y: Math.min(e.clientY - r.top, size.h - 300), t });
  };

  const onWheel = (e: React.WheelEvent) => {
    const [x, y] = evPos(e);
    const k = Math.exp(-e.deltaY * 0.0015);
    const nz = Math.min(16, Math.max(0.25, xf.zoom * k));
    const kk = nz / xf.zoom;
    // keep the point under the cursor fixed
    const cx = size.w / 2 + xf.panX,
      cy = size.h / 2 + xf.panY;
    setXf({ zoom: nz, panX: xf.panX + (x - cx) * (1 - kk), panY: xf.panY + (y - cy) * (1 - kk) });
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT") return;
      if (e.key === "Escape") {
        p.setLockedTarget(null);
        setMenu(null);
      }
      if ((e.key === "Delete" || e.key === "Backspace") && p.selection.size) p.onDeleteSelected();
      if (e.key === "f") setXf({ zoom: 1, panX: 0, panY: 0 });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <div className="viewer" ref={wrapRef} onContextMenu={(e) => e.preventDefault()}>
      <canvas
        ref={canvasRef}
        style={{ width: size.w, height: size.h, cursor: drag.current?.mode === "pan" ? "grabbing" : p.target ? "crosshair" : "default" }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={() => {
          setMouse(null);
          if (!menu) p.setTarget(null);
        }}
        onContextMenu={onContext}
        onWheel={onWheel}
      />
      <div className="viewer-hud">
        <span>{Math.round(sc * 100)}%</span>
        {p.solution && <span>{shown.length} pts</span>}
        {p.lockedTarget && <span className="hud-target">target locked · Esc to release · drag centre to move</span>}
      </div>
      {menu && (
        <div className="ctx-menu" style={{ left: menu.x, top: menu.y }} onMouseLeave={() => setMenu(null)}>
          {MENU.map((m, i) =>
            "sep" in m ? (
              <div key={i} className="ctx-sep" />
            ) : (
              <div
                key={m.id}
                className={"ctx-item" + (m.id === "delete" && !p.selection.size ? " disabled" : "")}
                onClick={() => {
                  setMenu(null);
                  if (m.id === "delete") p.onDeleteSelected();
                  else p.onContextAction(m.id, p.lockedTarget ?? menu.t);
                }}
              >
                {m.label}
              </div>
            ),
          )}
        </div>
      )}
    </div>
  );
}

type ToScreen = (u: number, v: number) => readonly [number, number];

function polyline(g: CanvasRenderingContext2D, cam: FrameCamera, pts: V3[], toScreen: ToScreen, close = false) {
  let started = false;
  g.beginPath();
  let ok = 0;
  for (const P of pts) {
    const pr = cam.project(P);
    if (!pr) {
      started = false;
      continue;
    }
    const [x, y] = toScreen(pr[0], pr[1]);
    if (!started) {
      g.moveTo(x, y);
      started = true;
    } else g.lineTo(x, y);
    ok++;
  }
  if (close && ok === pts.length) g.closePath();
  return ok;
}

function drawTarget(g: CanvasRenderingContext2D, sol: SolutionData, cam: FrameCamera, t: TargetState, toScreen: ToScreen, locked: boolean) {
  const xyz = sol.points.xyz;
  const P = (i: number): V3 => [xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]];
  // translucent triangle between the three defining points
  const tri = t.triangle.map(P);
  g.fillStyle = locked ? "rgba(255,40,40,0.28)" : "rgba(255,40,40,0.18)";
  g.strokeStyle = "rgba(255,80,80,0.9)";
  g.lineWidth = 1;
  if (polyline(g, cam, tri, toScreen, true) === 3) {
    g.fill();
    g.stroke();
  }
  // bullseye lying on the fitted plane (perspective-correct ellipses)
  const [bx, by] = planeBasis(t.normal, cam.axis(0));
  const ring = (r: number) => {
    const out: V3[] = [];
    for (let k = 0; k <= 64; k++) {
      const a = (k / 64) * Math.PI * 2;
      out.push(add(t.center, add(scale(bx, Math.cos(a) * r), scale(by, Math.sin(a) * r))));
    }
    return out;
  };
  const R = t.radius;
  const rings: [number, string][] = [
    [1.0, locked ? "rgba(255,30,30,0.55)" : "rgba(255,30,30,0.40)"],
    [0.72, "rgba(255,255,255,0.40)"],
    [0.46, locked ? "rgba(255,30,30,0.65)" : "rgba(255,30,30,0.50)"],
    [0.2, "rgba(255,255,255,0.55)"],
  ];
  for (const [f, col] of rings) {
    if (polyline(g, cam, ring(R * f), toScreen, true) > 3) {
      g.fillStyle = col;
      g.fill();
    }
  }
  g.strokeStyle = "rgba(255,40,40,0.95)";
  g.lineWidth = locked ? 2 : 1.4;
  polyline(g, cam, ring(R), toScreen, true);
  g.stroke();
  // orientation: in-plane axes + surface normal stem
  g.lineWidth = 2;
  g.strokeStyle = "#ff3b3b";
  polyline(g, cam, [t.center, add(t.center, scale(bx, R * 1.25))], toScreen);
  g.stroke();
  g.strokeStyle = "#5dff6a";
  polyline(g, cam, [t.center, add(t.center, scale(by, R * 1.25))], toScreen);
  g.stroke();
  g.strokeStyle = "#4da3ff";
  polyline(g, cam, [t.center, add(t.center, scale(t.normal, R * 1.1))], toScreen);
  g.stroke();
  const c = cam.project(t.center);
  if (c) {
    const [x, y] = toScreen(c[0], c[1]);
    g.fillStyle = "#ff2020";
    g.beginPath();
    g.arc(x, y, locked ? 4 : 3, 0, Math.PI * 2);
    g.fill();
    g.strokeStyle = "#fff";
    g.lineWidth = 1;
    g.stroke();
  }
}

function drawSceneOverlay(g: CanvasRenderingContext2D, sol: SolutionData, cam: FrameCamera, toScreen: ToScreen) {
  const M = sol.world.M;
  const Mi = invertSimilarity(M);
  const toRaw = (w: V3): V3 => [
    Mi[0][0] * w[0] + Mi[0][1] * w[1] + Mi[0][2] * w[2] + Mi[0][3],
    Mi[1][0] * w[0] + Mi[1][1] * w[1] + Mi[1][2] * w[2] + Mi[1][3],
    Mi[2][0] * w[0] + Mi[2][1] * w[1] + Mi[2][2] * w[2] + Mi[2][3],
  ];
  const s = sol.world.scale;
  // ground grid in world units (z = 0)
  const ext = 10 / Math.max(s, 1e-9) > 0 ? 5 : 5;
  const step = 1;
  g.strokeStyle = "rgba(120,200,255,0.22)";
  g.lineWidth = 1;
  for (let k = -ext; k <= ext; k += step) {
    const a: V3[] = [];
    const b: V3[] = [];
    for (let m = -ext; m <= ext; m += 0.5) {
      a.push(toRaw([k, m, 0]));
      b.push(toRaw([m, k, 0]));
    }
    polyline(g, cam, a, toScreen);
    g.stroke();
    polyline(g, cam, b, toScreen);
    g.stroke();
  }
  // world axes at origin
  const o = toRaw([0, 0, 0]);
  const cols = ["#ff4040", "#40ff40", "#4080ff"];
  for (let i = 0; i < 3; i++) {
    const e: V3 = [0, 0, 0];
    e[i] = 1;
    g.strokeStyle = cols[i];
    g.lineWidth = 2.5;
    polyline(g, cam, [o, toRaw(e)], toScreen);
    g.stroke();
  }
  // created objects
  for (const ob of sol.scene.objects) {
    const pos = ob.position as V3;
    const n = unit(ob.normal as V3);
    const [bx, by] = planeBasis(n, ob.x_hint as V3);
    const r = ob.size / s / (ob.type === "ground_plane" ? 0.5 : 2);
    g.lineWidth = 1.5;
    if (ob.type === "null" || ob.type === "anchor") {
      const L = ob.size / s;
      const ax: V3[] = ob.type === "null" ? [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map((v) => sub(toRaw(v as V3), o)) as V3[] : [bx, by, n];
      ax.forEach((d, i) => {
        g.strokeStyle = cols[i];
        polyline(g, cam, [pos, add(pos, scale(unit(d), L))], toScreen);
        g.stroke();
      });
    } else {
      g.strokeStyle = ob.type === "solid" ? "rgba(255,210,60,0.95)" : "rgba(255,120,60,0.95)";
      g.fillStyle = ob.type === "solid" ? "rgba(255,210,60,0.25)" : "rgba(255,120,60,0.15)";
      const c = [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
      ].map(([a, b]) => add(pos, add(scale(bx, a * r), scale(by, b * r))));
      if (polyline(g, cam, c, toScreen, true) === 4) {
        g.fill();
        g.stroke();
      }
    }
    const pr = cam.project(pos);
    if (pr) {
      const [x, y] = toScreen(pr[0], pr[1]);
      g.fillStyle = "#fff";
      g.font = "11px system-ui";
      g.fillText(ob.name, x + 6, y - 6);
    }
  }
  void dot;
}
