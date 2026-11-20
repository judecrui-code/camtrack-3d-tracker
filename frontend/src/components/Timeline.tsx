/** Timeline: frame ruler, work-area (frame range) handles, playhead, per-frame solve quality graph. */
import { useEffect, useRef, useState } from "react";
import type { SolutionData } from "../types";

interface Props {
  frameCount: number;
  fps: number;
  frame: number;
  setFrame: (f: number) => void;
  playing: boolean;
  setPlaying: (b: boolean) => void;
  loop: boolean;
  setLoop: (b: boolean) => void;
  rangeStart: number;
  rangeEnd: number;
  setRange: (a: number, b: number) => void;
  solution: SolutionData | null;
  threshold: number;
}

const tc = (f: number, fps: number) => {
  const s = f / fps;
  const m = Math.floor(s / 60);
  const ss = Math.floor(s % 60);
  const ff = Math.round(f - Math.floor(s) * fps);
  return `${String(m).padStart(2, "0")}:${String(ss).padStart(2, "0")}:${String(Math.min(ff, Math.ceil(fps) - 1)).padStart(2, "0")}`;
};

export default function Timeline(p: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(800);
  const drag = useRef<null | "scrub" | "in" | "out">(null);
  const H = 92;
  const pad = 14;
  const n = Math.max(1, p.frameCount);
  const x = (f: number) => pad + ((f + 0.5) / n) * (w - 2 * pad);
  const f = (px: number) => Math.max(0, Math.min(n - 1, Math.floor(((px - pad) / (w - 2 * pad)) * n)));

  useEffect(() => {
    const el = wrap.current!;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const cv = ref.current!;
    const dpr = window.devicePixelRatio || 1;
    cv.width = w * dpr;
    cv.height = H * dpr;
    const g = cv.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = "#1b1d21";
    g.fillRect(0, 0, w, H);
    // work area
    g.fillStyle = "rgba(90,140,220,0.18)";
    g.fillRect(x(p.rangeStart) - 2, 0, x(p.rangeEnd) - x(p.rangeStart) + 4, H);
    g.fillStyle = "#5a8cdc";
    g.fillRect(x(p.rangeStart) - 3, 0, 3, 18);
    g.fillRect(x(p.rangeEnd), 0, 3, 18);
    // ruler
    g.strokeStyle = "#555";
    g.fillStyle = "#9aa0a6";
    g.font = "10px ui-monospace, monospace";
    const pxPerFrame = (w - 2 * pad) / n;
    const steps = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000];
    const major = steps.find((s) => s * pxPerFrame > 60) ?? 10000;
    const minor = steps.find((s) => s * pxPerFrame > 8) ?? major;
    for (let k = 0; k < n; k += minor) {
      const xx = Math.round(x(k)) + 0.5;
      g.beginPath();
      g.moveTo(xx, 18);
      g.lineTo(xx, k % major === 0 ? 30 : 24);
      g.stroke();
      if (k % major === 0) g.fillText(String(k), xx + 2, 29);
    }
    // per-frame reprojection error graph
    const sol = p.solution;
    const gy0 = 36,
      gh = H - gy0 - 6;
    g.fillStyle = "#15171a";
    g.fillRect(pad, gy0, w - 2 * pad, gh);
    if (sol) {
      const thr = p.threshold;
      const maxE = Math.max(thr * 0.75, ...sol.frame_error.map((e) => e ?? 0));
      for (let i = 0; i < sol.frames.length; i++) {
        const fr = sol.frames[i];
        const e = sol.frame_error[i];
        const st = sol.frame_status[i];
        const x0 = x(fr) - pxPerFrame / 2;
        const bw = Math.max(1, pxPerFrame - (pxPerFrame > 3 ? 1 : 0));
        if (st === "interpolated" || e == null) {
          g.fillStyle = "rgba(255,50,50,0.8)";
          g.fillRect(x0, gy0, bw, gh);
          continue;
        }
        const hh = (e / maxE) * gh;
        g.fillStyle = e < 0.5 ? "#3fb950" : e < 1.0 ? "#d29922" : "#f85149";
        g.fillRect(x0, gy0 + gh - hh, bw, hh);
        if (st === "keyframe") {
          g.fillStyle = "#ffd54f";
          g.fillRect(x0, gy0, bw, 2);
        }
      }
      g.fillStyle = "#8b949e";
      g.fillText(`reproj. error / frame (max ${maxE.toFixed(2)}px)`, pad + 4, gy0 + 10);
    } else {
      g.fillStyle = "#555";
      g.fillText("no solve", pad + 4, gy0 + 12);
    }
    // playhead
    const px = Math.round(x(p.frame)) + 0.5;
    g.strokeStyle = "#ff3b3b";
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(px, 0);
    g.lineTo(px, H);
    g.stroke();
    g.fillStyle = "#ff3b3b";
    g.beginPath();
    g.moveTo(px - 5, 0);
    g.lineTo(px + 5, 0);
    g.lineTo(px, 7);
    g.fill();
  }, [w, p.frame, p.rangeStart, p.rangeEnd, p.solution, n, p.threshold]);

  const down = (e: React.MouseEvent) => {
    const r = ref.current!.getBoundingClientRect();
    const px = e.clientX - r.left;
    const py = e.clientY - r.top;
    if (py < 20 && Math.abs(px - x(p.rangeStart)) < 8) drag.current = "in";
    else if (py < 20 && Math.abs(px - x(p.rangeEnd)) < 8) drag.current = "out";
    else {
      drag.current = "scrub";
      p.setPlaying(false);
      p.setFrame(f(px));
    }
    const move = (ev: MouseEvent) => {
      const fx = f(ev.clientX - r.left);
      if (drag.current === "scrub") p.setFrame(fx);
      else if (drag.current === "in") p.setRange(Math.min(fx, p.rangeEnd), p.rangeEnd);
      else if (drag.current === "out") p.setRange(p.rangeStart, Math.max(fx, p.rangeStart));
    };
    const up = () => {
      drag.current = null;
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  };

  const step = (d: number) => {
    p.setPlaying(false);
    p.setFrame(Math.max(0, Math.min(n - 1, p.frame + d)));
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT" || (e.target as HTMLElement)?.tagName === "SELECT") return;
      if (e.key === " ") {
        e.preventDefault();
        p.setPlaying(!p.playing);
      } else if (e.key === "ArrowRight" || e.key === "PageDown") step(e.shiftKey ? 10 : 1);
      else if (e.key === "ArrowLeft" || e.key === "PageUp") step(e.shiftKey ? -10 : -1);
      else if (e.key === "Home") step(-n);
      else if (e.key === "End") step(n);
      else if (e.key === "b") p.setRange(Math.min(p.frame, p.rangeEnd), p.rangeEnd);
      else if (e.key === "n") p.setRange(p.rangeStart, Math.max(p.frame, p.rangeStart));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <div className="timeline">
      <div className="transport">
        <button title="First frame (Home)" onClick={() => step(-n)}>⏮</button>
        <button title="Previous frame (←)" onClick={() => step(-1)}>◀︎|</button>
        <button className="play" title="Play / pause (Space)" onClick={() => p.setPlaying(!p.playing)}>
          {p.playing ? "❚❚" : "▶"}
        </button>
        <button title="Next frame (→)" onClick={() => step(1)}>|▶︎</button>
        <button title="Last frame (End)" onClick={() => step(n)}>⏭</button>
        <button className={p.loop ? "on" : ""} title="Loop work area" onClick={() => p.setLoop(!p.loop)}>⟲</button>
        <div className="tc">
          <input
            type="number"
            value={p.frame}
            min={0}
            max={n - 1}
            onChange={(e) => p.setFrame(Math.max(0, Math.min(n - 1, parseInt(e.target.value || "0"))))}
          />
          <span>{tc(p.frame, p.fps)}</span>
          <span className="dim">/ {n - 1} · {p.fps.toFixed(3).replace(/\.?0+$/, "")} fps</span>
        </div>
        <div className="range">
          <span className="dim">Work area</span>
          <input type="number" value={p.rangeStart} min={0} max={p.rangeEnd} onChange={(e) => p.setRange(Math.max(0, parseInt(e.target.value || "0")), p.rangeEnd)} />
          <span>–</span>
          <input type="number" value={p.rangeEnd} min={p.rangeStart} max={n - 1} onChange={(e) => p.setRange(p.rangeStart, Math.min(n - 1, parseInt(e.target.value || "0")))} />
          <button title="Set in point at playhead (B)" onClick={() => p.setRange(Math.min(p.frame, p.rangeEnd), p.rangeEnd)}>[ In</button>
          <button title="Set out point at playhead (N)" onClick={() => p.setRange(p.rangeStart, Math.max(p.frame, p.rangeStart))}>Out ]</button>
        </div>
      </div>
      <div ref={wrap} className="tl-canvas-wrap">
        <canvas ref={ref} style={{ width: w, height: H }} onMouseDown={down} />
      </div>
    </div>
  );
}
