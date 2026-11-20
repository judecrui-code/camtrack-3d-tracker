/**
 * Interactive 3D viewport (Three.js): point cloud, camera trajectory, current
 * camera frustum, ground plane, origin axes, selected points, scene objects and
 * the live target. World space is Z-up (same as Blender export).
 */
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { FrameCamera, V3, planeBasis, seqIndexOf, toWorld, unit, worldDir } from "../geometry";
import type { DisplaySettings, SolutionData, TargetState } from "../types";

interface Props {
  solution: SolutionData | null;
  frame: number;
  selection: Set<number>;
  target: TargetState | null;
  lockedTarget: TargetState | null;
  display: DisplaySettings;
  selectedObject: string | null;
}

export default function SceneView(p: Props) {
  const mount = useRef<HTMLDivElement>(null);
  const ctx = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    dyn: THREE.Group; // rebuilt when solution changes
    live: THREE.Group; // per-frame things
    fitted: boolean;
    bounds: THREE.Box3 | null;
  } | null>(null);
  const [lookThrough, setLookThrough] = useState(false);

  // ---- init ----
  useEffect(() => {
    const el = mount.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x16181c);
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 5000);
    camera.up.set(0, 0, 1);
    camera.position.set(8, -12, 6);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.screenSpacePanning = true;
    const dyn = new THREE.Group();
    const live = new THREE.Group();
    scene.add(dyn, live);
    scene.add(new THREE.AmbientLight(0xffffff, 1));
    ctx.current = { renderer, scene, camera, controls, dyn, live, fitted: false, bounds: null };
    const resize = () => {
      const w = el.clientWidth,
        h = el.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / Math.max(1, h);
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();
    let raf = 0;
    const loop = () => {
      raf = requestAnimationFrame(loop);
      controls.update();
      renderer.render(scene, camera);
    };
    loop();
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      ctx.current = null;
    };
  }, []);

  // ---- static content: points, trajectory, ground, origin, objects ----
  useEffect(() => {
    const c = ctx.current;
    if (!c) return;
    disposeGroup(c.dyn);
    const sol = p.solution;
    if (!sol) {
      c.dyn.add(new THREE.GridHelper(20, 20, 0x444444, 0x2a2a2a).rotateX(Math.PI / 2));
      return;
    }
    const M = sol.world.M;
    const pts = sol.points;
    const n = pts.ids.length;
    const pos: number[] = [];
    const col: number[] = [];
    const selPos: number[] = [];
    const badPos: number[] = [];
    for (let i = 0; i < n; i++) {
      if (pts.deleted[i]) continue;
      const w = toWorld(M, [pts.xyz[3 * i], pts.xyz[3 * i + 1], pts.xyz[3 * i + 2]]);
      if (p.selection.has(pts.ids[i])) {
        selPos.push(...w);
        continue;
      }
      if (!pts.reliable[i]) {
        if (p.display.showUnreliable) badPos.push(...w);
        continue;
      }
      pos.push(...w);
      col.push(pts.colors[3 * i] / 255, pts.colors[3 * i + 1] / 255, pts.colors[3 * i + 2] / 255);
    }
    const mkPoints = (arr: number[], color: number | null, size: number, colors?: number[]) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(arr, 3));
      if (colors) g.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
      const m = new THREE.PointsMaterial({ size, sizeAttenuation: false, vertexColors: !!colors, color: color ?? 0xffffff });
      return new THREE.Points(g, m);
    };
    c.dyn.add(mkPoints(pos, null, Math.max(2, p.display.pointSize * 0.6), col));
    if (badPos.length) c.dyn.add(mkPoints(badPos, 0xff5a3c, 3));
    if (selPos.length) c.dyn.add(mkPoints(selPos, 0x00e5ff, 6));

    // trajectory
    const centres: THREE.Vector3[] = [];
    const kf: number[] = [];
    const interp: number[] = [];
    for (let i = 0; i < sol.frames.length; i++) {
      const cam = new FrameCamera(sol, i);
      const w = toWorld(M, cam.center);
      centres.push(new THREE.Vector3(...w));
      if (sol.frame_status[i] === "keyframe") kf.push(...w);
      if (sol.frame_status[i] === "interpolated") interp.push(...w);
    }
    c.dyn.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(centres), new THREE.LineBasicMaterial({ color: 0xff8c1a })));
    if (kf.length) c.dyn.add(mkPoints(kf, 0xffd54f, 5));
    if (interp.length) c.dyn.add(mkPoints(interp, 0xff3030, 6));

    // scene extent
    const box = new THREE.Box3().setFromPoints(centres);
    for (let i = 0; i < pos.length; i += 3 * Math.max(1, Math.floor(pos.length / 3 / 2000))) box.expandByPoint(new THREE.Vector3(pos[i], pos[i + 1], pos[i + 2]));
    const sz = box.getSize(new THREE.Vector3()).length() || 10;

    // ground plane (world z = 0) + grid + origin axes
    const gsize = Math.pow(10, Math.ceil(Math.log10(sz)));
    const grid = new THREE.GridHelper(gsize, 20, 0x5a6270, 0x30353d);
    grid.rotateX(Math.PI / 2);
    c.dyn.add(grid);
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(gsize * 0.5, gsize * 0.5),
      new THREE.MeshBasicMaterial({ color: 0xff3020, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false }),
    );
    c.dyn.add(plane);
    const axes = new THREE.AxesHelper(gsize * 0.08);
    (axes.material as THREE.LineBasicMaterial).linewidth = 2;
    c.dyn.add(axes);

    // objects
    for (const ob of sol.scene.objects) {
      const loc = toWorld(M, ob.position as V3);
      const nW = worldDir(M, ob.normal as V3);
      const xW = worldDir(M, ob.x_hint as V3);
      const [bx, by] = planeBasis(nW, xW);
      const basis = new THREE.Matrix4().makeBasis(new THREE.Vector3(...bx), new THREE.Vector3(...by), new THREE.Vector3(...nW));
      const hi = ob.id === p.selectedObject;
      let o3: THREE.Object3D;
      if (ob.type === "null") {
        o3 = new THREE.AxesHelper(ob.size);
      } else if (ob.type === "anchor") {
        o3 = new THREE.AxesHelper(ob.size);
        o3.setRotationFromMatrix(basis);
        const cone = new THREE.Mesh(new THREE.ConeGeometry(ob.size * 0.12, ob.size * 0.3, 12), new THREE.MeshBasicMaterial({ color: 0x4da3ff }));
        cone.rotateX(Math.PI / 2);
        cone.position.set(0, 0, ob.size);
        o3.add(cone);
      } else {
        const s = ob.type === "solid" ? ob.size : ob.size * 4;
        o3 = new THREE.Mesh(
          new THREE.PlaneGeometry(s, s),
          new THREE.MeshBasicMaterial({
            color: ob.type === "solid" ? 0xffd23c : 0xff783c,
            transparent: true,
            opacity: hi ? 0.7 : 0.4,
            side: THREE.DoubleSide,
          }),
        );
        o3.setRotationFromMatrix(basis);
      }
      o3.position.set(...loc);
      c.dyn.add(o3);
      if (hi) {
        const ring = new THREE.Mesh(new THREE.SphereGeometry(ob.size * 0.15, 12, 8), new THREE.MeshBasicMaterial({ color: 0x00e5ff, wireframe: true }));
        ring.position.set(...loc);
        c.dyn.add(ring);
      }
    }

    c.bounds = box.clone();
    if (!c.fitted) frameAll(c);
  }, [p.solution, p.selection, p.display.showUnreliable, p.display.pointSize, p.selectedObject]);

  // ---- per-frame: current camera frustum & target ----
  useEffect(() => {
    const c = ctx.current;
    if (!c) return;
    disposeGroup(c.live);
    const sol = p.solution;
    if (!sol) return;
    const M = sol.world.M;
    const i = seqIndexOf(sol, p.frame);
    if (i >= 0) {
      const cam = new FrameCamera(sol, i);
      const C = toWorld(M, cam.center);
      const right = worldDir(M, cam.axis(0));
      const down = worldDir(M, cam.axis(1));
      const fwd = worldDir(M, cam.axis(2));
      const W = sol.width,
        H = sol.height;
      const box = new THREE.Box3();
      const L = (sol.world.scale * 10) / 6;
      const corner = (u: number, v: number) => {
        const x = (u - cam.cx) / cam.f,
          y = (v - cam.cy) / cam.f;
        return new THREE.Vector3(C[0] + L * (fwd[0] + x * right[0] + y * down[0]), C[1] + L * (fwd[1] + x * right[1] + y * down[1]), C[2] + L * (fwd[2] + x * right[2] + y * down[2]));
      };
      const o = new THREE.Vector3(...C);
      const cs = [corner(0, 0), corner(W, 0), corner(W, H), corner(0, H)];
      const seg = [o, cs[0], o, cs[1], o, cs[2], o, cs[3], cs[0], cs[1], cs[1], cs[2], cs[2], cs[3], cs[3], cs[0]];
      // "up" marker triangle
      const top = new THREE.Vector3().addVectors(cs[0], cs[1]).multiplyScalar(0.5);
      const upv = new THREE.Vector3(-down[0], -down[1], -down[2]).multiplyScalar(L * 0.15);
      seg.push(cs[0].clone().lerp(cs[1], 0.35), top.clone().add(upv), top.clone().add(upv), cs[0].clone().lerp(cs[1], 0.65));
      const geo = new THREE.BufferGeometry().setFromPoints(seg);
      c.live.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x00e5ff })));
      const dot = new THREE.Mesh(new THREE.SphereGeometry(L * 0.04, 12, 8), new THREE.MeshBasicMaterial({ color: 0x00e5ff }));
      dot.position.copy(o);
      c.live.add(dot);
      void box;
      if (lookThrough) {
        // view exactly through the solved camera
        c.camera.position.copy(o);
        c.camera.up.set(-down[0], -down[1], -down[2]);
        c.camera.fov = (2 * Math.atan(H / 2 / cam.f) * 180) / Math.PI;
        c.camera.updateProjectionMatrix();
        const tgt = o.clone().add(new THREE.Vector3(...fwd));
        c.controls.target.copy(tgt);
        c.camera.lookAt(tgt);
      }
    }
    for (const t of [p.lockedTarget, p.target]) {
      if (!t) continue;
      const ctr = toWorld(M, t.center);
      const n = worldDir(M, t.normal);
      const r = t.radius * sol.world.scale;
      const disc = new THREE.Mesh(
        new THREE.RingGeometry(r * 0.2, r, 48),
        new THREE.MeshBasicMaterial({ color: 0xff2020, transparent: true, opacity: t === p.lockedTarget ? 0.75 : 0.45, side: THREE.DoubleSide, depthWrite: false }),
      );
      disc.position.set(...ctr);
      disc.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3(...unit(n)));
      c.live.add(disc);
      c.live.add(new THREE.ArrowHelper(new THREE.Vector3(...n), new THREE.Vector3(...ctr), r * 1.5, 0x4da3ff));
    }
  }, [p.solution, p.frame, p.target, p.lockedTarget, lookThrough]);

  useEffect(() => {
    const c = ctx.current;
    if (!c || lookThrough) return;
    c.camera.up.set(0, 0, 1);
    c.camera.fov = 50;
    c.camera.updateProjectionMatrix();
  }, [lookThrough]);

  return (
    <div className="scene-view">
      <div ref={mount} className="scene-mount" />
      <div className="scene-hud">
        <span>Orbit: LMB · Pan: RMB/Shift · Zoom: wheel</span>
        <label>
          <input type="checkbox" checked={lookThrough} onChange={(e) => setLookThrough(e.target.checked)} /> Look through solved camera
        </label>
        <button
          className="mini"
          onClick={() => {
            setLookThrough(false);
            if (ctx.current) frameAll(ctx.current);
          }}
        >
          Frame all
        </button>
        <span className="legend">
          <i style={{ background: "#ff8c1a" }} /> trajectory <i style={{ background: "#ffd54f" }} /> keyframes <i style={{ background: "#00e5ff" }} /> current camera / selection{" "}
          <i style={{ background: "#ff5a3c" }} /> unreliable
        </span>
      </div>
    </div>
  );
}

function frameAll(c: { camera: THREE.PerspectiveCamera; controls: OrbitControls; bounds: THREE.Box3 | null; fitted: boolean }) {
  if (!c.bounds) return;
  const sz = c.bounds.getSize(new THREE.Vector3()).length() || 10;
  const ctr = c.bounds.getCenter(new THREE.Vector3());
  c.camera.up.set(0, 0, 1);
  c.camera.fov = 50;
  c.controls.target.copy(ctr);
  c.camera.position.copy(ctr.clone().add(new THREE.Vector3(sz * 0.6, -sz * 0.8, sz * 0.5)));
  c.camera.near = sz / 5000;
  c.camera.far = sz * 50;
  c.camera.updateProjectionMatrix();
  c.fitted = true;
}

function disposeGroup(g: THREE.Group) {
  for (const ch of [...g.children]) {
    g.remove(ch);
    ch.traverse((o: any) => {
      o.geometry?.dispose?.();
      if (Array.isArray(o.material)) o.material.forEach((m: any) => m.dispose());
      else o.material?.dispose?.();
    });
  }
}
