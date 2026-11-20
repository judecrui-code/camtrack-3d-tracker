import { useState } from "react";
import { api } from "../api";
import type { DisplaySettings, ProjectSummary, SolutionData, TargetState } from "../types";
import { Row } from "./LeftPanel";

interface Props {
  project: ProjectSummary | null;
  solution: SolutionData | null;
  display: DisplaySettings;
  setDisplay: (d: DisplaySettings) => void;
  selection: Set<number>;
  setSelection: (s: Set<number>) => void;
  lockedTarget: TargetState | null;
  onDeleteSelected: () => void;
  onRestoreDeleted: () => void;
  onSceneChange: () => void;
  selectedObject: string | null;
  setSelectedObject: (id: string | null) => void;
  notify: (msg: string, kind?: "info" | "error") => void;
}

function Section({ title, children, open = true, extra }: { title: string; children: React.ReactNode; open?: boolean; extra?: React.ReactNode }) {
  const [o, setO] = useState(open);
  return (
    <div className="section">
      <div className="section-h" onClick={() => setO(!o)}>
        <span className="caret">{o ? "▾" : "▸"}</span> {title}
        {extra && <span className="section-x">{extra}</span>}
      </div>
      {o && <div className="section-b">{children}</div>}
    </div>
  );
}

const rating = (e: number) => (e < 0.3 ? ["Excellent", "q-ex"] : e < 0.5 ? ["Good", "q-good"] : e < 1.0 ? ["Fair", "q-fair"] : ["Poor", "q-poor"]);

export default function RightPanel(p: Props) {
  const sol = p.solution;
  const pid = p.project?.id;
  const [includeFootage, setIncludeFootage] = useState(true);
  const [frameOffset, setFrameOffset] = useState(1);
  const [scaleVal, setScaleVal] = useState<string>("");
  const st = sol?.stats;
  const d = p.display;
  const setD = (k: keyof DisplaySettings, v: any) => p.setDisplay({ ...d, [k]: v });
  const scene = sol?.scene;
  const ground = scene?.ground ?? scene?.auto_ground ?? null;
  const W = sol?.width ?? 1;

  const run = async (fn: () => Promise<any>, ok?: string) => {
    try {
      await fn();
      p.onSceneChange();
      if (ok) p.notify(ok);
    } catch (e: any) {
      p.notify(e.message, "error");
    }
  };

  const selectWhere = (pred: (i: number) => boolean) => {
    if (!sol) return;
    const s = new Set<number>();
    sol.points.ids.forEach((id, i) => !sol.points.deleted[i] && pred(i) && s.add(id));
    p.setSelection(s);
  };

  return (
    <div className="panel right">
      <Section title="Solve Quality">
        {!sol && <div className="dim">No camera solve yet.</div>}
        {sol && st && (
          <>
            <div className={"quality " + rating(st.mean_error_px)[1]}>
              <div className="q-big">{st.mean_error_px.toFixed(3)} px</div>
              <div className="q-lbl">
                avg. reprojection error · <b>{rating(st.mean_error_px)[0]}</b>
              </div>
            </div>
            <div className="kv">
              <span>Median / 95th pct</span>
              <b>
                {st.median_error_px.toFixed(3)} / {st.p95_error_px.toFixed(3)} px
              </b>
              <span>Solve mode</span>
              <b>
                {sol.mode} · {sol.engine}
              </b>
              <span>Solved frames</span>
              <b>
                {st.n_solved_frames}/{st.n_frames}
                {st.n_interpolated_frames ? ` (${st.n_interpolated_frames} interpolated)` : ""}
              </b>
              <span>3D points</span>
              <b>
                {st.n_points} ({sol.points.reliable.filter(Boolean).length} reliable)
              </b>
              <span>Focal length</span>
              <b>
                {st.focal_px.toFixed(1)} px · {((st.focal_px / W) * 36).toFixed(2)} mm
                <span className="dim"> (36mm)</span>
              </b>
              <span>Horizontal FOV</span>
              <b>{((2 * Math.atan(W / 2 / st.focal_px) * 180) / Math.PI).toFixed(2)}°</b>
              <span>Focal source</span>
              <b>{st.focal_estimated ? "estimated" : "user supplied"}</b>
              <span>Distortion k1 / k2</span>
              <b>
                {st.k1.toFixed(4)} / {st.k2.toFixed(4)}
              </b>
              {sol.mode !== "tripod" && (
                <>
                  <span>Median tri. angle</span>
                  <b>{st.median_triangulation_angle?.toFixed(2)}°</b>
                </>
              )}
            </div>
            {sol.warnings.length > 0 && (
              <div className="warnings">
                {sol.warnings.map((w, i) => (
                  <div key={i} className="warning">⚠ {w}</div>
                ))}
              </div>
            )}
            {sol.needs_resolve && <div className="note">Point deletions changed since the last solve — press Re-solve to update the camera.</div>}
            {!sol.needs_resolve && sol.deleted_count > 0 && <div className="hint">{sol.deleted_count} deleted point(s) excluded from this solve.</div>}
          </>
        )}
      </Section>

      <Section title="Display">
        <Row label="Track point size">
          <input type="range" min={2} max={14} step={0.5} value={d.pointSize} onChange={(e) => setD("pointSize", parseFloat(e.target.value))} />
          <span className="num">{d.pointSize}</span>
        </Row>
        <Row label="Track point density">
          <input type="range" min={0.05} max={1} step={0.05} value={d.density} onChange={(e) => setD("density", parseFloat(e.target.value))} />
          <span className="num">{Math.round(d.density * 100)}%</span>
        </Row>
        <Row label="Target size">
          <input type="range" min={0.2} max={4} step={0.1} value={d.targetSize} onChange={(e) => setD("targetSize", parseFloat(e.target.value))} />
          <span className="num">{d.targetSize.toFixed(1)}</span>
        </Row>
        <Row label="Show unreliable points">
          <input type="checkbox" checked={d.showUnreliable} onChange={(e) => setD("showUnreliable", e.target.checked)} />
        </Row>
        <Row label="Show raw 2D tracks">
          <input type="checkbox" checked={d.showRaw2D} onChange={(e) => setD("showRaw2D", e.target.checked)} />
        </Row>
        <Row label="Show scene overlay" hint="Ground grid, world axes and created objects projected through the solved camera">
          <input type="checkbox" checked={d.showOverlay} onChange={(e) => setD("showOverlay", e.target.checked)} />
        </Row>
      </Section>

      {sol && (
        <Section title={`Selection (${p.selection.size})`}>
          <div className="hint">Click the footage to lock a target. Shift/Ctrl-click toggles points; Shift-drag box-selects. Delete removes.</div>
          <div className="btn-grid">
            <button onClick={() => selectWhere((i) => !sol.points.reliable[i])}>Select unreliable</button>
            <button onClick={() => selectWhere((i) => sol.points.error[i] > sol.threshold * 0.5)}>Select high error</button>
            <button onClick={() => p.setSelection(new Set())}>Deselect</button>
            <button className="danger" disabled={!p.selection.size} onClick={p.onDeleteSelected}>
              Delete selected
            </button>
            <button disabled={!sol.deleted_count} onClick={p.onRestoreDeleted}>
              Restore deleted
            </button>
          </div>
          {p.selection.size === 2 && sol.mode !== "tripod" && (
            <div className="scale-dist">
              <div className="hint">Two points selected: set their real-world distance to define scene scale.</div>
              <Row label="Distance">
                <input type="number" placeholder="e.g. 1.5" value={scaleVal} onChange={(e) => setScaleVal(e.target.value)} />
                <button
                  disabled={!(parseFloat(scaleVal) > 0)}
                  onClick={() => {
                    const [a, b] = [...p.selection];
                    run(() => api.setScale(pid!, { mode: "distance", value: parseFloat(scaleVal), a, b }), "Scale set from point distance");
                  }}
                >
                  Set scale
                </button>
              </Row>
            </div>
          )}
        </Section>
      )}

      {sol && (
        <Section title="Target / Ground Plane">
          {p.lockedTarget ? (
            <div className="kv">
              <span>Target points</span>
              <b>{p.lockedTarget.pointIdx.length}</b>
              <span>Plane fit RMS</span>
              <b>{(p.lockedTarget.rms * sol.world.scale).toFixed(4)} units</b>
              <span>Normal (world)</span>
              <b>
                {(() => {
                  const M = sol.world.M;
                  const n = p.lockedTarget!.normal;
                  const w = [0, 1, 2].map((r) => M[r][0] * n[0] + M[r][1] * n[1] + M[r][2] * n[2]);
                  const l = Math.hypot(w[0], w[1], w[2]);
                  return w.map((v) => (v / l).toFixed(3)).join(", ");
                })()}
              </b>
            </div>
          ) : (
            <div className="hint">Hover over the footage: a target appears where neighbouring solved points define a plane. Right-click for actions.</div>
          )}
          <div className="kv">
            <span>Ground plane</span>
            <b>{ground ? (ground.source === "user" ? "user defined" : `auto estimate (${ground.inliers ?? "?"} pts)`) : sol.mode === "tripod" ? "n/a (tripod)" : "none (camera-relative)"}</b>
            <span>Origin</span>
            <b>{scene?.origin ? "user defined" : "ground target"}</b>
            <span>Scale</span>
            <b>
              {scene?.scale.mode === "camera_height"
                ? `camera height ${scene.scale.value} (assumed)`
                : scene?.scale.mode === "distance"
                  ? `points ${scene.scale.a}↔${scene.scale.b} = ${scene.scale.value}`
                  : `factor ${scene?.scale.value}`}
            </b>
          </div>
          <Row label="Camera height">
            <input
              type="number"
              step={0.1}
              defaultValue={scene?.scale.mode === "camera_height" ? scene.scale.value : 1.6}
              onKeyDown={(e) => {
                if (e.key === "Enter") run(() => api.setScale(pid!, { mode: "camera_height", value: parseFloat((e.target as HTMLInputElement).value) }), "Scale updated");
              }}
            />
            <span className="dim">units ↵</span>
          </Row>
          <div className="btn-row">
            <button onClick={() => run(() => api.resetScene(pid!), "Scene reset to automatic ground estimate")}>Reset scene</button>
          </div>
        </Section>
      )}

      {sol && (
        <Section title={`Scene Objects (${scene?.objects.length ?? 0})`}>
          {!scene?.objects.length && <div className="hint">Right-click a target to create nulls, solids, 3D anchors or ground-plane objects.</div>}
          {scene?.objects.map((o) => (
            <div key={o.id} className={"obj-item" + (o.id === p.selectedObject ? " active" : "")} onClick={() => p.setSelectedObject(o.id === p.selectedObject ? null : o.id)}>
              <span className={"obj-ico " + o.type} />
              <input
                className="obj-name"
                defaultValue={o.name}
                onClick={(e) => e.stopPropagation()}
                onBlur={(e) => e.target.value !== o.name && run(() => api.updateObject(pid!, o.id, { name: e.target.value }))}
              />
              <input
                className="obj-size"
                type="number"
                step={0.1}
                defaultValue={o.size.toFixed(2)}
                title="Size (world units)"
                onClick={(e) => e.stopPropagation()}
                onBlur={(e) => parseFloat(e.target.value) > 0 && run(() => api.updateObject(pid!, o.id, { size: parseFloat(e.target.value) }))}
              />
              <button className="x" onClick={(e) => (e.stopPropagation(), run(() => api.deleteObject(pid!, o.id)))}>
                ×
              </button>
            </div>
          ))}
        </Section>
      )}

      <Section title="Export">
        <button
          className="primary big blender"
          disabled={!sol}
          onClick={() => {
            window.location.href = api.exportUrl(pid!, "blender", `?include_footage=${includeFootage}&frame_offset=${frameOffset}`);
          }}
        >
          Export to Blender
        </button>
        <Row label="Include original footage">
          <input type="checkbox" checked={includeFootage} onChange={(e) => setIncludeFootage(e.target.checked)} />
        </Row>
        <Row label="Blender frame offset" hint="Blender frame = footage frame + offset (default 1: footage frame 0 → Blender frame 1)">
          <input type="number" value={frameOffset} onChange={(e) => setFrameOffset(parseInt(e.target.value || "0"))} />
        </Row>
        <div className="btn-grid">
          {[
            ["ply", "PLY point cloud"],
            ["json", "JSON tracking data"],
            ["points.csv", "CSV 3D points"],
            ["tracks.csv", "CSV 2D tracks"],
            ["camera.csv", "CSV camera"],
          ].map(([k, l]) => (
            <button key={k} disabled={!sol} onClick={() => (window.location.href = api.exportUrl(pid!, k, `?frame_offset=${frameOffset}`))}>
              {l}
            </button>
          ))}
        </div>
      </Section>
    </div>
  );
}
