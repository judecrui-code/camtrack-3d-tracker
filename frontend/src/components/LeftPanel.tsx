import { useRef, useState } from "react";
import type { Job, ProjectSummary, Settings } from "../types";

interface Props {
  projects: ProjectSummary[];
  project: ProjectSummary | null;
  openProject: (id: string) => void;
  deleteProject: (id: string) => void;
  onUpload: (f: File) => void;
  onImportPath: (p: string) => void;
  uploadProgress: number | null;
  settings: Settings | null;
  setSettings: (s: Settings) => void;
  onAnalyze: () => void;
  onResolve: () => void;
  onCancel: () => void;
  job: Job | null;
}

function Section({ title, children, open = true }: { title: string; children: React.ReactNode; open?: boolean }) {
  const [o, setO] = useState(open);
  return (
    <div className="section">
      <div className="section-h" onClick={() => setO(!o)}>
        <span className="caret">{o ? "▾" : "▸"}</span> {title}
      </div>
      {o && <div className="section-b">{children}</div>}
    </div>
  );
}

export const Row = ({ label, children, hint }: { label: string; children: React.ReactNode; hint?: string }) => (
  <label className="row" title={hint}>
    <span className="row-l">{label}</span>
    <span className="row-v">{children}</span>
  </label>
);

export default function LeftPanel(p: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [path, setPath] = useState("");
  const s = p.settings;
  const info = p.project?.info;
  const set = (k: keyof Settings, v: any) => s && p.setSettings({ ...s, [k]: v });
  const busy = p.job && (p.job.status === "running" || p.job.status === "queued") && p.job.kind !== "ingest";
  const procW = s && info ? (s.processing_width || (Math.max(info.width, info.height) <= 1280 ? info.width : Math.round((info.width * 1280) / Math.max(info.width, info.height)))) : 0;

  return (
    <div className="panel left">
      <Section title="Project">
        <div className="btn-row">
          <button className="primary" onClick={() => fileRef.current?.click()}>Import Footage…</button>
          <input
            ref={fileRef}
            type="file"
            accept="video/*,.mp4,.mov,.m4v,.avi,.mkv,.webm,.mxf,.mts,.m2ts"
            style={{ display: "none" }}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) p.onUpload(f);
              e.target.value = "";
            }}
          />
        </div>
        {p.uploadProgress != null && (
          <div className="progress small">
            <div style={{ width: `${p.uploadProgress * 100}%` }} />
            <span>Uploading {Math.round(p.uploadProgress * 100)}%</span>
          </div>
        )}
        <div className="path-import">
          <input placeholder="…or local file path on this machine" value={path} onChange={(e) => setPath(e.target.value)} />
          <button disabled={!path} onClick={() => p.onImportPath(path)}>Open</button>
        </div>
        <div className="project-list">
          {p.projects.map((pr) => (
            <div key={pr.id} className={"project-item" + (pr.id === p.project?.id ? " active" : "")} onClick={() => p.openProject(pr.id)}>
              <span className={"dot " + (pr.has_solution ? "ok" : pr.has_tracks ? "mid" : "")} />
              <span className="pname">{pr.name}</span>
              <span className="dim">{pr.info.width}×{pr.info.height}</span>
              <button
                className="x"
                title="Delete project"
                onClick={(e) => {
                  e.stopPropagation();
                  if (confirm(`Delete project "${pr.name}" and its cached data?`)) p.deleteProject(pr.id);
                }}
              >
                ×
              </button>
            </div>
          ))}
          {!p.projects.length && <div className="dim">No projects. Import an MP4/MOV to start.</div>}
        </div>
      </Section>

      {info && (
        <Section title="Footage">
          <div className="kv">
            <span>File</span>
            <b title={info.filename}>{info.filename}</b>
            <span>Resolution</span>
            <b>
              {info.width} × {info.height}
              {info.pixel_aspect !== 1 ? ` (PAR ${info.pixel_aspect.toFixed(3)})` : ""}
            </b>
            <span>Frame rate</span>
            <b>
              {info.fps.toFixed(3).replace(/\.?0+$/, "")} fps ({info.fps_num}/{info.fps_den})
            </b>
            <span>Duration</span>
            <b>
              {info.duration.toFixed(2)} s · {info.frame_count} frames
            </b>
            <span>Codec</span>
            <b>{info.codec}</b>
            <span>Display cache</span>
            <b>{p.project?.frames_cached ? "ready (full-res)" : "building…"}</b>
          </div>
        </Section>
      )}

      {s && info && (
        <Section title="Track Camera">
          <Row label="Engine" hint="Native: sequential video SfM (KLT tracks + bundle adjustment). COLMAP: SIFT + COLMAP incremental mapper.">
            <select value={s.engine} onChange={(e) => set("engine", e.target.value)}>
              <option value="native">Native video SfM</option>
              <option value="colmap">COLMAP (pycolmap)</option>
            </select>
          </Row>
          <Row label="Solve method">
            <select value={s.solve_method} onChange={(e) => set("solve_method", e.target.value)}>
              <option value="auto">Auto detect</option>
              <option value="typical">Typical (moving camera)</option>
              <option value="flat">Mostly flat scene</option>
              <option value="tripod">Tripod pan (rotation only)</option>
            </select>
          </Row>
          <Row label="Frame range">
            <input type="number" value={s.frame_start ?? 0} onChange={(e) => set("frame_start", parseInt(e.target.value || "0"))} />
            <span className="dim">–</span>
            <input type="number" value={s.frame_end ?? info.frame_count - 1} onChange={(e) => set("frame_end", parseInt(e.target.value || "0"))} />
          </Row>
          <Row label="Processing res." hint="Analysis resolution (never upscaled). Results are always reported in original pixels.">
            <select value={s.processing_width} onChange={(e) => set("processing_width", parseInt(e.target.value))}>
              <option value={0}>Auto ({procW}px wide)</option>
              {[640, 960, 1280, 1920, 2560, 3840].filter((v) => v <= info.width).map((v) => (
                <option key={v} value={v}>{v}px{v === info.width ? " (full)" : ""}</option>
              ))}
              {![640, 960, 1280, 1920, 2560, 3840].includes(info.width) && <option value={info.width}>{info.width}px (full)</option>}
            </select>
          </Row>
          <Row label="Frame sampling" hint="Track every Nth frame. Every frame is the most reliable. Auto picks about 30 fps and caps long clips at ~400 frames to save time; Blender interpolates the camera between solved frames.">
            <select value={s.frame_step} onChange={(e) => set("frame_step", parseInt(e.target.value))}>
              {[1, 2, 3, 4, 5].map((v) => (
                <option key={v} value={v}>{v === 1 ? "Every frame (most reliable)" : `Every ${v} frames`}</option>
              ))}
              <option value={0}>Auto (~30 fps, max 400 frames: faster, less reliable)</option>
            </select>
          </Row>
          <div className="subhead">Feature tracking</div>
          <Row label="Detection quality">
            <select value={s.detection_quality} onChange={(e) => set("detection_quality", e.target.value)}>
              <option value="draft">Draft</option>
              <option value="standard">Standard</option>
              <option value="high">High</option>
            </select>
          </Row>
          <Row label="Max points">
            <input type="number" step={250} value={s.max_points} onChange={(e) => set("max_points", parseInt(e.target.value || "0"))} />
          </Row>
          <Row label="Min track length">
            <input type="number" value={s.min_track_length} onChange={(e) => set("min_track_length", parseInt(e.target.value || "0"))} />
            <span className="dim">frames</span>
          </Row>
          <Row label="Sensitivity" hint="Higher = more features and more permissive matching; lower = stricter rejection.">
            <input type="range" min={0} max={1} step={0.05} value={s.tracking_sensitivity} onChange={(e) => set("tracking_sensitivity", parseFloat(e.target.value))} />
            <span className="num">{s.tracking_sensitivity.toFixed(2)}</span>
          </Row>
          <div className="subhead">Camera / lens</div>
          <Row label="Focal length">
            <select
              value={s.focal_length_px != null ? "px" : s.focal_length_mm != null ? "mm" : "auto"}
              onChange={(e) => {
                const v = e.target.value;
                p.setSettings({ ...s, focal_length_px: v === "px" ? Math.round(info.width * 0.9) : null, focal_length_mm: v === "mm" ? 35 : null });
              }}
            >
              <option value="auto">Estimate</option>
              <option value="px">Known (pixels)</option>
              <option value="mm">Known (mm)</option>
            </select>
          </Row>
          {s.focal_length_px != null && (
            <Row label="Focal (px)">
              <input type="number" value={s.focal_length_px} onChange={(e) => set("focal_length_px", parseFloat(e.target.value))} />
            </Row>
          )}
          {s.focal_length_mm != null && (
            <>
              <Row label="Focal (mm)">
                <input type="number" value={s.focal_length_mm} onChange={(e) => set("focal_length_mm", parseFloat(e.target.value))} />
              </Row>
              <Row label="Sensor width (mm)">
                <input type="number" value={s.sensor_width_mm} onChange={(e) => set("sensor_width_mm", parseFloat(e.target.value))} />
              </Row>
            </>
          )}
          <Row label="Variable focal (zoom)">
            <input type="checkbox" checked={s.variable_focal} onChange={(e) => set("variable_focal", e.target.checked)} />
          </Row>
          <Row label="Lens distortion">
            <select value={s.distortion} onChange={(e) => set("distortion", e.target.value)}>
              <option value="none">None</option>
              <option value="k1">Estimate k1</option>
              <option value="k1k2">Estimate k1 + k2</option>
              <option value="manual">Manual</option>
            </select>
          </Row>
          {s.distortion === "manual" && (
            <Row label="k1 / k2">
              <input type="number" step={0.01} value={s.k1} onChange={(e) => set("k1", parseFloat(e.target.value))} />
              <input type="number" step={0.01} value={s.k2} onChange={(e) => set("k2", parseFloat(e.target.value))} />
            </Row>
          )}
          <div className="subhead">Solver</div>
          <Row label="Reproj. threshold" hint="Observations with a larger reprojection error (processing px) are rejected as outliers.">
            <input type="number" step={0.25} value={s.reprojection_threshold} onChange={(e) => set("reprojection_threshold", parseFloat(e.target.value))} />
            <span className="dim">px</span>
          </Row>
          <Row label="Min. triangulation angle">
            <input type="number" step={0.25} value={s.min_triangulation_angle} onChange={(e) => set("min_triangulation_angle", parseFloat(e.target.value))} />
            <span className="dim">deg</span>
          </Row>
          <div className="btn-row">
            <button className="primary big" disabled={!!busy} onClick={p.onAnalyze} title="Track features then solve the camera">
              Analyze &amp; Solve
            </button>
            <button disabled={!!busy || !p.project?.has_tracks} onClick={p.onResolve} title="Re-run only the camera solve with existing tracks (after deleting points or changing solve settings)">
              Re-solve
            </button>
            {busy && (
              <button className="danger" onClick={p.onCancel}>
                Cancel
              </button>
            )}
          </div>
        </Section>
      )}

      {p.job && (
        <Section title="Analysis Log" open={true}>
          <div className="log">
            {p.job.log.map((l, i) => (
              <div key={i} className={l.includes("WARNING") ? "warn" : l.startsWith("Traceback") ? "err" : ""}>
                {l}
              </div>
            ))}
            {p.job.error && <div className="err">{p.job.error}</div>}
          </div>
        </Section>
      )}
    </div>
  );
}
