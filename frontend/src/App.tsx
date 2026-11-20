import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import LeftPanel from "./components/LeftPanel";
import RightPanel from "./components/RightPanel";
import SceneView from "./components/SceneView";
import Timeline from "./components/Timeline";
import Viewer from "./components/Viewer";
import type { DisplaySettings, Job, ProjectSummary, Settings, SolutionData, TargetState, ViewMode } from "./types";

export default function App() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [project, setProject] = useState<ProjectSummary | null>(null);
  const [settings, setSettingsState] = useState<Settings | null>(null);
  const [solution, setSolution] = useState<SolutionData | null>(null);
  const [frame, setFrameState] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loop, setLoop] = useState(true);
  const [view, setView] = useState<ViewMode>("footage");
  const [selection, setSelection] = useState<Set<number>>(new Set());
  const [target, setTarget] = useState<TargetState | null>(null);
  const [lockedTarget, setLockedTarget] = useState<TargetState | null>(null);
  const [selectedObject, setSelectedObject] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [ingestJob, setIngestJob] = useState<Job | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [toast, setToast] = useState<{ msg: string; kind: string } | null>(null);
  const [hoverInfo, setHoverInfo] = useState("");
  const [display, setDisplay] = useState<DisplaySettings>({
    pointSize: 4.5,
    density: 0.5,
    showUnreliable: true,
    showRaw2D: false,
    showOverlay: true,
    targetSize: 1,
  });
  const imageCache = useRef(new Map<number, HTMLImageElement>()).current;
  const saveTimer = useRef<number | undefined>(undefined);

  const notify = useCallback((msg: string, kind: "info" | "error" = "info") => {
    setToast({ msg, kind });
    window.setTimeout(() => setToast((t) => (t?.msg === msg ? null : t)), kind === "error" ? 7000 : 3500);
  }, []);

  const refreshProjects = useCallback(() => api.projects().then(setProjects).catch((e) => notify(e.message, "error")), [notify]);

  const loadSolution = useCallback(
    async (id: string) => {
      try {
        setSolution(await api.solution(id));
      } catch {
        setSolution(null);
      }
    },
    [],
  );

  const openProject = useCallback(
    async (id: string) => {
      try {
        const p = await api.project(id);
        imageCache.clear();
        setProject(p);
        setSettingsState(p.settings);
        setSelection(new Set());
        setLockedTarget(null);
        setTarget(null);
        setSelectedObject(null);
        setFrameState(p.settings.frame_start ?? 0);
        setJob(null);
        setIngestJob(null);
        for (const j of p.jobs ?? []) (j.kind === "ingest" ? setIngestJob : setJob)(j);
        if (p.has_solution) await loadSolution(id);
        else setSolution(null);
        localStorage.setItem("camtrack.project", id);
      } catch (e: any) {
        notify(e.message, "error");
      }
    },
    [imageCache, loadSolution, notify],
  );

  useEffect(() => {
    refreshProjects().then(() => {
      const last = localStorage.getItem("camtrack.project");
      if (last) openProject(last).catch(() => {});
    });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- job polling ----
  useEffect(() => {
    const active = [job, ingestJob].filter((j) => j && (j.status === "running" || j.status === "queued")) as Job[];
    if (!active.length) return;
    const h = window.setInterval(async () => {
      for (const j of active) {
        try {
          const nj = await api.job(j.id);
          (nj.kind === "ingest" ? setIngestJob : setJob)(nj);
          if (nj.status !== "running" && nj.status !== "queued") {
            if (nj.kind !== "ingest") {
              if (nj.status === "done") notify(`${nj.kind === "analyze" ? "Analysis" : "Solve"} finished`);
              else if (nj.status === "failed") notify(nj.error || "Job failed", "error");
              else if (nj.status === "cancelled") notify("Cancelled");
              if (project) {
                await loadSolution(project.id);
                setSelection(new Set());
              }
            }
            if (project) setProject(await api.project(project.id));
            refreshProjects();
          }
        } catch {
          /* transient */
        }
      }
    }, 400);
    return () => window.clearInterval(h);
  }, [job, ingestJob, project, loadSolution, notify, refreshProjects]);

  // ---- settings (debounced persist) ----
  const setSettings = (s: Settings) => {
    setSettingsState(s);
    window.clearTimeout(saveTimer.current);
    if (project) saveTimer.current = window.setTimeout(() => api.saveSettings(project.id, s).catch(() => {}), 400);
  };
  const frameCount = project?.info.frame_count ?? 1;
  const rangeStart = settings?.frame_start ?? 0;
  const rangeEnd = settings?.frame_end ?? frameCount - 1;
  const setRange = (a: number, b: number) => settings && setSettings({ ...settings, frame_start: a, frame_end: b });
  const setFrame = useCallback((f: number) => setFrameState(f), []);

  // ---- playback (real time, frame accurate, preloading) ----
  useEffect(() => {
    if (!playing || !project) return;
    const fps = project.info.fps || 24;
    let raf = 0;
    let last = performance.now();
    let acc = 0;
    let cur = frame;
    const preload = (from: number) => {
      for (let k = 1; k <= 24; k++) {
        const n = from + k;
        if (n > rangeEnd || imageCache.has(n)) continue;
        const im = new Image();
        im.src = api.frameUrl(project.id, n);
        imageCache.set(n, im);
        if (imageCache.size > 600) {
          const first = imageCache.keys().next().value as number;
          imageCache.delete(first);
        }
      }
    };
    const tick = (now: number) => {
      acc += now - last;
      last = now;
      const next = cur + 1 > rangeEnd ? (loop ? rangeStart : -1) : cur + 1;
      const ready = next >= 0 && imageCache.get(next)?.complete;
      if (acc >= 1000 / fps && ready) {
        acc = Math.min(acc - 1000 / fps, 1000 / fps);
        cur = next;
        setFrameState(cur);
        preload(cur);
      } else if (next < 0) {
        setPlaying(false);
        return;
      } else if (!imageCache.has(next)) preload(cur);
      raf = requestAnimationFrame(tick);
    };
    preload(cur);
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, project?.id, rangeStart, rangeEnd, loop]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- actions ----
  const onUpload = async (f: File) => {
    try {
      setUploadProgress(0);
      const r = await api.upload(f, setUploadProgress);
      setUploadProgress(null);
      await refreshProjects();
      await openProject(r.project.id);
      setIngestJob(r.job);
      notify(`Imported ${f.name}`);
    } catch (e: any) {
      setUploadProgress(null);
      notify(e.message, "error");
    }
  };
  const onImportPath = async (path: string) => {
    try {
      const r = await api.importPath(path);
      await refreshProjects();
      await openProject(r.project.id);
      setIngestJob(r.job);
    } catch (e: any) {
      notify(e.message, "error");
    }
  };
  const onAnalyze = async () => {
    if (!project || !settings) return;
    try {
      setJob(await api.analyze(project.id, settings));
      setLockedTarget(null);
    } catch (e: any) {
      notify(e.message, "error");
    }
  };
  const onResolve = async () => {
    if (!project || !settings) return;
    try {
      setJob(await api.solve(project.id, settings));
      setLockedTarget(null);
    } catch (e: any) {
      notify(e.message, "error");
    }
  };
  const onCancel = () => job && api.cancel(job.id);
  const onDeleteSelected = async () => {
    if (!project || !selection.size) return;
    await api.deletePoints(project.id, [...selection]);
    notify(`${selection.size} point(s) deleted. Re-solve to update the camera.`);
    setSelection(new Set());
    setLockedTarget(null);
    loadSolution(project.id);
  };
  const onRestoreDeleted = async () => {
    if (!project) return;
    await api.restorePoints(project.id);
    loadSolution(project.id);
  };
  const reloadScene = () => project && loadSolution(project.id);

  const onContextAction = async (action: string, t: TargetState) => {
    if (!project || !solution) return;
    const ids = t.pointIdx.map((i) => solution.points.ids[i]);
    try {
      if (action === "ground_origin" || action === "ground") {
        await api.setGround(project.id, { point: t.center, normal: t.normal, frame: t.frame, point_ids: ids, set_origin: action === "ground_origin" });
        if (action === "ground_origin") await api.setOrigin(project.id, t.center);
        notify(action === "ground" ? "Ground plane set" : "Ground plane and origin set");
      } else if (action === "origin") {
        await api.setOrigin(project.id, t.center);
        notify("Origin set");
      } else if (action === "select") {
        setSelection(new Set(ids));
        return;
      } else {
        const names: Record<string, string> = { null: "Null", solid: "Solid", anchor: "3D Anchor", ground_plane: "Ground Plane" };
        const n = (solution.scene.objects.filter((o) => o.type === action).length ?? 0) + 1;
        await api.createObject(project.id, {
          type: action,
          point: t.center,
          normal: t.normal,
          frame: t.frame,
          point_ids: ids,
          size_raw: t.radius * 2,
          name: `${names[action]} ${n}`,
        });
        notify(`${names[action]} created`);
      }
      setLockedTarget(null);
      await loadSolution(project.id);
    } catch (e: any) {
      notify(e.message, "error");
    }
  };

  const busy = job && (job.status === "running" || job.status === "queued");
  const activeJob = busy ? job : ingestJob && (ingestJob.status === "running" || ingestJob.status === "queued") ? ingestJob : null;

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand">
          <span className="logo">◎</span> CamTrack <span className="dim">3D Camera Tracker</span>
        </div>
        <div className="menu">
          <label className="tb-btn">
            Import
            <input
              type="file"
              accept="video/*,.mp4,.mov,.m4v,.avi,.mkv,.webm,.mxf,.mts,.m2ts"
              style={{ display: "none" }}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onUpload(f);
                e.target.value = "";
              }}
            />
          </label>
          <button className="tb-btn" disabled={!project || !!busy} onClick={onAnalyze}>
            Analyze
          </button>
          <button className="tb-btn" disabled={!project?.has_tracks || !!busy} onClick={onResolve}>
            Re-solve
          </button>
          {busy && (
            <button className="tb-btn danger" onClick={onCancel}>
              Cancel
            </button>
          )}
        </div>
        <div className="view-toggle">
          {(["footage", "3d", "split"] as ViewMode[]).map((v) => (
            <button key={v} className={view === v ? "on" : ""} onClick={() => setView(v)}>
              {v === "footage" ? "Footage" : v === "3d" ? "3D Scene" : "Split"}
            </button>
          ))}
        </div>
        <button
          className="tb-btn blender"
          disabled={!solution}
          onClick={() => project && (window.location.href = api.exportUrl(project.id, "blender", "?include_footage=true&frame_offset=1"))}
        >
          Export to Blender
        </button>
      </div>

      <div className="main">
        <LeftPanel
          projects={projects}
          project={project}
          openProject={openProject}
          deleteProject={async (id) => {
            try {
              await api.deleteProject(id);
              if (project?.id === id) {
                setProject(null);
                setSolution(null);
                setSettingsState(null);
              }
              refreshProjects();
            } catch (e: any) {
              notify(e.message, "error");
            }
          }}
          onUpload={onUpload}
          onImportPath={onImportPath}
          uploadProgress={uploadProgress}
          settings={settings}
          setSettings={setSettings}
          onAnalyze={onAnalyze}
          onResolve={onResolve}
          onCancel={onCancel}
          job={job}
        />
        <div className={"center view-" + view}>
          {!project && (
            <div className="empty">
              <div className="empty-logo">◎</div>
              <h2>Import footage to begin</h2>
              <p>MP4, MOV, MKV, AVI, WebM … The original file is kept untouched; analysis runs locally.</p>
              <label className="primary big tb-btn">
                Import Footage…
                <input
                  type="file"
                  accept="video/*"
                  style={{ display: "none" }}
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) onUpload(f);
                  }}
                />
              </label>
            </div>
          )}
          {project && (view === "footage" || view === "split") && (
            <Viewer
              project={project}
              solution={solution}
              frame={frame}
              display={display}
              selection={selection}
              setSelection={setSelection}
              target={target}
              setTarget={setTarget}
              lockedTarget={lockedTarget}
              setLockedTarget={setLockedTarget}
              onContextAction={onContextAction}
              onDeleteSelected={onDeleteSelected}
              imageCache={imageCache}
              onHoverInfo={setHoverInfo}
            />
          )}
          {project && (view === "3d" || view === "split") && (
            <SceneView
              solution={solution}
              frame={frame}
              selection={selection}
              target={target}
              lockedTarget={lockedTarget}
              display={display}
              selectedObject={selectedObject}
            />
          )}
        </div>
        <RightPanel
          project={project}
          solution={solution}
          display={display}
          setDisplay={setDisplay}
          selection={selection}
          setSelection={setSelection}
          lockedTarget={lockedTarget}
          onDeleteSelected={onDeleteSelected}
          onRestoreDeleted={onRestoreDeleted}
          onSceneChange={reloadScene}
          selectedObject={selectedObject}
          setSelectedObject={setSelectedObject}
          notify={notify}
        />
      </div>

      {project && (
        <Timeline
          frameCount={frameCount}
          fps={project.info.fps}
          frame={frame}
          setFrame={setFrame}
          playing={playing}
          setPlaying={setPlaying}
          loop={loop}
          setLoop={setLoop}
          rangeStart={rangeStart}
          rangeEnd={rangeEnd}
          setRange={setRange}
          solution={solution}
          threshold={settings?.reprojection_threshold ?? 2}
        />
      )}

      <div className="statusbar">
        {activeJob ? (
          <>
            <div className="progress">
              <div style={{ width: `${activeJob.progress * 100}%` }} />
              <span>
                {activeJob.stage} — {activeJob.message} ({Math.round(activeJob.progress * 100)}%)
              </span>
            </div>
            <span className="dim">{activeJob.elapsed.toFixed(0)}s</span>
          </>
        ) : (
          <span className="dim">
            {job?.status === "failed" ? <span className="err">Last job failed: {job.error}</span> : job?.status === "done" ? `Last ${job.kind} finished in ${job.elapsed.toFixed(1)}s` : "Ready"}
          </span>
        )}
        <span className="sb-right dim">
          {hoverInfo} {project ? `· frame ${frame}` : ""} {selection.size ? `· ${selection.size} selected` : ""}
        </span>
      </div>
      {toast && <div className={"toast " + toast.kind}>{toast.msg}</div>}
    </div>
  );
}
