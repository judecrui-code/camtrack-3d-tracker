import type { Job, ProjectSummary, SceneDef, Settings, SolutionData } from "./types";

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const b = await r.json();
      if (b.detail) msg = typeof b.detail === "string" ? b.detail : JSON.stringify(b.detail);
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }
  return r.json();
}

const post = (url: string, body?: unknown) =>
  fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

export const api = {
  health: () => fetch("/api/health").then(j<any>),
  projects: () => fetch("/api/projects").then(j<ProjectSummary[]>),
  project: (id: string) => fetch(`/api/projects/${id}`).then(j<ProjectSummary>),
  deleteProject: (id: string) => fetch(`/api/projects/${id}`, { method: "DELETE" }).then(j<any>),
  upload: (file: File, onProgress: (f: number) => void) =>
    new Promise<{ project: ProjectSummary; job: Job }>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const fd = new FormData();
      fd.append("file", file);
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
      xhr.onload = () => {
        try {
          const b = JSON.parse(xhr.responseText);
          if (xhr.status >= 300) reject(new Error(b.detail || xhr.statusText));
          else resolve(b);
        } catch (e) {
          reject(e);
        }
      };
      xhr.onerror = () => reject(new Error("Upload failed"));
      xhr.open("POST", "/api/projects");
      xhr.send(fd);
    }),
  importPath: (path: string) => post("/api/projects/from_path", { path }).then(j<{ project: ProjectSummary; job: Job }>),
  saveSettings: (id: string, s: Partial<Settings>) =>
    fetch(`/api/projects/${id}/settings`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(s),
    }).then(j<Settings>),
  analyze: (id: string, s: Partial<Settings>) => post(`/api/projects/${id}/analyze`, s).then(j<Job>),
  solve: (id: string, s: Partial<Settings>) => post(`/api/projects/${id}/solve`, s).then(j<Job>),
  job: (jid: string) => fetch(`/api/jobs/${jid}`).then(j<Job>),
  cancel: (jid: string) => post(`/api/jobs/${jid}/cancel`).then(j<any>),
  solution: (id: string) => fetch(`/api/projects/${id}/solution`).then(j<SolutionData>),
  tracks2d: (id: string, n: number) =>
    fetch(`/api/projects/${id}/tracks2d/${n}`).then(j<{ frame: number; tracks: [number, number, number, number, number][] }>),
  deletePoints: (id: string, ids: number[]) => post(`/api/projects/${id}/points/delete`, { ids }).then(j<any>),
  restorePoints: (id: string) => post(`/api/projects/${id}/points/restore`).then(j<any>),
  setGround: (id: string, body: any) => post(`/api/projects/${id}/scene/ground`, body).then(j<SceneDef>),
  setOrigin: (id: string, point: number[]) => post(`/api/projects/${id}/scene/origin`, { point }).then(j<SceneDef>),
  setScale: (id: string, body: any) => post(`/api/projects/${id}/scene/scale`, body).then(j<SceneDef>),
  resetScene: (id: string) => post(`/api/projects/${id}/scene/reset`).then(j<SceneDef>),
  createObject: (id: string, body: any) => post(`/api/projects/${id}/scene/objects`, body).then(j<SceneDef>),
  updateObject: (id: string, oid: string, body: any) =>
    fetch(`/api/projects/${id}/scene/objects/${oid}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(j<SceneDef>),
  deleteObject: (id: string, oid: string) =>
    fetch(`/api/projects/${id}/scene/objects/${oid}`, { method: "DELETE" }).then(j<SceneDef>),
  frameUrl: (id: string, n: number) => `/api/projects/${id}/frames/${n}`,
  exportUrl: (id: string, kind: string, q = "") =>
    kind === "blender" ? `/api/projects/${id}/export/blender${q}` : `/api/projects/${id}/export/${kind}${q}`,
};
