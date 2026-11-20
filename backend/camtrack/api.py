"""FastAPI application: REST API + static frontend."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import AnalysisSettings
from .export.formats import (blender_package, camera_csv, ply_bytes, points_csv, tracks2d_csv)
from .export.scene_data import build_scene_data
from .footage import SUPPORTED_EXTENSIONS, FrameCache
from .jobs import JobManager
from .project import ProjectStore, analyze, ingest, resolve
from .scene.scene import (default_scene, estimate_auto_ground, make_object, refine_plane_from_points,
                          world_transform, camera_axes_raw, _frame_index)
from .solver.io import intrinsics_original, to_original_px

from .paths import frontend_dist, user_data_dir  # noqa: E402

DATA_DIR = user_data_dir()
FRONTEND_DIST = frontend_dist()

app = FastAPI(title="CamTrack 3D Camera Tracker", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
store = ProjectStore(os.path.join(DATA_DIR, "projects"))
jobs = JobManager()


def _project(pid: str):
    p = store.get(pid)
    if p is None:
        raise HTTPException(404, "Project not found")
    return p


def _busy(p):
    act = [j for j in jobs.active_for(p.id) if j.kind != "ingest"]
    if act:
        raise HTTPException(409, f"A '{act[0].kind}' job is running for this project")


# ------------------------------------------------------------------ projects
@app.get("/api/health")
def health():
    try:
        import pycolmap  # noqa
        pc = getattr(pycolmap, "__version__", "?")
    except Exception as e:  # COLMAP engine unavailable; native engine still works
        pc = f"unavailable ({type(e).__name__})"
    return dict(ok=True, opencv=cv2.__version__, pycolmap=pc, data_dir=DATA_DIR)


@app.get("/api/projects")
def list_projects():
    return [p.summary() for p in sorted(store.projects.values(), key=lambda p: -p.meta.get("created", 0))]


@app.post("/api/projects")
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    os.makedirs(DATA_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=ext, dir=DATA_DIR)
    with os.fdopen(fd, "wb") as out:
        while chunk := await file.read(8 << 20):
            out.write(chunk)
    try:
        p = store.create(tmp, file.filename, move=True)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise HTTPException(400, f"Could not read video: {e}")
    job = jobs.submit("ingest", p.id, lambda rep: ingest(p, rep))
    return dict(project=p.summary(), job=job.to_dict())


@app.post("/api/projects/from_path")
def import_path(body: dict = Body(...)):
    """Import a file already on this machine (no upload copy through the browser)."""
    path = os.path.expanduser(body.get("path", ""))
    if not os.path.isfile(path):
        raise HTTPException(400, "File not found on the server machine")
    if os.path.splitext(path)[1].lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, "Unsupported file type")
    try:
        p = store.create(path, os.path.basename(path), move=False)
    except Exception as e:
        raise HTTPException(400, f"Could not read video: {e}")
    job = jobs.submit("ingest", p.id, lambda rep: ingest(p, rep))
    return dict(project=p.summary(), job=job.to_dict())


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    p = _project(pid)
    d = p.summary()
    d["jobs"] = [j.to_dict() for j in jobs.active_for(pid)]
    return d


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    p = _project(pid)
    _busy(p)
    store.delete(pid)
    return dict(ok=True)


@app.put("/api/projects/{pid}/settings")
def put_settings(pid: str, body: dict = Body(...)):
    p = _project(pid)
    s = AnalysisSettings.from_dict({**(p.meta.get("settings") or {}), **body})
    p.meta["settings"] = s.to_dict()
    p.save()
    return s.to_dict()


@app.get("/api/projects/{pid}/frames/{n}")
def frame(pid: str, n: int):
    p = _project(pid)
    info = p.info
    if n < 0 or n >= info.frame_count:
        raise HTTPException(404, "Frame out of range")
    cache = FrameCache(p.frames_dir)
    path = cache.path(n)
    if not os.path.exists(path):
        cap = cv2.VideoCapture(p.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(500, "Could not decode frame")
        # display only; frame cache builder will overwrite with the exact sequentially-decoded frame
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.get("/api/projects/{pid}/video")
def video(pid: str):
    p = _project(pid)
    return FileResponse(p.video_path, filename=p.info.filename)


# ------------------------------------------------------------------ jobs
@app.post("/api/projects/{pid}/analyze")
def start_analyze(pid: str, body: Optional[dict] = Body(None)):
    p = _project(pid)
    _busy(p)
    s = AnalysisSettings.from_dict({**(p.meta.get("settings") or {}), **(body or {})})
    job = jobs.submit("analyze", pid, lambda rep: analyze(p, s, rep, retrack=True))
    return job.to_dict()


@app.post("/api/projects/{pid}/solve")
def start_solve(pid: str, body: Optional[dict] = Body(None)):
    p = _project(pid)
    _busy(p)
    if p.tracks is None:
        raise HTTPException(400, "Run Analyze first (no tracking data)")
    s = AnalysisSettings.from_dict({**(p.meta.get("settings") or {}), **(body or {})})
    job = jobs.submit("solve", pid, lambda rep: resolve(p, s, rep))
    return job.to_dict()


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "Job not found")
    return j.to_dict()


@app.post("/api/jobs/{jid}/cancel")
def job_cancel(jid: str):
    return dict(ok=jobs.cancel(jid))


# ------------------------------------------------------------------ results
@app.get("/api/projects/{pid}/solution")
def solution(pid: str):
    p = _project(pid)
    sol = p.solution
    if sol is None:
        raise HTTPException(404, "Not solved yet")
    info = p.info
    intr = intrinsics_original(sol, info.width, info.height)
    Rs = np.array([cv2.Rodrigues(np.asarray(r, float))[0].ravel() for r in sol.rvecs])
    deleted = set(int(i) for i in p.meta.get("deleted_points", []))
    rel = sol.reliable_mask(p.settings.reprojection_threshold)
    first = [o[0][0] if o else 0 for o in sol.point_obs]
    last = [o[-1][0] if o else 0 for o in sol.point_obs]
    T = world_transform(sol, p.scene)
    sc = sol.scale
    return dict(
        mode=sol.mode, engine=sol.engine, width=info.width, height=info.height, intrinsics=intr,
        frames=sol.frames, frame_status=sol.frame_status, R=np.round(Rs, 9).ravel().tolist(),
        t=np.round(sol.tvecs, 9).ravel().tolist(),
        points=dict(ids=sol.point_ids.tolist(), xyz=np.round(sol.points, 6).ravel().tolist(),
                    colors=sol.colors.ravel().tolist(), error=np.round(sol.point_error / sc, 4).tolist(),
                    nobs=sol.point_nobs.tolist(), angle=np.round(sol.point_angle, 3).tolist(),
                    reliable=rel.tolist(), first=first, last=last,
                    deleted=[int(i) in deleted for i in sol.point_ids]),
        frame_error=[None if not np.isfinite(e) else round(float(e / sc), 4) for e in sol.frame_error],
        frame_npoints=sol.frame_npoints.tolist(),
        stats={**sol.stats, "mean_error_px": sol.stats["mean_error_px"] / sc,
               "median_error_px": sol.stats["median_error_px"] / sc, "rms_error_px": sol.stats["rms_error_px"] / sc,
               "p95_error_px": sol.stats["p95_error_px"] / sc, "focal_px": intr["f"],
               "processing_scale": sc},
        warnings=sol.warnings, world=dict(M=T["M"].tolist(), scale=T["s"]), scene=p.scene,
        threshold=p.settings.reprojection_threshold, deleted_count=len(deleted),
        needs_resolve=sorted(deleted) != sorted(int(i) for i in p.meta.get("solved_deleted", [])),
    )


@app.get("/api/projects/{pid}/tracks2d/{n}")
def tracks2d(pid: str, n: int):
    """Raw 2D tracks present at frame n (original pixels): [id, x, y, length, dynamic]."""
    p = _project(pid)
    ts = p.tracks
    if ts is None:
        return dict(frame=n, tracks=[])
    idx = getattr(p, "_frame_index", None)
    if idx is None:
        idx = {}
        for t in ts.tracks:
            L = len(t.frames)
            for fr, (x, y) in zip(t.frames, t.xy):
                idx.setdefault(fr, []).append((t.id, x, y, L, t.dynamic))
        p._frame_index = idx
    s = ts.scale
    return dict(frame=n, tracks=[[i, round((x + .5) / s - .5, 2), round((y + .5) / s - .5, 2), L, int(d)]
                                 for i, x, y, L, d in idx.get(n, [])])


@app.post("/api/projects/{pid}/points/delete")
def delete_points(pid: str, body: dict = Body(...)):
    p = _project(pid)
    ids = set(int(i) for i in body.get("ids", []))
    cur = set(int(i) for i in p.meta.get("deleted_points", []))
    p.meta["deleted_points"] = sorted(cur | ids)
    p.save()
    return dict(deleted=len(p.meta["deleted_points"]))


@app.post("/api/projects/{pid}/points/restore")
def restore_points(pid: str):
    p = _project(pid)
    p.meta["deleted_points"] = []
    p.save()
    return dict(deleted=0)


# ------------------------------------------------------------------ scene
def _need_solution(p):
    if p.solution is None:
        raise HTTPException(400, "Solve the camera first")
    return p.solution


@app.post("/api/projects/{pid}/scene/ground")
def set_ground(pid: str, body: dict = Body(...)):
    p = _project(pid)
    sol = _need_solution(p)
    if sol.mode == "tripod":
        raise HTTPException(400, "Tripod solves have no depth, so a ground plane cannot be measured")
    pt, n, rms, used = refine_plane_from_points(sol, body.get("point_ids", []), body["point"], body["normal"],
                                                grow=bool(body.get("grow", True)))
    fi = _frame_index(sol, body.get("frame"))
    right = camera_axes_raw(sol, fi)[1]
    p.scene["ground"] = dict(point=pt.tolist(), normal=n.tolist(), frame=int(sol.frames[fi]), x_hint=right.tolist(),
                             point_ids=body.get("point_ids", []), rms=rms, source="user", inliers=used)
    if body.get("set_origin", True):
        p.scene["origin"] = None   # origin follows the ground target
    p.save()
    return p.scene


@app.post("/api/projects/{pid}/scene/origin")
def set_origin(pid: str, body: dict = Body(...)):
    p = _project(pid)
    _need_solution(p)
    p.scene["origin"] = list(map(float, body["point"]))
    p.save()
    return p.scene


@app.post("/api/projects/{pid}/scene/scale")
def set_scale(pid: str, body: dict = Body(...)):
    p = _project(pid)
    mode = body.get("mode", "camera_height")
    if mode not in ("camera_height", "factor", "distance"):
        raise HTTPException(400, "bad scale mode")
    val = float(body.get("value", 1.0))
    if val <= 0:
        raise HTTPException(400, "Scale value must be positive")
    p.scene["scale"] = dict(mode=mode, value=val, a=body.get("a"), b=body.get("b"))
    p.save()
    return p.scene


@app.post("/api/projects/{pid}/scene/reset")
def reset_scene(pid: str):
    p = _project(pid)
    sol = p.solution
    p.meta["scene"] = default_scene()
    if sol is not None:
        p.meta["scene"]["auto_ground"] = estimate_auto_ground(sol)
    p.save()
    return p.scene


@app.post("/api/projects/{pid}/scene/objects")
def create_object(pid: str, body: dict = Body(...)):
    p = _project(pid)
    sol = _need_solution(p)
    pt, n = body["point"], body["normal"]
    if body.get("point_ids"):
        pt, n, _, _ = refine_plane_from_points(sol, body["point_ids"], pt, n, grow=False)
    T = world_transform(sol, p.scene)
    size_world = float(body.get("size_raw", 1.0)) * T["s"]
    try:
        ob = make_object(body.get("type", "null"), pt, n, body.get("frame"), size_world, sol, body.get("name"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    p.scene.setdefault("objects", []).append(ob)
    p.save()
    return p.scene


@app.patch("/api/projects/{pid}/scene/objects/{oid}")
def update_object(pid: str, oid: str, body: dict = Body(...)):
    p = _project(pid)
    for ob in p.scene.get("objects", []):
        if ob["id"] == oid:
            for k in ("name", "size", "position", "normal"):
                if k in body:
                    ob[k] = body[k]
            p.save()
            return p.scene
    raise HTTPException(404, "Object not found")


@app.delete("/api/projects/{pid}/scene/objects/{oid}")
def delete_object(pid: str, oid: str):
    p = _project(pid)
    p.scene["objects"] = [o for o in p.scene.get("objects", []) if o["id"] != oid]
    p.save()
    return p.scene


# ------------------------------------------------------------------ export
def _safe(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "camtrack"


@app.get("/api/projects/{pid}/export/blender")
def export_blender(pid: str, include_footage: bool = True, frame_offset: int = 1):
    p = _project(pid)
    _need_solution(p)
    path = blender_package(p, include_footage=include_footage, frame_offset=frame_offset)
    return FileResponse(path, filename=f"{_safe(p.meta['name'])}_blender.zip", media_type="application/zip")


@app.get("/api/projects/{pid}/export/{kind}")
def export_other(pid: str, kind: str, frame_offset: int = 1):
    p = _project(pid)
    _need_solution(p)
    name = _safe(p.meta["name"])
    data = build_scene_data(p, frame_offset=frame_offset, include_obs=kind in ("json", "tracks.csv"))
    if kind == "json":
        return Response(json.dumps(data, indent=1).encode(), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{name}_tracking.json"'})
    if kind == "ply":
        return Response(ply_bytes(data), media_type="application/octet-stream",
                        headers={"Content-Disposition": f'attachment; filename="{name}_points.ply"'})
    if kind == "points.csv":
        return Response(points_csv(data), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{name}_points.csv"'})
    if kind == "tracks.csv":
        return Response(tracks2d_csv(data), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{name}_tracks_2d.csv"'})
    if kind == "camera.csv":
        return Response(camera_csv(data), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{name}_camera.csv"'})
    raise HTTPException(404, "Unknown export type")


# ------------------------------------------------------------------ frontend
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/")
    def no_frontend():
        return JSONResponse({"message": "Frontend not built. Run `npm run build` in frontend/ or use the Vite dev server."})
