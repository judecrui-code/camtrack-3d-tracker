"""COLMAP engine (pycolmap): SIFT features + sequential matching with geometric
verification + COLMAP's incremental mapper with Ceres bundle adjustment.

Frames are written losslessly (PNG) at processing resolution. The COLMAP model
(SIMPLE_RADIAL / RADIAL camera shared by all frames) is converted back into the
common ``Solution`` format via the same statistics/normalisation code as the
native engine, so everything downstream (overlay, target, export) is identical.

COLMAP image coordinates put the top-left pixel *corner* at (0,0); we convert
to OpenCV convention (pixel centres at integers) by subtracting 0.5.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading

import cv2
import numpy as np

from ..config import AnalysisSettings
from ..footage import iter_frames, processing_size, write_image
from ..tracking.tracker import Track, TrackSet
from .sfm import SequentialSfM, SolveFailed


def solve_colmap(project, ts_native, settings: AnalysisSettings, rep, exclude=None):
    import pycolmap

    info = project.info
    pw, ph, scale = processing_size(info.width, info.height, settings.processing_width)
    start = settings.frame_start if settings.frame_start is not None else 0
    end = settings.frame_end if settings.frame_end is not None else info.frame_count - 1
    work = tempfile.mkdtemp(prefix="colmap_", dir=project.dir)
    img_dir = os.path.join(work, "images")
    os.makedirs(img_dir)
    token = pycolmap.CancellationToken()
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            if rep.cancelled():
                token.cancel()
                return
            stop.wait(0.3)

    threading.Thread(target=watch, daemon=True).start()
    try:
        rep.stage("COLMAP: writing frames", 0.0, 0.1)
        frames = []
        from ..config import effective_frame_step
        step = effective_frame_step(settings, info.fps, end - start + 1)
        total = (end - start) // step + 1
        for k, (n, fr) in enumerate(iter_frames(project.video_path, start, end, step)):
            rep.check()
            if scale != 1.0:
                fr = cv2.resize(fr, (pw, ph), interpolation=cv2.INTER_AREA)
            write_image(os.path.join(img_dir, f"{n:06d}.png"), fr)
            frames.append(n)
            rep(k / total, f"Frame {n}")
        db = os.path.join(work, "database.db")

        rep.stage("COLMAP: SIFT feature extraction", 0.1, 0.35)
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "SIMPLE_RADIAL" if settings.distortion != "k1k2" else "RADIAL"
        if settings.focal_length_px:
            f = settings.focal_length_px * scale
            reader.camera_params = f"{f},{(pw) / 2},{(ph) / 2},0" + (",0" if reader.camera_model == "RADIAL" else "")
        ext = pycolmap.FeatureExtractionOptions()
        ext.use_gpu = False
        ext.sift.max_num_features = int(max(2000, settings.max_points * 2))
        pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                                  extraction_options=ext, device=pycolmap.Device.cpu, cancellation_token=token)
        rep.check()

        rep.stage("COLMAP: sequential matching + geometric verification", 0.35, 0.6)
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = 12
        pairing.quadratic_overlap = True
        mopts = pycolmap.FeatureMatchingOptions()
        mopts.use_gpu = False
        pycolmap.match_sequential(db, matching_options=mopts, pairing_options=pairing, device=pycolmap.Device.cpu,
                                  cancellation_token=token)
        rep.check()

        rep.stage("COLMAP: incremental mapping + bundle adjustment", 0.6, 0.95)
        opts = pycolmap.IncrementalPipelineOptions()
        opts.multiple_models = False
        opts.ba_refine_focal_length = settings.focal_length_px is None and settings.focal_length_mm is None
        opts.ba_refine_extra_params = settings.distortion in ("k1", "k1k2")
        opts.ba_refine_principal_point = False
        opts.num_threads = os.cpu_count() or 4
        nreg = [0]

        def on_image():
            nreg[0] += 1
            rep(min(0.99, nreg[0] / max(1, len(frames))), f"Registered {nreg[0]} images")

        out = os.path.join(work, "sparse")
        os.makedirs(out)
        recs = pycolmap.incremental_mapping(db, img_dir, out, opts, next_image_callback=on_image,
                                            cancellation_token=token)
        rep.check()
        if not recs:
            raise SolveFailed("COLMAP could not reconstruct the scene (no initial pair found). "
                              "Try the native engine or Tripod mode.")
        rec = max(recs.values(), key=lambda r: r.num_reg_images())
        rep.log(f"COLMAP: {rec.num_reg_images()}/{len(frames)} images registered, {rec.num_points3D()} points, "
                f"mean reprojection error {rec.compute_mean_reprojection_error():.3f}px")
        return _to_solution(rec, frames, pw, ph, scale, settings, rep, exclude)
    finally:
        stop.set()
        shutil.rmtree(work, ignore_errors=True)


def _to_solution(rec, frames, pw, ph, scale, settings, rep, exclude):
    cam = next(iter(rec.cameras.values()))
    params = list(cam.params)
    f, cx, cy = params[0], params[1] - 0.5, params[2] - 0.5
    k1 = params[3] if len(params) > 3 else 0.0
    k2 = params[4] if len(params) > 4 else 0.0
    if abs(cx - (pw - 1) / 2) > 1e-3 or abs(cy - (ph - 1) / 2) > 1e-3:
        rep.log(f"COLMAP principal point ({cx:.1f},{cy:.1f}) differs from image centre")
    name_to_frame = {f"{n:06d}.png": n for n in frames}
    img_frame = {}
    poses = {}
    for iid, im in rec.images.items():
        n = name_to_frame.get(im.name)
        if n is None or not im.has_pose:
            continue
        img_frame[iid] = n
        M = im.cam_from_world().matrix()
        poses[n] = (np.asarray(M)[:, :3], np.asarray(M)[:, 3])
    # every COLMAP 3D point becomes a "track"
    tracks = []
    xyz = {}
    exclude = exclude or set()
    for pid, p in rec.points3D.items():
        if pid in exclude:
            continue
        obs = []
        for el in p.track.elements:
            n = img_frame.get(el.image_id)
            if n is None:
                continue
            xy = rec.images[el.image_id].points2D[el.point2D_idx].xy
            obs.append((n, float(xy[0]) - 0.5, float(xy[1]) - 0.5))
        if len(obs) < 2:
            continue
        obs.sort()
        # keep one observation per frame
        seen, o2 = set(), []
        for o in obs:
            if o[0] not in seen:
                seen.add(o[0])
                o2.append(o)
        t = Track(id=int(pid), frames=[o[0] for o in o2], xy=[(o[1], o[2]) for o in o2],
                  color=tuple(int(c) for c in p.color))
        tracks.append(t)
        xyz[int(pid)] = np.asarray(p.xyz, float)
    ts = TrackSet(tracks, frames, pw, ph, scale, dict(engine="colmap"))
    s2 = AnalysisSettings.from_dict({**settings.to_dict(), "min_track_length": 3})
    sfm = SequentialSfM(ts, s2, rep, None)
    sfm._reset(f, k1, k2)
    fpos = {n: i for i, n in enumerate(frames)}
    for n, (R, t) in poses.items():
        i = fpos[n]
        sfm.rv[i] = cv2.Rodrigues(R)[0].ravel()
        sfm.tv[i] = t
        sfm.reg[i] = True
    for ti, t in enumerate(sfm.tracks):
        sfm.pt[ti] = xyz[t.id]
        sfm.has_pt[ti] = True
    sfm.ref_frame = int(np.where(sfm.reg)[0][0])
    # flag observations COLMAP's own filtering would reject
    sfm.filter_outliers(1.0)
    sfm.focal_sweep = []
    kfs = []  # COLMAP has no keyframe notion; all registered frames are "solved"
    sol = sfm.build_solution("typical", kfs, settings.focal_length_px is None and settings.focal_length_mm is None)
    sol.engine = "colmap"
    return sol
