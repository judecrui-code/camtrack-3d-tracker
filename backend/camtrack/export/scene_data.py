"""Builds the exported scene description (world coordinates, Blender conventions).

Coordinate conventions
----------------------
* Solver ("raw") space: OpenCV camera axes of the first solved frame
  (x right, y down, z forward); world->camera ``Xc = R X + t``.
* World space (exported): right-handed, +Z up, ground plane at Z=0 when a
  ground plane is set, origin at the chosen origin. ``Xw = s Rw (X - o)``.
* Blender camera axes: +X right, +Y up, -Z viewing direction. A camera-to-world
  rotation is therefore ``Rw @ R_c^T @ diag(1,-1,-1)``.
* Image coordinates: pixel centres at integer coordinates, origin top-left
  (OpenCV). Blender shift_x/shift_y are derived from the principal point.
* Frames: exported frame number = original footage frame number + frame_offset
  (default 1, because Blender scenes conventionally start at frame 1 and a movie
  clip's first frame is clip frame 1).
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from ..scene.scene import object_world, world_transform
from ..solver.geometry import pose_center, rot_to_quat
from ..solver.io import intrinsics_original, to_original_px

FLIP = np.diag([1.0, -1.0, -1.0])


def build_scene_data(project, frame_offset: int = 1, sensor_width_mm: float = 36.0, include_obs: bool = False) -> dict:
    sol = project.solution
    if sol is None:
        raise RuntimeError("Nothing to export: the camera has not been solved yet")
    info = project.info
    scene = project.scene
    T = world_transform(sol, scene)
    Rw, o, s, M = T["R"], T["o"], T["s"], T["M"]
    intr = intrinsics_original(sol, info.width, info.height)
    W, H = info.width, info.height
    maxdim = max(W, H)

    def lens_mm(f_px):
        return f_px * sensor_width_mm / W

    cams = []
    for i, fr in enumerate(sol.frames):
        Rc = cv2.Rodrigues(np.asarray(sol.rvecs[i], float))[0]
        c = pose_center(sol.rvecs[i], sol.tvecs[i])
        loc = s * Rw @ (c - o)
        Rbl = Rw @ Rc.T @ FLIP
        f_px = intr["cam_focal"][i] if intr["cam_focal"] else intr["f"]
        cams.append(dict(frame=int(fr), blender_frame=int(fr) + frame_offset, location=loc.tolist(),
                         rotation_matrix=Rbl.tolist(), quaternion_wxyz=rot_to_quat(Rbl).tolist(),
                         focal_px=float(f_px), focal_mm=float(lens_mm(f_px)), status=sol.frame_status[i],
                         reprojection_error_px=None if not np.isfinite(sol.frame_error[i]) else float(sol.frame_error[i] / sol.scale)))

    rel = sol.reliable_mask(project.settings.reprojection_threshold)
    pts_w = (s * (Rw @ (sol.points - o).T)).T
    points = []
    for k in range(len(sol.points)):
        p = dict(id=int(sol.point_ids[k]), location=pts_w[k].tolist(), color=[int(c) for c in sol.colors[k]],
                 error_px=float(sol.point_error[k] / sol.scale), observations=int(sol.point_nobs[k]),
                 reliable=bool(rel[k]))
        if include_obs:
            p["track_2d"] = [[int(fr), to_original_px(sol, x), to_original_px(sol, y)] for fr, x, y in sol.point_obs[k]]
        points.append(p)

    # ground plane object: centred under the point cloud, sized to its extent
    gp = s * Rw @ (T["ground_point"] - o)
    gn = Rw @ T["ground_normal"]
    if len(pts_w):
        proj = pts_w - np.outer((pts_w - gp) @ gn, gn)
        ext = float(np.percentile(np.linalg.norm(proj - gp, axis=1), 90)) * 2
    else:
        ext = 10.0
    z = gn / np.linalg.norm(gn)
    x = np.array([1.0, 0, 0]) - z[0] * z
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([0, 1.0, 0])
    gro = np.stack([x, np.cross(z, x), z], 1)
    ground = dict(location=gp.tolist(), normal=gn.tolist(), rotation_matrix=gro.tolist(),
                  size=max(ext, 1e-3), source=(scene.get("ground") or scene.get("auto_ground") or {}).get("source", "default"))

    objects = [object_world(ob, T) for ob in scene.get("objects", [])]
    fs = sol.frames[0]
    fe = sol.frames[-1]
    stats = dict(sol.stats)
    for k in ("mean_error_px", "median_error_px", "rms_error_px", "p95_error_px"):
        if k in stats:
            stats[k + "_original_res"] = stats[k] / sol.scale
    return dict(
        format="camtrack-scene", version=1, created=time.strftime("%Y-%m-%d %H:%M:%S"),
        footage=dict(filename=info.filename, relative_path=f"footage/{info.filename}", width=W, height=H,
                     fps=info.fps, fps_num=info.fps_num, fps_den=info.fps_den, frame_count=info.frame_count,
                     pixel_aspect=info.pixel_aspect),
        frame_range=dict(start=int(fs), end=int(fe), blender_start=int(fs) + frame_offset,
                         blender_end=int(fe) + frame_offset, frame_offset=frame_offset,
                         frame_step=project.settings.frame_step),
        intrinsics=dict(focal_px=intr["f"], cx_px=intr["cx"], cy_px=intr["cy"], k1=intr["k1"], k2=intr["k2"],
                        distortion_model="polynomial_radial_k1k2 (OpenCV/Blender compatible, normalised coords)",
                        sensor_width_mm=sensor_width_mm, sensor_fit="HORIZONTAL", focal_mm=lens_mm(intr["f"]),
                        shift_x=((W - 1) / 2 - intr["cx"]) / maxdim, shift_y=(intr["cy"] - (H - 1) / 2) / maxdim,
                        variable_focal=intr["cam_focal"] is not None,
                        image_convention="pixel centres at integer coords, origin top-left"),
        camera=cams, points=points, ground_plane=ground,
        origin=dict(location=[0.0, 0.0, 0.0]), objects=objects,
        world_transform=dict(raw_to_world=M.tolist(), scale=s,
                             note="world = raw_to_world @ [x_raw, y_raw, z_raw, 1]; raw = solver space"),
        solve=dict(mode=sol.mode, engine=sol.engine, stats=stats, warnings=sol.warnings),
        units="world units (metres when scale is set from a real measurement; otherwise assumed camera height)",
        scale_reference=scene.get("scale"),
    )
