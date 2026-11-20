"""Export writers: Blender package (zip), PLY point cloud, CSV, JSON."""
from __future__ import annotations

import csv
import io
import json
import os
import zipfile

from .scene_data import build_scene_data

TEMPLATE = os.path.join(os.path.dirname(__file__), "blender_template.py")


def blender_script(data: dict) -> str:
    with open(TEMPLATE, encoding="utf-8") as fh:
        tpl = fh.read()
    js = json.dumps(data, separators=(",", ":"))
    assert '"""' not in js
    return tpl.replace("__SCENE_JSON__", js)


def ply_bytes(data: dict) -> bytes:
    pts = data["points"]
    out = io.StringIO()
    out.write("ply\nformat ascii 1.0\ncomment CamTrack reconstructed points (world space, Z up)\n")
    out.write(f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n")
    out.write("property uchar red\nproperty uchar green\nproperty uchar blue\nproperty float error_px\n"
              "property int track_id\nend_header\n")
    for p in pts:
        x, y, z = p["location"]
        r, g, b = p["color"]
        out.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {p['error_px']:.4f} {p['id']}\n")
    return out.getvalue().encode()


def points_csv(data: dict) -> bytes:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["track_id", "x", "y", "z", "r", "g", "b", "reprojection_error_px", "observations", "reliable"])
    for p in data["points"]:
        w.writerow([p["id"], *[f"{v:.6f}" for v in p["location"]], *p["color"], f"{p['error_px']:.4f}",
                    p["observations"], int(p["reliable"])])
    return out.getvalue().encode()


def tracks2d_csv(data: dict) -> bytes:
    """Per-frame 2D track positions (original pixels, top-left origin, pixel centres at integers)."""
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["frame", "track_id", "x_px", "y_px", "X", "Y", "Z"])
    rows = []
    for p in data["points"]:
        for fr, x, y in p.get("track_2d", []):
            rows.append((fr, p["id"], x, y, *p["location"]))
    rows.sort()
    for r in rows:
        w.writerow([r[0], r[1], f"{r[2]:.3f}", f"{r[3]:.3f}", f"{r[4]:.6f}", f"{r[5]:.6f}", f"{r[6]:.6f}"])
    return out.getvalue().encode()


def camera_csv(data: dict) -> bytes:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["frame", "blender_frame", "x", "y", "z", "qw", "qx", "qy", "qz", "focal_mm", "focal_px", "status",
                "reprojection_error_px"])
    for c in data["camera"]:
        w.writerow([c["frame"], c["blender_frame"], *[f"{v:.6f}" for v in c["location"]],
                    *[f"{v:.8f}" for v in c["quaternion_wxyz"]], f"{c['focal_mm']:.4f}", f"{c['focal_px']:.3f}",
                    c["status"], "" if c["reprojection_error_px"] is None else f"{c['reprojection_error_px']:.4f}"])
    return out.getvalue().encode()


README = """CamTrack export - Blender package
=================================

Contents
  build_scene.py   Blender script: builds animated camera, track points, ground plane,
                   origin, user objects and footage background. Self-contained.
  scene.json       Full scene data (world space, Z up, Blender camera conventions).
  points.ply       Reconstructed point cloud (world space) with colours and errors.
  points.csv       3D points as CSV.
  tracks_2d.csv    Per-frame 2D tracks (original pixels) with their 3D positions.
  camera.csv       Per-frame camera position / quaternion / focal length.
  footage/         The original, untouched footage file (if included).

Open in Blender
  1. Unzip anywhere (keep the folder structure).
  2. Blender > Scripting > Open > build_scene.py > Run Script
     or:  blender --python build_scene.py
     or:  blender -b --python build_scene.py -- --save scene.blend

Solve summary
  Mode: {mode}   Engine: {engine}
  Mean reprojection error: {err:.3f} px (original resolution)
  Focal length: {fpx:.1f} px = {fmm:.2f} mm on a {sw:.1f} mm wide sensor
  Lens distortion: k1={k1:.5f} k2={k2:.5f} (set on the movie clip; Blender's camera itself is a
  pinhole, so the background is shown undistorted when distortion is non-zero).
  Frames: footage {fs}-{fe}  ->  Blender {bs}-{be}  (offset {off})
{warn}
"""


def blender_package(project, include_footage: bool = True, frame_offset: int = 1) -> str:
    data = build_scene_data(project, frame_offset=frame_offset, include_obs=True)
    out_path = os.path.join(project.dir, f"{project.meta['name']}_blender.zip")
    lite = dict(data)
    lite["points"] = [{k: v for k, v in p.items() if k != "track_2d"} for p in data["points"]]
    st = data["solve"]["stats"]
    intr = data["intrinsics"]
    fr = data["frame_range"]
    warn = "\n".join("  WARNING: " + w for w in data["solve"]["warnings"])
    readme = README.format(mode=data["solve"]["mode"], engine=data["solve"]["engine"],
                           err=st.get("mean_error_px_original_res", float("nan")), fpx=intr["focal_px"],
                           fmm=intr["focal_mm"], sw=intr["sensor_width_mm"], k1=intr["k1"], k2=intr["k2"],
                           fs=fr["start"], fe=fr["end"], bs=fr["blender_start"], be=fr["blender_end"],
                           off=fr["frame_offset"], warn=warn)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("build_scene.py", blender_script(lite))
        z.writestr("scene.json", json.dumps(lite, indent=1))
        z.writestr("points.ply", ply_bytes(data))
        z.writestr("points.csv", points_csv(data))
        z.writestr("tracks_2d.csv", tracks2d_csv(data))
        z.writestr("camera.csv", camera_csv(data))
        z.writestr("README.txt", readme)
        if include_footage:
            z.write(project.video_path, f"footage/{data['footage']['filename']}", compress_type=zipfile.ZIP_STORED)
    return out_path
