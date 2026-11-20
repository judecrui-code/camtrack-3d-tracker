"""Blender export round-trip test (run with Blender's Python / the `bpy` module).

    python tests/test_blender.py EXPORT.zip [--gt GT.json] [--save out.blend]

1. Unzips the export package and runs its build_scene.py inside Blender.
2. Checks scene settings: resolution, fps, frame range, collections, camera
   animation, empties, ground plane, footage background.
3. Re-projects every track-point Empty through Blender's animated camera
   (bpy_extras.object_utils.world_to_camera_view) at sampled frames and compares
   with the original 2D feature tracks (tracks_2d.csv, original pixels). This
   validates coordinate systems, camera rotation conventions, image origin,
   frame numbering, focal length / sensor conversion and principal point.
4. Optionally compares the Blender camera trajectory to ground truth.
"""
import csv
import json
import math
import os
import sys
import tempfile
import zipfile

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


def main():
    args = sys.argv[1:]
    if "--" in args:
        args = args[args.index("--") + 1:]
    zpath = args[0]
    gt_path = args[args.index("--gt") + 1] if "--gt" in args else None
    save = args[args.index("--save") + 1] if "--save" in args else None
    work = tempfile.mkdtemp(prefix="camtrack_blender_")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(work)
    script = os.path.join(work, "build_scene.py")
    ns = {"__file__": script, "__name__": "camtrack_build"}
    exec(compile(open(script).read(), script, "exec"), ns)
    data = ns["build"]()
    scene = bpy.context.scene
    report = {}
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        report[name] = ("PASS" if cond else "FAIL") + (f" ({detail})" if detail else "")
        ok &= bool(cond)

    ft, fr, intr = data["footage"], data["frame_range"], data["intrinsics"]
    check("resolution", scene.render.resolution_x == ft["width"] and scene.render.resolution_y == ft["height"],
          f"{scene.render.resolution_x}x{scene.render.resolution_y}")
    check("fps", abs(scene.render.fps / scene.render.fps_base - ft["fps"]) < 1e-6,
          f"{scene.render.fps}/{scene.render.fps_base}")
    check("frame_range", scene.frame_start == fr["blender_start"] and scene.frame_end == fr["blender_end"],
          f"{scene.frame_start}-{scene.frame_end}")
    cam = scene.camera
    check("camera", cam is not None and cam.type == "CAMERA")
    check("collections", all(bpy.data.collections.get(n) for n in
                             ["CamTrack", "CamTrack Camera", "CamTrack Track Points", "CamTrack Scene", "CamTrack Objects"]))
    empties = [o for o in bpy.data.collections["CamTrack Track Points"].objects if o.type == "EMPTY"]
    check("point_empties", len(empties) == len(data["points"]), f"{len(empties)} empties / {len(data['points'])} points")
    check("ground_plane", bpy.data.objects.get("Ground Plane") is not None)
    check("world_origin", bpy.data.objects.get("World Origin") is not None)
    check("objects", len(bpy.data.collections["CamTrack Objects"].objects) == len(data["objects"]),
          f"{len(data['objects'])} user objects")
    bgs = cam.data.background_images
    check("footage_background", len(bgs) == 1 and bgs[0].clip is not None,
          bgs[0].clip.filepath if len(bgs) and bgs[0].clip else "missing")
    if len(bgs) and bgs[0].clip:
        clip = bgs[0].clip
        check("clip_length", clip.frame_duration == ft["frame_count"], f"{clip.frame_duration} frames")
        check("clip_alignment", clip.frame_start == fr["frame_offset"], f"clip.frame_start={clip.frame_start}")

    # ---- reprojection through Blender's camera ----
    obs = {}
    with open(os.path.join(work, "tracks_2d.csv")) as fh:
        for row in csv.DictReader(fh):
            obs.setdefault(int(row["frame"]), []).append((int(row["track_id"]), float(row["x_px"]), float(row["y_px"])))
    by_id = {int(o.name.split()[-1]): o for o in empties}
    W, H = ft["width"], ft["height"]
    f, cx, cy, k1, k2 = intr["focal_px"], intr["cx_px"], intr["cy_px"], intr["k1"], intr["k2"]
    frames = sorted(obs)
    sample = frames[:: max(1, len(frames) // 12)] + [frames[-1]]
    errs = []
    per_frame = {}
    for n in sample:
        scene.frame_set(n + fr["frame_offset"])
        fe = []
        for tid, u_obs, v_obs in obs[n]:
            e = by_id.get(tid)
            if e is None:
                continue
            co = world_to_camera_view(scene, cam, e.matrix_world.translation)
            if co.z <= 0:
                fe.append(1e9)
                continue
            # Blender NDC (0,0)=bottom-left -> OpenCV pixel coords (pixel centres at integers, top-left origin)
            u = co.x * W - 0.5
            v = (1.0 - co.y) * H - 0.5
            # apply the solved lens distortion (Blender camera is pinhole) to compare with raw tracks
            x, y = (u - cx) / f, (v - cy) / f
            r2 = x * x + y * y
            d = 1 + k1 * r2 + k2 * r2 * r2
            u, v = f * d * x + cx, f * d * y + cy
            fe.append(math.hypot(u - u_obs, v - v_obs))
        fe.sort()
        per_frame[n] = fe[len(fe) // 2] if fe else float("nan")
        errs += fe
    errs.sort()
    med = errs[len(errs) // 2]
    mean = sum(e for e in errs if e < 1e8) / max(1, len([e for e in errs if e < 1e8]))
    solve_err = data["solve"]["stats"].get("mean_error_px_original_res")
    report["reprojection_in_blender"] = dict(observations=len(errs), mean_px=round(mean, 4), median_px=round(med, 4),
                                            p95_px=round(errs[int(0.95 * len(errs))], 4),
                                            solve_reported_mean_px=round(solve_err, 4),
                                            per_frame_median={k: round(v, 3) for k, v in per_frame.items()})
    check("reprojection_matches_solve", abs(mean - solve_err) < 0.05 + 0.1 * solve_err,
          f"blender mean {mean:.3f}px vs solve {solve_err:.3f}px")

    # frame-offset sanity: projecting with a deliberately wrong frame must be much worse
    n = sample[len(sample) // 2]
    scene.frame_set(n + fr["frame_offset"] + 5)
    wrong = []
    for tid, u_obs, v_obs in obs[n][:300]:
        e = by_id.get(tid)
        if e is None:
            continue
        co = world_to_camera_view(scene, cam, e.matrix_world.translation)
        wrong.append(math.hypot(co.x * W - 0.5 - u_obs, (1 - co.y) * H - 0.5 - v_obs))
    wrong.sort()
    if wrong:
        report["frame_shift_control_median_px"] = round(wrong[len(wrong) // 2], 3)

    # ---- ground truth trajectory (synthetic footage) ----
    if gt_path:
        import numpy as np
        gt = json.load(open(gt_path))
        C_b, C_g, R_b, R_g = [], [], [], []
        flip = np.diag([1.0, -1.0, -1.0])
        for c in data["camera"]:
            scene.frame_set(c["blender_frame"])
            mw = cam.matrix_world
            C_b.append(np.array(mw.translation))
            R_b.append(np.array(mw.to_3x3()) @ flip)  # cam->world, OpenCV axes
            P = gt["poses"][c["frame"]]
            R = np.array(P["R"])
            C_g.append(-R.T @ np.array(P["t"]))
            R_g.append(R.T)
        C_b, C_g = np.array(C_b), np.array(C_g)
        acc = sum(rg @ rb.T for rg, rb in zip(R_g, R_b))
        U, _, Vt = np.linalg.svd(acc)
        Ra = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        rc = (Ra @ C_b.T).T
        mc, mg = rc.mean(0), C_g.mean(0)
        s = float(np.sum((rc - mc) * (C_g - mg)) / max(np.sum((rc - mc) ** 2), 1e-12))
        al = s * rc + (mg - s * mc)
        path = float(np.sum(np.linalg.norm(np.diff(C_g, axis=0), axis=1)))
        pe = np.linalg.norm(al - C_g, axis=1)
        re = [math.degrees(math.acos(max(-1, min(1, (np.trace(Ra @ rb @ rg.T) - 1) / 2)))) for rb, rg in zip(R_b, R_g)]
        lens_px = cam.data.lens / cam.data.sensor_width * W
        gp = bpy.data.objects["Ground Plane"].matrix_world
        n_gt = Ra @ np.array(gp.to_3x3().col[2])
        n_gt /= np.linalg.norm(n_gt)
        p_gt = s * Ra @ np.array(gp.translation) + (mg - s * mc)
        report["ground_plane_vs_truth"] = dict(
            source=data["ground_plane"]["source"],
            normal_error_deg=round(math.degrees(math.acos(min(1.0, abs(float(n_gt[2]))))), 4),
            height_error_pct_of_cam_height=round(100 * abs(float(p_gt[2])) / max(1e-9, float(np.mean(C_g[:, 2]))), 4))
        report["ground_truth"] = dict(
            position_error_pct_of_path=round(100 * float(pe.mean()) / max(path, 1e-9), 4) if path > 1e-6 else None,
            rotation_error_deg_mean=round(float(np.mean(re)), 4), focal_px_blender=round(lens_px, 2), focal_px_true=gt["f"])
    if save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(save))
        report["saved"] = os.path.abspath(save)
    print("BLENDER_TEST_REPORT " + json.dumps(report, indent=1))
    print("BLENDER_TEST_RESULT", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
