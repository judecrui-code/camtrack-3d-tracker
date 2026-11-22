"""End-to-end test through the HTTP API (same calls as the UI), followed by the
Blender round-trip test on the exported package.

    python tests/e2e_api.py FOOTAGE.mp4 [--gt GT.json] [--settings '{"solve_method":"auto"}']
           [--bpy /path/to/python-with-bpy] [--server http://127.0.0.1:8000]

Steps: import -> wait for frame cache -> analyze (track+solve) -> set ground plane on
the auto-estimated ground -> create a null + solid -> delete a few worst points and
re-solve -> export Blender package / PLY / JSON / CSV -> run tests/test_blender.py.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

ARGS = sys.argv[1:]
SERVER = ARGS[ARGS.index("--server") + 1] if "--server" in ARGS else "http://127.0.0.1:8000"


def call(method, path, body=None, raw=False):
    req = urllib.request.Request(SERVER + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    return data if raw else json.loads(data)


def wait(job):
    while True:
        j = call("GET", f"/api/jobs/{job['id']}")
        if j["status"] not in ("queued", "running"):
            if j["status"] != "done":
                raise SystemExit(f"job {j['kind']} {j['status']}: {j['error']}\n" + "\n".join(j["log"][-15:]))
            return j
        time.sleep(1)


def main():
    path = os.path.abspath(ARGS[0])
    gt = ARGS[ARGS.index("--gt") + 1] if "--gt" in ARGS else None
    settings = json.loads(ARGS[ARGS.index("--settings") + 1]) if "--settings" in ARGS else {}
    bpy_python = ARGS[ARGS.index("--bpy") + 1] if "--bpy" in ARGS else None
    t0 = time.time()
    r = call("POST", "/api/projects/from_path", {"path": path})
    pid = r["project"]["id"]
    print("project", pid, r["project"]["info"]["width"], "x", r["project"]["info"]["height"],
          r["project"]["info"]["frame_count"], "frames")
    wait(r["job"])
    j = wait(call("POST", f"/api/projects/{pid}/analyze", settings))
    print(f"analyze done in {j['elapsed']:.1f}s")
    sol = call("GET", f"/api/projects/{pid}/solution")
    st = sol["stats"]
    print(f"mode={sol['mode']} points={st['n_points']} solved={st['n_solved_frames']}/{st['n_frames']} "
          f"err={st['mean_error_px']:.3f}px f={st['focal_px']:.1f}px k1={st['k1']:.4f}")
    for w in sol["warnings"]:
        print("  WARNING:", w)
    # frame read check
    f0 = call("GET", f"/api/projects/{pid}/frames/{sol['frames'][0]}", raw=True)
    assert f0[:2] == b"\xff\xd8", "frame endpoint did not return a JPEG"
    g = sol["scene"]["auto_ground"]
    if sol["mode"] != "tripod" and not g:
        print("  (no automatic ground estimate for this clip; skipping ground/object steps)")
    if sol["mode"] != "tripod" and g:
        # use the points nearest the auto-ground point as the 'target' points
        import math
        xyz = sol["points"]["xyz"]
        # seed like a user hovering on the floor: nearest points that lie on the estimated ground
        on_plane = [i for i in range(len(sol["points"]["ids"]))
                    if abs(sum((xyz[3 * i + k] - g["point"][k]) * g["normal"][k] for k in range(3))) < 0.05]
        d = sorted(on_plane, key=lambda i: math.dist(xyz[3 * i:3 * i + 3], g["point"]))[:8]
        ids = [sol["points"]["ids"][i] for i in d]
        sc = call("POST", f"/api/projects/{pid}/scene/ground",
                  {"point": g["point"], "normal": g["normal"], "frame": sol["frames"][0], "point_ids": ids})
        print(f"ground set from target: {sc['ground']['inliers']} coplanar points (rms {sc['ground']['rms']:.4f})")
        call("POST", f"/api/projects/{pid}/scene/objects",
             {"type": "null", "point": g["point"], "normal": g["normal"], "frame": sol["frames"][0], "size_raw": 0.5})
        call("POST", f"/api/projects/{pid}/scene/objects",
             {"type": "solid", "point": g["point"], "normal": g["normal"], "frame": sol["frames"][0], "size_raw": 1.0})
    # delete the 5 worst points and re-solve (scene is carried over)
    errs = sol["points"]["error"]
    worst = sorted(range(len(errs)), key=lambda i: -errs[i])[:5]
    call("POST", f"/api/projects/{pid}/points/delete", {"ids": [sol["points"]["ids"][i] for i in worst]})
    j = wait(call("POST", f"/api/projects/{pid}/solve", settings))
    sol2 = call("GET", f"/api/projects/{pid}/solution")
    print(f"re-solve done in {j['elapsed']:.1f}s: err={sol2['stats']['mean_error_px']:.3f}px, "
          f"{sol2['deleted_count']} deleted points excluded, objects kept: {len(sol2['scene']['objects'])}")
    deleted = {sol["points"]["ids"][i] for i in worst}
    assert not deleted & set(sol2["points"]["ids"]), "deleted points reappeared after re-solve"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", f"e2e_{pid}")
    os.makedirs(out, exist_ok=True)
    for kind, fn in [("ply", "points.ply"), ("json", "tracking.json"), ("points.csv", "points.csv"),
                     ("tracks.csv", "tracks_2d.csv"), ("camera.csv", "camera.csv")]:
        data = call("GET", f"/api/projects/{pid}/export/{kind}", raw=True)
        open(os.path.join(out, fn), "wb").write(data)
        print(f"export {kind}: {len(data)} bytes")
    z = call("GET", f"/api/projects/{pid}/export/blender?include_footage=true&frame_offset=1", raw=True)
    zp = os.path.join(out, "blender_package.zip")
    open(zp, "wb").write(z)
    print(f"export blender: {len(z)} bytes -> {zp}")
    ok = True
    if bpy_python:
        cmd = [bpy_python, os.path.join(os.path.dirname(__file__), "test_blender.py"), zp,
               "--save", os.path.join(out, "scene.blend")] + (["--gt", gt] if gt else [])
        res = subprocess.run(cmd, capture_output=True, text=True)
        txt = res.stdout
        rep = txt[txt.find("BLENDER_TEST_REPORT") + 20: txt.find("BLENDER_TEST_RESULT")]
        print(rep)
        print([l for l in txt.splitlines() if l.startswith("BLENDER_TEST_RESULT")][0])
        ok = res.returncode == 0
    print(f"total {time.time() - t0:.1f}s")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
