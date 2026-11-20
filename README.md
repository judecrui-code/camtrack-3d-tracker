# CamTrack — 3D Camera Tracker

A local matchmoving application in the spirit of After Effects' 3D Camera Tracker.

Import footage, track features, solve the 3D camera (pose, focal length, lens distortion), and
inspect the reconstructed point cloud over the footage and in a 3D viewport. Place the ground plane
and origin with an interactive target, create nulls/solids/anchors, then **export the solved scene to
Blender** with an animated camera, track points, ground plane and footage background.

Everything runs locally: a Python/FastAPI backend does the computer vision, and a React + TypeScript +
Three.js frontend provides the UI.

---

## Download (Windows app)

Get the latest **`CamTrack-Setup-…-win64.exe`** (installer) or **`…-portable.zip`** from the
repository's **Releases** page. Windows 10/11 64-bit is required, and no Python or Node install is
needed.
- The app opens in its own window, using Microsoft Edge WebView2, which is preinstalled on Windows
  10/11.
- Exports are saved to your Downloads folder.
- Projects are stored in `%LOCALAPPDATA%\CamTrack`.
- The build is not code-signed, so SmartScreen may warn on first launch: choose *More info → Run
  anyway*.
- The *CamTrack (open in browser)* shortcut runs the same app in your default browser instead.

Releases are built by `.github/workflows/windows-release.yml` on a Windows runner:
1. Builds the frontend, then the app with PyInstaller (`packaging/camtrack.spec`), bundling `ffprobe`.
2. Smoke-tests the built `.exe`: renders a synthetic clip, analyses it, sets the ground, re-solves
   and exports through the HTTP API.
3. Packages an Inno Setup installer (`packaging/installer.iss`) and a portable zip.

Push a tag such as `v1.0.0` to publish a release; the *Run workflow* button builds without releasing.
To build locally on Windows:
`cd frontend && npm ci && npm run build && cd .. && pip install -r backend/requirements.txt pyinstaller pywebview && pyinstaller packaging/camtrack.spec`
(the output is in `dist\CamTrack\CamTrack.exe`).

## Install & run from source

Requirements: **Python 3.10+** (tested on 3.13), **Node.js 18+** (tested on 22), and `ffmpeg`/`ffprobe`
on the PATH (used for metadata probing; OpenCV handles decoding).

```bash
./run.sh
```

The script creates `.venv`, installs `backend/requirements.txt`, builds the frontend once, and serves
UI + API at **http://127.0.0.1:8000**.

Manual setup (any OS):

```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt   # Windows: .venv\Scripts\pip
cd frontend && npm install && npm run build && cd ..
cd backend && ../.venv/bin/uvicorn camtrack.api:app --host 127.0.0.1 --port 8000
```

Frontend development with hot reload: run the backend as above, then `cd frontend && npm run dev` and
open http://localhost:5173 (it proxies `/api` to port 8000).

Project data (the original footage, the display frame cache, tracks, solves and exports) is stored in
`data/projects/<id>/`. Set `CAMTRACK_DATA` to use another location.

---

## Workflow

1. **Import footage.** Use *Import* (browser upload) or paste a local path. MP4, MOV, MKV, AVI, WebM,
   MXF, MTS and similar are supported. The original file is stored untouched. Resolution, frame rate
   (exact rational), duration, frame count, codec and pixel aspect are shown. A full-resolution JPEG
   (q95) display cache is built in the background; it is used for display only.
2. **Set the frame range and settings.** Drag the blue work-area handles on the timeline, or use
   `[ In` / `Out ]` (keys `B` / `N`). The defaults are sensible; see *Settings* below.
3. **Analyze & Solve.** This runs feature tracking, then the camera solve, in a background job with
   progress, a log and *Cancel*. Results appear as:
   - Solved track points drawn over the footage. Colour encodes depth. Red × marks unreliable points;
     cyan marks the selection.
   - Solve quality: mean, median and 95th-percentile reprojection error, focal length (px, mm, FOV),
     distortion, solved/interpolated frames, and warnings.
   - A per-frame reprojection-error graph in the timeline. Yellow ticks are keyframes; red bars are
     unsolved (interpolated) frames.
4. **Clean up (optional).** Click, Shift/Ctrl-click or Shift-drag to select points, or use *Select
   unreliable* / *Select high error*. Press **Delete**, then **Re-solve**. Re-solve reuses the existing
   tracks and only reruns the solver; your ground plane and objects carry over.
5. **Target interaction.** Hover the footage: the app finds neighbouring solved points around the
   cursor and fits a local plane to their 3D positions. It then intersects the viewing ray with that
   plane and draws the triangle of defining points (translucent red), the red bullseye lying on the
   plane in correct perspective, its in-plane axes (red/green) and its normal (blue).
   - **Click** locks the target and selects its points.
   - **Drag the centre** of a locked target to slide it along its plane.
   - **Right-click** for: Set Ground Plane and Origin, Set Ground Plane, Set Origin, Create Null,
     Create Solid, Create 3D Anchor, Create Ground Plane Object, Select Target Points, Delete
     Selected Points.
   - Adjust *Target size* in the Display panel.
   - *Set Ground Plane* refines the local plane by region growing over all coplanar reconstructed
     points, so the floor orientation comes from the whole visible surface, not three points.
   - Changing the ground plane, origin or scale never re-solves the camera.
6. **Scale.** By default the world is scaled so the camera is 1.6 units above the ground; this is an
   assumption and is labelled as such. To use real units, select two points and enter their real
   distance, or set the camera height.
7. **3D Scene.** Shows the point cloud, camera trajectory (keyframes highlighted), current camera
   frustum, ground grid, origin axes, selected points, created objects and the live target. Orbit,
   pan and zoom with the mouse. *Look through solved camera* views from the solved camera; *Split*
   shows footage and 3D side by side.
8. **Export to Blender** (top-right button, or the Export panel). Also available: PLY point cloud,
   JSON tracking data, CSV 3D points, CSV 2D tracks, CSV camera.

Keyboard: `Space` play/pause · `←/→` step (Shift: ×10) · `Home/End` · `B/N` set in/out ·
`Esc` release target · `Delete` delete selected points · `F` fit view · mouse wheel zoom ·
middle-drag or Alt-drag pan.

---

## Blender export

`Export to Blender` downloads a zip containing:

| File | Contents |
|---|---|
| `build_scene.py` | Self-contained Blender script (data embedded; prefers `scene.json` next to it if present) |
| `scene.json` | Camera per frame, intrinsics, distortion, points, ground plane, origin, objects, footage reference, fps, resolution, frame range, solve statistics |
| `points.ply` | Point cloud (world space) with colours, errors and track ids |
| `points.csv`, `tracks_2d.csv`, `camera.csv` | 3D points; per-frame 2D tracks in original pixels; per-frame camera position, quaternion and focal |
| `footage/<file>` | The original footage, copied byte for byte (optional) |
| `README.txt` | Solve summary and warnings |

Open it in Blender via **Scripting → Open `build_scene.py` → Run Script**, or
`blender --python build_scene.py`, or headless with
`blender -b --python build_scene.py -- --save scene.blend`. The script:

1. Sets resolution, pixel aspect, exact fps (`fps = num`, `fps_base = den`), and the frame range.
2. Creates the **Solved Camera**, keyframed every solved frame (location + quaternion, linear
   interpolation), with focal length in mm on a 36 mm horizontal sensor and the principal-point
   shift. For zoom shots the lens is keyframed too.
3. Creates every reconstructed point as an **Empty** (plain axes; spheres for unreliable points),
   plus a coloured point-cloud mesh.
4. Creates the **Ground Plane** (marked as a shadow catcher), a **World Origin** empty, and all user
   objects (nulls, solids, 3D anchors, ground-plane objects).
5. Loads the original footage as a **movie clip** and sets it as the camera background. The clip's
   tracking camera receives focal length, principal point and k1/k2 distortion; the background is
   shown undistorted when distortion is non-zero.
6. Organises everything in collections: `CamTrack` → `Camera`, `Track Points`, `Scene`, `Objects`.

### Conventions

- **World:** right-handed, +Z up, the same as Blender. The ground plane is Z = 0 once set.
- **Camera:** the solver uses OpenCV axes (x right, y down, z forward). Blender cameras look down −Z
  with +Y up, so camera-to-world = `Rw · Rcᵀ · diag(1, −1, −1)`.
- **Image coordinates:** origin top-left, pixel centres at integers. Converted to Blender's
  bottom-left, pixel-edge convention where needed (principal point, verification). Processing at
  reduced resolution maps back with `x_orig = (x_proc + 0.5)/s − 0.5`.
- **Frames:** footage frame *n* (0-based, decoder order) becomes Blender frame *n + offset*. The
  offset defaults to 1, and the movie clip's `frame_start` equals the offset, so the plate lines up.
- **Lens:** pinhole + radial `d = 1 + k1 r² + k2 r⁴` in normalised coordinates. This is identical in
  OpenCV and in Blender's "Polynomial" model, so the coefficients transfer directly.

---

## How the tracking engine works

**Feature tracking** (`backend/camtrack/tracking/tracker.py`, OpenCV):
- Shi-Tomasi corners with sub-pixel refinement, kept spread out by a minimum-distance mask.
- Pyramidal Lucas-Kanade optical flow with a forward-backward consistency check.
- **Anchored refinement:** each track is re-aligned to its reference appearance in an anchor frame,
  which removes the drift that chained frame-to-frame flow accumulates. On synthetic footage this cut
  119-frame drift from 1.4 px to 0.4 px.
- **ORB descriptor verification** catches drift and feature swaps.
- **RANSAC fundamental matrix** (USAC/MAGSAC) checks against a frame 5 frames back. Tracks that are
  repeatedly epipolar outliers are classified as **moving objects**; they are removed, and their image
  region is masked against re-detection for a while.
- **Re-acquisition after occlusion:** new detections are matched (ORB Hamming distance plus a position
  predicted from neighbouring motion) to recently lost tracks, so a feature that was briefly hidden
  keeps its id.

**Camera solve, native engine** (`backend/camtrack/solver/`):
- Keyframe selection based on feature displacement and overlap.
- **Motion-model analysis:** essential matrix vs. homography vs. *pure rotation*. The rotation test
  fits a rotation to bearing vectors over wide keyframe pairs, across a fine focal-length search. It
  drives *Auto* mode: typical, flat, or tripod.
- Initial pair from essential-matrix RANSAC, scored by inliers × triangulation angle. Flat scenes also
  evaluate homography-decomposition hypotheses.
- **Focal length estimation:** a sweep of focal-length hypotheses, each seeding a short incremental
  reconstruction with focal-length bundle adjustment; the lowest-error one wins. The spread of the
  sweep detects poorly constrained focal length, which triggers a warning. A weak MAP prior (60°
  horizontal FOV) only matters when the footage carries no focal information; when the data
  constrains focal (orbit, indoor, tripod), the data wins (≤ 0.6% error in tests).
- Incremental registration: PnP RANSAC (SQPnP) + LM refinement; multi-view DLT triangulation with
  angle, cheirality and reprojection gates; periodic global bundle adjustment with outlier rejection.
- **Bundle adjustment** (`bundle.py`): Levenberg-Marquardt with analytic Jacobians, the Schur
  complement and robust Huber loss (IRLS). It refines poses, points, focal length, k1/k2, and
  per-frame focal length (zoom, with a random-walk smoothness prior). Large problems run on a
  spatially stratified point subset with capped observations per point; the remaining points are then
  refined with cameras fixed.
- Remaining frames registered, iterative final BA and outlier removal. Unsolvable frames are
  interpolated (SLERP) and **flagged**.
- **Tripod mode:** robust rotation estimation (Kabsch + RANSAC on bearing vectors), points as
  directions, rotation + focal + distortion BA.

**COLMAP engine** (`solver/colmap_engine.py`, pycolmap): lossless frames, SIFT features, sequential
matching with geometric verification, then COLMAP's incremental mapper with Ceres bundle adjustment.
Its output is converted to the same solution format, statistics and warnings. It is slower, and
useful as an independent cross-check.

**Honesty rules built in:**
- Reprojection errors are always reported in original-resolution pixels.
- The quality rating (Excellent / Good / Fair / Poor) follows the measured error.
- Warnings are raised for:
  - high error;
  - low parallax;
  - weakly constrained focal length;
  - few points;
  - weakly constrained frames;
  - interpolated frames;
  - large distortion;
  - tripod (no depth).

---

## Testing & validation

There is no public test footage in the repository, so `backend/tests/synth.py` renders **synthetic
footage with exact ground truth**. Scenes are textured planar geometry rendered in perspective, with
optional radial distortion, sensor noise and blur, encoded to H.264 (CRF 14). There are 9 scenarios:
forward dolly, sideways truck, orbit around an object, mostly-flat ground, outdoor with an
independently moving car, indoor room with barrel distortion, low texture, tripod pan, and zoom.

```bash
cd backend
../.venv/bin/python -m tests.synth ../test_footage                 # render the 9 test clips (~5 min)
../.venv/bin/python -m tests.validate ../test_footage --json tests/validation_results.json
```

`tests.validate` tracks and solves each clip, then measures:
- reprojection error;
- focal error;
- recovered k1;
- camera position error after similarity alignment (as % of path length);
- camera rotation error;
- for the outdoor clip, how many solved points lie on the moving car.

Results (1280×720, 120 frames each, default settings, Auto mode):

| Scenario | Mode chosen | Solved frames | Mean reproj. error | Focal error | Position error (% of path) | Rotation error | Notes |
|---|---|---|---|---|---|---|---|
| forward | typical | 120/120 | 0.438 px | 2.35% | 0.030% | 0.018° | focal weakly observable in a forward dolly (warned); 7764 pts, 19+68s |
| sideways | typical | 120/120 | 0.259 px | 0.84% | 0.014% | 0.029° | 4433 pts, 21+53s |
| orbit | typical | 120/120 | 0.496 px | 0.01% | 0.013% | 0.018° | 7500 pts, 62+41s |
| flat | typical | 120/120 | 0.386 px | 6.96% | 0.645% | 0.010° | planar scene + constant orientation: focal not observable (warned; weak prior); 5678 pts, 14+191s |
| outdoor | typical | 120/120 | 0.244 px | 0.19% | 0.022% | 0.006° | independently moving car: 4 of 4092 points on it; 4092 pts, 20+49s |
| indoor | typical | 120/120 | 0.264 px | 0.64% | 0.042% | 0.032° | barrel distortion recovered: k1 -0.0781 (true -0.08); 5018 pts, 13+50s |
| lowtexture | typical | 120/120 | 0.783 px | 1.41% | 0.107% | 0.021° | many short tracks; slower; 9986 pts, 190+47s |
| tripod | tripod | 120/120 | 0.223 px | 0.01% | n/a (no translation) | 0.003° | auto-detected rotation-only; 6105 pts, 14+41s |
| zoom | typical | 120/120 | 0.332 px | 7.20% | 0.123% | 0.201° | variable focal 967-1501px (true 900-1400px); 5303 pts, 11+44s |

**Blender round-trip** (`tests/test_blender.py`, run with Blender's Python or the `bpy` module):
1. Unzips an export and runs `build_scene.py` inside Blender 5.0.
2. Checks resolution, fps, frame range, collections, empties, ground plane, objects, the footage
   movie clip and its frame alignment.
3. Projects every track-point Empty through Blender's own animated camera
   (`world_to_camera_view`) and compares against the original 2D tracks.

Measured: Blender reprojection matches the solver within 0.01 px (e.g. 0.467 px vs 0.462 px over
32,873 observations on the orbit clip). A deliberate 5-frame shift raises the error to ~10–40 px,
which proves the frame numbering. Against ground truth, the Blender camera path is within 0.01–0.04%
of path length and 0.001–0.02° of rotation, and the focal length is within 0.6%. A user-set ground
plane is within 0.08–0.5° of the true floor.

`tests/e2e_api.py` drives the HTTP API exactly as the UI does:
- import, analyze, set ground, create objects;
- delete points and re-solve;
- export all formats, then run the Blender test.

It has passed on the orbit, indoor (lens distortion), tripod, and sideways clips; the sideways run used
a frame sub-range with every-2nd-frame sampling.

```bash
# with the server running:
../.venv/bin/python tests/e2e_api.py ../test_footage/indoor.mp4 --gt ../test_footage/indoor.gt.json --bpy /path/to/python-with-bpy
```

To get `bpy` without a Blender install: `pip install bpy` into a Python 3.11 venv.

---

## Settings

| Setting | Default | Meaning |
|---|---|---|
| Engine | Native | Native video SfM, or COLMAP (pycolmap) |
| Solve method | Auto | Auto / Typical / Mostly flat / Tripod |
| Frame range | whole clip | Original frame numbers are preserved |
| Processing resolution | long edge 1280 px | Never upscaled; results are reported in original pixels |
| Frame sampling | every frame | Track every N-th frame; Blender interpolates between keys |
| Detection quality | Standard | Draft / Standard / High (corner threshold, spacing, LK window, pyramid) |
| Max points | 3000 | Maximum simultaneously tracked features |
| Min track length | 8 frames | Shorter tracks are not used for the solve |
| Sensitivity | 0.5 | Higher accepts more and weaker features with looser checks |
| Focal length | Estimate | Or known focal in px, or mm + sensor width |
| Variable focal | off | Per-frame focal for zoom shots |
| Lens distortion | Estimate k1 | None / k1 / k1 + k2 / manual |
| Reprojection threshold | 2 px | Outlier cutoff (processing pixels) |
| Min triangulation angle | 1° | Points with less parallax are not triangulated |

---

## Limitations (measured or known, not hidden)

- **Validated on synthetic footage only.** This environment had no access to real sample footage.
  The synthetic clips include compression, noise, blur, distortion, low texture and a moving object,
  but not motion blur, rolling shutter, lighting changes, reflections, or real lens behaviour.
  Expect real footage to give higher errors than the table above.
- **Focal length can be unobservable.** This happens with pure translation over a single plane, and
  is weak for forward dollies. CamTrack warns, and the weak prior keeps the result plausible (7% /
  2.4% error in the flat / forward tests). Enter the real focal length when you know it.
- **Zoom shots** (variable focal) solved with a 7% focal scale bias and 0.12% path error in the
  synthetic test, because zoom and dolly are partly ambiguous.
- **Lens model** is radial k1/k2 with the principal point fixed at the image centre. There is no
  tangential, fisheye or anamorphic model, and no rolling-shutter correction.
- **Moving objects** are rejected by epipolar consistency and reprojection error. An object moving
  along the epipolar lines, or one dominating the frame, can still disturb a solve. In the outdoor
  test, 4 of 4,092 points (0.1%) on the moving car survived; they did not measurably affect the camera.
  Delete such points and re-solve.
- **Tripod solves** have no depth. Points lie on a sphere, so the ground plane and the target are
  unavailable.
- **Low-texture footage** solves, at ~0.8 px error in the test, but produces many short tracks and is
  slower.
- **Speed.** A 120-frame 720p clip takes about 1–4 minutes on 4 CPU cores with the native engine,
  and around 6 minutes with COLMAP. The solver favours accuracy over speed; there is no GPU path.
- **Blender** cameras are pinhole. Distortion is stored on the movie clip and the background is shown
  undistorted; to composite onto the original distorted plate, add a Movie Distortion node in the
  compositor (not created automatically). Thousands of Empties make big scenes heavy; set
  `CREATE_POINT_EMPTIES = False` or `ONLY_RELIABLE_POINTS = True` at the top of the script.
- **Frames before the display cache finishes** are fetched by seeking, which can be off by a frame
  for some long-GOP codecs. Analysis always decodes sequentially, and the cache replaces those frames.
- **Single-user local tool** with no authentication. Keep it bound to 127.0.0.1.

## Project layout

```
backend/camtrack/
  api.py            FastAPI routes (projects, frames, jobs, solution, scene, exports) + static UI
  jobs.py           background jobs: progress, logs, cancellation
  project.py        persistence + pipeline glue (ingest / analyze / re-solve, scene carry-over)
  footage.py        probing, frame-accurate decoding, display cache
  config.py         settings and defaults
  tracking/         feature tracking (KLT + anchors + ORB + RANSAC + re-acquisition)
  solver/           geometry, bundle adjustment, sequential SfM, COLMAP engine, (de)serialisation
  scene/            ground plane, origin, scale, objects, world transform
  export/           scene data builder, Blender script template, PLY/CSV/JSON/zip writers
backend/tests/      synthetic footage generator, validation, Blender round-trip, API end-to-end
frontend/src/       React + TypeScript + Three.js UI (Viewer, SceneView, Timeline, panels, geometry)
```
