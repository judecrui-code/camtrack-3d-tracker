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
  which removes the drift that chained frame-to-frame flow accumulates.
- **Slide detection:** if a track no longer matches its reference appearance, it is ended and a fresh
  track is detected, rather than re-anchored, which would lock the accumulated drift in. The tolerance
  adapts to each frame's noise level: about 0.4 px on clean footage, looser on faint, noisy texture.
  On a long walking clip with a plain floor, this raised the share of tracks within 1 px of truth
  during a corner turn from 36% to 72–82%.
- **Corner-strength floor:** blurred or dark frames (whip pans, a covered lens) no longer spawn
  thousands of noise "corners". The floor is deliberately low (6% of recent good-frame strength): a
  stricter one discarded the faint texture of a plain epoxy floor on a real phone clip (142 vs 324
  usable tracks in the same frames).
- **Contrast boost:** frames are contrast-equalised (CLAHE) before detection and tracking, which brings
  out the faint texture of plain painted or polished floors: about twice the usable tracks in the plain
  stretch of that clip.
- **Fast re-acquisition:** matching lost tracks to new corners uses spatial search trees. It had been
  80% of tracking time on faint texture (203 s for 125 frames, now about 22 s). Every lost track stays a
  candidate however short: skipping fragments under 5 frames made it faster still, but broke solves on
  noisy or blurred clips (a dead-zone test clip went from 0.14% to 7% path error), because stitched short
  fragments are what keep such footage solvable.
- **Gap bridging:** about three times a second, the tracker also extracts SIFT features, which can be
  matched across large viewpoint changes. Where frame-to-frame tracking breaks (a dead zone) *or is too
  weak to constrain the camera* (fewer than 150 usable tracks, or tracks bunched into a thin strip of the
  image, such as a plain floor with only a pillar along one edge), link frames within about 4 s on either
  side are matched directly to each other. Each match is checked
  against the epipolar geometry and a cycle-consistency test. These link tracks tie the footage after
  a dead zone to the 3D points before it. They are only used across detected breaks, so footage that
  tracks continuously is solved exactly as before.
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
  constrains focal (orbit, indoor, tripod), the data wins (≤ 0.6% error in tests). The prior and the
  hypotheses are measured across the image's **long edge**, so phone portrait video is handled like
  landscape. The short trials cannot separate lenses when the opening frames are mostly pan or tilt, so
  the leading distinct candidates are re-tested over a much longer stretch with the lens frozen, ranked
  by how many keyframes they register and then by error.
- **Best-next-frame registration** (as in COLMAP): the solver always adds the unregistered keyframe that
  sees the most reconstructed 3D points, so one hard frame cannot strand everything behind it. Pose
  estimation is PnP RANSAC (SQPnP) with LM refinement. It tolerates a low share of correct matches
  (15%), so frames bridged across a dead zone, which see mostly wide-baseline matches, still register.
- Multi-view DLT triangulation with angle, cheirality and reprojection gates, and periodic global
  bundle adjustment with outlier rejection.
- **Focal length is held fixed while the reconstruction grows**, then refined once over the whole clip.
  Re-estimating it on a partial reconstruction let it drift on long walking shots and broke the chain.
- **Segment recovery:** if a long stretch still cannot be chained, it is reconstructed as a separate
  segment and joined to the rest. The join uses shared 3D points or frames located in both segments
  (measured). If they share nothing, the join comes from camera-motion continuity; that is estimated,
  and the app warns about it.
- **Bundle adjustment** (`bundle.py`): Levenberg-Marquardt with analytic Jacobians, the Schur
  complement and robust Huber loss (IRLS). It refines poses, points, focal length, k1/k2, and
  per-frame focal length (zoom, with a random-walk smoothness prior). Large problems run on a
  spatially stratified point subset with capped observations per point; the remaining points are then
  refined with cameras fixed.
- Remaining frames registered, iterative final BA and outlier removal. Frames with nothing trackable
  are filled with a smooth curve (cubic position, eased rotation) between the solved frames on either
  side. They are **flagged**, and the warning lists their frame ranges.
- **Floor detection** searches only planes within 35° of horizontal (relative to the average camera
  up-direction) that the camera stays above. On long clips the walls, which usually hold far more
  points, can no longer crowd the floor out.
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
  - interpolated frames (with ranges);
  - segments joined by estimation rather than measurement;
  - large distortion;
  - tripod (no depth).

---

## Testing & validation

There is no public test footage in the repository, so `backend/tests/synth.py` renders **synthetic
footage with exact ground truth**. Scenes are textured planar geometry rendered in perspective, with
optional radial distortion, sensor noise and blur, encoded to H.264 (CRF 14). There are 9 scenarios:
forward dolly, sideways truck, orbit around an object, mostly-flat ground, outdoor with an
independently moving car, indoor room with barrel distortion, low texture, tripod pan, and zoom.
There are also two **long-clip** tests: a 20 s, 600-frame handheld walk down a street and round a
corner, with and without a **2-second dead zone** (heavy motion blur and underexposure) where nothing
can be tracked. They use large non-repeating textures. An earlier version with exactly tiled walls
made wide-baseline matching ambiguous by construction: a feature and its copy one tile away are
indistinguishable.

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

Results (1280×720, 120 frames each; the two long clips have 600; default settings, Auto mode):

| Scenario | Mode chosen | Solved frames | Mean reproj. error | Focal error | Position error (% of path) | Rotation error | Notes |
|---|---|---|---|---|---|---|---|
| forward | typical | 120/120 | 0.278 px | 1.46% | 0.017% | 0.012° | focal weakly observable in a forward dolly (warned); 11695 pts, 14+51s |
| sideways | typical | 120/120 | 0.187 px | 0.10% | 0.009% | 0.010° | 6109 pts, 17+37s |
| orbit | typical | 120/120 | 0.315 px | 0.01% | 0.017% | 0.029° | 12067 pts, 19+69s |
| flat | typical | 120/120 | 0.436 px | 3.49% | 0.331% | 0.007° | planar scene + constant orientation: focal not observable (warned; weak prior); 8482 pts, 19+139s |
| outdoor | typical | 120/120 | 0.168 px | 0.05% | 0.041% | 0.006° | independently moving car: 27 of 4734 points on it; 4734 pts, 18+63s |
| indoor | typical | 120/120 | 0.239 px | 0.54% | 0.057% | 0.033° | barrel distortion recovered: k1 -0.0782 (true -0.08); 5885 pts, 13+52s |
| lowtexture | typical | 120/120 | 0.600 px | 3.44% | 0.245% | 0.164° | many short tracks; slower; 2452 pts, 26+17s |
| tripod | tripod | 120/120 | 0.214 px | 0.13% | n/a (no translation) | 0.028° | auto-detected rotation-only; 6957 pts, 11+36s |
| zoom | typical | 120/120 | 0.257 px | 7.34% | 0.053% | 0.207° | variable focal 967-1504px (true 900-1400px); 7212 pts, 12+48s |
| longwalk | typical | 540/600 | 0.390 px | 0.16% | 0.136% | 0.517° | 20 s walk with 2 s dead zone (frames 300-359 interpolated); floor 2.82° off; 23264 pts, 81+289s |
| longwalk_nogap | typical | 600/600 | 0.384 px | 0.71% | 0.286% | 0.201° | 20 s walk round a corner, no dead zone; floor 0.67° off; 24753 pts, 62+665s |

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
| Min track length | 8 frames | Shorter tracks are not used for the solve (lowered automatically, down to 4, on sparse footage) |
| Sensitivity | 0.5 | Higher accepts more and weaker features with looser checks |
| Focal length | Estimate | Or known focal in px, or mm + sensor width |
| Variable focal | off | Per-frame focal for zoom shots |
| Lens distortion | Estimate k1 | None / k1 / k1 + k2 / manual |
| Reprojection threshold | 2 px | Outlier cutoff (processing pixels) |
| Min triangulation angle | 1° | Points with less parallax are not triangulated |

---

## Limitations (measured or known, not hidden)

- **Mostly validated on synthetic footage.** The accuracy figures above come from synthetic clips, where
  the true camera is known. They include compression, noise, blur, distortion, low texture and a moving
  object, but not rolling shutter, lighting changes, reflections, or real lens behaviour. Expect real
  footage to give higher errors than the table above. One real clip (an 18 s portrait iPhone video, mostly
  plain glossy floor with reflections, a pillar and a turn) was used to find and fix problems; with no
  true camera for it, it was judged by reprojection error and by drawing the detected floor onto the
  frames. It solves all 552 frames. Its last 67 frames form a separate segment, joined to the rest through
  21 frames located in both parts (measured, but weakly constrained), and its lens is uncertain (see
  below).
- **Low parallax makes the lens unmeasurable.** Pan-and-tilt footage with little sideways movement cannot
  pin down focal length, and the estimate trades off against lens distortion. On that clip, estimates
  ranged from about 1,390 px (a standard phone 26 mm lens, supplied) to about 1,850 px (estimated), with
  opposite-signed distortion, and both fit the footage. **Enter the real focal length when you know it**
  (*Known (mm)*: a 35mm-equivalent value such as 26 for an iPhone main camera at 1x, 52 at 2x; the sensor
  size refers to the image's long edge, 36 by default). Supplying it made the clip register 161 of 174
  keyframes on the first pass, against 75 with the lens estimated.
- **Focal length can be unobservable.** This happens with pure translation over a single plane, and
  is weak for forward dollies. CamTrack warns, and the weak prior keeps the result plausible (see the
  flat and forward rows above). Enter the real focal length when you know it.
- **Zoom shots** (variable focal) solve with a few percent of focal scale bias, because zoom and dolly
  are partly ambiguous.
- **Dead zones are interpolated, not measured.** Frames where nothing can be tracked (heavy blur, a
  covered lens, a blank wall) get a smooth curve between the solved frames either side, and the warning
  lists them. Bridging the solve across a dead zone needs the scenery before and after it to overlap
  within about 4 seconds. If the camera ends up looking at entirely new scenery, that part is solved
  as a separate segment and joined by camera-motion continuity: an estimate, flagged in the warnings.
- **Exactly repeating patterns** (identical tiles, parking bays, rows of windows) can produce wrong
  matches across a dead zone, because a feature and its identical copy cannot be told apart. Bridging
  checks the geometry, but a perfectly regular pattern can still fool it.
- **Long clips:** tested up to 600 frames (20 s at 30 fps) with a corner turn and a 2 s dead zone. A
  600-frame clip takes roughly 2 minutes to track and 4–8 minutes to solve on 4 cores.
- **Floor detection** picks the lowest well-supported horizontal surface beneath the camera. On a
  sloped street, a ramp, or a floor with almost no texture, check the plane and set it with the target
  if needed.
- **Lens model** is radial k1/k2 with the principal point fixed at the image centre. There is no
  tangential, fisheye or anamorphic model, and no rolling-shutter correction.
- **Moving objects** are rejected by epipolar consistency and reprojection error. An object moving
  along the epipolar lines, or one dominating the frame, can still disturb a solve. In the outdoor
  test, 33 of 4,653 points (0.7%) on the moving car survived; they did not measurably affect the camera.
  Delete such points and re-solve.
- **Tripod solves** have no depth. Points lie on a sphere, so the ground plane and the target are
  unavailable.
- **Low-texture footage** solves (see the table) but is the most fragile case. Its accuracy varied
  noticeably with tracker settings during tuning.
- **Speed.** A 120-frame 720p clip takes about 1–4 minutes on 4 CPU cores with the native engine,
  a 600-frame clip roughly 5–8 minutes,
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
