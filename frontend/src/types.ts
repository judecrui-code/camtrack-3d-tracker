export interface FootageInfo {
  path: string;
  filename: string;
  width: number;
  height: number;
  fps: number;
  fps_num: number;
  fps_den: number;
  frame_count: number;
  duration: number;
  codec: string;
  pixel_aspect: number;
}

export interface Settings {
  engine: "native" | "colmap";
  frame_start: number | null;
  frame_end: number | null;
  processing_width: number;
  frame_step: number;
  detection_quality: "draft" | "standard" | "high";
  max_points: number;
  min_track_length: number;
  tracking_sensitivity: number;
  solve_method: "auto" | "typical" | "flat" | "tripod";
  variable_focal: boolean;
  focal_length_px: number | null;
  focal_length_mm: number | null;
  sensor_width_mm: number;
  distortion: "none" | "k1" | "k1k2" | "manual";
  k1: number;
  k2: number;
  reprojection_threshold: number;
  min_triangulation_angle: number;
}

export interface SceneObject {
  id: string;
  type: "null" | "solid" | "anchor" | "ground_plane";
  name: string;
  position: number[];
  normal: number[];
  x_hint: number[];
  frame: number;
  size: number;
}

export interface GroundDef {
  point: number[];
  normal: number[];
  frame: number;
  x_hint: number[];
  point_ids?: number[];
  rms?: number | null;
  source: string;
  inliers?: number;
}

export interface SceneDef {
  ground: GroundDef | null;
  auto_ground: GroundDef | null;
  origin: number[] | null;
  scale: { mode: "camera_height" | "factor" | "distance"; value: number; a: number | null; b: number | null };
  objects: SceneObject[];
}

export interface ProjectSummary {
  id: string;
  name: string;
  info: FootageInfo;
  settings: Settings;
  state: string;
  created: number;
  scene: SceneDef;
  deleted_points: number;
  has_tracks: boolean;
  has_solution: boolean;
  frames_cached: boolean;
  last_track_stats?: Record<string, number>;
  jobs?: Job[];
}

export interface Job {
  id: string;
  kind: string;
  project_id: string;
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  stage: string;
  progress: number;
  message: string;
  error: string | null;
  log: string[];
  elapsed: number;
}

export interface Intrinsics {
  width: number;
  height: number;
  f: number;
  cx: number;
  cy: number;
  k1: number;
  k2: number;
  cam_focal: number[] | null;
}

export interface SolutionData {
  mode: string;
  engine: string;
  width: number;
  height: number;
  intrinsics: Intrinsics;
  frames: number[];
  frame_status: string[];
  R: number[];
  t: number[];
  points: {
    ids: number[];
    xyz: number[];
    colors: number[];
    error: number[];
    nobs: number[];
    angle: number[];
    reliable: boolean[];
    first: number[];
    last: number[];
    deleted: boolean[];
  };
  frame_error: (number | null)[];
  frame_npoints: number[];
  stats: Record<string, any>;
  warnings: string[];
  world: { M: number[][]; scale: number };
  scene: SceneDef;
  threshold: number;
  deleted_count: number;
  needs_resolve: boolean;
}

/** A plane hypothesis under the cursor, computed from reconstructed 3D points. */
export interface TargetState {
  center: [number, number, number]; // raw solve coords
  normal: [number, number, number];
  pointIdx: number[]; // indices into solution.points used for the fit
  triangle: number[]; // 3 indices forming the displayed triangle
  rms: number; // plane fit residual (raw units)
  frame: number;
  radius: number; // raw units
}

export type ViewMode = "footage" | "3d" | "split";

export interface DisplaySettings {
  pointSize: number;
  density: number; // 0..1
  showUnreliable: boolean;
  showRaw2D: boolean;
  showOverlay: boolean;
  targetSize: number; // relative
}
