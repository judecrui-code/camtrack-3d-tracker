"""Render tests/validation_results.json as the Markdown table used in README.md."""
import json
import os
import sys

NOTES = {
    "forward": "focal weakly observable in a forward dolly (warned)",
    "sideways": "",
    "orbit": "",
    "flat": "planar scene + constant orientation: focal not observable (warned; weak prior)",
    "outdoor": "independently moving car: {car} of {pts} points on it",
    "indoor": "barrel distortion recovered: k1 {k1:.4f} (true {k1t})",
    "lowtexture": "many short tracks; slower",
    "tripod": "auto-detected rotation-only",
    "zoom": "variable focal {fr0:.0f}-{fr1:.0f}px (true {ft0:.0f}-{ft1:.0f}px)",
}


def main(path):
    rows = json.load(open(path))
    out = []
    for d in rows:
        if "error" in d:
            out.append(f"| {d['scenario']} | — | failed | — | — | — | — | {d['error']} |")
            continue
        note = NOTES.get(d["scenario"], "").format(
            car=d.get("points_on_moving_object", "?"), pts=d["points"], k1=d["k1_solved"], k1t=d["k1_true"],
            fr0=(d.get("focal_solved_range_px") or [0, 0])[0], fr1=(d.get("focal_solved_range_px") or [0, 0])[1],
            ft0=(d.get("focal_true_range_px") or [0, 0])[0], ft1=(d.get("focal_true_range_px") or [0, 0])[1])
        pos = d.get("pos_error_mean_pct_of_path")
        out.append(
            f"| {d['scenario']} | {d['mode']} | {d['solved_frames']}/{d['frames']} | {d['reproj_mean_px']:.3f} px | "
            f"{d['focal_error_pct']:.2f}% | {'n/a (no translation)' if pos is None else f'{pos:.3f}%'} | "
            f"{d['rot_error_deg_mean']:.3f}° | {note}{'; ' if note else ''}{d['points']} pts, "
            f"{d['track_time_s']:.0f}+{d['solve_time_s']:.0f}s |")
    return "\n".join(out)


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "validation_results.json")
    print(main(p))
