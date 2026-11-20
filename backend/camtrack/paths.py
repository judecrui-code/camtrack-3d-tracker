"""Filesystem locations that work both from source and from a PyInstaller build."""
from __future__ import annotations

import os
import shutil
import sys

FROZEN = getattr(sys, "frozen", False)
# folder holding bundled resources (PyInstaller: sys._MEIPASS) or the repo root
BUNDLE_DIR = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def user_data_dir() -> str:
    if os.environ.get("CAMTRACK_DATA"):
        return os.environ["CAMTRACK_DATA"]
    if not FROZEN:
        return os.path.join(BUNDLE_DIR, "data")
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "CamTrack")


def frontend_dist() -> str:
    if os.environ.get("CAMTRACK_FRONTEND"):
        return os.environ["CAMTRACK_FRONTEND"]
    for cand in (os.path.join(BUNDLE_DIR, "frontend_dist"), os.path.join(BUNDLE_DIR, "frontend", "dist")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(BUNDLE_DIR, "frontend", "dist")


def ffprobe_exe() -> str | None:
    """Bundled ffprobe (next to the executable / in the bundle) or one on PATH."""
    name = "ffprobe.exe" if sys.platform == "win32" else "ffprobe"
    for d in (BUNDLE_DIR, os.path.dirname(sys.executable)):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return shutil.which("ffprobe")
