# PyInstaller spec for CamTrack (one-folder build; windowed).
# Build:  pyinstaller packaging/camtrack.spec --noconfirm   (run from the repo root, after `npm run build`)
import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

datas = [
    (os.path.join(ROOT, "frontend", "dist"), "frontend_dist"),
    (os.path.join(ROOT, "backend", "camtrack", "export", "blender_template.py"), os.path.join("camtrack", "export")),
]
binaries = []
for name in ("ffprobe.exe", "ffprobe"):
    p = os.path.join(ROOT, "packaging", "bin", name)
    if os.path.isfile(p):
        binaries.append((p, "."))
hiddenimports = collect_submodules("uvicorn") + collect_submodules("camtrack") + ["python_multipart", "multipart"]
for pkg in ("pycolmap", "webview"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        print(f"warning: could not collect {pkg}: {e}")

a = Analysis(
    [os.path.join(ROOT, "packaging", "camtrack_app.py")],
    pathex=[os.path.join(ROOT, "backend")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "PyQt5", "PyQt6", "PySide2", "PySide6"],
    noarchive=False,
)
pyz = PYZ(a.pure)
icon = os.path.join(ROOT, "packaging", "camtrack.ico")
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="CamTrack", console=False,
          icon=icon if os.path.isfile(icon) else None, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="CamTrack", upx=False)
