"""Desktop launcher: runs the backend on a free local port and opens the UI in a
native window (pywebview / Edge WebView2). Falls back to the default browser.

Used as the entry point of the packaged application (CamTrack.exe).
Command line:  CamTrack.exe [--browser] [--port N]
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import socket
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _setup_logging(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    log_path = os.path.join(data_dir, "camtrack.log")
    # windowed builds have no console: route everything to a log file
    if sys.stdout is None or sys.stderr is None:
        f = open(log_path, "a", buffering=1, encoding="utf-8")
        sys.stdout = sys.stdout or f
        sys.stderr = sys.stderr or f
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return log_path


def _error_box(msg: str):
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, msg, "CamTrack", 0x10)
            return
    except Exception:
        pass
    print(msg, file=sys.stderr)


def main():
    multiprocessing.freeze_support()
    args = sys.argv[1:]
    from .paths import user_data_dir
    data_dir = user_data_dir()
    log_path = _setup_logging(data_dir)
    port = int(args[args.index("--port") + 1]) if "--port" in args else _free_port()
    url = f"http://127.0.0.1:{port}/"
    try:
        import uvicorn
        from .api import app
    except Exception:
        _error_box("CamTrack failed to start:\n\n" + traceback.format_exc()[-1500:] + f"\n\nLog: {log_path}")
        raise
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None, access_log=False)
    server = uvicorn.Server(config)
    th = threading.Thread(target=server.run, daemon=True, name="camtrack-server")
    th.start()
    for _ in range(300):  # up to 30 s
        try:
            urllib.request.urlopen(url + "api/health", timeout=1)
            break
        except Exception:
            if not th.is_alive():
                _error_box(f"CamTrack server stopped during start-up. See log:\n{log_path}")
                return 1
            time.sleep(0.1)
    logging.info("CamTrack running at %s (data: %s)", url, data_dir)

    use_browser = "--browser" in args
    if not use_browser:
        try:
            import webview  # pywebview
            try:
                webview.settings["ALLOW_DOWNLOADS"] = True  # exports go to the Downloads folder
            except Exception:
                pass
            webview.create_window("CamTrack — 3D Camera Tracker", url, width=1600, height=960,
                                  min_size=(1100, 700), maximized=True, text_select=False)
            webview.start()  # blocks until the window is closed
            server.should_exit = True
            th.join(timeout=5)
            return 0
        except Exception:
            logging.warning("Native window unavailable, falling back to browser:\n%s", traceback.format_exc())
    webbrowser.open(url)
    print(f"CamTrack is running at {url}  (close this window or press Ctrl+C to quit)")
    try:
        while th.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.should_exit = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
