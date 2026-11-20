"""PyInstaller entry point for the CamTrack desktop application."""
import sys

from camtrack.desktop import main

if __name__ == "__main__":
    sys.exit(main())
