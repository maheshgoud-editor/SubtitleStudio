import os
import sys
import socket
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import uvicorn
import app

def resource_path(relative):
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative

os.chdir(resource_path("."))

PORT = 8000

def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0

def wait_for_server_then_open(url, timeout_seconds=60):
    # A fixed sleep(2) before opening the browser is unreliable: loading the
    # Whisper model or a slow machine can easily take longer than that,
    # showing a "can't connect" error on first launch. Poll the server
    # instead and only open the browser once it actually responds.
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                webbrowser.open(url)
                return
        except Exception:
            time.sleep(0.3)
    # Server never came up in time - open anyway so the user at least sees
    # the browser tab and can refresh once it's ready.
    webbrowser.open(url)

if __name__ == "__main__":
    if port_in_use(PORT):
        print(
            f"Port {PORT} is already in use - Subtitle Studio may already be "
            f"running. Open http://127.0.0.1:{PORT} in your browser, or close "
            "the other instance and try again."
        )
        input("Press Enter to exit...")
        sys.exit(1)

    threading.Thread(
        target=wait_for_server_then_open, args=(f"http://127.0.0.1:{PORT}",), daemon=True
    ).start()
    # Pass the imported FastAPI object directly. A string target ("app:app")
    # is invisible to PyInstaller's static import analysis and can omit app.py.
    uvicorn.run(app.app, host="127.0.0.1", port=PORT, log_level="info")
