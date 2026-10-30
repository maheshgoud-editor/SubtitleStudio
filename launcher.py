import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn
import app

def resource_path(relative):
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative

os.chdir(resource_path("."))

def open_browser():
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    # Pass the imported FastAPI object directly. A string target ("app:app")
    # is invisible to PyInstaller's static import analysis and can omit app.py.
    uvicorn.run(app.app, host="127.0.0.1", port=8000, log_level="info")
