# PyInstaller build spec for the local Subtitle Studio desktop launcher.
from PyInstaller.utils.hooks import collect_all

datas = [("static", "static")]
binaries = []
hiddenimports = [
    "app",
    "uvicorn.logging", "uvicorn.loops", "uvicorn.loops.auto",
    "uvicorn.protocols", "uvicorn.protocols.http",
    "uvicorn.protocols.websockets", "uvicorn.lifespan",
    "uvicorn.lifespan.on", "multipart",
]

# Collect runtime data and hidden imports for packages that rely on dynamic
# imports or ship native/runtime resources.
for package in [
    "fastapi", "starlette", "uvicorn", "anyio", "multipart",
    "faster_whisper", "ctranslate2", "av", "tokenizers",
    "huggingface_hub", "onnxruntime", "imageio_ffmpeg",
]:
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(["launcher.py"], pathex=["."], binaries=binaries, datas=datas,
             hiddenimports=hiddenimports, hookspath=[], hooksconfig={},
             runtime_hooks=[], excludes=[], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="SubtitleStudio",
          debug=False, bootloader_ignore_signals=False, strip=False,
          upx=False, console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               name="SubtitleStudio")
