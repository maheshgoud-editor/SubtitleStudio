import os, tempfile, shutil, json, subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import imageio_ffmpeg
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

ROOT = Path(__file__).parent
TMP = Path(tempfile.gettempdir()) / "subtitle-studio"
TMP.mkdir(exist_ok=True)
app = FastAPI(title="Subtitle Studio")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
_models = {}

@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")

def get_model(size):
    if size not in _models:
        _models[size] = WhisperModel(size, device="cpu", compute_type="int8")
    return _models[size]

def romanize_devanagari(text):
    consonants = {'क':'k','ख':'kh','ग':'g','घ':'gh','ङ':'ng','च':'ch','छ':'chh','ज':'j','झ':'jh','ञ':'ny','ट':'t','ठ':'th','ड':'d','ढ':'dh','ण':'n','त':'t','थ':'th','द':'d','ध':'dh','न':'n','प':'p','फ':'ph','ब':'b','भ':'bh','म':'m','य':'y','र':'r','ल':'l','व':'v','श':'sh','ष':'sh','स':'s','ह':'h'}
    vowels = {'अ':'a','आ':'aa','इ':'i','ई':'ee','उ':'u','ऊ':'oo','ऋ':'ri','ए':'e','ऐ':'ai','ओ':'o','औ':'au','ा':'aa','ि':'i','ी':'ee','ु':'u','ू':'oo','ृ':'ri','े':'e','ै':'ai','ो':'o','ौ':'au','ं':'n','ँ':'n','ः':'h','्':''}
    out=[]
    for ch in text:
        if ch in consonants: out.append(consonants[ch]+'a')
        elif ch in vowels: out.append(vowels[ch])
        elif ch != '़': out.append(ch)
    s=''.join(out)
    for a,b in [('a ',' '),('a,',','),('a.','.'),('a?','?'),('a!','!')]: s=s.replace(a,b)
    return s

def translate_text(text, target, source_language):
    if target == "original": return text
    if target == "hindi" and source_language == "hi": return text
    if target == "hinglish" and source_language == "hi": return romanize_devanagari(text)
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="auto", target="hi").translate(text)
        return romanize_devanagari(translated) if target == "hinglish" else translated
    except Exception as exc:
        raise RuntimeError("Online Hindi translation failed. Check internet connection and retry.") from exc

@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("auto"),
                     output: str = Form("original"), model_size: str = Form("small")):
    allowed = {".mp3",".mp4",".wav",".m4a",".mov",".mkv",".webm",".aac",".flac",".ogg",".avi"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed: raise HTTPException(400, "Unsupported media format.")
    if output not in {"original","hindi","hinglish","english"}: output = "original"
    if model_size not in {"base","small","medium"}: model_size = "small"
    tmp_path = TMP / (next(tempfile._get_candidate_names()) + suffix)
    try:
        with tmp_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        lang = None if language == "auto" else language
        model = get_model(model_size)
        task = "translate" if output == "english" else "transcribe"
        segments, info = model.transcribe(str(tmp_path), language=lang, task=task,
                                          word_timestamps=True, vad_filter=True)
        rows=[]
        for seg in segments:
            original = seg.text.strip()
            source_lang = "hi" if language == "hi" else info.language
            converted = translate_text(original, output, source_lang) if output in {"hindi","hinglish"} else original
            words=[]
            if seg.words:
                for w in seg.words:
                    words.append({"start":round(w.start,3),"end":round(w.end,3),"text":w.word.strip()})
            rows.append({"id":len(rows)+1,"start":round(seg.start,3),"end":round(seg.end,3),
                         "text":converted,"words":words})
        return {"language":info.language,"language_probability":round(info.language_probability,3),
                "segments":rows,"message":"Transcription complete"}
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")
    finally:
        try: tmp_path.unlink(missing_ok=True)
        except Exception: pass


@app.post("/api/export/mp4")
async def export_mp4(file: UploadFile = File(...), captions: str = Form(...),
                     template: str = Form("classic"), font_size: int = Form(26)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}:
        raise HTTPException(400, "Upload a video file for MP4 export.")
    try:
        rows = json.loads(captions)
        if not isinstance(rows, list) or not rows:
            raise ValueError("No captions to export.")
    except Exception as exc:
        raise HTTPException(400, "Invalid caption data.") from exc
    job = Path(tempfile.mkdtemp(prefix="subtitle-export-"))
    source = job / ("source" + suffix)
    output = job / "SubtitleStudio-captioned.mp4"
    ass = job / "captions.ass"
    try:
        with source.open("wb") as dst:
            shutil.copyfileobj(file.file, dst)
        presets = {
            "classic": ("&H00FFFFFF", "&H80000000", 2),
            "bold": ("&H0000FFFF", "&H80000000", 3),
            "outline": ("&H00FFFFFF", "&H00000000", 2),
            "minimal": ("&H00FFFFFF", "&H00000000", 1),
            "neon": ("&H0000FFFF", "&H00000000", 3),
            "boxed": ("&H00FFFFFF", "&H80000000", 3),
        }
        primary, back, border = presets.get(template, presets["classic"])
        size = max(16, min(int(font_size), 54))
        def ts(value):
            centis = max(0, round(float(value) * 100))
            hh, centis = divmod(centis, 360000)
            mm, centis = divmod(centis, 6000)
            ss, centis = divmod(centis, 100)
            return f"{hh}:{mm:02d}:{ss:02d}.{centis:02d}"
        def esc(value):
            return str(value).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")
        lines = [
            "[Script Info]", "ScriptType: v4.00+", "PlayResX: 1920", "PlayResY: 1080", "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Default,Arial,{size},{primary},&H000000FF,&H00000000,{back},1,0,0,0,100,100,0,0,{border},2,1,2,80,80,55,1",
            "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
        ]
        for row in rows:
            text = esc(row.get("text", ""))
            if text.strip():
                start = max(0, float(row.get("start", 0)))
                end = max(start + 0.1, float(row.get("end", start + 1)))
                lines.append(f"Dialogue: 0,{ts(start)},{ts(end)},Default,,0,0,0,,{text}")
        ass.write_text("\n".join(lines), encoding="utf-8-sig")
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg, "-y", "-i", str(source), "-vf", "subtitles=captions.ass",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)]
        proc = subprocess.run(cmd, cwd=str(job), capture_output=True, text=True, timeout=3600)
        if proc.returncode or not output.exists():
            raise RuntimeError((proc.stderr or "FFmpeg render failed")[-1800:])
        return FileResponse(output, media_type="video/mp4", filename="SubtitleStudio-captioned.mp4")
    except HTTPException:
        shutil.rmtree(job, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(job, ignore_errors=True)
        raise HTTPException(500, f"MP4 export failed: {exc}") from exc
