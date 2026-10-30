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
    """Readable Roman Hindi (Hinglish), not English translation."""
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        result = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
        # Turn scholarly ITRANS nasal markers into familiar chat-style spelling.
        result = result.replace(".N", "n").replace("M", "n")
        result = result.replace("RRi", "ri").replace("R^i", "ri")
        result = result.replace("A", "aa").replace("I", "ee").replace("U", "oo")
        result = result.replace("~N", "n").replace("~n", "n")
        result = result.replace("Ch", "chh").replace("chh", "chh")
        return result
    except Exception:
        # Safe fallback if transliteration support is unavailable.
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="hi", target="en").translate(text)
def translate_text(text, target, source_language):
    if target == "original": return text
    # Whisper can occasionally emit Urdu/Arabic script for Hindi speech.
    has_urdu_script = any(0x0600 <= ord(ch) <= 0x06ff for ch in text)
    if target == "hindi" and source_language == "hi" and not has_urdu_script:
        return text
    if target == "hinglish" and source_language == "hi" and not has_urdu_script:
        return romanize_devanagari(text)
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="auto", target="hi").translate(text)
        return romanize_devanagari(translated) if target == "hinglish" else translated
    except Exception as exc:
        raise RuntimeError("Hindi script conversion/translation needs an internet connection. Retry online.") from exc
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
                    word_text = w.word.strip()
                    if output == "hinglish" and source_lang == "hi":
                        word_text = romanize_devanagari(word_text)
                    elif output == "hindi" and source_lang == "hi" and any(0x0600 <= ord(ch) <= 0x06ff for ch in word_text):
                        word_text = translate_text(word_text, "hindi", "ur")
                    words.append({"start":round(w.start,3),"end":round(w.end,3),"text":word_text})
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
                     template: str = Form("simple"), caption_mode: str = Form("sentence"),
                     font_size: int = Form(26), position_x: float = Form(50),
                     position_y: float = Form(86)):
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
            "simple": ("&H00FFFFFF", "&H80000000", 1, 2, 1),
            "word-highlight": ("&H00FFFFFF", "&H0000DFFF", 3, 3, 1),
            "karaoke": ("&H00FFFFFF", "&H00A000FF", 3, 3, 1),
            "bold": ("&H0000FFFF", "&H80000000", 3, 3, 1),
            "rounded": ("&H00FFFFFF", "&HAA202020", 3, 3, 1),
            "bar": ("&H0000FF00", "&HAA101010", 3, 3, 1),
            "dual": ("&H00FFFFFF", "&HAA000000", 3, 3, 1),
            "gradient": ("&H00FFB0FF", "&HAA5000A0", 3, 3, 1),
            "outline": ("&H00FFFFFF", "&H00000000", 1, 4, 1),
            "sentence-box": ("&H00FFFFFF", "&HAA101010", 3, 3, 1),
            "brush": ("&H00000000", "&H00FFFFFF", 3, 2, 1),
            "pop3d": ("&H00FFFFFF", "&HAA000000", 3, 4, 2),
            "minimal": ("&H00FFFFFF", "&H00000000", 1, 1, 0),
            "keyword": ("&H00FFFFFF", "&H0000DFFF", 3, 3, 1),
            "bounce": ("&H00FFFFFF", "&HAA0077FF", 3, 3, 1),
            "emoji": ("&H00FFFFFF", "&HAA101010", 3, 3, 1),
            "glass": ("&H00FFFFFF", "&HAA00AADD", 3, 3, 1),
            "typewriter": ("&H00FFFFFF", "&H00000000", 1, 1, 0),
            "subtle": ("&H00FFFFFF", "&H88101010", 3, 2, 1),
            "blocks": ("&H00FFFFFF", "&HAA202020", 3, 2, 1),
        }
        primary, back, border, outline, shadow = presets.get(template, presets["simple"])
        size = max(16, min(int(font_size), 64))
        px = max(5, min(float(position_x), 95)) * 1920 / 100
        py = max(5, min(float(position_y), 95)) * 1080 / 100
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
            f"Style: Default,Arial,{size},{primary},&H000000FF,&H00000000,{back},1,0,0,0,100,100,0,0,{border},{outline},{shadow},5,80,80,55,1",
            "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
        ]
        for row in rows:
            start = max(0, float(row.get("start", 0)))
            end = max(start + 0.1, float(row.get("end", start + 1)))
            words = row.get("words") or []
            if caption_mode == "word" and words:
                for word in words:
                    word_text = esc(word.get("text", ""))
                    ws = max(start, float(word.get("start", start)))
                    we = min(end, max(ws + 0.06, float(word.get("end", end))))
                    if word_text.strip():
                        lines.append(f"Dialogue: 0,{ts(ws)},{ts(we)},Default,,0,0,0,,{{\\an5\\pos({px:.0f},{py:.0f})}}{word_text}")
            elif caption_mode == "karaoke" and words:
                karaoke = []
                for word in words:
                    token = esc(word.get("text", ""))
                    ws = max(start, float(word.get("start", start)))
                    we = min(end, max(ws + 0.06, float(word.get("end", end))))
                    duration = max(1, round((we - ws) * 100))
                    if token.strip():
                        karaoke.append(f"{{\\k{duration}}}{token}")
                text = " ".join(karaoke)
                if text.strip():
                    lines.append(f"Dialogue: 0,{ts(start)},{ts(end)},Default,,0,0,0,,{{\\an5\\pos({px:.0f},{py:.0f})}}{text}")
            else:
                text = esc(row.get("text", ""))
                if text.strip():
                    lines.append(f"Dialogue: 0,{ts(start)},{ts(end)},Default,,0,0,0,,{{\\an5\\pos({px:.0f},{py:.0f})}}{text}")
        ass.write_text("\n".join(lines), encoding="utf-8-sig")
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg, "-y", "-i", str(source), "-vf", "subtitles=captions.ass",
               "-c:v", "libx264", "-preset", "veryslow", "-crf", "12",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "320k",
               "-movflags", "+faststart", str(output)]
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
