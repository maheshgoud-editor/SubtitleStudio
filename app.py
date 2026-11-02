import os, time, tempfile, shutil, json, subprocess, re, requests
from collections import defaultdict, deque, OrderedDict
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
import imageio_ffmpeg
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

ROOT = Path(__file__).parent
TMP = Path(tempfile.gettempdir()) / "subtitle-studio"
TMP.mkdir(exist_ok=True)
app = FastAPI(title="Subtitle Studio")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
_models = OrderedDict()
# Cap how many Whisper models stay loaded in RAM at once. Users can pick
# different model sizes (base/small/medium/large-v3); without a cap, every
# distinct size ever requested stays cached forever - large-v3 alone is
# several GB, so this grows unbounded over time. Override with MAX_CACHED_MODELS.
MAX_CACHED_MODELS = int(os.getenv("MAX_CACHED_MODELS", "2"))

# ---------------------------------------------------------------------------
# Upload size limit. A public tool with no cap lets anyone upload huge files
# and exhaust server disk/memory. Override with the MAX_UPLOAD_MB env var.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
UPLOAD_CHUNK = 1024 * 1024

async def save_upload_limited(upload: UploadFile, dest: Path, max_bytes: int = MAX_UPLOAD_BYTES):
    """Stream an upload to disk, aborting with 413 if it exceeds max_bytes.
    Using file.file + shutil.copyfileobj (as this project used to) reads an
    unbounded amount of data with no size check at all."""
    total = 0
    with dest.open("wb") as out:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    413,
                    f"File is too large. Maximum allowed size is {max_bytes // (1024*1024)} MB.",
                )
            out.write(chunk)
    await upload.seek(0)
    return total

# ---------------------------------------------------------------------------
# Very small in-memory per-IP rate limiter for the expensive endpoints
# (transcription, MP4 rendering). This is intentionally simple (no Redis,
# no external service) - good enough to stop one visitor or script from
# hammering the server endlessly; it resets if the process restarts.
# Override with RATE_LIMIT_MAX / RATE_LIMIT_WINDOW_SECONDS env vars.
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "8"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "600"))
_request_log = defaultdict(deque)

def enforce_rate_limit(request: Request):
    # Behind a reverse proxy (Hugging Face Spaces, Docker behind nginx,
    # etc.) request.client.host is the proxy's own address, not the
    # visitor's - which would turn this into one shared limit for every
    # visitor combined. Prefer the forwarded header when present.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    log = _request_log[client_ip]
    while log and now - log[0] > RATE_LIMIT_WINDOW_SECONDS:
        log.popleft()
    if len(log) >= RATE_LIMIT_MAX:
        raise HTTPException(
            429,
            f"Too many requests. Limit is {RATE_LIMIT_MAX} per "
            f"{RATE_LIMIT_WINDOW_SECONDS // 60} minutes - please wait and try again.",
        )
    log.append(now)

@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")

def get_model(size):
    # Cache loaded models (LRU, capped at MAX_CACHED_MODELS) to avoid
    # repeated cold starts while still bounding RAM usage.
    if size in _models:
        _models.move_to_end(size)
        return _models[size]

    while len(_models) >= MAX_CACHED_MODELS:
        _models.popitem(last=False)

    cpu_threads = max(1, min(os.cpu_count() or 2, 8))
    # Prefer GPU when available - "auto" (default) tries CUDA first and
    # silently falls back to CPU if no GPU/CUDA runtime is present. The
    # previous version hardcoded device="cpu", so a GPU-enabled host would
    # never actually be used and transcription stayed slow regardless of
    # what hardware the server had. Force it with WHISPER_DEVICE=cpu/cuda.
    device_pref = os.getenv("WHISPER_DEVICE", "auto")
    if device_pref in ("auto", "cuda"):
        try:
            model = WhisperModel(size, device="cuda", compute_type="float16")
            _models[size] = model
            return model
        except Exception:
            if device_pref == "cuda":
                raise

    _models[size] = WhisperModel(size, device="cpu", compute_type="int8",
                                 cpu_threads=cpu_threads, num_workers=1)
    return _models[size]

def split_caption_rows(rows, max_words=6):
    """Split transcript segments into short, readable caption groups."""
    result = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        tokens = [
            word for word in (row.get("words") or [])
            if str(word.get("text", "")).strip()
        ]
        if tokens:
            groups, current = [], []
            for token in tokens:
                current.append(token)
                token_text = str(token.get("text", "")).strip()
                sentence_end = bool(re.search(r"[.!?।][\"'”’)]*$", token_text))
                if len(current) >= max_words or (sentence_end and len(current) >= 3):
                    groups.append(current)
                    current = []
            if current:
                groups.append(current)
            for group in groups:
                group_start = float(group[0].get("start", row.get("start", 0)))
                group_end = float(group[-1].get("end", row.get("end", group_start + 1)))
                group_text = " ".join(str(word.get("text", "")).strip() for word in group)
                result.append({
                    "id": len(result) + 1,
                    "start": round(group_start, 3),
                    "end": round(max(group_start + 0.08, group_end), 3),
                    "text": group_text,
                    "words": group,
                })
        else:
            parts = text.split()
            if not parts:
                continue
            start_time = float(row.get("start", 0))
            end_time = max(start_time + 0.08, float(row.get("end", start_time + 1)))
            duration = end_time - start_time
            for index in range(0, len(parts), max_words):
                chunk = parts[index:index + max_words]
                chunk_start = start_time + duration * index / len(parts)
                chunk_end = start_time + duration * min(index + len(chunk), len(parts)) / len(parts)
                result.append({
                    "id": len(result) + 1,
                    "start": round(chunk_start, 3),
                    "end": round(max(chunk_start + 0.08, chunk_end), 3),
                    "text": " ".join(chunk),
                    "words": [],
                })
    return result

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
        # Preserve the Hindi transcript if optional transliteration is unavailable.
        return text
def translate_text(text, target, source_language):
    if target == "original" or not text:
        return text
    has_devanagari = any(0x0900 <= ord(ch) <= 0x097f for ch in text)
    has_urdu_script = any(0x0600 <= ord(ch) <= 0x06ff for ch in text)
    if target == "hindi" and source_language == "hi" and has_devanagari:
        return text
    if target == "hinglish" and source_language == "hi" and has_devanagari:
        return romanize_devanagari(text)
    # Enforce requested output even if a speech engine returns source script.
    if target == "english" and not (has_devanagari or has_urdu_script):
        return text
    if target == "hindi" and source_language == "hi" and not has_urdu_script:
        return text
    try:
        from deep_translator import GoogleTranslator
        dest = "en" if target == "english" else "hi"
        translated = GoogleTranslator(source="auto", target=dest).translate(text)
        return romanize_devanagari(translated) if target == "hinglish" else translated
    except Exception:
        # Translation is optional: do not discard a successful speech transcript.
        return text
@app.post("/api/transcribe")
async def transcribe(request: Request, file: UploadFile = File(...), language: str = Form("auto"),
                     output: str = Form("original"), model_size: str = Form("small"),
                     punctuation: str = Form("true"), quality_mode: str = Form("balanced"),
                     vocabulary: str = Form(""), word_timestamps: str = Form("true"), engine: str = Form("local")):
    enforce_rate_limit(request)
    allowed = {".mp3",".mp4",".wav",".m4a",".mov",".mkv",".webm",".aac",".flac",".ogg",".avi"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed: raise HTTPException(400, "Unsupported media format.")
    if output not in {"original","hindi","hinglish","english"}: output = "original"
    if model_size not in {"base","small","medium","large-v3"}: model_size = "small"
    tmp_path = TMP / (next(tempfile._get_candidate_names()) + suffix)
    try:
        await save_upload_limited(file, tmp_path)
        lang = None if language == "auto" else language
        if engine == "sarvam":
            api_key = os.getenv("SARVAM_API_KEY")
            if not api_key:
                raise HTTPException(400, "Sarvam engine selected but SARVAM_API_KEY is missing on the server.")
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            probe = subprocess.run([ffmpeg, "-i", str(tmp_path)], capture_output=True, text=True, timeout=30)
            match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", probe.stderr or "")
            if not match:
                raise HTTPException(400, "Could not read media duration.")
            duration = int(match.group(1))*3600 + int(match.group(2))*60 + float(match.group(3))
            lang_code = {"hi":"hi-IN","en":"en-IN","mr":"mr-IN","ur":"ur-IN"}.get(language, "unknown")
            mode = "translate" if output == "english" else ("translit" if output == "hinglish" else ("codemix" if output == "original" else "transcribe"))
            rows = []
            for offset in range(0, max(1, int(duration)+1), 25):
                wav = TMP / (next(tempfile._get_candidate_names()) + ".wav")
                try:
                    conv = subprocess.run([ffmpeg,"-y","-ss",str(offset),"-t","25","-i",str(tmp_path),"-vn","-ac","1","-ar","16000","-c:a","pcm_s16le",str(wav)],capture_output=True,text=True,timeout=120)
                    if conv.returncode or not wav.exists() or wav.stat().st_size < 1000:
                        continue
                    with wav.open("rb") as audio:
                        data = {"model":"saaras:v4","language_code":lang_code,"mode":mode,"with_timestamps":"true"}
                        terms = [x.strip() for x in (vocabulary or "").split(",") if x.strip()][:50]
                        if terms: data["keyterms"] = json.dumps(terms)
                        resp = requests.post("https://api.sarvam.ai/speech-to-text",headers={"api-subscription-key":api_key},files={"file":(wav.name,audio,"audio/wav")},data=data,timeout=90)
                    if not resp.ok:
                        raise RuntimeError("Sarvam API "+str(resp.status_code)+": "+resp.text[:400])
                    result = resp.json()
                    stamps = result.get("timestamps") or {}
                    phrases, starts, ends = stamps.get("words") or [], stamps.get("start_time_seconds") or [], stamps.get("end_time_seconds") or []
                    if phrases and len(phrases)==len(starts)==len(ends):
                        for phrase, start, end in zip(phrases,starts,ends):
                            if str(phrase).strip():
                                rows.append({"id":len(rows)+1,"start":round(offset+float(start),3),"end":round(offset+float(end),3),"text":str(phrase).strip(),"words":[]})
                    elif result.get("transcript","").strip():
                        rows.append({"id":len(rows)+1,"start":float(offset),"end":min(duration,offset+25),"text":result["transcript"].strip(),"words":[]})
                finally:
                    wav.unlink(missing_ok=True)
            if not rows:
                raise RuntimeError("Sarvam returned no transcript.")
            detected = language if language != "auto" else "hi"
            if output in {"english", "hindi", "hinglish"}:
                for row in rows:
                    row["text"] = translate_text(row["text"], output, detected)
            rows = split_caption_rows(rows, 6)
            return {"language":detected,"language_probability":None,"segments":rows,"message":"Sarvam transcription complete","engine":"sarvam"}
        model = get_model(model_size)
        task = "translate" if output == "english" else "transcribe"
        prompt_parts = []
        if output == "english":
            # task="translate" already asks Whisper for an English
            # translation. Never feed a Devanagari-script instruction here -
            # it fights the translate task and can drag the output back
            # into Hindi script even though English was requested.
            prompt_parts.append("Translate the speech into clear, natural English.")
        elif language == "hi":
            prompt_parts.append("यह स्वाभाविक बोलचाल की हिंदी है। हिंदी शब्दों को देवनागरी में लिखें और अंग्रेज़ी शब्दों को अनावश्यक रूप से हिंदी में न बदलें।")
        elif language == "en":
            prompt_parts.append("Transcribe clearly with natural English punctuation.")
        if punctuation.lower() != "true":
            prompt_parts.append("Use minimal punctuation; preserve the spoken words.")
        vocab = (vocabulary or "").strip()
        if vocab:
            prompt_parts.append("Names and domain vocabulary (use these spellings when heard): " + vocab[:800])
        prompt = " ".join(prompt_parts) or None
        quality_mode = quality_mode if quality_mode in {"fast", "balanced", "accurate"} else "balanced"
        beam = {"fast": 1, "balanced": 3, "accurate": 5}[quality_mode]
        if model_size == "base":
            beam = min(beam, 2)
        def run_local_transcription():
            # faster-whisper's transcribe() + segment iteration is pure CPU
            # work and takes real time (seconds to minutes). Running it
            # directly inside this async endpoint would block the single
            # asyncio event loop, freezing the whole server - including
            # every other visitor's page loads and requests - for as long
            # as this one transcription runs. Doing it in a worker thread
            # (via run_in_threadpool below) keeps the server responsive for
            # everyone else while this request is processed.
            segments, info = model.transcribe(
                str(tmp_path), language=lang, task=task,
                word_timestamps=word_timestamps.lower() == "true", vad_filter=True,
                # Lower VAD threshold + shorter min-silence + more pre/post padding
                # so quiet or quick words right at a pause aren't trimmed away as
                # "silence" before Whisper ever sees them.
                vad_parameters={
                    "threshold": 0.35,
                    "min_silence_duration_ms": 300,
                    "speech_pad_ms": 400,
                },
                beam_size=beam, best_of=1, patience=1, temperature=0,
                # Default thresholds are tuned for clean studio audio and will
                # silently drop low-confidence segments (which looks like
                # "missing words" in the captions). Relax them so marginal
                # speech is still transcribed instead of discarded.
                no_speech_threshold=0.75,
                log_prob_threshold=-1.5,
                compression_ratio_threshold=2.6,
                condition_on_previous_text=(quality_mode == "accurate"), initial_prompt=prompt
            )
            local_rows = []
            for seg in segments:
                original = seg.text.strip()
                source_lang = "hi" if language == "hi" else info.language
                converted = translate_text(original, output, source_lang) if output in {"hindi","hinglish","english"} else original
                words = []
                if seg.words:
                    for w in seg.words:
                        word_text = w.word.strip()
                        if output == "hinglish" and source_lang == "hi":
                            word_text = romanize_devanagari(word_text)
                        elif output == "hindi" and source_lang == "hi" and any(0x0600 <= ord(ch) <= 0x06ff for ch in word_text):
                            word_text = translate_text(word_text, "hindi", "ur")
                        words.append({"start":round(w.start,3),"end":round(w.end,3),"text":word_text})
                local_rows.append({"id":len(local_rows)+1,"start":round(seg.start,3),"end":round(seg.end,3),
                             "text":converted,"words":words})
            return local_rows, info

        rows, info = await run_in_threadpool(run_local_transcription)
        caption_rows = split_caption_rows(rows, 6)
        if not caption_rows:
            raise RuntimeError("No speech was detected. Try another audio file or select the spoken source language manually.")
        return {"language":info.language,"language_probability":round(info.language_probability,3),
                "segments":caption_rows,"message":"Transcription complete"}
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")
    finally:
        try: tmp_path.unlink(missing_ok=True)
        except Exception: pass


@app.post("/api/export/mp4")
async def export_mp4(request: Request, file: UploadFile = File(...), captions: str = Form(...),
                     template: str = Form("simple"), caption_mode: str = Form("sentence"),
                     font_size: int = Form(26), position_x: float = Form(50),
                     position_y: float = Form(86), text_color: str = Form("#ffffff"),
                     font_name: str = Form("Arial"), preview_height: float = Form(0)):
    enforce_rate_limit(request)
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
        await save_upload_limited(file, source)
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
        # Probe source dimensions so ASS coordinates work for portrait, square and landscape.
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        probe = subprocess.run([ffmpeg, "-i", str(source)], capture_output=True, text=True, timeout=30)
        probe_text = (probe.stderr or "") + (probe.stdout or "")
        dimensions = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", probe_text)
        width, height = (int(dimensions.group(1)), int(dimensions.group(2))) if dimensions else (1920, 1080)
        width, height = max(2, width), max(2, height)
        preview_h = float(preview_height or 0)
        scale = height / preview_h if preview_h > 40 else height / 360.0
        size = max(16, min(round(int(font_size) * scale), 240))
        px = max(2, min(float(position_x), 98)) * width / 100
        py = max(2, min(float(position_y), 98)) * height / 100
        clean_color = text_color if re.fullmatch(r"#[0-9a-fA-F]{6}", text_color or "") else "#ffffff"
        rgb = clean_color.lstrip("#")
        primary = f"&H00{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}"
        safe_font = re.sub(r"[^A-Za-z0-9 _-]", "", font_name or "Arial")[:40] or "Arial"
        def ts(value):
            centis = max(0, round(float(value) * 100))
            hh, centis = divmod(centis, 360000)
            mm, centis = divmod(centis, 6000)
            ss, centis = divmod(centis, 100)
            return f"{hh}:{mm:02d}:{ss:02d}.{centis:02d}"
        def esc(value):
            return str(value).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")
        lines = [
            "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}", "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Default,{safe_font},{size},{primary},&H000000FF,&H00000000,{back},1,0,0,0,100,100,0,0,{border},{outline},{shadow},5,80,80,55,1",
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
                    # Distinct ASS treatments for still-image export parity with the live templates.
                    style_tags = {
                        "simple": r"\\bord2\\shad1",
                        "word-highlight": r"\\bord3\\3c&H00DFFF&",
                        "karaoke": r"\\bord3\\3c&H8000FF&",
                        "bold": r"\\b1\\fs" + str(size+4) + r"\\c&H00FFFF&\\bord3\\3c&H000000&",
                        "rounded": r"\\bord8\\3c&H202020&\\be2",
                        "bar": r"\\bord10\\3c&H131307&\\c&H00FFFF&",
                        "dual": r"\\bord4\\3c&H000000&\\shad2",
                        "gradient": r"\\bord4\\3c&H800050&\\c&HFFB0FF&",
                        "outline": r"\\bord4\\3c&H000000&\\shad0",
                        "sentence-box": r"\\bord8\\3c&H101010&\\be1",
                        "brush": r"\\b1\\bord5\\3c&HFFFFFF&\\c&H000000&",
                        "pop3d": r"\\b1\\bord3\\3c&H2222AA&\\shad4",
                        "minimal": r"\\bord1\\3c&H000000&\\shad0",
                        "keyword": r"\\b1\\bord3\\3c&H00DFFF&",
                        "bounce": r"\\b1\\bord4\\3c&HFF7700&\\shad2",
                        "emoji": r"\\b1\\bord6\\3c&H101010&",
                        "glass": r"\\bord5\\3c&HFFDD00&\\be3",
                        "typewriter": r"\\fnCourier New\\fsp2\\bord1\\shad0",
                        "subtle": r"\\bord5\\3c&H2A1A12&\\shad1",
                        "blocks": r"\\b1\\bord3\\3c&H202020&"
                    }
                    tag = style_tags.get(template, style_tags["simple"])
                    lines.append(f"Dialogue: 0,{ts(start)},{ts(end)},Default,,0,0,0,,{{\\an5\\pos({px:.0f},{py:.0f}){tag}}}{text}")
        ass.write_text("\n".join(lines), encoding="utf-8-sig")
        cmd = [ffmpeg, "-y", "-i", str(source), "-vf", "subtitles=captions.ass",
               "-c:v", "libx264", "-preset", "medium", "-crf", "18",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
               "-movflags", "+faststart", str(output)]
        # Video encoding can take minutes for longer clips. Run it in a
        # worker thread so it doesn't block the server's event loop (and
        # every other visitor) for the whole duration of the encode.
        proc = await run_in_threadpool(
            subprocess.run, cmd, cwd=str(job), capture_output=True, text=True, timeout=3600
        )
        if proc.returncode or not output.exists():
            raise RuntimeError((proc.stderr or "FFmpeg render failed")[-1800:])
        # Delete the whole job folder (source copy + rendered mp4 + ass file)
        # only AFTER the response has finished streaming to the client -
        # otherwise every successful export permanently leaks a full video
        # copy on disk and the server eventually runs out of space.
        return FileResponse(
            output, media_type="video/mp4", filename="SubtitleStudio-captioned.mp4",
            background=BackgroundTask(shutil.rmtree, job, ignore_errors=True),
        )
    except HTTPException:
        shutil.rmtree(job, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(job, ignore_errors=True)
        raise HTTPException(500, f"MP4 export failed: {exc}") from exc

