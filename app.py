import os, tempfile, shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
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
            converted = translate_text(original, output, info.language) if output in {"hindi","hinglish"} else original
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
