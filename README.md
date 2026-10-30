# Subtitle Studio — Auto Caption Generator

A dark, responsive video/audio transcription workspace powered by faster-whisper.

## Features
- Upload MP4, MOV, MKV, AVI, MP3, WAV, M4A, FLAC, AAC, OGG and WebM
- Auto language detection or manual language selection
- Original transcript, Hindi translation and basic Hinglish transliteration
- Video preview with timed caption overlay
- Editable transcript segments and search
- Caption styles, font size and text color preview
- Export SRT, VTT and TXT

## Run locally (Windows)
1. Install Python 3.10 or 3.11.
2. Install FFmpeg and ensure `ffmpeg` is available in PATH.
3. Open a terminal in this folder and run:
   ```bat
   python -m venv .venv
   .venv\\Scripts\\activate
   pip install -r requirements.txt
   uvicorn app:app --reload
   ```
4. Open http://127.0.0.1:8000

The first transcription downloads the selected Whisper model. The `small` model is a balanced default; `base` is faster and `medium` needs more RAM and time.

## Current limitations
- Hindi translation uses the optional online Google Translate service via deep-translator; network access is required.
- Hinglish is a basic Devanagari transliteration and needs proofreading.
- Speaker diarization, batch processing, waveform editing and burned-in MP4 export are not implemented yet.
- Word timestamps are requested from Whisper; the UI edits and exports segment-level captions.
- This is an initial implementation. Run and test it locally before production use.
