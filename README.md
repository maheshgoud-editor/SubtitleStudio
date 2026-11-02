---
title: Subtitle Studio
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Subtitle Studio — Auto Caption Generator

A responsive video/audio transcription workspace powered by faster-whisper.

## Features
- Upload MP4, MOV, MKV, AVI, MP3, WAV, M4A, FLAC, AAC, OGG and WebM
- Automatic language detection or manual selection
- Original speech transcription, English translation, Hindi translation and Hinglish transliteration
- Video preview with timed captions, 20 caption styles with a live preview
- Editable transcript, search and caption styling
- SRT, VTT and TXT export
- Burned-in styled MP4 export (captions rendered permanently into the video)

## Hosting
This repository includes a Dockerfile and Hugging Face Spaces Docker metadata. To publish it, create a new Docker Space and import this repository. The first transcription downloads the Whisper model, so the first request can take several minutes.

### Configuration (environment variables)
- `MAX_UPLOAD_MB` (default `500`) - maximum upload size in MB for both transcription and MP4 export.
- `RATE_LIMIT_MAX` / `RATE_LIMIT_WINDOW_SECONDS` (default `8` requests / `600` seconds) - simple per-IP rate limit on the transcription and export endpoints. This is in-memory only and resets on restart; for serious production traffic, put a real rate limiter/WAF in front instead.
- `MAX_CACHED_MODELS` (default `2`) - how many Whisper model sizes stay loaded in RAM at once (LRU eviction beyond this).
- `WHISPER_DEVICE` (default `cpu`) - CPU works everywhere with no setup. Set to `cuda` only on a host you've confirmed has a working CUDA + cuBLAS install (GPU is not auto-detected, since a broken/missing CUDA runtime fails lazily on the first transcription instead of at startup).
- `SARVAM_API_KEY` - only needed if you select the optional, paid Sarvam engine from the UI. Not required for the default local Whisper engine.

## Local Windows run
1. Install Python 3.10 or 3.11 and FFmpeg.
2. In this folder run:
   ```bat
   python -m venv .venv
   .venv\\Scripts\\activate
   pip install -r requirements.txt
   uvicorn app:app --reload
   ```
3. Open http://127.0.0.1:8000

## Limitations
- Hindi/English translation falls back to an online Google Translate service when Whisper's own output needs cleanup; this needs internet access on the server.
- Hinglish is basic transliteration and may need proofreading.
- Speaker diarization, batch processing and waveform editing are not implemented.
- The optional Sarvam engine (Saaras v4) is paid (free credits on signup, then pay-as-you-go) and requires your own `SARVAM_API_KEY`; the default local Whisper engine is free and does not need it.
- The per-IP rate limit is in-memory only (resets on restart) and is meant to stop casual abuse, not to replace proper infrastructure for a production/public launch.
- This is an evolving project; test with real media before relying on it for production use.
