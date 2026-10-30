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
- Original speech transcription, Hindi translation and Hinglish transliteration
- Video preview with timed captions
- Editable transcript, search and caption styling
- SRT, VTT and TXT export

## Hosting
This repository includes a Dockerfile and Hugging Face Spaces Docker metadata. To publish it, create a new Docker Space and import this repository. The first transcription downloads the Whisper model, so the first request can take several minutes.

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
- Hindi translation uses an online Google Translate service.
- Hinglish is basic transliteration and may need proofreading.
- Speaker diarization, batch processing, waveform editing and burned-in MP4 export are not implemented.
- The interface requests word timestamps, but exported subtitle lines currently use Whisper segment timestamps.
- This is an initial implementation; test with real media before production use.
