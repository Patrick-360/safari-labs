# Copilot Instructions — Music Intelligence AI

You are helping build a cross-device web app for real-time chord recognition.

## Architecture
- Frontend: Next.js (TypeScript) using Web Audio API to capture microphone audio.
- Backend: FastAPI (Python) that receives audio chunks and returns chord + confidence + key estimate.
- Audio streaming: send 2-second audio chunks every 0.5 seconds.
- Initial inference: rule-based chord detection using chroma features + chord templates.
- Chord set: 24 major/minor triads + "N" (no chord).
- Key detection: sliding-window chroma histogram matched to major/minor key templates.
- Smoothing: median filter / majority vote over recent chord predictions to reduce flicker.

## Backend requirements
- POST /stream endpoint
- Accept WAV/PCM audio chunk, target 22050 Hz
- Use librosa: HPSS (optional), chroma_cqt extraction, normalization
- Return JSON: { chord, confidence, key, timestamp, debug? }

## Frontend requirements
- Start/Stop mic button
- Collect mic audio with Web Audio API
- Buffer and send chunks to backend on interval
- Display: current chord, key, confidence bar, last 12 chords timeline, smoothing toggle

## Coding standards
- Prefer small files, clear functions, type hints in Python, TypeScript types in frontend.
- Add docstrings and comments for audio pipeline steps.
- Include basic error handling and logging.