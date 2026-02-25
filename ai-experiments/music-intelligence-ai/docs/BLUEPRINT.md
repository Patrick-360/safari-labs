# Music Intelligence AI — Build Spec

Goal: A cross-device web app that performs real-time chord recognition from microphone input, plus key detection and progression tracking.

MVP:
- Frontend (Next.js): mic capture, sends 2s chunks every 0.5s, displays chord/key/confidence + timeline.
- Backend (FastAPI): /stream endpoint, chroma features, chord template matching, key detection, smoothing.

Chord vocabulary: 24 major/minor triads + N
Key vocabulary: 12 major + 12 minor

Outputs:
- chord: string (e.g., C, Dm, N)
- confidence: 0..1
- key: string (e.g., C major, A minor)