# Music Intelligence AI

Real-time chord recognition + key detection web app.

## Repo layout
- backend/: FastAPI service for audio inference
- frontend/: Next.js web app for microphone capture + UI
- ml/: training & experiments (future)
- docs/: specifications

## MVP flow
1) Frontend captures mic audio and sends 2-second chunks every 0.5 seconds to backend.
2) Backend extracts chroma features and returns chord + confidence + key estimate.
3) Frontend displays chord live with smoothing.

## Next steps
Build backend MVP first, then connect frontend streaming.