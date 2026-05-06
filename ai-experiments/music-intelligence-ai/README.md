# Music Intelligence AI

A practical **MVP** for musicians: **live** chord recognition from the microphone and **full-track analysis** from an audio file—key, tempo, beat-aligned chords, coarse sections, and practice-oriented UI (timeline, progression, section looping).

No auth, payments, or training pipeline in this repo path; focus is inference + a simple Next.js client.

## What the app does today

| Mode | Description |
|------|-------------|
| **Live microphone** | Browser captures short WAV chunks (~0.5 s), `POST /stream` returns **chord**, **key** (tonally smoothed), confidences, and debug scores. Good for jamming and quick feedback. |
| **Analyze file** | `POST /analyze` on a WAV/MP3 (etc.): **duration**, **tempo**, **global key**, **chord timeline**, **beats**, **sections** (heuristic), **rhythm hints** (assumed 4/4 bar grouping from beats). Frontend: big “now” chord, next chord, section navigation, beat-aware timeline, loop-by-section, full progression. |

**Stack:** FastAPI + librosa (backend), Next.js 15 App Router (frontend).

## Repository layout

```
backend/     # FastAPI API: /health, /stream, /analyze
frontend/    # Next.js UI: live + analyze-file modes
docs/        # Specs + manual review checklist
ml/          # Reserved for future training/experiments
```

## Prerequisites

- **Python** 3.11+ recommended (3.13 works with current deps in many setups)
- **Node.js** 18+ and npm

## Quick start (local review / demo)

### 1. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the API (from `backend/` so `app` resolves):

```bash
uvicorn app.main:app --reload --port 8000
```

- API: **http://localhost:8000**
- Health: **GET** http://localhost:8000/health

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

- App: **http://localhost:3000**
- By default the UI calls **http://localhost:8000** (see `NEXT_PUBLIC_API_URL` below).

### 3. Wire frontend ↔ backend (when not on localhost defaults)

| Variable | Where | Purpose |
|----------|--------|---------|
| `NEXT_PUBLIC_API_URL` | `frontend/.env.local` | API base URL, **no trailing slash** (e.g. `https://api.example.com`). Example file: `frontend/.env.local.example`. |
| `CORS_ORIGINS` | Shell or `backend/.env` + your loader | Comma-separated **browser origins** allowed to call the API. Default if unset: `http://localhost:3000`. Example: `backend/.env.example`. |

**Production-style example:**

```bash
export CORS_ORIGINS="https://your-app.vercel.app,http://localhost:3000"
# start uvicorn …
```

```bash
# frontend/.env.local
NEXT_PUBLIC_API_URL=https://your-api.example.com
```

FastAPI does not load `.env` automatically; use `export`, a process manager, or add `python-dotenv` later if you want file-based loading.

## Testing (automated)

From **`backend/`** with venv activated:

```bash
python -m unittest discover -v -s tests -p 'test_*.py'
```

Covers: **`/health`**, **`/stream`** JSON contract (silent WAV), **`/analyze`** shape (including sections + rhythm), and a small key-scoring sanity check.

Optional script (HTTP against a **running** server):

```bash
bash backend/scripts/smoke_analyze.sh
```

Frontend production build:

```bash
cd frontend && npm run build
```

## Manual review checklist

For a full walkthrough (live silence, key stability, analyze playback, loop, etc.), use:

**[docs/MANUAL_REVIEW_CHECKLIST.md](docs/MANUAL_REVIEW_CHECKLIST.md)**

## Roadmap (future phases)

Ideas only—not committed scope:

- **Audio:** melody/bass line or separation (larger DSP / ML scope)
- **Analysis:** stronger meter / downbeat, richer section labels (verse/chorus) if still lightweight
- **Practice:** metronome click, speed control, export lead sheets
- **Ops:** Docker image, CI matrix, pinned prod configs

## Limitations (MVP)

- Chords/keys are **heuristic** (chroma + templates / Krumhansl); errors on dense harmony, tuning, or noise are expected.
- **Live key** is smoothed for stability; it is not a real-time Roman-numeral analysis engine.
- **Analyze** sections and **rhythm** bar lines are **heuristic** (e.g. assumed 4/4 from beat indices).

## License / ownership

Add your license here if applicable.
