# Music Intelligence AI — Beta

> **Upload a song. Get a practice roadmap.**

AI-assisted chord roadmap with key, tempo, simple practice progression, loopable sections, speed control, and piano basics.

**Beta notice:** Results are AI-assisted and should be checked by ear. Works best with clear audio — piano, guitar, and simple arrangements. Not a replacement for professional sheet music or exact transcription.

---

## What the app does

| Feature | Description |
|---------|-------------|
| **Analyze File** | Upload a WAV or MP3 and get a full practice roadmap — key, tempo, simple chord progression, loopable sections, and piano basics. **This is the main product feature.** |
| **Simple Practice Progression** | Beginner-friendly chord roadmap derived from the detailed timeline. Strips color tones (maj7, 7ths) and filters short passing chords. |
| **Detailed Detected Progression** | Full chord summary including color tones and transitions — available in a collapsible section. |
| **Chord Timeline** | Beat-aligned chord-by-chord timeline used for "Now" / "Next" chord display and section looping. |
| **Practice Sections** | Auto-detected A/B sections merged into practice parts for looping. |
| **Speed Control** | Playback rate control (0.5×–1.0×) for slow-down practice. |
| **Piano Basics** | Heuristic note spellings and practice hints for each chord. |
| **Live Mic (experimental)** | Rough chord sketches from the microphone — useful for quick checks, not for full song analysis. |

**Stack:** FastAPI + librosa (backend), Next.js 15 App Router (frontend).

---

## Known limitations

- Chords are **heuristic** (chroma + templates / Krumhansl) — errors expected on dense harmony, noise, or unusual tuning.
- Works best with **clear audio** — piano, guitar, and simple arrangements.
- **Does not work well** with noisy recordings, rap, dense mixes, or heavy reverb.
- Live modes are **experimental** — rougher than file analysis.
- Sections and rhythm bar lines are heuristic (assumes 4/4).
- Not a perfect transcription — check all results by ear.

---

## Quick start (local)

### Prerequisites

- **Python** 3.11+ (3.13 tested)
- **Node.js** 18+ and npm

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Start the API (run from `backend/` so `app` resolves):

```bash
uvicorn app.main:app --reload --port 8000
```

- API: **http://localhost:8000**
- Health check: **GET** http://localhost:8000/health → `{"status": "ok"}`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

- App: **http://localhost:3000**
- By default calls **http://localhost:8000** (see `NEXT_PUBLIC_API_URL` below).

---

## Environment variables

### Frontend

| Variable | Required for | Default | Example |
|----------|-------------|---------|---------|
| `NEXT_PUBLIC_API_URL` | Deployed frontend → deployed backend | `http://localhost:8000` | `https://your-api.example.com` |

Set in `frontend/.env.local` (not committed):

```bash
NEXT_PUBLIC_API_URL=https://your-api.example.com
```

Example file: `frontend/.env.local.example`

### Backend

| Variable | Required for | Default | Example |
|----------|-------------|---------|---------|
| `CORS_ORIGINS` | Deployed backend to allow deployed frontend | `http://localhost:3000` | `https://your-app.vercel.app,http://localhost:3000` |
| `PORT` | Deployment platforms (Render, Railway) | 8000 | `8000` |

FastAPI does **not** auto-load `.env` — use `export`, a process manager, or add `python-dotenv`.

Example:

```bash
export CORS_ORIGINS="https://your-app.vercel.app,http://localhost:3000"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Example file: `backend/.env.example`

---

## Deployment

See **[docs/deployment.md](docs/deployment.md)** for full deployment instructions including Vercel (frontend) and Render/Railway (backend).

---

## Testing

### Backend unit tests

```bash
cd backend
source .venv/bin/activate
python -m unittest discover -v -s tests -p 'test_*.py'
```

Covers: `/health`, `/stream` contract, `/analyze` schema, key-scoring sanity.

### Backend smoke test (requires running server)

```bash
bash backend/scripts/smoke_analyze.sh
```

### Frontend production build

```bash
cd frontend
npm run build
```

---

## Manual review checklist

Full walkthrough (silence handling, key stability, analyze playback, loop, etc.):

**[docs/MANUAL_REVIEW_CHECKLIST.md](docs/MANUAL_REVIEW_CHECKLIST.md)**

---

## Repository layout

```
backend/     FastAPI API: /health, /stream, /analyze, /live-transcribe
frontend/    Next.js UI: Analyze File (primary) + Live mic (experimental)
docs/        Specs, deployment guide, manual review checklist
ml/          Reserved for future training/experiments
```

---

## Roadmap (future, not committed)

- Stronger meter / downbeat detection
- Verse/chorus section labels
- Export to PDF / lead sheet
- Mobile app (React Native / Expo — not in this repo)
- Docker image and CI matrix
