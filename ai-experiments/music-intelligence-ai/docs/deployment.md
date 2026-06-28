# Deployment Guide — Music Intelligence AI Beta

This guide covers deploying the FastAPI backend and Next.js frontend for beta testing.

Recommended stack:
- **Frontend:** Vercel (zero-config Next.js deployment)
- **Backend:** Render or Railway (Python, persistent processes)

---

## Architecture

```
Browser → Next.js frontend (Vercel)
              ↓ fetch POST /analyze
         FastAPI backend (Render / Railway)
              ↓
         librosa audio analysis (no external ML services)
```

All chord analysis is done locally in the backend process. No external ML APIs are called.

---

## Environment variables

### Frontend (Vercel)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | Yes (production) | `http://localhost:8000` | Backend API base URL, **no trailing slash** |

Set this in your Vercel project dashboard under **Settings → Environment Variables**.

Example value: `https://your-backend.onrender.com`

### Backend (Render / Railway)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CORS_ORIGINS` | Yes (production) | `http://localhost:3000` | Comma-separated allowed browser origins |
| `PORT` | Render/Railway inject this | 8000 | Port uvicorn listens on |

Example `CORS_ORIGINS` value:
```
https://your-app.vercel.app,http://localhost:3000
```

---

## Local development

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Health check: `curl http://localhost:8000/health` → `{"status": "ok"}`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Create `frontend/.env.local` for local config:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Deploy: Frontend on Vercel

1. Connect your GitHub repo to [vercel.com](https://vercel.com).
2. Set the **Root Directory** to `frontend`.
3. Vercel auto-detects Next.js. Build command: `npm run build`. Output: `.next`.
4. Add environment variable:
   - `NEXT_PUBLIC_API_URL` = `https://your-backend.onrender.com` (or Railway URL)
5. Deploy. Vercel gives you a URL like `https://your-app.vercel.app`.

**Vercel settings summary:**

| Setting | Value |
|---------|-------|
| Framework | Next.js (auto-detected) |
| Root Directory | `frontend` |
| Build Command | `npm run build` |
| Output Directory | `.next` |
| Install Command | `npm install` |
| Node.js Version | 18+ |

---

## Deploy: Backend on Render

1. Create a new **Web Service** on [render.com](https://render.com).
2. Connect your GitHub repo.
3. Set:
   - **Root Directory:** `backend`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables:
   - `CORS_ORIGINS` = `https://your-app.vercel.app,http://localhost:3000`
5. Deploy. Render gives you a URL like `https://your-backend.onrender.com`.

**Render settings summary:**

| Setting | Value |
|---------|-------|
| Root Directory | `backend` |
| Runtime | Python 3.11+ |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/health` |
| Plan | Starter (1 CPU, 512MB RAM is a minimum; 1GB+ recommended for librosa) |

> **Note on Render free tier:** The free tier spins down after inactivity. First requests after spin-down take 30–60 seconds. For beta testing, use the Starter paid plan or accept cold starts.

---

## Deploy: Backend on Railway

1. Create a new project on [railway.app](https://railway.app).
2. Deploy from GitHub, set root to `backend`.
3. Railway detects Python automatically.
4. Add environment variables:
   - `CORS_ORIGINS` = `https://your-app.vercel.app,http://localhost:3000`
5. Set the start command in `Procfile` or Railway settings:
   ```
   uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```
6. Deploy. Railway gives you a URL like `https://your-backend.up.railway.app`.

---

## CORS wiring

The backend reads `CORS_ORIGINS` at startup:

```python
# backend/app/cors_config.py
def cors_allow_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
    return [o.strip() for o in raw.split(",") if o.strip()]
```

If `CORS_ORIGINS` is not set, only `http://localhost:3000` is allowed. Your deployed Vercel domain will be blocked. **Always set `CORS_ORIGINS` in production.**

---

## Beta memory limits and audio trimming

The backend is tuned for Render Starter (512MB RAM) instances.  Two constants in
`backend/app/core/config.py` control this:

| Constant | Default | Purpose |
|----------|---------|---------|
| `BETA_MAX_UPLOAD_SIZE_MB` | 30 | Files larger than this are rejected with a friendly 400 error |
| `BETA_MAX_ANALYSIS_DURATION_SEC` | 90 | Only the first N seconds of a long file are decoded and analyzed |

**Why trimming helps memory:**
librosa decodes audio into a float32 numpy array.  A 5-minute mono track at 22050 Hz is
~26 MB of RAM before HPSS and chroma processing.  Capping at 90 s keeps the array
under 8 MB, which fits comfortably in 512 MB alongside librosa's processing buffers.

**Frontend behavior:**
When `analysis_window.was_trimmed === true` in the API response, the UI shows a note:
"Beta note: This song was longer than 90 seconds, so we analyzed the first 90 seconds."

**How to increase limits:**
Upgrade the Render plan to 2 GB RAM, then raise both constants and redeploy.
Recommended: `BETA_MAX_ANALYSIS_DURATION_SEC = 300` (5 min) on 2 GB, or 600 on 4 GB.

**Upload format guidance for beta testers:**
- Recommended: MP3 under 30 MB (roughly 30 minutes at 128 kbps)
- WAV works but is larger per minute — keep under 30 MB (≈ 3 min of 16-bit 44.1 kHz stereo)
- Full album uploads will be rejected; single-song uploads are fine

---

## Build verification

Before deploying, run locally:

```bash
# Backend unit tests
cd backend
source .venv/bin/activate
python -m unittest discover -v -s tests -p 'test_*.py'

# Frontend production build
cd frontend
npm run build
```

Both should complete without errors.

---

## Beta testing checklist

- [ ] `/health` returns `{"status": "ok"}` on deployed backend
- [ ] Frontend loads at Vercel URL
- [ ] Upload a WAV or MP3 file
- [ ] Analysis completes (key, tempo, chords appear)
- [ ] Simple Practice Progression appears
- [ ] Detailed Detected Progression is available in collapsible section
- [ ] Current chord / Next chord updates during playback
- [ ] Practice sections work and loop
- [ ] Speed control works (0.5×, 0.75×, 1×)
- [ ] Piano basics appear for a chord
- [ ] Error message appears for a bad file (empty or corrupt)
- [ ] CORS is working (no blocked requests in browser console)

---

## Troubleshooting

**Frontend shows no data after analysis:**
- Check browser console for CORS errors
- Verify `NEXT_PUBLIC_API_URL` is set and has no trailing slash
- Check that `CORS_ORIGINS` includes your Vercel domain

**Backend 500 or import errors on deploy:**
- Check Python version (3.11+ required)
- Verify `pip install -r requirements.txt` completed successfully
- librosa requires `soundfile` and `numba` — both are in requirements.txt

**Analysis takes very long:**
- Files over 5 minutes may take 20–40 seconds on a small Render instance
- Consider upgrading to a higher-RAM plan (1GB+ recommended for librosa)

**CORS blocked:**
- Add your exact Vercel domain to `CORS_ORIGINS` (with `https://`, no trailing slash)
- Redeploy backend after updating env vars (Render/Railway require a restart)
