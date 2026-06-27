# Deployment Checklist

Deploy backend first, then frontend. Backend URL is required before the frontend env var can be set.

---

## Step 1 — Deploy backend (Render)

Go to [render.com](https://render.com) → New → Web Service → connect your GitHub repo.

| Field | Value |
|-------|-------|
| **Root Directory** | `backend` |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| **Health Check Path** | `/health` |

> Render injects `PORT` automatically. The Procfile in `backend/` also works if Render picks it up.

### Environment variables (Render dashboard → Environment)

Set these **before** first deploy:

| Key | Value |
|-----|-------|
| `CORS_ORIGINS` | `http://localhost:3000` ← placeholder; update after Vercel deploy |

Leave `PORT` alone — Render sets it.

### After deploy

- Copy the service URL: `https://your-backend.onrender.com`
- Test: `curl https://your-backend.onrender.com/health` → `{"status": "ok"}`

---

## Step 2 — Deploy frontend (Vercel)

Go to [vercel.com](https://vercel.com) → New Project → import your GitHub repo.

| Field | Value |
|-------|-------|
| **Root Directory** | `frontend` |
| **Framework Preset** | Next.js (auto-detected) |
| **Build Command** | `npm run build` *(leave as default)* |
| **Output Directory** | `.next` *(leave as default)* |
| **Install Command** | `npm install` *(leave as default)* |
| **Node.js Version** | 18.x or 20.x |

### Environment variables (Vercel dashboard → Settings → Environment Variables)

| Key | Value |
|-----|-------|
| `NEXT_PUBLIC_API_URL` | `https://your-backend.onrender.com` ← use the URL from Step 1, no trailing slash |

### After deploy

- Copy the Vercel URL: `https://your-app.vercel.app`
- Open it — you should see "Upload a song. Get a practice roadmap."

---

## Step 3 — Update CORS on backend

Go back to Render → your backend service → Environment → edit `CORS_ORIGINS`:

```
https://your-app.vercel.app,http://localhost:3000
```

Click **Save** — Render will redeploy automatically.

---

## Step 4 — Smoke test the live deployment

Run these in order:

```bash
# 1. Backend health
curl https://your-backend.onrender.com/health
# Expected: {"status":"ok"}

# 2. CORS preflight
curl -I -X OPTIONS \
  -H "Origin: https://your-app.vercel.app" \
  -H "Access-Control-Request-Method: POST" \
  https://your-backend.onrender.com/analyze
# Expected: access-control-allow-origin: https://your-app.vercel.app

# 3. Analyze a file
curl -X POST https://your-backend.onrender.com/analyze \
  -F "file=@/path/to/any.mp3" \
  | python3 -m json.tool | grep -E '"key"|"tempo"|"simple_practice'
# Expected: key, tempo, and simple_practice_progression in the JSON
```

In the browser at your Vercel URL:

- [ ] Page loads, shows "Upload a song. Get a practice roadmap."
- [ ] "Analyze File" tab is active by default
- [ ] Upload an MP3 or WAV → click Analyze → results appear
- [ ] "Simple practice progression" block shows chords
- [ ] "Detailed detected progression" drawer is collapsed but opens
- [ ] No CORS errors in browser console (F12 → Console)

---

## Railway alternative (if not using Render)

| Field | Value |
|-------|-------|
| **Root Directory** | `backend` |
| **Start Command** | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| **`CORS_ORIGINS`** | `https://your-app.vercel.app,http://localhost:3000` |

Railway also injects `PORT`. Everything else is the same.

---

## Local dev (unchanged)

```bash
# Terminal 1 — backend
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm run dev
# Frontend reads NEXT_PUBLIC_API_URL from frontend/.env.local
# Default if unset: http://localhost:8000
```

`frontend/.env.local` (not committed):
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Known gotchas

**CORS blocked after deploy:**
- The `CORS_ORIGINS` backend env var must include `https://your-app.vercel.app` exactly, with `https://` and no trailing slash.
- After updating env vars on Render, wait for the automatic redeploy to finish before testing.

**Render free tier cold starts:**
- First request after ~15 min idle takes 30–60 s. Use a paid plan or accept it for beta.

**`$PORT` not substituted:**
- If you paste the start command with `$PORT` literally into some UIs, verify the platform interpolates it. Both Render and Railway do.

**`NEXT_PUBLIC_API_URL` baked at build time:**
- Next.js bakes `NEXT_PUBLIC_*` vars at build time. If you change it in Vercel, trigger a redeploy (Vercel does this automatically when you save the env var).
