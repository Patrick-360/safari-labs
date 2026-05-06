# Manual review checklist

Use this when doing an end-to-end review or demo dry-run. Assume backend on **:8000** and frontend on **:3000** unless configured otherwise.

## Setup

- [ ] Backend: venv active, `pip install -r requirements.txt`, `uvicorn app.main:app --reload --port 8000` from `backend/`
- [ ] Frontend: `npm install`, `npm run dev` from `frontend/`
- [ ] If frontend and API are on different hosts/ports, set `NEXT_PUBLIC_API_URL` and matching `CORS_ORIGINS` (see README)

## Live microphone mode

- [ ] **Silence**: Start recording; with little or no input, chord display should not flicker random symbols indefinitely; after sustained silence, UI tends toward empty/“N” behavior per app logic
- [ ] **Chord responsiveness**: Play clear triads; chord updates within roughly chunk latency (~0.5 s chunks)
- [ ] **Key stability**: Stay in one key; play several **diatonic** chords; reported **key** should not jump every chord (tonal-center smoothing)
- [ ] **Stop**: Stop recording; post-stop behavior is acceptable (fade / clear)

## Analyze file mode

- [ ] **Upload + analyze**: Choose audio (WAV/MP3), run analysis; loading state appears; results render without errors
- [ ] **Playback**: Audio plays; time and **current chord** track playback
- [ ] **Next chord**: “Next chord” / coming-up hints match upcoming timeline
- [ ] **Timeline**: Chord blocks, beat ticks (taller on assumed downbeats), section markers, playhead move correctly
- [ ] **Section navigation**: Previous/next section and section pills seek; loop target updates when jumping (if implemented)
- [ ] **Section loop**: Select section, enable loop; playback wraps at section end
- [ ] **Progression**: Chips scroll/highlight; **Now** / **Next** badges; tap-to-seek works

## API / health

- [ ] `GET /health` returns `{"status":"ok"}`
- [ ] `POST /analyze` returns expected JSON (duration, tempo, key, chords, beats, sections with `index`, `rhythm`)

## Automated smoke (optional)

From `backend/`:

```bash
.venv/bin/python -m unittest discover -v -s tests -p 'test_*.py'
```
