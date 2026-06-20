# Chord accuracy benchmarks (Analyze File)

Repeatable offline evaluation for **POST /analyze** chord engines (`stable`, `theory`, `experimental`). This does not change app behavior — it calls `run_analysis` directly, same as the API.

## Folder layout

```
backend/benchmarks/
  audio/          # WAV, MP3, etc. (one file per clip)
  annotations/    # One JSON ground-truth file per clip
  results/        # Written by the evaluator (latest.json)
```

## Annotation format

One JSON file per clip in `annotations/`. Files whose names start with `_` are ignored (templates only).

```json
{
  "title": "My song clip",
  "audio_file": "my_song.wav",
  "key": "C major",
  "chords": [
    { "start": 0.0, "end": 2.0, "label": "C" },
    { "start": 2.0, "end": 4.0, "label": "G" },
    { "start": 4.0, "end": 6.0, "label": "Am" },
    { "start": 6.0, "end": 8.0, "label": "F" }
  ]
}
```

| Field | Notes |
|-------|--------|
| `title` | Human-readable name (shown in reports). |
| `audio_file` | Filename under `audio/` (e.g. `clip.wav` or `clip.mp3`). |
| `key` | Expected key, e.g. `C major`, `A minor` (compared to analyze `key.label`). |
| `chords` | Segments in **seconds**; labels use the same symbols as the UI (`C`, `Am`, `F#m`, `Gsus4`, …). |

Tips:

- Segments should cover the harmonic parts you care about; gaps with no label are not scored.
- Use `N` only if you explicitly want “no chord” in the reference for a span.
- Align times to what you hear (rough boundaries are fine; the script also reports boundary timing error when labels agree).

Copy `_template.json` when adding a new clip.

## Run the benchmark

From `backend/` with the project venv active:

```bash
cd backend
source .venv/bin/activate
python scripts/evaluate_chord_accuracy.py
```

Quick start (generates a built-in synthetic C–G–Am–F clip, then evaluates all engines):

```bash
python scripts/evaluate_chord_accuracy.py --write-example
python scripts/evaluate_chord_accuracy.py
```

Options (optional):

```bash
python scripts/evaluate_chord_accuracy.py --engines stable,theory,experimental
python scripts/evaluate_chord_accuracy.py --results benchmarks/results/latest.json
```

Output:

- Terminal table: per-clip and per-engine metrics.
- `benchmarks/results/latest.json`: full JSON for diffing or CI.

## Add a new test clip

1. Put the audio file in `benchmarks/audio/`, e.g. `my_clip.wav`.
2. Create `benchmarks/annotations/my_clip.json` using the format above (`audio_file` must match the filename).
3. Label chords and key by ear (or from a DAW chart). Keep segment times monotonic; small overlaps at boundaries are OK.
4. Run `python scripts/evaluate_chord_accuracy.py`.
5. Compare **weighted accuracy %** and **key match** across engines in the printed summary.

## How scoring works (short)

- Timeline is sampled every 50 ms (configurable). For each sample where the annotation has a chord label, the predicted label at that time is compared.
- **Weighted accuracy** = fraction of annotated time where labels match (primary metric).
- **Key match** = predicted global key string matches annotation (normalized).
- **Timing error** = mean absolute start/end offset when a predicted segment overlaps a matching ground-truth segment.
- **Confusion** = seconds of annotated time spent in each (ground truth → predicted) label pair.

See `scripts/evaluate_chord_accuracy.py` for details.
