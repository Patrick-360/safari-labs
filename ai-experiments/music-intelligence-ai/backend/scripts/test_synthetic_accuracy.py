#!/usr/bin/env python3
"""
Sanity-check chord detection with simple synthetic triads (sine stacks).

Run from repo root or backend/:
  backend/.venv/bin/python backend/scripts/test_synthetic_accuracy.py

This is not a rigorous test suite — it prints what the heuristic pipeline hears
so you can compare expected triads vs merged segment labels.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
	sys.path.insert(0, str(BACKEND_ROOT))

from app.audio.analyze_pipeline import run_analysis  # noqa: E402

SR = 22050
CHUNK_SEC = 1.8


def _hz(note: str) -> float:
	return float(librosa.note_to_hz(note))


def synth_triad(n1: str, n2: str, n3: str, seconds: float = CHUNK_SEC) -> np.ndarray:
	t = np.linspace(0.0, seconds, int(SR * seconds), endpoint=False, dtype=np.float64)
	sig = (
		np.sin(2 * np.pi * _hz(n1) * t)
		+ np.sin(2 * np.pi * _hz(n2) * t)
		+ np.sin(2 * np.pi * _hz(n3) * t)
	)
	sig *= 0.32
	# gentle fade to reduce edge clicks
	if sig.size > SR // 8:
		ramp = int(SR * 0.02)
		sig[:ramp] *= np.linspace(0.0, 1.0, ramp)
		sig[-ramp:] *= np.linspace(1.0, 0.0, ramp)
	return sig.astype(np.float32)


def to_wav_bytes(y: np.ndarray) -> bytes:
	buf = io.BytesIO()
	sf.write(buf, y, SR, format="WAV", subtype="PCM_16")
	return buf.getvalue()


def main() -> None:
	specs: list[tuple[str, np.ndarray, str | None]] = [
		("C_major", synth_triad("C4", "E4", "G4"), "C"),
		("G_major", synth_triad("G3", "B3", "D4"), "G"),
		("A_minor", synth_triad("A3", "C4", "E4"), "Am"),
		("F_major", synth_triad("F3", "A3", "C4"), "F"),
	]
	prog_y = np.concatenate(
		[
			synth_triad("C4", "E4", "G4", seconds=1.1),
			synth_triad("G3", "B3", "D4", seconds=1.1),
			synth_triad("A3", "C4", "E4", seconds=1.1),
			synth_triad("F3", "A3", "C4", seconds=1.1),
		],
	)
	specs.append(("C_G_Am_F_progression", prog_y, None))

	for name, y, expected in specs:
		raw = to_wav_bytes(y)
		out = run_analysis(raw, debug=False)
		labels = [str(c.get("label", "")) for c in out.get("chords") or []]
		print(f"\n=== {name} ===")
		print(f"  duration_s: {out.get('duration')}  tempo: {out.get('tempo')}  key: {out.get('key')}")
		preview = labels[:16]
		suffix = " …" if len(labels) > 16 else ""
		print(f"  chords ({len(labels)} segments): {preview}{suffix}")
		if expected:
			ok = expected in labels
			print(f"  expect label in segments: {expected!r} → {ok}")


if __name__ == "__main__":
	main()
