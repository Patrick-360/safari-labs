#!/usr/bin/env python3
"""
POST a short synthetic C-major triad mixture to /stream (no browser).

Usage (server on http://127.0.0.1:8000):

    pip install httpx   # if not already in your venv

    python scripts/test_stream_chord.py
    python scripts/test_stream_chord.py --url http://localhost:8000/stream

Requires: httpx (same venv as the API), plus stdlib for WAV generation.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import struct
import sys
import wave

import httpx


def build_c_major_chord_wav(
	*,
	sr: int = 22050,
	duration_sec: float = 1.2,
) -> bytes:
	"""C4 + E4 + G4 sine mixture — not a real instrument, enough to exercise chroma."""
	buf = io.BytesIO()
	n = int(sr * duration_sec)
	# Equal-tempered approximations (Hz)
	freqs = (261.63, 329.63, 392.0)
	with wave.open(buf, "wb") as w:
		w.setnchannels(1)
		w.setsampwidth(2)
		w.setframerate(sr)
		frames = bytearray()
		for i in range(n):
			t = i / sr
			sample = 0.0
			for f in freqs:
				sample += (1.0 / len(freqs)) * 0.28 * math.sin(2.0 * math.pi * f * t)
			s = int(max(-1.0, min(1.0, sample)) * 32767)
			frames.extend(struct.pack("<h", s))
		w.writeframes(bytes(frames))
	return buf.getvalue()


def main() -> int:
	parser = argparse.ArgumentParser(description="POST synthetic chord WAV to /stream")
	parser.add_argument(
		"--url",
		default="http://127.0.0.1:8000/stream",
		help="Full /stream URL",
	)
	args = parser.parse_args()
	wav = build_c_major_chord_wav()
	print(f"POST {args.url} ({len(wav)} bytes WAV, synthetic C-E-G)")
	try:
		with httpx.Client(timeout=30.0) as client:
			r = client.post(args.url, files={"file": ("chord.wav", wav, "audio/wav")})
	except httpx.RequestError as e:
		print(f"Request failed: {e}", file=sys.stderr)
		return 1
	if not r.is_success:
		print(f"HTTP {r.status_code}: {r.text}", file=sys.stderr)
		return 1
	try:
		data = r.json()
	except json.JSONDecodeError:
		print("Non-JSON body:", r.text[:500], file=sys.stderr)
		return 1
	print(json.dumps(data, indent=2))
	ch = data.get("chord", "")
	if ch and ch != "N":
		print(f"\n--> display chord: {ch!r}")
	else:
		print("\n--> display chord empty or N (try longer/denser synthetic mix or check server logs)")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
