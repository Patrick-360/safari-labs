"""Tiny synthetic WAV bytes for API smoke tests."""

from __future__ import annotations

import io
import math
import struct
import wave


def one_second_silent_wav() -> bytes:
	buf = io.BytesIO()
	with wave.open(buf, "wb") as w:
		w.setnchannels(1)
		w.setsampwidth(2)
		w.setframerate(22050)
		w.writeframes(b"\x00\x00" * 22050)
	return buf.getvalue()


def n_seconds_silent_wav(duration_sec: float, sr: int = 22050) -> bytes:
	"""Mono PCM16 silence of exactly `duration_sec` seconds."""
	buf = io.BytesIO()
	n_frames = int(sr * duration_sec)
	with wave.open(buf, "wb") as w:
		w.setnchannels(1)
		w.setsampwidth(2)
		w.setframerate(sr)
		w.writeframes(b"\x00\x00" * n_frames)
	return buf.getvalue()


def half_second_tone_wav(*, freq_hz: float = 261.63, sr: int = 22050, duration_sec: float = 0.6) -> bytes:
	"""Mono PCM16 sine — enough samples for chroma / stream path smoke tests."""
	buf = io.BytesIO()
	n = int(sr * duration_sec)
	with wave.open(buf, "wb") as w:
		w.setnchannels(1)
		w.setsampwidth(2)
		w.setframerate(sr)
		frames = bytearray()
		for i in range(n):
			t = i / sr
			x = 0.25 * math.sin(2.0 * math.pi * freq_hz * t)
			s = int(max(-1.0, min(1.0, x)) * 32767)
			frames.extend(struct.pack("<h", s))
		w.writeframes(bytes(frames))
	return buf.getvalue()
