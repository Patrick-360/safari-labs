"""
Smoke test: POST /stream accepts WAV chunks and returns JSON (live microphone path).

Run from `backend/`:

    pip install -r requirements.txt
    python -m unittest discover -v -s tests -p 'test_*.py'
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import app

from tests.wav_fixtures import half_second_tone_wav, one_second_silent_wav


class TestStreamSmoke(unittest.TestCase):
	def test_stream_accepts_tone_wav(self) -> None:
		client = TestClient(app)
		wav_bytes = half_second_tone_wav()
		res = client.post("/stream", files={"file": ("chunk.wav", wav_bytes, "audio/wav")})
		self.assertEqual(res.status_code, 200, res.text)
		data = res.json()
		self.assertIn("chord", data)
		self.assertIn("confidence", data)
		self.assertIn("key", data)
		self.assertIn("key_confidence", data)
		self.assertIsInstance(data["chord"], str)
		self.assertIn("debug", data)

	def test_stream_accepts_silent_wav(self) -> None:
		client = TestClient(app)
		wav_bytes = one_second_silent_wav()
		res = client.post("/stream", files={"file": ("chunk.wav", wav_bytes, "audio/wav")})
		self.assertEqual(res.status_code, 200, res.text)
		data = res.json()
		# Silence path: debug raw is N (display chord may hold last valid from prior requests in-process).
		self.assertEqual(data.get("debug", {}).get("raw_chord"), "N")


if __name__ == "__main__":
	unittest.main()
