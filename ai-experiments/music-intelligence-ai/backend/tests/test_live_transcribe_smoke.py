"""Smoke: POST /live-transcribe returns contract."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import app

from tests.wav_fixtures import half_second_tone_wav


class TestLiveTranscribeSmoke(unittest.TestCase):
	def test_live_transcribe_accepts_tone_wav(self) -> None:
		client = TestClient(app)
		wav_bytes = half_second_tone_wav()
		res = client.post(
			"/live-transcribe",
			files={"file": ("win.wav", wav_bytes, "audio/wav")},
			params={"window_start": 0.0, "session_id": "t1"},
		)
		self.assertEqual(res.status_code, 200, res.text)
		data = res.json()
		self.assertIn("key", data)
		self.assertIn("chords", data)
		self.assertIn("core_progression", data)
		self.assertIn("current_chord", data)
		self.assertIn("summary", data)
		self.assertIn("status", data)


if __name__ == "__main__":
	unittest.main()
