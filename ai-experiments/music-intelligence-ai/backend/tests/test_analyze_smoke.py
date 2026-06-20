"""
Smoke test: POST /analyze accepts a short WAV and returns duration, tempo, key, chords.

Requires `httpx` (listed in requirements.txt for TestClient).

Run from `backend/`:

    pip install -r requirements.txt
    python -m unittest discover -v -s tests -p 'test_*.py'

Optional HTTP check with server running on :8000:

    bash scripts/smoke_analyze.sh
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import app

from tests.wav_fixtures import one_second_silent_wav


class TestAnalyzeSmoke(unittest.TestCase):
	def test_analyze_accepts_short_wav(self) -> None:
		client = TestClient(app)
		wav_bytes = one_second_silent_wav()
		res = client.post("/analyze", files={"file": ("test.wav", wav_bytes, "audio/wav")})
		self.assertEqual(res.status_code, 200, res.text)
		data = res.json()
		self.assertIn("duration", data)
		self.assertEqual(data.get("chord_engine"), "theory")
		self.assertIn("tempo", data)
		self.assertIn("key", data)
		self.assertIn("label", data["key"])
		self.assertIn("confidence", data["key"])
		self.assertIsInstance(data["key"]["confidence"], (int, float))
		self.assertIn("chords", data)
		self.assertIsInstance(data["chords"], list)
		self.assertIn("beats", data)
		self.assertIsInstance(data["beats"], list)
		for b in data["beats"]:
			self.assertIn("time", b)
		self.assertIn("sections", data)
		self.assertIsInstance(data["sections"], list)
		for s in data["sections"]:
			self.assertIn("index", s)
			self.assertIn("start", s)
			self.assertIn("end", s)
			self.assertIn("label", s)
		self.assertIn("rhythm", data)
		self.assertIn("assumed_beats_per_bar", data["rhythm"])
		self.assertIn("bar_start_times", data["rhythm"])
		self.assertIsInstance(data["rhythm"]["bar_start_times"], list)
		if data["sections"]:
			self.assertIn("repeat_group", data["sections"][0])
		if data["chords"]:
			c0 = data["chords"][0]
			self.assertIn("notes", c0)
			self.assertIn("practice_hint", c0)
			self.assertIn("confidence", c0)
			self.assertIn("low_confidence", c0)
			self.assertIn("is_passing", c0)
			self.assertIn("chord_role", c0)

	def test_analyze_engine_stable_parameter(self) -> None:
		client = TestClient(app)
		wav_bytes = one_second_silent_wav()
		res = client.post("/analyze", params={"engine": "stable"}, files={"file": ("test.wav", wav_bytes, "audio/wav")})
		self.assertEqual(res.status_code, 200, res.text)
		self.assertEqual(res.json().get("chord_engine"), "stable")


if __name__ == "__main__":
	unittest.main()
