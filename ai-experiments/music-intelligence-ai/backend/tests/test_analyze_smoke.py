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

from app.core.config import BETA_ANALYSIS_DURATION_SEC
from app.main import app

from tests.wav_fixtures import n_seconds_silent_wav, one_second_silent_wav


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

	def test_short_file_analysis_window_not_trimmed(self) -> None:
		"""A short file (1 s) must return analysis_window with was_trimmed=False."""
		client = TestClient(app)
		wav_bytes = one_second_silent_wav()
		res = client.post("/analyze", files={"file": ("test.wav", wav_bytes, "audio/wav")})
		self.assertEqual(res.status_code, 200, res.text)
		data = res.json()
		self.assertIn("analysis_window", data)
		aw = data["analysis_window"]
		self.assertIsNotNone(aw)
		self.assertFalse(aw["was_trimmed"])
		self.assertIsNone(aw["reason"])
		self.assertGreater(aw["duration_analyzed"], 0.0)
		self.assertLess(aw["duration_analyzed"], BETA_ANALYSIS_DURATION_SEC)

	def test_long_file_analysis_window_trimmed(self) -> None:
		"""A file longer than BETA_ANALYSIS_DURATION_SEC must be trimmed and report was_trimmed=True."""
		client = TestClient(app)
		long_sec = BETA_ANALYSIS_DURATION_SEC + 40.0
		wav_bytes = n_seconds_silent_wav(long_sec)
		res = client.post("/analyze", files={"file": ("long.wav", wav_bytes, "audio/wav")})
		self.assertEqual(res.status_code, 200, res.text)
		data = res.json()
		self.assertIn("analysis_window", data)
		aw = data["analysis_window"]
		self.assertTrue(aw["was_trimmed"])
		self.assertEqual(aw["reason"], "beta_duration_limit")
		self.assertAlmostEqual(aw["duration_analyzed"], BETA_ANALYSIS_DURATION_SEC, delta=1.0)
		self.assertIsNotNone(aw["original_duration"])
		self.assertGreater(aw["original_duration"], BETA_ANALYSIS_DURATION_SEC)

	def test_oversized_file_returns_friendly_error(self) -> None:
		"""A file exceeding the 30MB limit must return 400 with error=file_too_large."""
		from app.core.config import BETA_MAX_UPLOAD_SIZE_MB
		client = TestClient(app)
		# Create a fake payload just over the limit (use raw bytes, not a valid WAV).
		oversize = b"\x00" * (BETA_MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1)
		res = client.post("/analyze", files={"file": ("big.wav", oversize, "audio/wav")})
		self.assertEqual(res.status_code, 400)
		detail = res.json().get("detail", {})
		self.assertEqual(detail.get("error"), "file_too_large")


if __name__ == "__main__":
	unittest.main()
