"""Lightweight contract tests for /health and /stream (live path)."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import app

from tests.wav_fixtures import one_second_silent_wav


class TestHealthAndStreamSmoke(unittest.TestCase):
	def test_health_ok(self) -> None:
		client = TestClient(app)
		res = client.get("/health")
		self.assertEqual(res.status_code, 200, res.text)
		data = res.json()
		self.assertEqual(data.get("status"), "ok")

	def test_stream_accepts_wav_returns_contract(self) -> None:
		client = TestClient(app)
		wav_bytes = one_second_silent_wav()
		res = client.post("/stream", files={"file": ("chunk.wav", wav_bytes, "audio/wav")})
		self.assertEqual(res.status_code, 200, res.text)
		data = res.json()
		self.assertIn("chord", data)
		self.assertIn("confidence", data)
		self.assertIn("key", data)
		self.assertIn("key_confidence", data)
		self.assertIn("timestamp", data)
		self.assertIn("debug", data)
		self.assertIn("raw_chord", data["debug"])
		self.assertIn("scores_top3", data["debug"])


if __name__ == "__main__":
	unittest.main()
