"""Unit tests for tempo half/double normalization in analyze_pipeline."""

from __future__ import annotations

import unittest

from app.audio.analyze_pipeline import _normalize_tempo_from_candidates


class TestTempoNormalize(unittest.TestCase):
	def test_halves_obviously_fast_ballad_misread(self) -> None:
		bpm, reason = _normalize_tempo_from_candidates(168.0, [84.0, 168.0])
		self.assertAlmostEqual(bpm, 84.0, places=1)
		self.assertIn("halved", reason)

	def test_keeps_practical_tempo(self) -> None:
		bpm, reason = _normalize_tempo_from_candidates(72.0, [72.0, 144.0])
		self.assertAlmostEqual(bpm, 72.0, places=1)
		self.assertEqual(reason, "beat_track_in_practical_range")

	def test_doubles_slow_misread(self) -> None:
		bpm, reason = _normalize_tempo_from_candidates(36.0, [36.0, 72.0])
		self.assertAlmostEqual(bpm, 72.0, places=1)
		self.assertIn("doubled", reason)


if __name__ == "__main__":
	unittest.main()
