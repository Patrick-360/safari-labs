"""Sanity check for key profile scoring (deterministic, used by live key hysteresis)."""

from __future__ import annotations

import unittest

import numpy as np

from app.audio.features import estimate_key, score_key_profile_fit


class TestKeyProfileFit(unittest.TestCase):
	def test_score_matches_argmax_of_estimate_key(self) -> None:
		hist = np.array(
			[0.45, 0.02, 0.08, 0.02, 0.12, 0.15, 0.02, 0.06, 0.02, 0.02, 0.02, 0.02],
			dtype=float,
		)
		best_raw, _ = estimate_key(hist)
		best_fit = score_key_profile_fit(hist, best_raw)
		# Every other key should score <= best (same profiles as estimate_key).
		pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
		for name in pitch_classes:
			for mode in (":maj", ":min"):
				raw = f"{name}{mode}"
				fit = score_key_profile_fit(hist, raw)
				self.assertLessEqual(fit, best_fit + 1e-6, msg=raw)


if __name__ == "__main__":
	unittest.main()
