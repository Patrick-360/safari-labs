"""Unit tests for music_theory helpers (synthetic vectors, no audio)."""

from __future__ import annotations

import unittest

import numpy as np

from app.audio.music_theory import (
	PITCH_NAMES_SHARP,
	blend_chroma_mean_max,
	chroma_single_note_penalty,
	diatonic_template_keys,
	likely_passing_segment,
	major_scale_pcs,
	parse_key_raw,
	pick_chord_with_theory,
	progression_pair_bonus,
	roman_numeral_for_chord_in_key,
)
from app.models.chords import build_chord_templates, _normalize_vector


class TestMusicTheory(unittest.TestCase):
	def test_major_scale_pcs(self) -> None:
		pcs = major_scale_pcs(0)
		self.assertEqual(pcs, (0, 2, 4, 5, 7, 9, 11))

	def test_parse_key(self) -> None:
		self.assertEqual(parse_key_raw("C:maj"), (0, "maj"))
		self.assertEqual(parse_key_raw("A:min"), (9, "min"))
		self.assertIsNone(parse_key_raw(None))
		self.assertIsNone(parse_key_raw("nope"))

	def test_diatonic_keys_c_major(self) -> None:
		di = diatonic_template_keys("C:maj")
		self.assertIn("C:maj", di)
		self.assertIn("D:min", di)
		self.assertIn("G:maj", di)

	def test_roman_and_progression(self) -> None:
		self.assertEqual(roman_numeral_for_chord_in_key("C:maj", "C:maj"), "I")
		self.assertEqual(roman_numeral_for_chord_in_key("G:maj", "C:maj"), "V")
		b = progression_pair_bonus("C:maj", "G:maj", "C:maj")
		self.assertGreater(b, 0.0)

	def test_single_note_penalty(self) -> None:
		v = np.zeros(12)
		v[5] = 1.0
		self.assertGreater(chroma_single_note_penalty(v), 0.0)
		u = np.ones(12) / np.sqrt(12.0)
		self.assertLess(chroma_single_note_penalty(u), 0.02)

	def test_blend_chroma_mean_max(self) -> None:
		ch = np.zeros((12, 3))
		ch[0, 0] = 1.0
		ch[4, 2] = 1.0
		out = blend_chroma_mean_max(ch)
		self.assertEqual(out.shape, (12,))
		self.assertGreater(float(out[0] + out[4]), 0.5)

	def test_pick_chord_prefers_close_fit(self) -> None:
		tpl = build_chord_templates(include_sevenths=False, include_extended=False)
		root = PITCH_NAMES_SHARP.index("C")
		v = np.zeros(12)
		for i in (0, 4, 7):
			v[(root + i) % 12] = 1.0
		v = _normalize_vector(v)
		name, label, best, _, conf = pick_chord_with_theory(v, tpl, key_raw="C:maj", prev_internal=None, normalize_vector=_normalize_vector)
		self.assertEqual(name, "C:maj")
		self.assertEqual(label, "C")
		self.assertGreater(best, 0.85)
		self.assertGreater(conf, 0.1)

	def test_likely_passing_segment(self) -> None:
		self.assertTrue(
			likely_passing_segment(0.25, "Dm", "C", "C", 0.12, max_dur=0.48, conf_hi=0.34),
		)
		self.assertFalse(
			likely_passing_segment(0.25, "F", "C", "G", 0.12, max_dur=0.48, conf_hi=0.34),
		)


if __name__ == "__main__":
	unittest.main()
