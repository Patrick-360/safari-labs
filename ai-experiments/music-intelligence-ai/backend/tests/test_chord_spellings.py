"""Lightweight spell_chord_tones coverage — mirrors frontend/lib/chordSpelling.ts examples."""

import unittest

from app.audio.chord_spellings import spell_chord_tones


class TestChordSpellings(unittest.TestCase):
	def test_triads_and_colon(self) -> None:
		self.assertEqual(spell_chord_tones("C"), ["C", "E", "G"])
		self.assertEqual(spell_chord_tones("Am"), ["A", "C", "E"])
		self.assertEqual(spell_chord_tones("G"), ["G", "B", "D"])
		self.assertEqual(spell_chord_tones("F"), ["F", "A", "C"])
		self.assertEqual(spell_chord_tones("Dm"), ["D", "F", "A"])
		self.assertEqual(spell_chord_tones("Bb"), ["Bb", "D", "F"])
		self.assertEqual(spell_chord_tones("F#"), ["F#", "A#", "C#"])
		self.assertEqual(spell_chord_tones("C:maj"), ["C", "E", "G"])
		self.assertEqual(spell_chord_tones("A:min"), ["A", "C", "E"])

	def test_sharp_root_minor_stays_sharp(self) -> None:
		self.assertEqual(spell_chord_tones("F#m"), ["F#", "A", "C#"])

	def test_sevenths(self) -> None:
		self.assertEqual(spell_chord_tones("C7"), ["C", "E", "G", "Bb"])
		self.assertEqual(spell_chord_tones("Cmaj7"), ["C", "E", "G", "B"])
		self.assertEqual(spell_chord_tones("Cm7"), ["C", "Eb", "G", "Bb"])
		self.assertEqual(spell_chord_tones("Am7"), ["A", "C", "E", "G"])

	def test_slash_no_and_n(self) -> None:
		self.assertEqual(spell_chord_tones("C/E"), ["C", "E", "G"])
		self.assertEqual(spell_chord_tones("N"), [])
		self.assertEqual(spell_chord_tones(""), [])

	def test_short_root_minor_seventh_regression(self) -> None:
		self.assertEqual(spell_chord_tones("Cm7"), ["C", "Eb", "G", "Bb"])


if __name__ == "__main__":
	unittest.main()
