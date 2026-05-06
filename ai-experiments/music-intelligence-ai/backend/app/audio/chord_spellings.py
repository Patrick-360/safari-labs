"""
Reliable close-position chord spellings for practice UI (not voicings / inversions).

Single source of truth for /analyze `notes` + `practice_hint`. Unknown symbols → no fake notes.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

_PC_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_PC_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

_ROOT_ALIASES: Dict[str, int] = {
	"C": 0,
	"C#": 1,
	"Db": 1,
	"D": 2,
	"D#": 3,
	"Eb": 3,
	"E": 4,
	"F": 5,
	"F#": 6,
	"Gb": 6,
	"G": 7,
	"G#": 8,
	"Ab": 8,
	"A": 9,
	"A#": 10,
	"Bb": 10,
	"B": 11,
}


def _pc_display(pc: int, prefer_flat: bool) -> str:
	table = _PC_FLAT if prefer_flat else _PC_SHARP
	return table[int(pc) % 12]


def _root_prefers_flat(body: str) -> bool:
	"""Flat spelling when the root token uses a flat accidental (e.g. Bb, Eb)."""
	return "b" in body.lower()


def _normalize_colon_chord(token: str) -> str | None:
	"""Turn C:maj / A:min style into plain token (C, Am). Unknown quality → None."""
	if ":" not in token:
		return token.strip()
	root, qual_raw = token.split(":", 1)
	root, qual_raw = root.strip(), qual_raw.strip()
	qual = qual_raw.lower()
	if root not in _ROOT_ALIASES:
		return None
	if qual in ("maj", "major") or qual_raw == "M":
		return root
	if qual in ("min", "minor", "m"):
		return f"{root}m"
	return None


def _root_has_sharp(body: str) -> bool:
	return "#" in body


def _prefer_flat_spelling(body: str, intervals: Tuple[int, ...]) -> bool:
	"""
	Flat spellings for roots with 'b' (always). For chords with ♭3 / ♭7 family colors,
	prefer flats when the root is not a sharp-letter token so outputs match C7 → Bb, Cm7 → Eb.
	Sharp roots (e.g. F#m7) stay in sharp spelling.
	"""
	if _root_prefers_flat(body):
		return True
	if _root_has_sharp(body):
		return False
	if 3 in intervals or 10 in intervals:
		return True
	return False


def spell_chord_tones(symbol: str) -> List[str]:
	"""
	Spell chord tones from a display-style symbol. Slash uses left side only (e.g. C/E → C major).
	Returns [] for N, empty, or unrecognized.
	"""
	if not symbol or not str(symbol).strip():
		return []
	raw = str(symbol).strip()
	if raw.upper() == "N":
		return []

	chord_side = raw.split("/")[0].strip()
	core = _normalize_colon_chord(chord_side)
	if core is None:
		return []

	cl = core.lower()
	body: str
	intervals: Tuple[int, ...]

	if cl.endswith("maj7"):
		body, intervals = core[:-4], (0, 4, 7, 11)
	elif cl.endswith("m7b5"):
		body, intervals = core[:-5], (0, 3, 6, 10)
	elif cl.endswith("m7"):
		# Strip literal "m7" (two chars). core[:-3] breaks short roots like "Cm7" → "".
		body, intervals = core[:-2], (0, 3, 7, 10)
	elif cl.endswith("sus4"):
		body, intervals = core[:-4], (0, 5, 7)
	elif cl.endswith("sus2"):
		body, intervals = core[:-4], (0, 2, 7)
	elif cl.endswith("dim"):
		body, intervals = core[:-3], (0, 3, 6)
	elif cl.endswith("aug"):
		body, intervals = core[:-3], (0, 4, 8)
	elif len(core) > 1 and core.endswith("+") and not core.lower().endswith("aug"):
		body, intervals = core[:-1], (0, 4, 8)
	elif len(cl) >= 3 and cl.endswith("maj") and not cl.endswith("maj7"):
		body, intervals = core[:-3], (0, 4, 7)
	elif len(core) >= 2 and core.endswith("7") and not cl.endswith("m7") and not cl.endswith("maj7"):
		body, intervals = core[:-1], (0, 4, 7, 10)
	elif len(core) > 1 and core.endswith("m") and not cl.endswith("maj"):
		body, intervals = core[:-1], (0, 3, 7)
	else:
		body, intervals = core, (0, 4, 7)

	if body not in _ROOT_ALIASES:
		return []
	root_pc = _ROOT_ALIASES[body]
	prefer_flat = _prefer_flat_spelling(body, intervals)
	return [_pc_display(root_pc + iv, prefer_flat) for iv in intervals]


def playable_triad_notes_and_hint(symbol: str) -> Tuple[List[str], str]:
	"""
	Map a chord symbol to close-position note names + short hint line for beginners.

	Slash chords: tones follow the chord (left of /); bass for LH is a separate UI concern.
	Unknown → empty notes and “check by ear” (no invented spellings).
	"""
	notes = spell_chord_tones(symbol)
	if not notes:
		if not symbol or str(symbol).strip().upper() in ("", "N"):
			return [], "—"
		return [], "Check this one by ear — unfamiliar or abbreviated symbol."
	line = " · ".join(notes)
	return notes, f"{line} (simplified close spelling, any octave; check by ear if unsure)"
