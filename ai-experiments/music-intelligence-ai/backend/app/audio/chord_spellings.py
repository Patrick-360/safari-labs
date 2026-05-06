"""
Heuristic playable chord spellings for practice (triads in close position spelling, no voicing/inversion).

These are simplified theory helpers for beginners—not exact piano voicings or transcriptions.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Match chroma / template roots (sharp names).
_PC_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
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


def _pc_name(pc: int) -> str:
	return _PC_SHARP[pc % 12]


def playable_triad_notes_and_hint(symbol: str) -> Tuple[List[str], str]:
	"""
	Map a display chord symbol to close-position *roots* (no inversion), for practice hints.

	Supports triads, common sevenths, sus2/sus4, dim/aug, and half-diminished (m7b5).
	Unknown spellings return a generic line — heuristic, not a full fake book.
	"""
	if not symbol or symbol == "N":
		return [], "—"

	s = symbol.strip()
	sl = s.lower()

	if sl.endswith("maj7"):
		body = s[: -4].strip()
		if body not in _ROOT_ALIASES:
			return [], f"Play {symbol} (spelling not automated)"
		root = _ROOT_ALIASES[body]
		intervals = (0, 4, 7, 11)
		notes = [_pc_name(root + iv) for iv in intervals]
		hint = f"Play {symbol}: {' — '.join(notes)} (major 7 — close guide, any octave)"
		return notes, hint

	if sl.endswith("m7b5"):
		body = s[: -5].strip()
		if body not in _ROOT_ALIASES:
			return [], f"Play {symbol} (spelling not automated)"
		root = _ROOT_ALIASES[body]
		intervals = (0, 3, 6, 10)
		notes = [_pc_name(root + iv) for iv in intervals]
		hint = f"Play {symbol}: {' — '.join(notes)} (half-dim 7 — close guide, any octave)"
		return notes, hint

	if len(sl) >= 3 and sl.endswith("m7") and not sl.endswith("maj7"):
		body = s[: -2].strip()
		if body not in _ROOT_ALIASES:
			return [], f"Play {symbol} (spelling not automated)"
		root = _ROOT_ALIASES[body]
		intervals = (0, 3, 7, 10)
		notes = [_pc_name(root + iv) for iv in intervals]
		hint = f"Play {symbol}: {' — '.join(notes)} (minor 7 — close guide, any octave)"
		return notes, hint

	if len(s) >= 2 and s[-1] == "7":
		body = s[: -1].strip()
		if body not in _ROOT_ALIASES:
			return [], f"Play {symbol} (spelling not automated)"
		root = _ROOT_ALIASES[body]
		intervals = (0, 4, 7, 10)
		notes = [_pc_name(root + iv) for iv in intervals]
		hint = f"Play {symbol}: {' — '.join(notes)} (dominant 7 — close guide, any octave)"
		return notes, hint

	if sl.endswith("sus4"):
		body = s[: -4].strip()
		if body not in _ROOT_ALIASES:
			return [], f"Play {symbol} (spelling not automated)"
		root = _ROOT_ALIASES[body]
		intervals = (0, 5, 7)
		notes = [_pc_name(root + iv) for iv in intervals]
		hint = f"Play {symbol}: {' — '.join(notes)} (sus4 — close guide, any octave)"
		return notes, hint

	if sl.endswith("sus2"):
		body = s[: -4].strip()
		if body not in _ROOT_ALIASES:
			return [], f"Play {symbol} (spelling not automated)"
		root = _ROOT_ALIASES[body]
		intervals = (0, 2, 7)
		notes = [_pc_name(root + iv) for iv in intervals]
		hint = f"Play {symbol}: {' — '.join(notes)} (sus2 — close guide, any octave)"
		return notes, hint

	if sl.endswith("dim"):
		body = s[: -3].strip()
		if body not in _ROOT_ALIASES:
			return [], f"Play {symbol} (spelling not automated)"
		root = _ROOT_ALIASES[body]
		intervals = (0, 3, 6)
		notes = [_pc_name(root + iv) for iv in intervals]
		hint = f"Play {symbol}: {' — '.join(notes)} (diminished triad — close guide, any octave)"
		return notes, hint

	if sl.endswith("aug") or s.endswith("+"):
		body = s[: -3].strip() if sl.endswith("aug") else s[: -1].strip()
		if body not in _ROOT_ALIASES:
			return [], f"Play {symbol} (spelling not automated)"
		root = _ROOT_ALIASES[body]
		intervals = (0, 4, 8)
		notes = [_pc_name(root + iv) for iv in intervals]
		hint = f"Play {symbol}: {' — '.join(notes)} (augmented triad — close guide, any octave)"
		return notes, hint

	is_minor = len(s) > 1 and s.endswith("m") and not s.endswith("maj")
	root_token = s[:-1] if is_minor else s
	if root_token not in _ROOT_ALIASES:
		return [], f"Play {symbol} (spelling not automated)"

	root = _ROOT_ALIASES[root_token]
	intervals = (0, 3, 7) if is_minor else (0, 4, 7)
	notes = [_pc_name(root + iv) for iv in intervals]
	qual = "minor" if is_minor else "major"
	hint = f"Play {symbol}: {' — '.join(notes)} ({qual} triad — close spelling, any octave)"
	return notes, hint

