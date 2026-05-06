"""
Lightweight music-theory helpers for /analyze chord interpretation (heuristics, not a rules engine).

Biases template matching toward plausible harmony; audio evidence still wins when strong.
Optional future: harmonic stem from `source_separation.separate_harmonic_stems()` before chroma.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from app.models.chords import CHROMA_BINS

# Align with analyze_pipeline thresholds (avoid importing analyze_pipeline — circular).
THEORY_SEVENTH_SIMPLIFY_MARGIN = 0.065
THEORY_WEAK_AUDIO_CAP = 0.55

PITCH_NAMES_SHARP: Tuple[str, ...] = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

_NAME_TO_PC: Dict[str, int] = {
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


def pc_to_sharp_name(pc: int) -> str:
	return PITCH_NAMES_SHARP[int(pc) % 12]


def parse_key_raw(key_raw: str | None) -> Tuple[int, str] | None:
	if not key_raw or ":" not in key_raw:
		return None
	root_s, mode = key_raw.split(":", 1)
	if mode not in ("maj", "min"):
		return None
	pc = _NAME_TO_PC.get(root_s)
	if pc is None:
		return None
	return pc, mode


def major_scale_pcs(tonic_pc: int) -> Tuple[int, ...]:
	return tuple((tonic_pc + s) % 12 for s in (0, 2, 4, 5, 7, 9, 11))


def natural_minor_scale_pcs(tonic_pc: int) -> Tuple[int, ...]:
	return tuple((tonic_pc + s) % 12 for s in (0, 2, 3, 5, 7, 8, 10))


def diatonic_template_keys_major(tonic_pc: int) -> set[str]:
	pcs = major_scale_pcs(tonic_pc)
	quals = ("maj", "min", "min", "maj", "maj", "min", "min")
	return {f"{pc_to_sharp_name(pcs[i])}:{quals[i]}" for i in range(7)}


def diatonic_template_keys_minor(tonic_pc: int) -> set[str]:
	pcs = natural_minor_scale_pcs(tonic_pc)
	quals = ("min", "min", "maj", "min", "min", "maj", "maj")
	return {f"{pc_to_sharp_name(pcs[i])}:{quals[i]}" for i in range(7)}


def diatonic_template_keys(key_raw: str | None) -> set[str]:
	k = parse_key_raw(key_raw)
	if not k:
		return set()
	tonic, mode = k
	if mode == "maj":
		return diatonic_template_keys_major(tonic)
	return diatonic_template_keys_minor(tonic)


def _triad_quality_for_roman(chord_internal: str) -> str | None:
	"""Map dom7 / maj7 / min7 to triad quality for scale degree lookup."""
	if ":" not in chord_internal:
		return None
	_, q = chord_internal.split(":", 1)
	if q in ("maj", "7", "maj7", "sus2", "sus4"):
		return "maj"
	if q in ("min", "min7"):
		return "min"
	if q == "dim" or q == "m7b5":
		return "dim"
	if q == "aug":
		return "aug"
	return None


def roman_numeral_for_chord_in_key(chord_internal: str, key_raw: str | None) -> str | None:
	if not chord_internal or chord_internal == "N":
		return None
	k = parse_key_raw(key_raw)
	if not k or ":" not in chord_internal:
		return None
	root_s, q_in = chord_internal.split(":", 1)
	pc_c = _NAME_TO_PC.get(root_s)
	if pc_c is None:
		return None
	q = _triad_quality_for_roman(chord_internal)
	if q is None:
		return None
	tonic_pc, mode = k
	interval = (pc_c - tonic_pc) % 12
	if mode == "maj":
		deg_map = {
			(0, "maj"): "I",
			(2, "min"): "ii",
			(4, "min"): "iii",
			(5, "maj"): "IV",
			(7, "maj"): "V",
			(9, "min"): "vi",
			(11, "dim"): "viio",
			(11, "min"): "vii",
		}
	else:
		deg_map = {
			(0, "min"): "i",
			(2, "dim"): "iio",
			(2, "min"): "ii",
			(3, "maj"): "III",
			(5, "min"): "iv",
			(7, "min"): "v",
			(7, "maj"): "V",
			(8, "maj"): "VI",
			(10, "maj"): "VII",
		}
	return deg_map.get((interval, q))


_PROGRESSION_BONUS: Dict[Tuple[str, str], float] = {
	("I", "IV"): 0.028,
	("I", "V"): 0.03,
	("I", "vi"): 0.022,
	("IV", "V"): 0.028,
	("V", "I"): 0.034,
	("vi", "IV"): 0.024,
	("ii", "V"): 0.032,
	("I", "ii"): 0.018,
	("IV", "I"): 0.02,
	("vi", "I"): 0.018,
	("i", "iv"): 0.026,
	("i", "V"): 0.024,
	("i", "VII"): 0.02,
	("VII", "VI"): 0.018,
	("VI", "VII"): 0.018,
}


def progression_pair_bonus(prev_internal: str | None, cand_internal: str, key_raw: str | None) -> float:
	if not prev_internal or prev_internal == "N" or cand_internal == "N":
		return 0.0
	r0 = roman_numeral_for_chord_in_key(prev_internal, key_raw)
	r1 = roman_numeral_for_chord_in_key(cand_internal, key_raw)
	if not r0 or not r1:
		return 0.0
	return float(_PROGRESSION_BONUS.get((r0, r1), 0.0))


def key_fit_bonus(
	chord_internal: str,
	key_diats: set[str],
	*,
	audio_margin: float,
	audio_best: float,
	max_bonus: float = 0.045,
) -> float:
	if chord_internal == "N" or not chord_internal:
		return 0.0
	if chord_internal in key_diats:
		scale = 0.55 + 0.45 * min(1.0, audio_margin * 8.0) * min(1.0, audio_best)
		return float(max_bonus * scale)
	return 0.0


def chroma_single_note_penalty(chroma_12: np.ndarray) -> float:
	v = np.asarray(chroma_12, dtype=float).reshape(-1)
	n = float(np.linalg.norm(v))
	if n < 1e-12:
		return 0.0
	u = v / n
	mx = float(np.max(u))
	excess = max(0.0, mx - 0.48)
	return float(min(0.14, excess * 0.35))


def blend_chroma_mean_max(chroma_win: np.ndarray, w_mean: float = 0.58, w_max: float = 0.42) -> np.ndarray:
	"""Mean + max over time: sustained harmony vs arpeggiated tones (heuristic)."""
	if chroma_win.size == 0:
		return np.zeros(CHROMA_BINS, dtype=float)
	m = np.mean(chroma_win, axis=1)
	mx = np.max(chroma_win, axis=1)
	out = w_mean * m + w_max * mx
	n = float(np.linalg.norm(out))
	return out / n if n > 1e-12 else out


def _is_seventh_quality(name: str) -> bool:
	return name.endswith(":7") or name.endswith(":maj7") or name.endswith(":min7")


def _is_color_extension(name: str) -> bool:
	if ":" not in name:
		return False
	q = name.split(":", 1)[1]
	return q in ("dim", "aug", "sus2", "sus4", "m7b5")


def format_internal_chord_label(name: str) -> str:
	if name == "N":
		return "N"
	root, quality = name.split(":", 1)
	if quality == "maj":
		return root
	if quality == "min":
		return f"{root}m"
	if quality == "7":
		return f"{root}7"
	if quality == "maj7":
		return f"{root}maj7"
	if quality == "min7":
		return f"{root}m7"
	if quality == "dim":
		return f"{root}dim"
	if quality == "aug":
		return f"{root}aug"
	if quality == "sus2":
		return f"{root}sus2"
	if quality == "sus4":
		return f"{root}sus4"
	if quality == "m7b5":
		return f"{root}m7b5"
	return root


# Same chord as previous slot → small continuity bias (not a lock; weak audio can still change).
CONTINUITY_WITH_PREV_BONUS = 0.02


def pick_chord_with_theory(
	chroma_hist: np.ndarray,
	templates: Dict[str, np.ndarray],
	*,
	key_raw: str | None,
	prev_internal: str | None,
	normalize_vector,
	top_k: int = 16,
	key_bonus_max: float = 0.045,
	prog_bonus_cap: float = 0.04,
	continuity_bonus: float = CONTINUITY_WITH_PREV_BONUS,
) -> Tuple[str, str, float, float, float]:
	"""
	Returns: internal_name, display_label, best_audio_dot, second_best_audio_dot, confidence-like value.

	Scoring (simple, tunable): combined = audio_dot + key_fit_bonus + progression_bonus
	− chroma_single_note_penalty + continuity_with_previous_template_bonus.
	Confidence is derived from separation of *combined* scores, capped when audio_dot is weak
	(`THEORY_WEAK_AUDIO_CAP`). Extended chord templates can simplify to triads when audio lead is tiny.
	"""
	from app.models.chords import _validate_chroma

	chroma_vec = normalize_vector(_validate_chroma(chroma_hist))
	if float(np.linalg.norm(chroma_vec)) < 1e-12:
		return "N", "N", 0.0, 0.0, 0.0

	scored: List[Tuple[str, float]] = []
	for name, template in templates.items():
		if name == "N":
			continue
		tpl = normalize_vector(_validate_chroma(template))
		scored.append((name, float(np.dot(chroma_vec, tpl))))
	scored.sort(key=lambda x: x[1], reverse=True)
	if not scored:
		return "N", "N", 0.0, 0.0, 0.0

	top = scored[: max(6, min(top_k, len(scored)))]
	second_audio = float(top[1][1]) if len(top) > 1 else 0.0
	mel_pen = chroma_single_note_penalty(chroma_hist)
	key_diats = diatonic_template_keys(key_raw)

	combined: List[Tuple[str, float, float]] = []
	for name, aud in top:
		amargin = aud - second_audio
		kb = key_fit_bonus(name, key_diats, audio_margin=amargin, audio_best=aud, max_bonus=key_bonus_max)
		pb = min(prog_bonus_cap, progression_pair_bonus(prev_internal, name, key_raw))
		cont = float(continuity_bonus) if (prev_internal and name == prev_internal) else 0.0
		comb = aud + kb + pb - mel_pen + cont
		combined.append((name, comb, aud))

	combined.sort(key=lambda x: x[1], reverse=True)
	best_name, best_comb, best_aud = combined[0]
	second_comb = combined[1][1] if len(combined) > 1 else 0.0

	extended = _is_seventh_quality(best_name) or _is_color_extension(best_name)
	if extended:
		simple_candidates = [
			x for x in combined if not _is_seventh_quality(x[0]) and not _is_color_extension(x[0])
		]
		if simple_candidates:
			simp_name, simp_comb, simp_aud = simple_candidates[0]
			if best_aud - simp_aud < THEORY_SEVENTH_SIMPLIFY_MARGIN and simp_comb >= best_comb - 0.02:
				best_name, best_comb, best_aud = simp_name, simp_comb, simp_aud

	if best_aud < 0.19:
		return "N", "N", best_aud, second_audio, 0.0

	margin = (best_comb - second_comb) / (abs(best_comb) + 1e-8)
	conf = float(max(0.0, min(1.0, margin)))
	if best_aud < THEORY_WEAK_AUDIO_CAP:
		conf = min(conf, best_aud / THEORY_WEAK_AUDIO_CAP)

	label = format_internal_chord_label(best_name)
	return best_name, label, best_aud, second_audio, conf


def chord_template_combined_candidates_debug(
	chroma_hist: np.ndarray,
	templates: Dict[str, np.ndarray],
	*,
	key_raw: str | None,
	prev_internal: str | None,
	normalize_vector,
	top_k: int = 8,
	key_bonus_max: float = 0.045,
	prog_bonus_cap: float = 0.04,
	continuity_bonus: float = CONTINUITY_WITH_PREV_BONUS,
) -> List[dict]:
	"""
	Debug-only: mirror pick_chord_with_theory scoring without seventh collapse / N cutoff.
	Returns top `top_k` entries by combined score with a short breakdown.
	"""
	from app.models.chords import _validate_chroma

	chroma_vec = normalize_vector(_validate_chroma(chroma_hist))
	out: List[dict] = []
	if float(np.linalg.norm(chroma_vec)) < 1e-12:
		return out

	scored: List[Tuple[str, float]] = []
	for name, template in templates.items():
		if name == "N":
			continue
		tpl = normalize_vector(_validate_chroma(template))
		scored.append((name, float(np.dot(chroma_vec, tpl))))
	scored.sort(key=lambda x: x[1], reverse=True)
	if not scored:
		return out

	second_audio_global = float(scored[1][1]) if len(scored) > 1 else 0.0
	mel_pen = chroma_single_note_penalty(chroma_hist)
	key_diats = diatonic_template_keys(key_raw)

	combined_rows: List[Tuple[str, float, float, float, float, float, float]] = []
	for name, aud in scored:
		amargin = aud - second_audio_global
		kb = key_fit_bonus(name, key_diats, audio_margin=amargin, audio_best=aud, max_bonus=key_bonus_max)
		pb = min(prog_bonus_cap, progression_pair_bonus(prev_internal, name, key_raw))
		cont = float(continuity_bonus) if (prev_internal and name == prev_internal) else 0.0
		comb = aud + kb + pb - mel_pen + cont
		combined_rows.append((name, comb, aud, kb, pb, mel_pen, cont))

	combined_rows.sort(key=lambda x: x[1], reverse=True)
	for row in combined_rows[:top_k]:
		name, comb, aud, kb, pb, mel_pen_v, cont = row
		out.append(
			{
				"internal": name,
				"label": format_internal_chord_label(name),
				"combined": round(float(comb), 6),
				"audio_dot": round(float(aud), 6),
				"key_bonus": round(float(kb), 6),
				"prog_bonus": round(float(pb), 6),
				"melody_penalty": round(float(mel_pen_v), 6),
				"continuity_bonus": round(float(cont), 6),
			},
		)
	return out


def likely_passing_segment(
	dur_sec: float,
	label: str,
	prev_label: str,
	next_label: str,
	confidence: float,
	max_dur: float = 0.42,
	conf_hi: float = 0.34,
) -> bool:
	if dur_sec > max_dur or label == "N":
		return False
	if prev_label == "N" or next_label == "N":
		return False
	if label == prev_label or label == next_label:
		return False
	if prev_label != next_label:
		return False
	if confidence > conf_hi:
		return False
	return True


__all__ = [
	"PITCH_NAMES_SHARP",
	"blend_chroma_mean_max",
	"chroma_single_note_penalty",
	"diatonic_template_keys",
	"likely_passing_segment",
	"parse_key_raw",
	"pick_chord_with_theory",
	"progression_pair_bonus",
	"roman_numeral_for_chord_in_key",
	"format_internal_chord_label",
]
