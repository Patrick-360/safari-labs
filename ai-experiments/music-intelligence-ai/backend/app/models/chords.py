from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


CHROMA_BINS = 12
# Seventh partial is slightly down-weighted in templates so a lone melodic seventh
# does not routinely beat the triad on noisy / vocal-heavy HPSS chroma (heuristic, not robust separation).
SEVENTH_BIN_WEIGHT = 0.72


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
	norm = float(np.linalg.norm(vector))
	if norm == 0.0:
		return vector.copy()
	return vector / norm


def _validate_chroma(vector: np.ndarray) -> np.ndarray:
	if vector is None:
		raise ValueError("Input vector is None.")

	array = np.asarray(vector, dtype=float).reshape(-1)
	if array.size != CHROMA_BINS:
		raise ValueError(f"Expected {CHROMA_BINS} elements, got {array.size}.")
	return array


def build_chord_templates(
	*,
	include_sevenths: bool = False,
	include_extended: bool = False,
) -> Dict[str, np.ndarray]:
	"""
	Chord prototypes as sparse chroma vectors (not a full chord-recognition model).

	include_sevenths: dom7 / maj7 / min7 (+ /analyze).
	include_extended: dim, aug, sus2, sus4, half-diminished m7b5 (+ /analyze when True).
	Live /stream uses defaults (triads only) so behavior stays predictable.
	"""
	templates: Dict[str, np.ndarray] = {}
	pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
	w7 = float(SEVENTH_BIN_WEIGHT)
	w_part = 0.88

	for root_index, root_name in enumerate(pitch_classes):
		major = np.zeros(CHROMA_BINS, dtype=float)
		minor = np.zeros(CHROMA_BINS, dtype=float)

		major_intervals = [0, 4, 7]
		minor_intervals = [0, 3, 7]

		for interval in major_intervals:
			major[(root_index + interval) % CHROMA_BINS] = 1.0
		for interval in minor_intervals:
			minor[(root_index + interval) % CHROMA_BINS] = 1.0

		templates[f"{root_name}:maj"] = major
		templates[f"{root_name}:min"] = minor

		if include_extended:
			dim = np.zeros(CHROMA_BINS, dtype=float)
			for interval in (0, 3, 6):
				dim[(root_index + interval) % CHROMA_BINS] = 1.0
			aug = np.zeros(CHROMA_BINS, dtype=float)
			for interval in (0, 4, 8):
				aug[(root_index + interval) % CHROMA_BINS] = 1.0
			sus2 = np.zeros(CHROMA_BINS, dtype=float)
			for interval in (0, 2, 7):
				sus2[(root_index + interval) % CHROMA_BINS] = 1.0
			sus4 = np.zeros(CHROMA_BINS, dtype=float)
			for interval in (0, 5, 7):
				sus4[(root_index + interval) % CHROMA_BINS] = 1.0
			m7b5 = np.zeros(CHROMA_BINS, dtype=float)
			for interval, weight in ((0, 1.0), (3, 1.0), (6, w_part), (10, w7)):
				m7b5[(root_index + interval) % CHROMA_BINS] = weight
			templates[f"{root_name}:dim"] = dim
			templates[f"{root_name}:aug"] = aug
			templates[f"{root_name}:sus2"] = sus2
			templates[f"{root_name}:sus4"] = sus4
			templates[f"{root_name}:m7b5"] = m7b5

		if include_sevenths:
			dom7 = np.zeros(CHROMA_BINS, dtype=float)
			maj7 = np.zeros(CHROMA_BINS, dtype=float)
			min7 = np.zeros(CHROMA_BINS, dtype=float)
			for interval, weight in ((0, 1.0), (4, 1.0), (7, 1.0), (10, w7)):
				dom7[(root_index + interval) % CHROMA_BINS] = weight
			for interval, weight in ((0, 1.0), (4, 1.0), (7, 1.0), (11, w7)):
				maj7[(root_index + interval) % CHROMA_BINS] = weight
			for interval, weight in ((0, 1.0), (3, 1.0), (7, 1.0), (10, w7)):
				min7[(root_index + interval) % CHROMA_BINS] = weight
			templates[f"{root_name}:7"] = dom7
			templates[f"{root_name}:maj7"] = maj7
			templates[f"{root_name}:min7"] = min7

	templates["N"] = np.zeros(CHROMA_BINS, dtype=float)
	return templates


def build_analyze_mvp_templates() -> Dict[str, np.ndarray]:
	"""
	Major/minor triads + N only for /analyze.

	Dim/sus were easy to trigger from vocal-heavy or sparse chroma; a small reliable
	vocabulary beats a larger wrong one. (Live /stream keeps its own template set.)
	"""
	return build_chord_templates(include_sevenths=False, include_extended=False)


def build_analyze_heuristic_templates() -> Dict[str, np.ndarray]:
	"""
	/analyze sliding-window heuristic set: majors, minors + dim / aug / sus2 / sus4 + N.

	Seventh chords and half-dim m7b5 stay out — extra ambiguity without clearer evidence.
	File-mode scoring gates exotic qualities with conservative thresholds before falling back
	to triadic families (`build_analyze_mvp_templates` behavior when gates fail).
	"""
	full = build_chord_templates(include_sevenths=False, include_extended=True)
	return {k: v for k, v in full.items() if k == "N" or not str(k).endswith(":m7b5")}


def build_analyze_theory_templates() -> Dict[str, np.ndarray]:
	"""
	Theory-enhanced /analyze set: maj/min + guarded dom7 / maj7 / min7 + dim / aug / sus + N.

	Sevenths use stricter scoring gates than triads; m7b5 stays out (too easy to mislabel).
	"""
	full = build_chord_templates(include_sevenths=True, include_extended=True)
	return {k: v for k, v in full.items() if k == "N" or not str(k).endswith(":m7b5")}


def chord_score(chroma: np.ndarray, template: np.ndarray) -> float:
	chroma_vec = _normalize_vector(_validate_chroma(chroma))
	template_vec = _normalize_vector(_validate_chroma(template))
	return float(np.dot(chroma_vec, template_vec))


def best_chord(
    chroma: np.ndarray,
    templates: Dict[str, np.ndarray],
) -> Tuple[str, float, float, float]:
    """
    Returns:
      best_name: chord label with highest score
      confidence: margin confidence in [0, 1]-ish (higher = more certain)
      best_score: top score
      second_score: second-best score
    """
    if not templates:
        raise ValueError("Templates dictionary is empty.")

    chroma_vec = _normalize_vector(_validate_chroma(chroma))

    # Score every chord
    scored: list[tuple[str, float]] = []
    for name, template in templates.items():
        template_vec = _normalize_vector(_validate_chroma(template))
        score = float(np.dot(chroma_vec, template_vec))
        scored.append((name, score))

    # Sort by score (descending)
    scored.sort(key=lambda x: x[1], reverse=True)

    best_name, best_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else float("-inf")

    # Margin confidence: how much better is best than second-best?
    eps = 1e-8
    if not np.isfinite(second_score):
        confidence = 1.0
    else:
        confidence = (best_score - second_score) / (abs(best_score) + eps)

    # Clamp to [0, 1] to keep it clean
    confidence = float(max(0.0, min(1.0, confidence)))

    return best_name, confidence, best_score, second_score
