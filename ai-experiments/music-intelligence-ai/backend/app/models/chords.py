from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


CHROMA_BINS = 12


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


def build_chord_templates() -> Dict[str, np.ndarray]:
	templates: Dict[str, np.ndarray] = {}
	pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

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

	templates["N"] = np.zeros(CHROMA_BINS, dtype=float)
	return templates


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
