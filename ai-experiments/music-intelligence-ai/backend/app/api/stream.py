from __future__ import annotations

import time
from collections import Counter, deque
from typing import Deque, Dict, List, Tuple

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.audio.features import (
	aggregate_chroma,
	estimate_key,
	extract_chroma_cqt,
	load_audio_bytes_wav,
)
from app.models.chords import build_chord_templates, chord_score

from app.models.chords import _normalize_vector, _validate_chroma


router = APIRouter()

CHORD_HISTORY: Deque[str] = deque(maxlen=2)
CHROMA_HISTORY: Deque[np.ndarray] = deque(maxlen=2)

CHORD_TEMPLATES = build_chord_templates()
CONFIDENCE_THRESHOLD = 0.05
EPS = 1e-8


def _map_chord_label(label: str) -> str:
	if label == "N":
		return "N"
	root, quality = label.split(":", 1)
	return root if quality == "maj" else f"{root}m"


def _majority_vote(items: Deque[str], fallback: str) -> str:
	if not items:
		return fallback
	counts = Counter(items)
	top_count = max(counts.values())
	candidates = [item for item, count in counts.items() if count == top_count]
	return fallback if fallback in candidates else candidates[0]


def _score_chords(chroma_hist: np.ndarray) -> Tuple[str, float, list[tuple[str, float]]]:
	"""
	Returns:
	  best_name: highest-scoring chord label (e.g. 'A:min' or 'C:maj')
	  confidence: margin confidence based on best vs second-best score
	  top3: list of top-3 (name, score) pairs for debugging
	"""
	# CHORD_TEMPLATES should be a dict[str, np.ndarray] created once (12-dim templates)
	# If your code uses a different name, keep it consistent.
	scores: list[tuple[str, float]] = []

	for name, template in CHORD_TEMPLATES.items():
		# Make sure both are normalized 12-d vectors
		chroma_vec = _normalize_vector(_validate_chroma(chroma_hist))
		template_vec = _normalize_vector(_validate_chroma(template))
		score = float(np.dot(chroma_vec, template_vec))
		scores.append((name, score))

	# Sort by score descending
	scores.sort(key=lambda x: x[1], reverse=True)

	best_name, best_score = scores[0]
	second_score = scores[1][1] if len(scores) > 1 else float("-inf")

	# Margin confidence: best vs second-best
	eps = 1e-8
	if not np.isfinite(second_score):
		confidence = 1.0
	else:
		confidence = (best_score - second_score) / (abs(best_score) + eps)

	# Clamp to [0, 1]
	confidence = float(max(0.0, min(1.0, confidence)))

	top3 = scores[:3]
	return best_name, confidence, top3


def _smooth_key(chroma_hist: np.ndarray) -> Tuple[str, float]:
	CHROMA_HISTORY.append(chroma_hist)
	stacked = np.stack(list(CHROMA_HISTORY), axis=0)
	smoothed = np.mean(stacked, axis=0)
	key_label, key_confidence = estimate_key(smoothed)
	if key_label.endswith(":maj"):
		return key_label.replace(":maj", " major"), key_confidence
	return key_label.replace(":min", " minor"), key_confidence


@router.post("/stream")
async def stream_audio(file: UploadFile = File(...)) -> Dict[str, object]:
	try:
		audio_bytes = await file.read()
		y, sr = load_audio_bytes_wav(audio_bytes)
		chroma = extract_chroma_cqt(y, sr, use_hpss=True)
		chroma_hist = aggregate_chroma(chroma)
	except Exception as exc:  # noqa: BLE001 - return HTTP 400 for any load/feature errors
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	best_name, confidence, top3 = _score_chords(chroma_hist)
	mapped = _map_chord_label(best_name)
	best_score = float(top3[0][1]) if top3 else 0.0

	raw_chord = mapped
	if confidence < CONFIDENCE_THRESHOLD and best_score < 0.5:
		mapped = "N"

	CHORD_HISTORY.append(mapped)
	smoothed_chord = _majority_vote(CHORD_HISTORY, mapped)

	key_label, key_confidence = _smooth_key(chroma_hist)

	return {
		"chord": smoothed_chord,
		"confidence": float(confidence),
		"key": key_label,
		"key_confidence": float(key_confidence),
		"timestamp": time.time(),
		"debug": {
			"raw_chord": raw_chord,
			"scores_top3": [(name, float(score)) for name, score in top3],
		},
	}
