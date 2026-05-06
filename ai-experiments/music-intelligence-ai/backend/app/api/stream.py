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
	score_key_profile_fit,
	waveform_rms,
)
from app.models.chords import build_chord_templates

from app.models.chords import _normalize_vector, _validate_chroma


router = APIRouter()

# Single-slot history → output tracks the latest chunk vote (faster chord changes; robustness from gates + LAST_VALID_CHORD).
CHORD_HISTORY: Deque[str] = deque(maxlen=1)

LAST_VALID_KEY: str = ""
LAST_VALID_KEY_CONFIDENCE: float = 0.0
LAST_VALID_CHORD: str = ""
# Consecutive silent chunks before LAST_VALID_CHORD is cleared (~2 s at 0.5 s chunks).
SILENCE_STREAK: int = 0
SILENCE_STREAK_CLEAR = 4

# --- Live key stability (tonal center; chord path stays unchanged) ---
# Chord path uses raw(ish) chunk chroma; key path uses slow EMA + rolling window so one chord ≠ key change.
KEY_CHROMA_EMA: np.ndarray | None = None
# Last N valid (non-silent) normalized chunk chroma vectors — medium-term harmonic context (~6 s at 0.5 s/chunk).
KEY_CHROMA_RING: Deque[np.ndarray] = deque(maxlen=12)
# Internal key as estimate_key label, e.g. "C:maj" (24-way Krumhansl winner space).
LAST_KEY_RAW: str = ""
# Candidate key that must "win" for KEY_WIN_STREAK consecutive comparable chunks before we switch.
KEY_PENDING_RAW: str = ""
KEY_WIN_STREAK: int = 0
# EMA update per valid chunk (lower = slower drift toward latest harmony).
KEY_EMA_ALPHA = 0.065
# Blend for key scoring: weight on EMA vs mean(ring); rest is rolling mean (tonal center, not last chord only).
KEY_CONTEXT_BLEND_EMA = 0.58
# New key must beat the locked key's profile fit by at least this margin (Krumhansl dot-product scale ~0–1).
KEY_SWITCH_MARGIN = 0.068
# Extra margin when the *heard* chord is diatonic in the locked key — blocks I–IV–V from looking like key changes.
KEY_DIATONIC_CHALLENGE_MARGIN = 0.095
# How many consecutive chunks the pending key must lead before replacing LAST_KEY_RAW (~5 s at 0.5 s/chunk).
KEY_SWITCH_CONSECUTIVE = 10
# Boost current-key fit when chord is diatonic in locked key (prefer stable tonal center).
KEY_DIATONIC_INERTIA_BONUS = 0.048

CHORD_TEMPLATES = build_chord_templates()
CONFIDENCE_THRESHOLD = 0.05
# Live mic chunks are short and often quieter than mastered tracks — avoid classifying all room tone as silence.
LIVE_SILENCE_RMS_THRESHOLD = 2.8e-4
# Stricter triad cosine floor: below this, harmonic evidence is too weak (noise / diffuse chroma).
# Live uses short windows (~1 s) — slightly lower than offline full-track so audible input still registers.
MIN_BEST_SCORE_ACCEPT = 0.30
# Very low margin + mediocre best score → ambiguous chunk (near-tied templates, no clear chord).
AMBIGUITY_MARGIN = 0.055
AMBIGUITY_BEST_MAX = 0.52
EPS = 1e-8

# Chord labels from _map_chord_label use sharp roots only; allow enharmonic spellings for diatonic check.
_ROOT_NAME_TO_PC: Dict[str, int] = {
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


def _parse_mapped_chord_root_quality(label: str) -> Tuple[int, str] | None:
	"""Root pitch class and triad quality for live labels like 'C', 'F#m'."""
	if not label or label == "N":
		return None
	s = label.strip()
	if len(s) > 1 and s.endswith("m"):
		root_name = s[:-1]
		qual = "min"
	else:
		root_name = s
		qual = "maj"
	pc = _ROOT_NAME_TO_PC.get(root_name)
	if pc is None:
		return None
	return pc, qual


def _diatonic_triad_set(tonic_pc: int, major_key: bool) -> set[Tuple[int, str]]:
	"""Natural minor triads for minor keys; major-key includes vii° as 'dim'."""
	if major_key:
		spec = [
			(0, "maj"),
			(2, "min"),
			(4, "min"),
			(5, "maj"),
			(7, "maj"),
			(9, "min"),
			(11, "dim"),
		]
	else:
		spec = [
			(0, "min"),
			(2, "dim"),
			(3, "maj"),
			(5, "min"),
			(7, "min"),
			(8, "maj"),
			(10, "maj"),
		]
	return {((tonic_pc + off) % 12, q) for off, q in spec}


def _chord_is_diatonic_in_key(mapped_chord: str, key_raw: str) -> bool:
	if not key_raw or ":" not in key_raw:
		return False
	parsed = _parse_mapped_chord_root_quality(mapped_chord)
	if parsed is None:
		return False
	root_pc, qual = parsed
	tonic_name, mode = key_raw.split(":", 1)
	if mode not in ("maj", "min"):
		return False
	tonic_pc = _ROOT_NAME_TO_PC.get(tonic_name)
	if tonic_pc is None:
		return False
	triads = _diatonic_triad_set(tonic_pc, mode == "maj")
	return (root_pc, qual) in triads


def _format_key_display(key_raw: str) -> str:
	if key_raw.endswith(":maj"):
		return key_raw.replace(":maj", " major")
	return key_raw.replace(":min", " minor")


def _key_context_vector() -> np.ndarray:
	"""
	Vector used only for key: blends slow EMA with mean of recent valid chunks.
	Chord stays on per-chunk chroma; key tracks medium-term tonal center.
	"""
	if KEY_CHROMA_EMA is None:
		raise RuntimeError("KEY_CHROMA_EMA unset in _key_context_vector.")
	if not KEY_CHROMA_RING:
		return KEY_CHROMA_EMA.copy()
	stacked = np.stack(list(KEY_CHROMA_RING), axis=0)
	rolling = _normalize_vector(np.mean(stacked, axis=0))
	ema = KEY_CHROMA_EMA
	blended = KEY_CONTEXT_BLEND_EMA * ema + (1.0 - KEY_CONTEXT_BLEND_EMA) * rolling
	return _normalize_vector(blended)


def _resolve_stable_live_key(chroma_hist: np.ndarray, mapped_chord: str) -> Tuple[str, float]:
	"""
	Tonal center (medium-term): rolling chroma + slow EMA → one context vector → Krumhansl.
	Hysteresis + diatonic inertia: do not swap key just because the 24-way winner flickers on
	another diatonic chord; require sustained margin over the *locked* key, with a stricter bar
	when the current chord still fits the locked key diatonically.
	"""
	global KEY_CHROMA_EMA, LAST_KEY_RAW, KEY_PENDING_RAW, KEY_WIN_STREAK

	h = _normalize_vector(_validate_chroma(chroma_hist))
	if KEY_CHROMA_EMA is None:
		KEY_CHROMA_EMA = h.copy()
	else:
		KEY_CHROMA_EMA = (1.0 - KEY_EMA_ALPHA) * KEY_CHROMA_EMA + KEY_EMA_ALPHA * h
	KEY_CHROMA_RING.append(h.copy())

	kv = _key_context_vector()
	cand_raw, margin_conf = estimate_key(kv)

	if not LAST_KEY_RAW:
		LAST_KEY_RAW = cand_raw
		KEY_PENDING_RAW = ""
		KEY_WIN_STREAK = 0
		return _format_key_display(cand_raw), margin_conf

	fit_curr = score_key_profile_fit(kv, LAST_KEY_RAW)
	fit_cand = score_key_profile_fit(kv, cand_raw)
	diatonic_in_locked = _chord_is_diatonic_in_key(mapped_chord, LAST_KEY_RAW)
	if diatonic_in_locked:
		fit_curr += KEY_DIATONIC_INERTIA_BONUS

	if cand_raw == LAST_KEY_RAW:
		KEY_PENDING_RAW = ""
		KEY_WIN_STREAK = 0
		return _format_key_display(LAST_KEY_RAW), margin_conf

	# Challenger must clear a higher bar if harmony still "belongs" to the locked key on paper.
	required_margin = KEY_SWITCH_MARGIN
	if diatonic_in_locked:
		required_margin = KEY_SWITCH_MARGIN + KEY_DIATONIC_CHALLENGE_MARGIN

	if cand_raw != KEY_PENDING_RAW:
		KEY_PENDING_RAW = cand_raw
		KEY_WIN_STREAK = 1
	elif fit_cand > fit_curr + required_margin:
		KEY_WIN_STREAK += 1
	else:
		KEY_WIN_STREAK = max(0, KEY_WIN_STREAK - 1)

	if KEY_WIN_STREAK >= KEY_SWITCH_CONSECUTIVE:
		LAST_KEY_RAW = cand_raw
		KEY_PENDING_RAW = ""
		KEY_WIN_STREAK = 0
		return _format_key_display(LAST_KEY_RAW), margin_conf

	return _format_key_display(LAST_KEY_RAW), margin_conf


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

	chroma_vec = _normalize_vector(_validate_chroma(chroma_hist))

	for name, template in CHORD_TEMPLATES.items():
		if name == "N":
			continue
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


def _fallback_key() -> Tuple[str, float]:
	"""When the chunk is invalid for key (treated as N): last valid key or empty."""
	if LAST_VALID_KEY:
		return LAST_VALID_KEY, LAST_VALID_KEY_CONFIDENCE
	return "", 0.0


@router.post("/stream")
async def stream_audio(file: UploadFile = File(...)) -> Dict[str, object]:
	global SILENCE_STREAK, LAST_VALID_CHORD, LAST_VALID_KEY, LAST_VALID_KEY_CONFIDENCE
	try:
		audio_bytes = await file.read()
		y, sr = load_audio_bytes_wav(audio_bytes)
		chroma = extract_chroma_cqt(y, sr, use_hpss=True)
		chroma_hist = aggregate_chroma(chroma)
	except Exception as exc:  # noqa: BLE001 - return HTTP 400 for any load/feature errors
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	rms_chunk = float(waveform_rms(y))

	if rms_chunk < LIVE_SILENCE_RMS_THRESHOLD:
		SILENCE_STREAK += 1
		if SILENCE_STREAK >= SILENCE_STREAK_CLEAR:
			LAST_VALID_CHORD = ""
		CHORD_HISTORY.append("N")
		key_label, key_confidence = _fallback_key()
		chord_out = LAST_VALID_CHORD if LAST_VALID_CHORD else "N"
		return {
			"chord": chord_out,
			"confidence": 0.0,
			"key": key_label,
			"key_confidence": float(key_confidence),
			"timestamp": time.time(),
			"debug": {
				"raw_chord": "N",
				"scores_top3": [],
				"waveform_rms": rms_chunk,
				"silence": True,
			},
		}

	SILENCE_STREAK = 0

	best_name, confidence, top3 = _score_chords(chroma_hist)
	mapped = _map_chord_label(best_name)
	best_score = float(top3[0][1]) if top3 else 0.0

	raw_chord = mapped
	reject_original = confidence < CONFIDENCE_THRESHOLD and best_score < 0.5
	reject_weak = best_score < MIN_BEST_SCORE_ACCEPT
	reject_ambiguous = confidence < AMBIGUITY_MARGIN and best_score < AMBIGUITY_BEST_MAX
	if reject_original or reject_weak or reject_ambiguous:
		mapped = "N"

	CHORD_HISTORY.append(mapped)
	smoothed_chord = _majority_vote(CHORD_HISTORY, mapped)

	if mapped != "N":
		LAST_VALID_CHORD = mapped
	if mapped == "N":
		key_label, key_confidence = _fallback_key()
	else:
		key_label, key_confidence = _resolve_stable_live_key(chroma_hist, mapped)
		LAST_VALID_KEY = key_label
		LAST_VALID_KEY_CONFIDENCE = float(key_confidence)

	if mapped == "N" and LAST_VALID_CHORD:
		display_chord = LAST_VALID_CHORD
	else:
		display_chord = smoothed_chord

	return {
		"chord": display_chord,
		"confidence": float(confidence),
		"key": key_label,
		"key_confidence": float(key_confidence),
		"timestamp": time.time(),
		"debug": {
			"raw_chord": raw_chord,
			"scores_top3": [(name, float(score)) for name, score in top3],
			"waveform_rms": rms_chunk,
			"silence": False,
		},
	}
