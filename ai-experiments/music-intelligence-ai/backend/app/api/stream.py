"""
Live microphone `/stream` — short WAV chunks, triad templates, heuristic gates only.

Query: `mode` preset — `instrument` (default), `song`, or `debug` (aliases: clean, playback, raw, …).

Testing (manual):
  Valid: sustained piano/guitar chord, clean chord from speakers (stable + harmonic).
  Invalid: silence, speech, claps/taps, rubbing mic, diffuse room noise (should reject or hold).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.audio.features import (
	chroma_hist_entropy_bits,
	chroma_temporal_stability_mean_cos,
	count_strong_chroma_bins,
	estimate_key,
	extract_chroma_cqt,
	load_audio_bytes_wav,
	score_key_profile_fit,
	waveform_peak_abs,
	waveform_rms,
)
from app.audio.live_thresholds import (
	LIVE_ROUTE_INSTANT_LIVE,
	SEMANTIC_INSTANT_LIVE_CLEAN,
	SEMANTIC_INSTANT_LIVE_DEBUG,
	SEMANTIC_INSTANT_LIVE_SONG,
	canon_stream_rejection,
)
from app.audio.music_theory import blend_chroma_mean_max
from app.audio.live_window_energy import hpss_harmonic_rms, waveform_non_silent_ratio
from app.models.chords import build_chord_templates

from app.models.chords import _normalize_vector, _validate_chroma


router = APIRouter()

LAST_VALID_KEY: str = ""
LAST_VALID_KEY_CONFIDENCE: float = 0.0
LAST_VALID_CHORD: str = ""
SILENCE_STREAK: int = 0
# Sub-threshold RMS/peak bursts (noise floor, not deep silence): clear held harmony quickly.
TOO_QUIET_STREAK: int = 0

KEY_CHROMA_EMA: np.ndarray | None = None
KEY_CHROMA_RING: list[np.ndarray] = []
LAST_KEY_RAW: str = ""
KEY_PENDING_RAW: str = ""
KEY_WIN_STREAK: int = 0
KEY_EMA_ALPHA = 0.065
KEY_CONTEXT_BLEND_EMA = 0.58
KEY_SWITCH_MARGIN = 0.068
KEY_DIATONIC_CHALLENGE_MARGIN = 0.095
KEY_SWITCH_CONSECUTIVE = 12
KEY_DIATONIC_INERTIA_BONUS = 0.048
KEY_RECENT_CHORD_INERTIA_MAX = 0.048

CHORD_TEMPLATES = build_chord_templates()


@dataclass(frozen=True)
class LiveStreamSensPreset:
	"""
	Instant Live `/stream` threshold bundle — **not** used by Analyze File.

	Microphone chunks lack the temporal context `/analyze` has; RMS + HPSS harmonic gates stay
	per-preset below. See ``app/audio/live_thresholds.py`` module doc for the split rationale.
	"""

	preset_id: str
	display_name: str
	live_route: str

	silence_rms_threshold: float
	min_signal_rms: float
	min_signal_peak: float
	min_hpss_harmonic_rms: float
	min_non_silent_ratio: float
	peak_frac_for_non_silent: float

	silence_streak_clear: int
	too_quiet_streak_clear: int

	max_crest_ratio: float
	min_best_score_accept: float
	ambiguity_margin: float
	ambiguity_best_max: float
	strong_best_score: float
	strong_margin: float
	max_chroma_entropy: float
	min_chroma_stability: float
	min_strong_chroma_bins: int
	single_note_escape_best_score: float
	weak_confirm_chunks: int
	invalid_streak_clear_display: int
	medium_fast_best_score: float | None
	medium_fast_margin: float | None
	fast_stable_best_score: float | None
	fast_stable_margin: float | None
	fast_stable_chroma_stability: float | None


# Instant Live presets (`LIVE_ROUTE_INSTANT_LIVE`); Analyze File unaffected.
PRESET_CLEAN_INSTRUMENT = LiveStreamSensPreset(
	preset_id="instrument",
	display_name="Clean instrument (strict)",
	live_route=LIVE_ROUTE_INSTANT_LIVE,
	silence_rms_threshold=4.65e-4,
	min_signal_rms=7.1e-4,
	min_signal_peak=1.18e-3,
	min_hpss_harmonic_rms=1.42e-4,
	min_non_silent_ratio=0.05,
	peak_frac_for_non_silent=0.065,
	silence_streak_clear=3,
	too_quiet_streak_clear=2,
	max_crest_ratio=20.8,
	min_best_score_accept=0.375,
	ambiguity_margin=0.064,
	ambiguity_best_max=0.475,
	strong_best_score=0.445,
	strong_margin=0.094,
	max_chroma_entropy=2.04,
	min_chroma_stability=0.665,
	min_strong_chroma_bins=2,
	single_note_escape_best_score=0.445,
	weak_confirm_chunks=2,
	invalid_streak_clear_display=2,
	medium_fast_best_score=None,
	medium_fast_margin=None,
	fast_stable_best_score=0.395,
	fast_stable_margin=0.074,
	fast_stable_chroma_stability=0.725,
)


PRESET_SONG_PLAYBACK = LiveStreamSensPreset(
	preset_id="song",
	display_name="Song playback (experimental — phone/speaker → mic)",
	live_route=LIVE_ROUTE_INSTANT_LIVE,
	silence_rms_threshold=3.15e-4,
	min_signal_rms=5.75e-4,
	min_signal_peak=9.5e-4,
	min_hpss_harmonic_rms=1.22e-4,
	min_non_silent_ratio=0.05,
	peak_frac_for_non_silent=0.052,
	silence_streak_clear=3,
	too_quiet_streak_clear=2,
	max_crest_ratio=24.8,
	min_best_score_accept=0.34,
	ambiguity_margin=0.05,
	ambiguity_best_max=0.522,
	strong_best_score=0.394,
	strong_margin=0.074,
	max_chroma_entropy=2.12,
	min_chroma_stability=0.56,
	min_strong_chroma_bins=2,
	single_note_escape_best_score=0.392,
	weak_confirm_chunks=2,
	invalid_streak_clear_display=3,
	medium_fast_best_score=0.322,
	medium_fast_margin=0.049,
	fast_stable_best_score=None,
	fast_stable_margin=None,
	fast_stable_chroma_stability=None,
)


PRESET_DEBUG_RAW = LiveStreamSensPreset(
	preset_id="debug",
	display_name="Debug / raw (permissive — labeling only)",
	live_route=LIVE_ROUTE_INSTANT_LIVE,
	silence_rms_threshold=9.5e-5,
	min_signal_rms=1.85e-4,
	min_signal_peak=3.05e-4,
	min_hpss_harmonic_rms=5.8e-5,
	min_non_silent_ratio=0.018,
	peak_frac_for_non_silent=0.032,
	silence_streak_clear=4,
	too_quiet_streak_clear=4,
	max_crest_ratio=33.0,
	min_best_score_accept=0.23,
	ambiguity_margin=0.026,
	ambiguity_best_max=0.645,
	strong_best_score=0.31,
	strong_margin=0.042,
	max_chroma_entropy=2.55,
	min_chroma_stability=0.415,
	min_strong_chroma_bins=1,
	single_note_escape_best_score=0.26,
	weak_confirm_chunks=1,
	invalid_streak_clear_display=4,
	medium_fast_best_score=0.27,
	medium_fast_margin=0.036,
	fast_stable_best_score=0.34,
	fast_stable_margin=0.058,
	fast_stable_chroma_stability=0.56,
)

LIVE_STREAM_PRESETS: Dict[str, LiveStreamSensPreset] = {
	PRESET_CLEAN_INSTRUMENT.preset_id: PRESET_CLEAN_INSTRUMENT,
	PRESET_SONG_PLAYBACK.preset_id: PRESET_SONG_PLAYBACK,
	PRESET_DEBUG_RAW.preset_id: PRESET_DEBUG_RAW,
}
_SEMANTIC_BY_STREAM_PRESET = {
	PRESET_CLEAN_INSTRUMENT.preset_id: SEMANTIC_INSTANT_LIVE_CLEAN,
	PRESET_SONG_PLAYBACK.preset_id: SEMANTIC_INSTANT_LIVE_SONG,
	PRESET_DEBUG_RAW.preset_id: SEMANTIC_INSTANT_LIVE_DEBUG,
}


def _normalize_stream_mode(raw: str | None) -> str:
	"""Map query string to preset key; unknown values fall back to instrument (backward compatible)."""
	if not raw:
		return PRESET_CLEAN_INSTRUMENT.preset_id
	v = raw.strip().lower()
	if v in (
		PRESET_CLEAN_INSTRUMENT.preset_id,
		"clean",
		"clean_instrument",
	):
		return PRESET_CLEAN_INSTRUMENT.preset_id
	if v in (
		PRESET_SONG_PLAYBACK.preset_id,
		"playback",
		"speaker",
		"phone",
	):
		return PRESET_SONG_PLAYBACK.preset_id
	if v in (PRESET_DEBUG_RAW.preset_id, "raw"):
		return PRESET_DEBUG_RAW.preset_id
	return PRESET_CLEAN_INSTRUMENT.preset_id


def _get_live_stream_preset(mode: str | None) -> LiveStreamSensPreset:
	key = _normalize_stream_mode(mode)
	return LIVE_STREAM_PRESETS[key]


WEAK_PENDING_LABEL: str = ""
WEAK_PENDING_STREAK: int = 0

# Recent *accepted* chord symbols (mapped labels) for tonal-center inertia — not used for gates.
RECENT_ACCEPTED_CHORDS: list[str] = []

# After this many hard rejects in a row, clear the held chord (client clears UI); limit set per preset.
NO_ACCEPT_STREAK: int = 0

# When the client switches sensitivity preset, drop pending weak state (thresholds changed).
LAST_STREAM_MODE: str = ""

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
EPS = 1e-8


def _preset_tune_debug(p: LiveStreamSensPreset) -> Dict[str, object]:
	"""Compact tuning fields for Live debug / API consumers."""
	out: Dict[str, object] = {
		"live_preset_semantic": _SEMANTIC_BY_STREAM_PRESET.get(p.preset_id, ""),
		"live_route_expected": p.live_route,
		"preset_weak_confirm_chunks": p.weak_confirm_chunks,
		"preset_silence_streak_clear": p.silence_streak_clear,
		"preset_too_quiet_streak_clear": p.too_quiet_streak_clear,
		"preset_invalid_streak_clear": p.invalid_streak_clear_display,
		"preset_strong_best": p.strong_best_score,
		"preset_strong_margin": p.strong_margin,
	}
	if p.medium_fast_best_score is not None and p.medium_fast_margin is not None:
		out["preset_medium_fast_best"] = p.medium_fast_best_score
		out["preset_medium_fast_margin"] = p.medium_fast_margin
	if (
		p.fast_stable_best_score is not None
		and p.fast_stable_margin is not None
		and p.fast_stable_chroma_stability is not None
	):
		out["preset_fast_stable_best"] = p.fast_stable_best_score
		out["preset_fast_stable_margin"] = p.fast_stable_margin
		out["preset_fast_stable_chroma_stability"] = p.fast_stable_chroma_stability
	return out


def _push_recent_accepted_chord(mapped: str) -> None:
	global RECENT_ACCEPTED_CHORDS
	t = str(mapped).strip()
	if not t or t.upper() == "N":
		return
	RECENT_ACCEPTED_CHORDS.append(t)
	if len(RECENT_ACCEPTED_CHORDS) > 14:
		RECENT_ACCEPTED_CHORDS[:] = RECENT_ACCEPTED_CHORDS[-14:]


def _clear_recent_accepted_chords() -> None:
	global RECENT_ACCEPTED_CHORDS
	RECENT_ACCEPTED_CHORDS.clear()


def _recent_chord_key_inertia_bonus(key_raw: str) -> float:
	"""Extra fit on the locked key when recent accepted triads mostly stay diatonic (I–IV–V stability)."""
	if not key_raw or not RECENT_ACCEPTED_CHORDS:
		return 0.0
	n = len(RECENT_ACCEPTED_CHORDS)
	diat = sum(1 for c in RECENT_ACCEPTED_CHORDS if _chord_is_diatonic_in_key(c, key_raw))
	ratio = diat / float(max(1, n))
	if ratio >= 0.88:
		return KEY_RECENT_CHORD_INERTIA_MAX
	if ratio >= 0.66:
		return KEY_RECENT_CHORD_INERTIA_MAX * 0.62
	return 0.0


def _immediate_accept_tier(
	preset: LiveStreamSensPreset,
	best_score: float,
	confidence: float,
	*,
	stability: float,
	strong_bins: int,
) -> tuple[bool, str]:
	"""Return (immediate_accept, commit_kind_hint). commit_kind_hint is diagnostic only when immediate."""
	if best_score >= preset.strong_best_score and confidence >= preset.strong_margin:
		return True, "immediate_strong"
	fs_b = preset.fast_stable_best_score
	fs_m = preset.fast_stable_margin
	fs_s = preset.fast_stable_chroma_stability
	if (
		fs_b is not None
		and fs_m is not None
		and fs_s is not None
		and strong_bins >= 2
		and best_score >= fs_b
		and confidence >= fs_m
		and stability + 1e-9 >= fs_s
	):
		return True, "immediate_fast_stable"
	mb = preset.medium_fast_best_score
	mm = preset.medium_fast_margin
	if mb is not None and mm is not None and best_score >= mb and confidence >= mm:
		return True, "immediate_medium_fast"
	return False, "pending"


def _parse_mapped_chord_root_quality(mapped: str) -> Tuple[int, str] | None:
	"""
	Map /stream chord labels to (pitch_class, quality) for diatonic-in-key checks only.
	Supports e.g. C, Am, F#, Bb, C:maj, A:min; slash chords use the left side; N/empty/unknown -> None.
	"""
	if not mapped or not str(mapped).strip():
		return None
	s = str(mapped).strip()
	if s.upper() == "N":
		return None
	if "/" in s:
		s = s.split("/", 1)[0].strip()
	if ":" in s:
		left, right = s.split(":", 1)
		root_name = left.strip()
		tag_raw = right.strip()
		if not tag_raw:
			return None
		tag_l = tag_raw.lower()
		if tag_raw == "M" or tag_l in ("maj", "major"):
			qual = "maj"
		elif tag_l in ("min", "minor", "m"):
			qual = "min"
		elif tag_l == "dim":
			qual = "dim"
		else:
			return None
		pc = _ROOT_NAME_TO_PC.get(root_name)
		if pc is None:
			return None
		return pc, qual
	if len(s) > 3 and s.endswith("dim"):
		root_name = s[:-3]
		qual = "dim"
	elif len(s) > 1 and s.endswith("m") and not s.endswith("maj"):
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
	global KEY_CHROMA_EMA, KEY_CHROMA_RING
	if KEY_CHROMA_EMA is None:
		raise RuntimeError("KEY_CHROMA_EMA unset in _key_context_vector.")
	if not KEY_CHROMA_RING:
		return KEY_CHROMA_EMA.copy()
	stacked = np.stack(KEY_CHROMA_RING, axis=0)
	rolling = _normalize_vector(np.mean(stacked, axis=0))
	ema = KEY_CHROMA_EMA
	blended = KEY_CONTEXT_BLEND_EMA * ema + (1.0 - KEY_CONTEXT_BLEND_EMA) * rolling
	return _normalize_vector(blended)


def _resolve_stable_live_key(chroma_hist: np.ndarray, mapped_chord: str) -> Tuple[str, float]:
	global KEY_CHROMA_EMA, LAST_KEY_RAW, KEY_PENDING_RAW, KEY_WIN_STREAK, KEY_CHROMA_RING

	h = _normalize_vector(_validate_chroma(chroma_hist))
	if KEY_CHROMA_EMA is None:
		KEY_CHROMA_EMA = h.copy()
	else:
		KEY_CHROMA_EMA = (1.0 - KEY_EMA_ALPHA) * KEY_CHROMA_EMA + KEY_EMA_ALPHA * h
	KEY_CHROMA_RING.append(h.copy())
	if len(KEY_CHROMA_RING) > 12:
		KEY_CHROMA_RING.pop(0)

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
	fit_curr += _recent_chord_key_inertia_bonus(LAST_KEY_RAW)

	if cand_raw == LAST_KEY_RAW:
		KEY_PENDING_RAW = ""
		KEY_WIN_STREAK = 0
		return _format_key_display(LAST_KEY_RAW), margin_conf

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


def _score_chords(chroma_hist: np.ndarray) -> Tuple[str, float, list[tuple[str, float]]]:
	scores: list[tuple[str, float]] = []
	chroma_vec = _normalize_vector(_validate_chroma(chroma_hist))

	for name, template in CHORD_TEMPLATES.items():
		if name == "N":
			continue
		template_vec = _normalize_vector(_validate_chroma(template))
		score = float(np.dot(chroma_vec, template_vec))
		scores.append((name, score))

	scores.sort(key=lambda x: x[1], reverse=True)
	best_name, best_score = scores[0]
	second_score = scores[1][1] if len(scores) > 1 else float("-inf")
	if not np.isfinite(second_score):
		confidence = 1.0
	else:
		confidence = (best_score - second_score) / (abs(best_score) + EPS)
	confidence = float(max(0.0, min(1.0, confidence)))
	top3 = scores[:3]
	return best_name, confidence, top3


def _fallback_key() -> Tuple[str, float]:
	if LAST_VALID_KEY:
		return LAST_VALID_KEY, LAST_VALID_KEY_CONFIDENCE
	return "", 0.0


def _gates_harmonic(
	preset: LiveStreamSensPreset,
	*,
	harmonic_hpss_rms: float,
	peak_chunk: float,
	rms_chunk: float,
	entropy: float,
	stability: float,
	strong_bins: int,
	best_score: float,
	confidence: float,
) -> str | None:
	"""Return internal rejection code or None if hard-gate passed (canonicalized separately for UI/debug)."""
	if harmonic_hpss_rms + 1e-15 < preset.min_hpss_harmonic_rms:
		return "not_harmonic"
	if peak_chunk < preset.min_signal_peak:
		return "weak_signal"
	crest = peak_chunk / (rms_chunk + 1e-10)
	if crest > preset.max_crest_ratio:
		return "transient"
	if entropy > preset.max_chroma_entropy:
		return "not_harmonic"
	if stability < preset.min_chroma_stability:
		return "not_harmonic"
	if strong_bins < preset.min_strong_chroma_bins and best_score < preset.single_note_escape_best_score:
		return "not_harmonic"
	if best_score < preset.min_best_score_accept:
		return "weak_signal"
	if confidence < preset.ambiguity_margin and best_score < preset.ambiguity_best_max:
		return "ambiguous"
	return None


def _stream_response_debug(
	preset: LiveStreamSensPreset,
	*,
	raw_mapped: str,
	final_chord: str,
	rejection_reason_canon: str,
	accepted: bool,
	clear_display: bool,
	key_updated_this_chunk: bool,
	scores_top3: List[Tuple[str, float]],
	rms_chunk: float,
	peak_chunk: float,
	harmonic_hpss_rms: float,
	non_silent_ratio: float | None,
	best_score: float,
	second_score: float,
	confidence: float,
	entropy: float,
	stability: float,
	strong_bins: int,
	silence: bool,
	key_display_source: str,
	instant_key_raw: str | None,
	instant_key_confidence: float | None,
	mode_fields: Dict[str, str],
	chord_commit_kind: str,
	displayed_chord: str | None = None,
) -> Dict[str, object]:
	held_last_valid = (
		(not accepted)
		and bool(final_chord)
		and final_chord != "N"
		and bool(LAST_VALID_CHORD)
		and final_chord == LAST_VALID_CHORD
	)
	ik_conf = None if instant_key_confidence is None else round(float(instant_key_confidence), 4)
	dc = displayed_chord if displayed_chord is not None else final_chord
	nsr_round = None if non_silent_ratio is None else round(float(non_silent_ratio), 5)
	out: Dict[str, object] = {
		"live_route_active": LIVE_ROUTE_INSTANT_LIVE,
		"raw_chord": raw_mapped,
		"final_chord": final_chord,
		"displayed_chord": dc,
		"chord_commit_kind": chord_commit_kind,
		"rejection_reason": rejection_reason_canon,
		"accepted": accepted,
		"key_updated_this_chunk": bool(key_updated_this_chunk),
		"clear_display": clear_display,
		"held_last_valid_chord": held_last_valid,
		"key_display_source": key_display_source,
		"instant_key_raw": instant_key_raw,
		"instant_key_confidence": ik_conf,
		"smoothed_key_raw_internal": LAST_KEY_RAW,
		"scores_top3": [(name, float(score)) for name, score in scores_top3],
		"waveform_rms": rms_chunk,
		"waveform_peak": peak_chunk,
		"harmonic_hpss_rms": round(float(harmonic_hpss_rms), 8),
		"non_silent_ratio": nsr_round,
		"best_score": best_score,
		"second_score": second_score,
		"confidence": float(confidence),
		"chroma_entropy": entropy,
		"chroma_stability": stability,
		"strong_chroma_bins": strong_bins,
		"silence": silence,
		**mode_fields,
		**_preset_tune_debug(preset),
	}
	return out


@router.post("/stream")
async def stream_audio(
	file: UploadFile = File(...),
	mode: str | None = Query(
		default=None,
		description="Live sensitivity preset: instrument (default), song, or debug",
	),
) -> Dict[str, object]:
	global SILENCE_STREAK, LAST_VALID_CHORD, LAST_VALID_KEY, LAST_VALID_KEY_CONFIDENCE
	global WEAK_PENDING_LABEL, WEAK_PENDING_STREAK, NO_ACCEPT_STREAK, LAST_STREAM_MODE, LAST_KEY_RAW
	global TOO_QUIET_STREAK

	preset = _get_live_stream_preset(mode)
	if preset.preset_id != LAST_STREAM_MODE:
		WEAK_PENDING_LABEL = ""
		WEAK_PENDING_STREAK = 0
		NO_ACCEPT_STREAK = 0
		TOO_QUIET_STREAK = 0
		_clear_recent_accepted_chords()
	LAST_STREAM_MODE = preset.preset_id

	mode_fields = {
		"input_mode": preset.preset_id,
		"preset_name": preset.display_name,
		"preset_live_route": preset.live_route,
	}

	try:
		audio_bytes = await file.read()
		y, sr = load_audio_bytes_wav(audio_bytes)
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	rms_chunk = float(waveform_rms(y))
	peak_chunk = float(waveform_peak_abs(y))

	if rms_chunk < preset.silence_rms_threshold:
		SILENCE_STREAK += 1
		TOO_QUIET_STREAK = 0
		WEAK_PENDING_LABEL = ""
		WEAK_PENDING_STREAK = 0
		if SILENCE_STREAK >= preset.silence_streak_clear:
			LAST_VALID_CHORD = ""
			NO_ACCEPT_STREAK = 0
			_clear_recent_accepted_chords()
		clear_display = SILENCE_STREAK >= preset.silence_streak_clear
		chord_out = "N"
		key_label, key_confidence = _fallback_key()
		return {
			"chord": chord_out,
			"confidence": 0.0,
			"key": key_label,
			"key_confidence": float(key_confidence),
			"timestamp": time.time(),
			"debug": _stream_response_debug(
				preset,
				raw_mapped="N",
				final_chord=chord_out,
				rejection_reason_canon=canon_stream_rejection("silence"),
				accepted=False,
				clear_display=clear_display,
				key_updated_this_chunk=False,
				scores_top3=[],
				rms_chunk=rms_chunk,
				peak_chunk=peak_chunk,
				harmonic_hpss_rms=0.0,
				non_silent_ratio=None,
				best_score=0.0,
				second_score=0.0,
				confidence=0.0,
				entropy=0.0,
				stability=0.0,
				strong_bins=0,
				silence=True,
				key_display_source="fallback_last_valid",
				instant_key_raw=None,
				instant_key_confidence=None,
				mode_fields=mode_fields,
				chord_commit_kind="silence",
				displayed_chord=chord_out,
			),
		}

	SILENCE_STREAK = 0

	harms_rms_chunk = hpss_harmonic_rms(y)
	ns_ratio = waveform_non_silent_ratio(y, peak_chunk, preset.peak_frac_for_non_silent)
	too_noise = (
		rms_chunk < preset.min_signal_rms
		or peak_chunk < preset.min_signal_peak
		or ns_ratio + 1e-9 < preset.min_non_silent_ratio
		or harms_rms_chunk + 1e-15 < preset.min_hpss_harmonic_rms
	)

	if too_noise:
		TOO_QUIET_STREAK += 1
		WEAK_PENDING_LABEL = ""
		WEAK_PENDING_STREAK = 0
		if TOO_QUIET_STREAK >= preset.too_quiet_streak_clear:
			LAST_VALID_CHORD = ""
			NO_ACCEPT_STREAK = 0
			_clear_recent_accepted_chords()
		clear_display = TOO_QUIET_STREAK >= preset.too_quiet_streak_clear
		chord_out = "N"
		key_label, key_confidence = _fallback_key()
		return {
			"chord": chord_out,
			"confidence": 0.0,
			"key": key_label,
			"key_confidence": float(key_confidence),
			"timestamp": time.time(),
			"debug": _stream_response_debug(
				preset,
				raw_mapped="N",
				final_chord=chord_out,
				rejection_reason_canon=canon_stream_rejection("too_quiet"),
				accepted=False,
				clear_display=clear_display,
				key_updated_this_chunk=False,
				scores_top3=[],
				rms_chunk=rms_chunk,
				peak_chunk=peak_chunk,
				harmonic_hpss_rms=harms_rms_chunk,
				non_silent_ratio=ns_ratio,
				best_score=0.0,
				second_score=0.0,
				confidence=0.0,
				entropy=0.0,
				stability=0.0,
				strong_bins=0,
				silence=False,
				key_display_source="fallback_last_valid",
				instant_key_raw=None,
				instant_key_confidence=None,
				mode_fields=mode_fields,
				chord_commit_kind="too_quiet",
				displayed_chord=chord_out,
			),
		}

	TOO_QUIET_STREAK = 0

	try:
		chroma = extract_chroma_cqt(y, sr, use_hpss=True)
		chroma_hist = blend_chroma_mean_max(chroma)
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	entropy = float(chroma_hist_entropy_bits(chroma_hist))
	stability = float(chroma_temporal_stability_mean_cos(chroma))
	strong_bins = int(count_strong_chroma_bins(chroma_hist))

	best_name, confidence, top3 = _score_chords(chroma_hist)
	raw_mapped = _map_chord_label(best_name)
	best_score = float(top3[0][1]) if top3 else 0.0
	second_score = float(top3[1][1]) if len(top3) > 1 else 0.0
	instant_key_raw, instant_key_confidence = estimate_key(chroma_hist)

	gate = _gates_harmonic(
		preset,
		harmonic_hpss_rms=harms_rms_chunk,
		peak_chunk=peak_chunk,
		rms_chunk=rms_chunk,
		entropy=entropy,
		stability=stability,
		strong_bins=strong_bins,
		best_score=best_score,
		confidence=confidence,
	)

	if gate is not None:
		WEAK_PENDING_LABEL = ""
		WEAK_PENDING_STREAK = 0
		NO_ACCEPT_STREAK += 1
		commit_gate = "gate_reject"
		chord_out = "N"
		if NO_ACCEPT_STREAK >= preset.invalid_streak_clear_display:
			LAST_VALID_CHORD = ""
			_clear_recent_accepted_chords()
			clear_display = True
			commit_gate = "cleared_invalid_streak"
		else:
			clear_display = False
		key_label, key_confidence = _fallback_key()
		return {
			"chord": chord_out,
			"confidence": float(confidence),
			"key": key_label,
			"key_confidence": float(key_confidence),
			"timestamp": time.time(),
			"debug": _stream_response_debug(
				preset,
				raw_mapped=raw_mapped,
				final_chord=chord_out,
				rejection_reason_canon=canon_stream_rejection(gate),
				accepted=False,
				clear_display=clear_display,
				key_updated_this_chunk=False,
				scores_top3=top3,
				rms_chunk=rms_chunk,
				peak_chunk=peak_chunk,
				harmonic_hpss_rms=harms_rms_chunk,
				non_silent_ratio=ns_ratio,
				best_score=best_score,
				second_score=second_score,
				confidence=float(confidence),
				entropy=entropy,
				stability=stability,
				strong_bins=strong_bins,
				silence=False,
				key_display_source="fallback_last_valid",
				instant_key_raw=instant_key_raw,
				instant_key_confidence=instant_key_confidence,
				mode_fields=mode_fields,
				chord_commit_kind=commit_gate,
				displayed_chord=chord_out,
			),
		}

	is_immediate, immediate_kind = _immediate_accept_tier(
		preset,
		best_score,
		confidence,
		stability=stability,
		strong_bins=strong_bins,
	)
	accepted = False
	rejection_raw = "accepted"
	commit_kind = "hold_pending_weak"

	if is_immediate:
		accepted = True
		commit_kind = immediate_kind
		WEAK_PENDING_LABEL = ""
		WEAK_PENDING_STREAK = 0
	else:
		if raw_mapped == WEAK_PENDING_LABEL:
			WEAK_PENDING_STREAK += 1
		else:
			WEAK_PENDING_LABEL = raw_mapped
			WEAK_PENDING_STREAK = 1
		if WEAK_PENDING_STREAK >= preset.weak_confirm_chunks:
			accepted = True
			commit_kind = "confirmed_weak"
			WEAK_PENDING_LABEL = ""
			WEAK_PENDING_STREAK = 0
		else:
			rejection_raw = "pending_weak_confirm"

	if accepted:
		NO_ACCEPT_STREAK = 0
		LAST_VALID_CHORD = raw_mapped
		_push_recent_accepted_chord(raw_mapped)
		key_label, key_confidence = _resolve_stable_live_key(chroma_hist, raw_mapped)
		LAST_VALID_KEY = key_label
		LAST_VALID_KEY_CONFIDENCE = float(key_confidence)
		return {
			"chord": raw_mapped,
			"confidence": float(confidence),
			"key": key_label,
			"key_confidence": float(key_confidence),
			"timestamp": time.time(),
			"debug": _stream_response_debug(
				preset,
				raw_mapped=raw_mapped,
				final_chord=raw_mapped,
				rejection_reason_canon=canon_stream_rejection("accepted"),
				accepted=True,
				clear_display=False,
				key_updated_this_chunk=True,
				scores_top3=top3,
				rms_chunk=rms_chunk,
				peak_chunk=peak_chunk,
				harmonic_hpss_rms=harms_rms_chunk,
				non_silent_ratio=ns_ratio,
				best_score=best_score,
				second_score=second_score,
				confidence=float(confidence),
				entropy=entropy,
				stability=stability,
				strong_bins=strong_bins,
				silence=False,
				key_display_source="smoothed_engine",
				instant_key_raw=instant_key_raw,
				instant_key_confidence=instant_key_confidence,
				mode_fields=mode_fields,
				chord_commit_kind=commit_kind,
				displayed_chord=raw_mapped,
			),
		}

	# Pending weak confirmation: smoother UX—show last-valid triad hint while accumulating evidence.
	NO_ACCEPT_STREAK = 0
	chord_out = LAST_VALID_CHORD if LAST_VALID_CHORD else "N"
	key_label, key_confidence = _fallback_key()
	return {
		"chord": chord_out,
		"confidence": float(confidence),
		"key": key_label,
		"key_confidence": float(key_confidence),
		"timestamp": time.time(),
		"debug": _stream_response_debug(
			preset,
			raw_mapped=raw_mapped,
			final_chord=chord_out,
			rejection_reason_canon=canon_stream_rejection(rejection_raw),
			accepted=False,
			clear_display=False,
			key_updated_this_chunk=False,
			scores_top3=top3,
			rms_chunk=rms_chunk,
			peak_chunk=peak_chunk,
			harmonic_hpss_rms=harms_rms_chunk,
			non_silent_ratio=ns_ratio,
			best_score=best_score,
			second_score=second_score,
			confidence=float(confidence),
			entropy=entropy,
			stability=stability,
			strong_bins=strong_bins,
			silence=False,
			key_display_source="fallback_last_valid",
			instant_key_raw=instant_key_raw,
			instant_key_confidence=instant_key_confidence,
			mode_fields=mode_fields,
			chord_commit_kind=commit_kind,
			displayed_chord=chord_out,
		),
	}
