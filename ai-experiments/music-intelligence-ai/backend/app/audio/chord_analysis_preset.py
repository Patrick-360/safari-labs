"""
Presets for Analyze File chord engines (`engine` query on POST /analyze).

Keeps a single implementation in ``analyze_pipeline``; presets only vary vocabulary,
smoothing, bass/arpeggio features, and exotic-gate strictness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

VocabularyKind = Literal["maj_min", "maj_min_dim_aug_sus"]

# Default matches current production path (theory-enhanced heuristics).
DEFAULT_CHORD_ENGINE = "theory"


@dataclass(frozen=True)
class ChordAnalysisPreset:
	"""Serializable tuning bundle selected by ``engine``."""

	engine: str
	display_name: str
	vocabulary: VocabularyKind
	use_bass_chroma: bool
	use_arpeggio_aggregate: bool
	"""Blend mean/max + temporal max for staggered chord tones; stable uses classic blend only."""
	chroma_time_smooth_frames: int
	arpeggio_conf_scale: float
	bass_template_dot_bonus: float
	"""Subtract from exotic quality floors (mc_min, margins). 0 disables relief."""
	exotic_threshold_relief: float
	"""Theory: dom7 / maj7 / min7 with guarded gates; stable/experimental omit."""
	include_sevenths: bool = False
	"""Stronger vocal hold + lower conf on sparse chroma (Theory only)."""
	vocal_resistance: bool = False
	"""Sticky hysteresis multiplier when vocal slots are flagged."""
	vocal_sticky_margin_mult: float = 1.28
	# --- Per-preset confidence calibration ---
	# Theory's 7th-chord templates add 36 extra competing templates → smaller margins everywhere.
	# These thresholds must be calibrated to the effective margin scale of each template set.
	"""Raw margin floor for sticky hysteresis to allow a chord change (best-second)/best."""
	sticky_min_raw_margin: float = 0.026
	"""Confidence below which a segment is flagged low_confidence."""
	low_conf_cutoff: float = 0.18
	"""Confidence below which a short (<0.5 s) blip is snapped to the previous chord."""
	snap_conf_threshold: float = 0.26
	"""Max wall-clock seconds sticky may hold one chord before forcing a release. 0 = unlimited."""
	max_sticky_hold_sec: float = 0.0
	"""After a forced cap release, hold the new chord for this many seconds so refine cannot snap it away.
	Uses majority-vote on the look-ahead window to pick the best non-prev chord. 0 = single-slot release."""
	sticky_forced_window_sec: float = 0.0
	"""After refine_chord_timeline, split any segment longer than this using pre-refine boundaries.
	0 = disabled. Theory: use 36.0 as safety net after forced window already limits single-chord dominance."""
	max_returned_segment_sec: float = 0.0


_CHORD_PRESETS: dict[str, ChordAnalysisPreset] = {
	"stable": ChordAnalysisPreset(
		engine="stable",
		display_name="Stable",
		vocabulary="maj_min",
		use_bass_chroma=False,
		use_arpeggio_aggregate=False,
		chroma_time_smooth_frames=5,
		arpeggio_conf_scale=0.0,
		bass_template_dot_bonus=0.0,
		exotic_threshold_relief=0.0,
	),
	"theory": ChordAnalysisPreset(
		engine="theory",
		display_name="Theory enhanced",
		vocabulary="maj_min_dim_aug_sus",
		use_bass_chroma=True,
		use_arpeggio_aggregate=True,
		chroma_time_smooth_frames=3,
		arpeggio_conf_scale=0.088,
		bass_template_dot_bonus=0.026,
		exotic_threshold_relief=0.0,
		include_sevenths=True,
		# 7th templates (36 extra) reduce best-vs-second margins. Calibrate thresholds down so
		# real chord changes pass the sticky gate and correctly-detected chords survive the snap.
		sticky_min_raw_margin=0.012,
		low_conf_cutoff=0.12,
		snap_conf_threshold=0.14,
		# Cap how long sticky may hold one chord: prevents single chord dominating a full song
		# when all frames have margins below the floor (common with 109-template set).
		# 30s avoids splitting genuine 25-30s chord holds while still catching 192s pathologies.
		max_sticky_hold_sec=30.0,
		# After cap release, force this many seconds of the majority non-prev chord so the
		# released segment is long enough to survive refine's snap/collapse steps (> 0.5s).
		sticky_forced_window_sec=6.0,
		# Safety-net guardrail: if any final segment is still > 36s, split using pre-refine.
		max_returned_segment_sec=36.0,
	),
	# Experimental: same vocabulary as theory but looser exotic gates + slightly stronger arpeggio/bass nudges.
	"experimental": ChordAnalysisPreset(
		engine="experimental",
		display_name="Experimental",
		vocabulary="maj_min_dim_aug_sus",
		use_bass_chroma=True,
		use_arpeggio_aggregate=True,
		chroma_time_smooth_frames=3,
		arpeggio_conf_scale=0.102,
		bass_template_dot_bonus=0.034,
		exotic_threshold_relief=0.016,
		# No 7ths, but exotic templates still add competition versus stable's triad-only set.
		sticky_min_raw_margin=0.020,
		low_conf_cutoff=0.15,
		snap_conf_threshold=0.22,
	),
}


def normalize_chord_engine(raw: str | None) -> ChordAnalysisPreset:
	"""Map client string to preset; unknown → theory (backward compatible)."""
	if not raw:
		return _CHORD_PRESETS[DEFAULT_CHORD_ENGINE]
	key = str(raw).strip().lower()
	return _CHORD_PRESETS.get(key, _CHORD_PRESETS[DEFAULT_CHORD_ENGINE])


def preset_for_debug(p: ChordAnalysisPreset) -> dict:
	"""Flatten for JSON debug (explicit numbers for A/B compares)."""

	return {
		"engine": p.engine,
		"display_name": p.display_name,
		"vocabulary": p.vocabulary,
		"use_bass_chroma": p.use_bass_chroma,
		"use_arpeggio_aggregate": p.use_arpeggio_aggregate,
		"chroma_time_smooth_frames": p.chroma_time_smooth_frames,
		"arpeggio_conf_scale": p.arpeggio_conf_scale,
		"bass_template_dot_bonus": p.bass_template_dot_bonus,
		"exotic_threshold_relief": p.exotic_threshold_relief,
		"include_sevenths": p.include_sevenths,
		"vocal_resistance": p.vocal_resistance,
		"vocal_sticky_margin_mult": p.vocal_sticky_margin_mult,
		"sticky_min_raw_margin": p.sticky_min_raw_margin,
		"low_conf_cutoff": p.low_conf_cutoff,
		"snap_conf_threshold": p.snap_conf_threshold,
		"max_sticky_hold_sec": p.max_sticky_hold_sec,
		"sticky_forced_window_sec": p.sticky_forced_window_sec,
		"max_returned_segment_sec": p.max_returned_segment_sec,
	}


def chord_vocab_description(vocabulary: VocabularyKind, *, include_sevenths: bool = False) -> str:
	if vocabulary == "maj_min":
		return "Major and minor triads + N only (dim/aug/sus omitted)."
	if include_sevenths:
		return "Major, minor, guarded dom7 / maj7 / min7, dim / aug / sus2 / sus4 + N."
	return "Major, minor, guarded dim / aug / sus2 / sus4 + N."

