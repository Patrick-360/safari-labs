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
	}


def chord_vocab_description(vocabulary: VocabularyKind, *, include_sevenths: bool = False) -> str:
	if vocabulary == "maj_min":
		return "Major and minor triads + N only (dim/aug/sus omitted)."
	if include_sevenths:
		return "Major, minor, guarded dom7 / maj7 / min7, dim / aug / sus2 / sus4 + N."
	return "Major, minor, guarded dim / aug / sus2 / sus4 + N."

