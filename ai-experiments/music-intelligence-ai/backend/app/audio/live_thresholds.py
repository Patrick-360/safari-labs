"""
Live-input threshold **documentation** vs offline `/analyze`.

**Why this file exists**

- **`/analyze`** (Analyze File) works on buffered, full-context audio: windows can aggregate
  pitch-class evidence over hundreds of milliseconds, tolerate quiet passages in a dense mix,
  and use guarded exotic templates (`analyze_pipeline`). Those knobs stay **only** there.

- **`/stream`** (Instant Live) and **`/live-transcribe`** see **microphone-shaped** chunks:
  room noise, speech, taps, silence, and highly variable RMS. Copying Analyze-style permissive
  gates would hallucinate chords on noise — so live paths use **strict, separate** presets.

This module holds **typed preset bundles** referenced by `/stream` and `/live-transcribe`.
It does **not** import `analyze_pipeline` (avoid circular deps and accidental sharing).
Analyze File tuning remains in ``app/audio/analyze_pipeline.py``.

Public symbols are small dataclasses + dict lookups so ``stream.py`` and ``live_transcribe``
can share semantics without coupling to file analysis.

**Preset semantics (avoid crossing wires with Analyze File)**

- ``instant_live_clean`` → ``stream`` preset_id ``instrument`` (`LiveStreamSensPreset`) — handheld
  melodic instrument close to mic; narrow crest / harmonic floors.
- ``instant_live_song`` → ``stream`` preset_id ``song`` — phone/speaker bleed; softer RMS floors but
  still **stricter than /analyze** so silence and room hiss do not mint triads every chunk.
- ``instant_live_debug`` → ``stream`` preset_id ``debug`` — permissive **dev only**.
- ``live_transcription`` → ``LiveTranscriptionPreflightPreset`` (this module only) enforced in
  ``evaluate_live_transcription_window`` **before** ``run_analysis``; never weakens Analyze File presets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# Semantic aliases documented for API reviewers (stream.py resolves mode query → preset objects).
SEMANTIC_INSTANT_LIVE_CLEAN: Final[str] = "instant_live_clean"  # stream: instrument/clean
SEMANTIC_INSTANT_LIVE_SONG: Final[str] = "instant_live_song"  # stream: song/playback
SEMANTIC_INSTANT_LIVE_DEBUG: Final[str] = "instant_live_debug"  # stream: debug/raw
SEMANTIC_LIVE_TRANSCRIPTION: Final[str] = "live_transcription"  # /live-transcribe preflight only

# --- Mode identifiers (API / debug stable strings) ---
# Analyze File knobs live exclusively in analyze_pipeline — never referenced by live presets below.
ANALYZE_FILE_THRESHOLD_MODULE = "app.audio.analyze_pipeline"
LIVE_ROUTE_ANALYZE_FILE: Final[str] = "analyze_file"
LIVE_ROUTE_INSTANT_LIVE: Final[str] = "instant_live"
LIVE_ROUTE_LIVE_TRANSCRIPTION: Final[str] = "live_transcription"


@dataclass(frozen=True)
class LiveTranscriptionPreflightPreset:
	"""
	Energy gates before running `run_analysis` on rolling mic buffers.

	If any check fails, the server returns ``status=listening`` and **does not** call the
	off-line pipeline — avoiding fake chords from nearly-silent/noise windows while keeping
	the Analyze pipeline unchanged when it *does* run.
	"""

	preset_id: str
	min_window_rms: float
	min_peak_abs: float
	"""Samples above noise floor counted as ``active``; scaled from peak to avoid brittle fixed dB."""
	non_silent_ratio_min: float
	peak_frac_for_floor: float
	min_hpss_harmonic_rms: float
	"""Cheap HPSS harmonic stem RMS — speech/taps often fail here vs sustained harmony."""
	min_window_sec_analysis: float
	"""Match ``run_analysis`` short-audio cutoff (~0.2s)."""


# Song mode defaults; still tighter than Analyze File gates — rolling mic silence must not enqueue analysis.
PRESET_LT_SONG_DEFAULT = LiveTranscriptionPreflightPreset(
	preset_id="song_default",
	min_window_rms=3.45e-4,
	min_peak_abs=9.25e-4,
	non_silent_ratio_min=0.092,
	peak_frac_for_floor=0.05,
	min_hpss_harmonic_rms=1.42e-4,
	min_window_sec_analysis=0.2,
)


LIVE_TRANSCRIPTION_PRESETS: dict[str, LiveTranscriptionPreflightPreset] = {
	PRESET_LT_SONG_DEFAULT.preset_id: PRESET_LT_SONG_DEFAULT,
}


def live_transcription_preset(mode: str | None) -> LiveTranscriptionPreflightPreset:
	"""Rolling-window transcription currently uses one conservative bundle; ``mode`` reserved."""
	_ = (mode or "").strip().lower()
	return PRESET_LT_SONG_DEFAULT


def canon_stream_rejection(internal: str) -> str:
	"""
	Map internal gate codes to stable debug/UI strings.

	Target vocabulary: silence | too_quiet | weak_harmony | ambiguous | transient_noise | accepted
	"""

	m = internal.strip().lower()
	if m in ("accepted", ""):
		return "accepted"
	if m == "silence":
		return "silence"
	if m == "too_quiet":
		return "too_quiet"
	if m in ("weak_signal", "not_harmonic", "weak_harmonics", "harmonic_weak"):
		return "weak_harmony"
	if m == "ambiguous":
		return "ambiguous"
	if m in ("transient", "transient_noise"):
		return "transient_noise"
	if m.startswith("pending"):
		return "ambiguous"
	return internal
