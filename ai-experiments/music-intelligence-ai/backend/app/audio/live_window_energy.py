"""Shared cheap energy metrics for microphone windows (instant live + rolling transcribe).

Kept independent from ``analyze_pipeline`` so Analyze File thresholds never leak here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import librosa
import numpy as np

from app.audio.features import waveform_peak_abs, waveform_rms
from app.audio.live_thresholds import LiveTranscriptionPreflightPreset


def waveform_non_silent_ratio(y: np.ndarray, peak: float, peak_frac_floor: float) -> float:
	"""Fraction of samples above ``max(peak * frac, epsilon)`` — sparse clicks score low."""
	a = np.abs(np.asarray(y, dtype=float).reshape(-1))
	if a.size == 0:
		return 0.0
	pk = max(float(peak), float(np.max(a)))
	floor_v = max(pk * peak_frac_floor, 1.0e-10)
	return float(np.mean(a > floor_v))


def hpss_harmonic_rms(y: np.ndarray) -> float:
	"""Approximate harmonic series energy (HPSS harmonic stem RMS)."""
	w = np.asarray(y, dtype=float).reshape(-1)
	if w.size == 0:
		return 0.0
	harmonic, _ = librosa.effects.hpss(w, margin=(2.75, 2.0))
	return float(waveform_rms(harmonic))


@dataclass(frozen=True)
class LiveWindowEnergyResult:
	ok: bool
	reason_code: str
	detail: Dict[str, Any]


def evaluate_live_transcription_window(
	y: np.ndarray,
	sr: int,
	preset: LiveTranscriptionPreflightPreset,
	*,
	audio_duration_sec: float,
) -> LiveWindowEnergyResult:
	"""Return ok=False before heavy analysis when the buffer is silence/noise-only."""
	yy = np.asarray(y, dtype=float).reshape(-1)
	rms = float(waveform_rms(yy))
	peak = float(waveform_peak_abs(yy))

	metrics: Dict[str, Any] = {
		"waveform_rms": round(rms, 8),
		"waveform_peak": round(peak, 8),
		"non_silent_ratio": 0.0,
		"hpss_harmonic_rms": 0.0,
	}

	if audio_duration_sec + 1e-9 < preset.min_window_sec_analysis:
		return LiveWindowEnergyResult(False, "waiting_for_more_audio", metrics)

	if rms + 1e-15 < preset.min_window_rms:
		metrics["non_silent_ratio"] = round(waveform_non_silent_ratio(yy, peak, preset.peak_frac_for_floor), 6)
		return LiveWindowEnergyResult(False, "input_too_quiet", metrics)

	if peak + 1e-15 < preset.min_peak_abs:
		metrics["non_silent_ratio"] = round(waveform_non_silent_ratio(yy, peak, preset.peak_frac_for_floor), 6)
		return LiveWindowEnergyResult(False, "input_too_quiet", metrics)

	nsr = waveform_non_silent_ratio(yy, peak, preset.peak_frac_for_floor)
	metrics["non_silent_ratio"] = round(nsr, 6)
	if nsr + 1e-9 < preset.non_silent_ratio_min:
		return LiveWindowEnergyResult(False, "input_too_quiet", metrics)

	h_rms = hpss_harmonic_rms(yy)
	metrics["hpss_harmonic_rms"] = round(h_rms, 8)
	if h_rms + 1e-15 < preset.min_hpss_harmonic_rms:
		return LiveWindowEnergyResult(False, "not_enough_harmonic_signal", metrics)

	return LiveWindowEnergyResult(True, "", metrics)
