"""
Stem separation adapter for preprocessing before harmony / melody analysis.

When a neural separator (Demucs-class) is available, accompaniment-focused stems
reduce vocal/drum leakage into chord chroma. Until then, callers always receive a
deterministic fallback that mirrors today's full-mix behaviour.

Important:
    • No automatic model downloads — wire ``get_model`` / checkpoints only behind
      explicit ops controls when you integrate Demucs.
    • Torch + demucs stay optional dependencies; import failures never crash ``/analyze``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StemBundle:
	"""Per-stem mono float waveforms (~same length as input, same ``sr``)."""

	vocals: np.ndarray | None
	drums: np.ndarray | None
	bass: np.ndarray | None
	other: np.ndarray
	full_mix: np.ndarray


@dataclass(frozen=True)
class SeparationResult:
	"""Outcome of ``separate_sources`` — always safe to consume (fallback fills ``other``)."""

	stems: StemBundle
	separation_requested: bool
	"""True when isolated stems materially changed accompaniment for chords (future: Demucs success)."""
	separation_used: bool
	warning: str | None
	stem_available: dict[str, bool]
	backend: str
	meta: dict[str, Any]


def _passthrough_fallback(
	audio: np.ndarray,
	*,
	separation_requested: bool,
	reason_code: str,
	warning_msg: str | None,
	backend_label: str,
	meta: dict[str, Any] | None = None,
) -> SeparationResult:
	bundle = StemBundle(vocals=None, drums=None, bass=None, other=audio, full_mix=audio)
	tag = dict(meta or {})
	tag.setdefault("fallback_reason_code", reason_code)
	return SeparationResult(
		stems=bundle,
		separation_requested=bool(separation_requested),
		separation_used=False,
		warning=warning_msg,
		stem_available={
			"vocals": False,
			"drums": False,
			"bass": False,
			# ``other`` is always present — passthrough duplicates full mix
			"other": True,
			"full_mix": True,
		},
		backend=backend_label,
		meta=tag,
	)


def _optional_neural_separation(audio: np.ndarray, sr: int) -> SeparationResult:
	"""
	Demucs / successor integration point (**lazy-import only**).

	When wired:
	  - Load checkpoints from an explicit path (no silent hub download).
	  - ``apply_model`` → fill ``StemBundle``; ``other`` becomes **vocals-drums-attenuated** accompaniment.
	  - Return ``SeparationResult(... separation_used=True, ...)``.

	Currently: verifies optional deps briefly, returns safe fallback messages.
	"""
	import importlib.util

	if importlib.util.find_spec("torch") is None:
		return _passthrough_fallback(
			audio,
			separation_requested=True,
			reason_code="torch_not_installed",
			warning_msg="Source separation requested but PyTorch is not installed — using full mix.",
			backend_label="fallback_torch_missing",
		)
	if importlib.util.find_spec("demucs") is None:
		return _passthrough_fallback(
			audio,
			separation_requested=True,
			reason_code="demucs_not_installed",
			warning_msg="Source separation requested but demucs is not installed — using full mix.",
			backend_label="fallback_demucs_missing",
		)

	# torch + demucs importable — forward pass still off until weights/policy are hooked up
	return _passthrough_fallback(
		audio,
		separation_requested=True,
		reason_code="demucs_stub_no_inference_yet",
		warning_msg=(
			"Demucs tooling is partially available but inference is not enabled "
			"in this build — using full mix."
		),
		backend_label="fallback_demucs_stub",
		meta={
			"hint": (
				"Implement apply_model/get_model behind explicit checkpoint paths inside "
				"_optional_neural_separation — avoid silent weight downloads."
			),
			"sr": int(sr),
		},
	)


def separate_sources(y: np.ndarray, sr: int, *, enabled: bool = False) -> SeparationResult:
	"""
	Split mixture into semantic stems when ``enabled`` and optional engines succeed.

	:param enabled: caller intent — False returns immediate passthrough (zero extra cost).
	"""
	audio = np.asarray(y, dtype=np.float32).reshape(-1)
	if audio.size == 0:
		raise ValueError("separate_sources: empty waveform")

	if not enabled:
		return _passthrough_fallback(
			audio,
			separation_requested=False,
			reason_code="disabled",
			warning_msg=None,
			backend_label="disabled",
		)
	return _optional_neural_separation(audio, sr)
