"""
Pitch / note transcription interface (ML-ready).

Purpose (when enabled later):
    Note-level timelines help with melody transcription, bass line extraction,
    arpeggiated patterns (where chord chroma averages poorly), and auditable overlays
    on top of aggregate chord guesses.

Plug-in points:
    - Spotify Basic Pitch (lightweight ONNX / torch — optional dependency later)
    - MT3 and similar MIR transformers (heavy; opt-in installs)

Current behavior:
    Returns an empty list so callers do not branch on absent infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoteEvent:
	start: float
	end: float
	pitch: float  # MIDI pitch number preferred
	note: str
	confidence: float
	source: str


def transcribe_pitch(y: np.ndarray, sr: int, *, enabled: bool = False) -> list[NoteEvent]:
	"""
	Lift pitches from ``y`` into timed note events.

	:param y: mono float waveform (often ``StemBundle.other`` after separation)
	:param sr: sample rate
	:param enabled: when False, skip work and return []
	"""
	if not enabled:
		return []
	_ = np.asarray(y, dtype=float), sr  # placeholders for future backends
	return []
