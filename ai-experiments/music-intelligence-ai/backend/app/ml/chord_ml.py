"""
Learned chord recognition interface (ML-ready).

Purpose (when enabled later):
    Template chroma is limited on minor/diminished/aug7/altered harmony, rap textures,
    and passing chords. A trained model over mel-CQT/logits can emit richer chord
    dictionaries and sharper boundaries once labeled data exists.

Plug-in points:
    - Convolutional / recurrent models on frame-wise features (export ONNX)
    - Transformer sequence taggers with smoothing / CRFs

Current behavior:
    Empty list preserves the heuristic analyze pipeline verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChordPrediction:
	start: float
	end: float
	label: str
	confidence: float
	source: str


def predict_chords_ml(y: np.ndarray, sr: int, *, enabled: bool = False) -> list[ChordPrediction]:
	"""
	Output ML chord spans (seconds, chord symbol, probability).

	:param y: mono float waveform (often ``StemBundle.other`` harmonics-first path)
	:param sr: sample rate
	:param enabled: when False, skip work and return []
	"""
	if not enabled:
		return []
	_ = np.asarray(y, dtype=float), sr  # placeholders for future checkpoints
	return []
