"""
Optional future hook: harmonic-focused waveform before chroma.

MVP uses librosa HPSS inside `analyze_pipeline.extract_chroma_track`.
A heavier separator (e.g. Demucs, open-unmix) could be implemented here and
called from that pipeline without changing chroma template code.

Not wired by default — keep imports and weights out of the cold path.
"""


from __future__ import annotations

import numpy as np


def separate_harmonic_stems(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
	"""
	Placeholder: returns (harmonic, percussive) stems.

	Replace with model-backed separation when available; until then callers should
	continue using `librosa.effects.hpss` directly (as analyze_pipeline does).
	"""
	raise NotImplementedError(
		"source_separation.separate_harmonic_stems is a future extension; use HPSS in analyze_pipeline.",
	)
