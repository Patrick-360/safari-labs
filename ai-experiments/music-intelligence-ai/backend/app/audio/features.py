from __future__ import annotations

import io
from typing import Any, List, Tuple

import librosa
import numpy as np
import soundfile as sf


CHROMA_BINS = 12
EPS = 1e-8

# Mono float waveform ~[-1, 1] after decode/resample; RMS below this → silence gate (stream chord path).
# Slightly conservative: short mic chunks can sit below 1e-3 at normal speaking/music distance.
SILENCE_RMS_THRESHOLD = 4e-4


def _validate_vector(vector: np.ndarray, size: int) -> np.ndarray:
	if vector is None:
		raise ValueError("Input vector is None.")
	array = np.asarray(vector, dtype=float).reshape(-1)
	if array.size != size:
		raise ValueError(f"Expected {size} elements, got {array.size}.")
	return array


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
	norm = float(np.linalg.norm(vector))
	if norm == 0.0:
		return vector.copy()
	return vector / (norm + EPS)


def load_audio_bytes_wav(audio_bytes: bytes, sr: int = 22050) -> Tuple[np.ndarray, int]:
	if not audio_bytes:
		raise ValueError("audio_bytes is empty.")

	with io.BytesIO(audio_bytes) as buffer:
		y, native_sr = sf.read(buffer, dtype="float32", always_2d=False)

	if y.ndim > 1:
		y = np.mean(y, axis=1)

	if native_sr != sr:
		y = librosa.resample(y, orig_sr=native_sr, target_sr=sr)
		native_sr = sr

	return y.astype(np.float32, copy=False), native_sr


def waveform_rms(y: np.ndarray) -> float:
	"""Root mean square of a 1-D float waveform (linear scale)."""
	array = np.asarray(y, dtype=float).reshape(-1)
	if array.size == 0:
		return 0.0
	return float(np.sqrt(np.mean(np.square(array))))


def waveform_peak_abs(y: np.ndarray) -> float:
	"""Peak absolute sample (linear); pairs with RMS for crest / level gates."""
	array = np.asarray(y, dtype=float).reshape(-1)
	if array.size == 0:
		return 0.0
	return float(np.max(np.abs(array)))


def chroma_hist_entropy_bits(chroma_hist: np.ndarray) -> float:
	"""
	Shannon entropy (natural log) of normalized non-negative chroma mass.
	Uniform spread ≈ ln(12) ≈ 2.485; one sharp chord is lower (more peaked).
	"""
	v = np.maximum(np.asarray(chroma_hist, dtype=float).reshape(-1), 0.0)
	s = float(np.sum(v))
	if s < 1e-12:
		return 0.0
	p = v / s
	return float(-np.sum(p * np.log(p + 1e-12)))


def chroma_temporal_stability_mean_cos(chroma: np.ndarray) -> float:
	"""Mean cosine similarity between adjacent chroma frames (12 x T). Higher = steadier / less transient."""
	c = np.asarray(chroma, dtype=float)
	if c.ndim != 2 or c.shape[0] != CHROMA_BINS:
		raise ValueError("Chroma must have shape (12, T).")
	if c.shape[1] < 2:
		return 1.0
	col_norm = np.linalg.norm(c, axis=0, keepdims=True) + EPS
	n = c / col_norm
	sims: list[float] = []
	for i in range(n.shape[1] - 1):
		sims.append(float(np.dot(n[:, i], n[:, i + 1])))
	return float(np.mean(sims)) if sims else 1.0


def count_strong_chroma_bins(chroma_hist: np.ndarray, threshold: float = 0.2) -> int:
	"""Count L2-normalized chroma bins above threshold; single-speech-pitch often has ~1 strong bin."""
	h = _normalize_vector(_validate_vector(chroma_hist, CHROMA_BINS))
	return int(np.sum(h > threshold))


def extract_chroma_cqt(y: np.ndarray, sr: int, use_hpss: bool = False) -> np.ndarray:
	if y is None:
		raise ValueError("Waveform is None.")
	if y.size == 0:
		raise ValueError("Waveform is empty.")

	waveform = np.asarray(y, dtype=float)

	if use_hpss:
		# Wider harmonic margin (aligned with /analyze HPSS) → steadier triad chroma for live mic.
		harmonic, _ = librosa.effects.hpss(waveform, margin=(2.75, 2.0))
		waveform = harmonic

	chroma = librosa.feature.chroma_cqt(
		y=waveform,
		sr=sr,
		hop_length=512,
		norm=2,
		threshold=0.025,
	)
	if chroma.shape[0] != CHROMA_BINS:
		raise ValueError(f"Expected {CHROMA_BINS} chroma bins, got {chroma.shape[0]}.")
	return chroma


def aggregate_chroma(chroma: np.ndarray) -> np.ndarray:
	if chroma is None:
		raise ValueError("Chroma is None.")
	array = np.asarray(chroma, dtype=float)
	if array.ndim != 2 or array.shape[0] != CHROMA_BINS:
		raise ValueError("Chroma must have shape (12, T).")

	chroma_hist = np.mean(array, axis=1)
	return _normalize_vector(chroma_hist)


def estimate_key(chroma_hist: np.ndarray) -> Tuple[str, float]:
	hist = _normalize_vector(_validate_vector(chroma_hist, CHROMA_BINS))

	major_profile = np.array(
		[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
		dtype=float,
	)
	minor_profile = np.array(
		[6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
		dtype=float,
	)

	major_profile = _normalize_vector(major_profile)
	minor_profile = _normalize_vector(minor_profile)

	pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

	scores = []
	labels = []

	for i, name in enumerate(pitch_classes):
		scores.append(float(np.dot(hist, np.roll(major_profile, i))))
		labels.append(f"{name}:maj")

		scores.append(float(np.dot(hist, np.roll(minor_profile, i))))
		labels.append(f"{name}:min")

	scores_array = np.array(scores, dtype=float)
	best_index = int(np.argmax(scores_array))

	sorted_scores = np.sort(scores_array)[::-1]
	best_score = float(sorted_scores[0])
	second_score = float(sorted_scores[1]) if sorted_scores.size > 1 else float("-inf")

	margin = best_score - second_score
	confidence = max(0.0, min(1.0, margin / (abs(best_score) + EPS)))

	return labels[best_index], confidence


def key_ranked_candidates(chroma_hist: np.ndarray, top_k: int = 8) -> List[dict[str, Any]]:
	"""
	Debug-only: ranked Krumhansl key candidates (same dot scores as estimate_key).
	Each entry: raw label, profile_dot, margin_to_next (among sorted candidates).
	"""
	hist = _normalize_vector(_validate_vector(chroma_hist, CHROMA_BINS))

	major_profile = np.array(
		[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
		dtype=float,
	)
	minor_profile = np.array(
		[6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
		dtype=float,
	)

	major_profile = _normalize_vector(major_profile)
	minor_profile = _normalize_vector(minor_profile)

	pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

	scores: list[float] = []
	labels: list[str] = []

	for i, name in enumerate(pitch_classes):
		scores.append(float(np.dot(hist, np.roll(major_profile, i))))
		labels.append(f"{name}:maj")

		scores.append(float(np.dot(hist, np.roll(minor_profile, i))))
		labels.append(f"{name}:min")

	pairs = sorted(zip(scores, labels, strict=True), key=lambda x: x[0], reverse=True)
	k = max(1, min(int(top_k), len(pairs)))
	out: List[dict[str, Any]] = []
	for i in range(k):
		s_i, lab = pairs[i]
		s_next = pairs[i + 1][0] if i + 1 < len(pairs) else float("-inf")
		margin = float(s_i - s_next) if np.isfinite(s_next) else float(s_i)
		conf = max(0.0, min(1.0, margin / (abs(float(s_i)) + EPS)))
		out.append(
			{
				"raw": lab,
				"profile_dot": round(float(s_i), 6),
				"margin_to_next": round(float(margin), 6),
				"margin_confidence": round(float(conf), 4),
			},
		)
	return out


def score_key_profile_fit(chroma_hist: np.ndarray, key_label_raw: str) -> float:
	"""
	Dot product of normalized chroma with the Krumhansl profile for a single key,
	e.g. key_label_raw == 'C:maj' or 'A:min'. Same profiles as estimate_key.
	"""
	hist = _normalize_vector(_validate_vector(chroma_hist, CHROMA_BINS))

	major_profile = np.array(
		[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
		dtype=float,
	)
	minor_profile = np.array(
		[6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
		dtype=float,
	)

	major_profile = _normalize_vector(major_profile)
	minor_profile = _normalize_vector(minor_profile)

	pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

	for i, name in enumerate(pitch_classes):
		if key_label_raw == f"{name}:maj":
			return float(np.dot(hist, np.roll(major_profile, i)))
		if key_label_raw == f"{name}:min":
			return float(np.dot(hist, np.roll(minor_profile, i)))

	raise ValueError(f"Unknown key label: {key_label_raw!r}")
