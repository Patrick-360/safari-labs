from __future__ import annotations

import io
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf


CHROMA_BINS = 12
EPS = 1e-8


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


def extract_chroma_cqt(y: np.ndarray, sr: int, use_hpss: bool = False) -> np.ndarray:
	if y is None:
		raise ValueError("Waveform is None.")
	if y.size == 0:
		raise ValueError("Waveform is empty.")

	waveform = np.asarray(y, dtype=float)

	if use_hpss:
		harmonic, _ = librosa.effects.hpss(waveform)
		waveform = harmonic

	chroma = librosa.feature.chroma_cqt(y=waveform, sr=sr)
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
