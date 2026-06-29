"""
Offline /analyze pipeline (heuristic — not ML transcription):

1. **Load audio**: mono @ ANALYSIS_SR via librosa.
2. **Harmonic extraction**: HPSS harmonic stem feeds chord chroma; percussive energy is not used for harmony
   (rhythm uses the full waveform only for beat_track elsewhere).
   Future: optional `source_separation.separate_harmonic_stems()` could replace / blend this step.
3. **Time–frequency**: CQT chroma on harmonic stem + light temporal smoothing; per-slot histogram blends
   mean/max over time plus per-class maxima so staggered chord tones still vote (non-ML arpeggio cue).
   A low-register CQT chroma track supplies optional **bass_root_hint** (small template bonus only).
4. **Beats**: librosa.beat_track on full `y` → boundaries for sections / optional timing snap (not mandatory chord grid).
5. **Chord candidates**: cosine vs **triads + guarded dim/aug/sus2/sus4** templates for file analyze; exotic
   qualities need strong score + margin vs major/minor or we fall back; sqrt compression still tames single-bin vocals.
6. **Timeline**: overlapping sliding windows (short sec-tuned window + hop) → light median + sticky hysteresis (vocal-aware; strong margins can override) → boundaries shifted slightly earlier + harmonic cusp refine; beat snap only when already near-grid.
7. **Key**: whole-track Krumhansl estimate (global); biases scoring softly, never hard-forces.
8. **Sections**: beat- or time-grid chroma similarity merge + repetition fingerprint.
9. **Core progression** (frontend): derived from merged runs; excludes passing / low-confidence per client helpers.

**ML hooks** (defaults off in ``app/core/config.py``):
    Future Demucs/source separation + note transcription + chord classifiers integrate via
    ``app/ml/*.py``. ``run_analysis`` calls those interfaces so fusion can be layered in later
    without reshaping the heuristic core; empty fallbacks intentionally preserve today's output.

Heuristic limitations (vocals, rap, dense mixes, etc.) are unchanged until those hooks are fused.
"""

from __future__ import annotations

import io
import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np

from app.audio.chord_analysis_preset import (
	ChordAnalysisPreset,
	chord_vocab_description,
	normalize_chord_engine,
	preset_for_debug,
)
from app.audio.chord_spellings import playable_triad_notes_and_hint
from app.audio.features import (
	CHROMA_BINS,
	aggregate_chroma,
	chroma_hist_entropy_bits,
	count_strong_chroma_bins,
	estimate_key,
	key_ranked_candidates,
)
from app.audio.music_theory import (
	PITCH_NAMES_SHARP,
	blend_chroma_mean_max,
	chord_template_combined_candidates_debug,
	format_internal_chord_label,
	likely_passing_segment,
	pick_chord_with_theory,
)
from app.models.chords import (
	_normalize_vector,
	_validate_chroma,
	build_analyze_heuristic_templates,
	build_analyze_mvp_templates,
	build_analyze_theory_templates,
)
from app.audio.simplify_progression import compute_simple_practice_progression
from app.core.config import (
	BETA_ANALYSIS_DURATION_SEC,
	BETA_ANALYSIS_SAMPLE_RATE,
	BETA_MAX_UPLOAD_SIZE_MB,
	ENABLE_ML_CHORDS,
	ENABLE_PITCH_TRANSCRIPTION,
	ENABLE_SOURCE_SEPARATION,
)
from app.ml import StemBundle, predict_chords_ml, separate_sources, transcribe_pitch

log = logging.getLogger(__name__)

# Chroma / segmentation (same hop as librosa CQT frames in this module).
HOP_LENGTH = 512
ANALYSIS_SR = 22050

# --- /analyze chord path (single tuning block) ---
# Shorter analysis window + hop → boundaries track harmony changes sooner (offline / median still smooths single-frame noise).
# Previous long median (5 slots × ~55 ms hop) plus sticky hysteresis noticeably delayed visible changes.
CHORD_WINDOW_SEC = 0.16
CHORD_HOP_SEC = 0.042
# Odd median width: 3 clears isolated slot spikes with less temporal blur than 5.
CHORD_LABEL_MEDIAN_SLOTS = 3
# Lighter per-frame chroma smoothing (fewer CQT frames ~23 ms apart at 22050/512 hop).
CHROMA_TIME_SMOOTH = 3
# Refine passes: lower floor so we mostly merge only very brief flutter, not brief real changes.
MIN_STABLE_REGION_SEC = 0.18
# Single-PC / vocal-heavy chroma gates (L2-normalized peak share = max bin size).
CHROMA_VOCAL_PEAK_RATIO = 0.52
CHROMA_VOCAL_PEAK_RATIO_LOOSE = 0.48
CHROMA_VOCAL_ENTROPY_MAX = 1.24
CHROMA_VOCAL_MAX_STRONG_BINS = 2
CHROMA_TRIAD_COVER_MIN = 0.54
VOCALChord_SWITCH_MIN_SCORE = 0.52
VOCALChord_SWITCH_MIN_MARGIN = 0.038
VOCALChord_SWITCH_MIN_STRONG_BINS = 2
# If true chord change is clear, allow switch even when vocal_heuristic fired.
VOCAL_STRONG_SWITCH_MIN_BS = 0.56
VOCAL_STRONG_SWITCH_MIN_MARGIN = 0.10
CHORD_ANALYZE_MIN_AUDIO_DOT = 0.165
# Drop quiet CQT bins before chroma aggregation (reduces transient / broad-band bleed — heuristic gate).
CHROMA_CQT_THRESHOLD = 0.025
# Key diatonic bias moved to music_theory.key_fit_bonus.
STICKY_MIN_BEST_SCORE = 0.39
STICKY_MIN_RAW_MARGIN = 0.026
STICKY_CONF_CAP = 0.22
VOCAL_STICKY_MARGIN_MULT = 1.28
# Fallback when chroma is very short.
SEGMENT_SECONDS = 0.35
# Merge adjacent beat- or time-windows when mean-chroma cosine similarity >= this (fewer, longer sections).
SECTION_MERGE_COS = 0.66
# Match distant section bodies to same "Section A" when chroma prototypes agree (or chroma+chord pattern agree).
SECTION_REPEAT_SIM_THRESHOLD = 0.82
# If chroma alone is marginal, still repeat-label when chord fingerprints align.
SECTION_REPEAT_CHROMA_LO = 0.76
SECTION_REPEAT_FP_SIM = 0.58
# Absorb only very short segments (vocal flutter / clicks), not quarter-note level changes at moderate tempos.
MIN_CHORD_SEGMENT_SEC = 0.20
# Replace isolated one-beat-wonder labels shorter than this when sandwiched.
SPIKE_ISOLATE_MAX_SEC = 0.40
# Low-confidence segment shorter than this whose chroma matches previous chord → keep previous (melody note).
SNAP_WEAK_MAX_SEC = 0.50
SNAP_WEAK_TO_PREV_CHROMA_SIM = 0.88
# Also snap short segments when marginal confidence is very low even if low_confidence was not pre-flagged.
SNAP_EXTRA_CONF_THRESHOLD = 0.26
# Merge adjacent *different* labels only when aggregate chroma is almost identical (avoid killing real changes).
CHORD_NEIGHBOR_MERGE_COS = 0.92
# Below this margin confidence, flag segment as low_confidence (honest UI).
CHORD_LOW_CONF_CUTOFF = 0.18
# Below this template cosine, cap confidence (weak evidence).
CHORD_WEAK_SCORE_CAP = 0.55
# After beat-based sectioning, merge sections shorter than this into neighbors (practice chunks, not fragments).
MIN_SECTION_DURATION_SEC = 8.0
# If a section stays longer than this after merges, split near beats into practice-sized pieces (~8–16s typical).
MAX_SECTION_PREFERRED_SEC = 18.0
# When splitting a long section, aim near this length (then snap to closest beat).
TARGET_PRACTICE_SECTION_SEC = 12.0
# No-beat fallback: coarse time grid for initial intervals (then chroma-merge as usual).
EQUAL_TIME_WINDOW_SEC = 10.0
# Snap segment edges to nearest beat only when boundary and beat already agree (~140 ms).
BEAT_SNAP_MAX_SEC = 0.14
# Search ± this many chroma frames around a boundary for strongest harmonic change.
HARMONIC_CUSP_RADIUS_FRAMES = 8
PASSING_MAX_SEGMENT_SEC = 0.48
# If the same chord label totals this much duration across ≥ PASSING_REPEAT_MIN_COUNT segments,
# suppress passing tags on tiny sandwich slices — likely a real harmony, not a grace passing sonority.
PASSING_REPEAT_MIN_TOTAL_DUR_SEC = 0.92
PASSING_REPEAT_MIN_COUNT = 2
# Small cosine boost when low-register chroma peak matches template root (does not pick quality alone).
BASS_TEMPLATE_DOT_BONUS = 0.026
# Confidence lift when staggered-evidence score is high and triad cover is decent.
ARPEGGIO_CONF_SCALE = 0.088
# Heuristic grouping for practice UI: assume 4/4, first detected beat = bar downbeat (not aligned to real meter).
DEFAULT_BEATS_PER_BAR = 4
# Mean-heavy chroma blend in sliding windows: de-emphasize single-frame melodic peaks vs sustained harmony.
CHROMA_BLEND_W_MEAN = 0.64
CHROMA_BLEND_W_MAX = 0.36
# Single-run segment boundary blend: slightly earlier than midpoint so a new chord appears sooner (reduces felt lag).
CHORD_RUN_BOUNDARY_LEFT_BIAS = 0.58
# Tempo normalization (half/double-time guardrails — ballads often misread as 2× BPM).
TEMPO_PRACTICAL_MIN_BPM = 55.0
TEMPO_PRACTICAL_MAX_BPM = 180.0
TEMPO_HALF_IF_ABOVE_BPM = 160.0
TEMPO_DOUBLE_IF_BELOW_BPM = 45.0
TEMPO_RETRACK_RATIO = 0.12
# Main progression eligibility (backend flags for frontend `exclude_from_core`).
CORE_MIN_SEGMENT_SEC = 0.38
CORE_MIN_CONFIDENCE = 0.20
CORE_ONE_OFF_MAX_SEC = 0.95
CORE_RARE_LABEL_TOTAL_SEC = 0.50


def load_audio_bytes(
	data: bytes,
	sr: int = ANALYSIS_SR,
	duration: float | None = None,
) -> Tuple[np.ndarray, int]:
	"""Load WAV/MP3/etc. via librosa; mono, resampled to `sr`.

	When `duration` is set, decode only the first N seconds.  For WAV/FLAC this is
	seek-efficient (soundfile reads only the needed frames); for MP3 decoding stops
	early via audioread.  Either way the returned array is bounded to `duration` seconds,
	which avoids allocating a full-song numpy buffer for long uploads.
	"""
	if not data:
		raise ValueError("audio_bytes is empty.")
	with io.BytesIO(data) as buf:
		y, _ = librosa.load(buf, sr=sr, mono=True, duration=duration)
	if y.size == 0:
		raise ValueError("Decoded waveform is empty.")
	return y.astype(np.float32, copy=False), sr


def _probe_audio_duration_sec(data: bytes) -> float | None:
	"""Return audio duration from the file header without full decode.

	Uses soundfile.info (supports WAV, FLAC, OGG, AIFF).  Returns None for formats
	that soundfile cannot read (e.g. MP3), which is acceptable — callers treat None
	as 'unknown original duration'.
	"""
	try:
		import soundfile as sf
		with io.BytesIO(data) as buf:
			info = sf.info(buf)
			return float(info.duration)
	except Exception:
		return None


def load_audio_from_path(
	path: str,
	sr: int = ANALYSIS_SR,
	duration: float | None = None,
) -> Tuple[np.ndarray, int]:
	"""Load audio directly from a file path without creating a BytesIO buffer.

	For WAV/FLAC/OGG, soundfile can seek to decode only `duration` seconds without
	reading the full file into RAM.  For MP3, audioread decodes progressively and
	stops at `duration` seconds.  Either way the peak in-memory footprint is bounded
	to the decoded numpy array size, not the compressed file size.
	"""
	y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
	if y.size == 0:
		raise ValueError("Decoded waveform is empty.")
	return y.astype(np.float32, copy=False), sr


def _probe_audio_duration_from_path(path: str) -> float | None:
	"""Return audio duration from a file header without full decode.

	Works for WAV, FLAC, OGG, AIFF via soundfile.  Returns None for MP3 (soundfile
	does not support MP3 directly) — callers treat None as 'unknown'.
	"""
	try:
		import soundfile as sf
		info = sf.info(path)
		return float(info.duration)
	except Exception:
		return None


def _count_label_runs(seq: list) -> int:
	"""Count distinct consecutive runs in a label sequence."""
	n = 0
	prev: str | None = None
	for lbl in seq:
		if lbl != prev:
			n += 1
			prev = lbl
	return n


def _expand_tempo_octave_candidates(values: Iterable[float]) -> List[float]:
	out: set[float] = set()
	for v in values:
		if not np.isfinite(v) or float(v) <= 1e-6:
			continue
		vf = float(v)
		for scale in (1.0, 0.5, 2.0):
			c = vf * scale
			if TEMPO_PRACTICAL_MIN_BPM <= c <= TEMPO_PRACTICAL_MAX_BPM:
				out.add(round(c, 2))
	return sorted(out)


def _normalize_tempo_from_candidates(raw_bpm: float, onset_candidates: List[float]) -> Tuple[float, str]:
	if not np.isfinite(raw_bpm) or raw_bpm <= 0:
		if onset_candidates:
			med = float(np.median(onset_candidates))
			if TEMPO_PRACTICAL_MIN_BPM <= med <= TEMPO_PRACTICAL_MAX_BPM:
				return med, "invalid_raw_used_onset_median"
		return 120.0, "invalid_raw_default_120"

	raw = float(raw_bpm)
	if raw > TEMPO_HALF_IF_ABOVE_BPM:
		half = raw / 2.0
		if TEMPO_PRACTICAL_MIN_BPM <= half <= TEMPO_PRACTICAL_MAX_BPM:
			near = any(abs(c - half) <= 5.0 for c in onset_candidates)
			if near or raw >= 168.0:
				return half, f"halved_fast_tempo_{raw:.1f}_to_{half:.1f}"

	if raw < TEMPO_DOUBLE_IF_BELOW_BPM:
		dbl = raw * 2.0
		if TEMPO_PRACTICAL_MIN_BPM <= dbl <= TEMPO_PRACTICAL_MAX_BPM:
			near = any(abs(c - dbl) <= 5.0 for c in onset_candidates)
			if near or raw <= 38.0:
				return dbl, f"doubled_slow_tempo_{raw:.1f}_to_{dbl:.1f}"

	if TEMPO_PRACTICAL_MIN_BPM <= raw <= TEMPO_PRACTICAL_MAX_BPM:
		return raw, "beat_track_in_practical_range"

	pool = _expand_tempo_octave_candidates([raw, *onset_candidates])
	if pool:
		best = min(
			pool,
			key=lambda c: min(abs(c - raw), abs(c - raw / 2.0), abs(c - raw * 2.0)),
		)
		if abs(best - raw) > 0.5:
			return best, f"octave_snap_{raw:.1f}_to_{best:.1f}"
		return best, "selected_practical_candidate"

	clamped = float(np.clip(raw, TEMPO_PRACTICAL_MIN_BPM, TEMPO_PRACTICAL_MAX_BPM))
	return clamped, "clamped_to_practical_range"


def estimate_tempo_and_beats(y: np.ndarray, sr: int) -> Tuple[float, List[float], Dict[str, Any]]:
	"""
	Global tempo (BPM) + beat times (seconds) using librosa.beat.beat_track.

	Normalizes half/double-time misreads into a practical 55–180 BPM window for ballads and
	up-tempo tracks. Returns debug metadata alongside normalized BPM.
	"""
	onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
	tempo_arr, beat_frames = librosa.beat.beat_track(
		y=y,
		sr=sr,
		hop_length=HOP_LENGTH,
		onset_envelope=onset_env,
	)
	arr = np.asarray(tempo_arr, dtype=float).reshape(-1)
	raw_bpm = float(arr[0]) if arr.size > 0 and np.isfinite(arr[0]) else 120.0

	onset_cands: List[float] = []
	try:
		t_multi = librosa.feature.tempo(
			onset_envelope=onset_env,
			sr=sr,
			hop_length=HOP_LENGTH,
			aggregate=None,
		)
		onset_cands.extend(
			float(x)
			for x in np.asarray(t_multi).reshape(-1)
			if np.isfinite(x) and float(x) > 0.0
		)
	except TypeError:
		pass

	t_med = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH, aggregate=np.median)
	tmed_a = np.asarray(t_med, dtype=float).reshape(-1)
	if tmed_a.size > 0 and np.isfinite(tmed_a[0]):
		onset_cands.append(float(tmed_a[0]))

	if not np.isfinite(raw_bpm) or raw_bpm < 30.0 or raw_bpm > 320.0:
		if onset_cands:
			raw_bpm = float(np.median(onset_cands))
		else:
			raw_bpm = 120.0

	norm_bpm, reason = _normalize_tempo_from_candidates(raw_bpm, onset_cands)

	if abs(norm_bpm - raw_bpm) / max(raw_bpm, 1e-9) > TEMPO_RETRACK_RATIO:
		try:
			_, beat_frames = librosa.beat.beat_track(
				y=y,
				sr=sr,
				hop_length=HOP_LENGTH,
				onset_envelope=onset_env,
				start_bpm=norm_bpm,
				bpm=norm_bpm,
			)
		except TypeError:
			_, beat_frames = librosa.beat.beat_track(
				y=y,
				sr=sr,
				hop_length=HOP_LENGTH,
				onset_envelope=onset_env,
				start_bpm=norm_bpm,
			)

	beat_times_arr = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
	beat_list = [round(float(t), 4) for t in np.asarray(beat_times_arr).reshape(-1)]
	meta: Dict[str, Any] = {
		"raw_tempo_bpm": round(raw_bpm, 2),
		"normalized_tempo_bpm": round(norm_bpm, 2),
		"tempo_reason": reason,
		"tempo_candidates": [round(c, 2) for c in sorted(set(onset_cands))[:12]],
	}
	return norm_bpm, beat_list, meta


def extract_chroma_track(y: np.ndarray, sr: int, use_hpss: bool = True) -> np.ndarray:
	"""
	Shape (12, T_frames). Prefer harmonic HPSS stem for steadier tonal features vs full mix.

	Future: optional `app.audio.source_separation.separate_harmonic_stems` could feed chroma
	instead of HPSS (not enabled in MVP).
	"""
	waveform = np.asarray(y, dtype=float)
	if use_hpss:
		# Stronger harmonic emphasis vs percussive residual (approximate; not Demucs).
		harmonic, _ = librosa.effects.hpss(waveform, margin=(3.05, 2.12))
		waveform = harmonic
	chroma = librosa.feature.chroma_cqt(
		y=waveform,
		sr=sr,
		hop_length=HOP_LENGTH,
		norm=2,
		threshold=float(CHROMA_CQT_THRESHOLD),
	)
	if chroma.shape[0] != CHROMA_BINS:
		raise ValueError(f"Expected {CHROMA_BINS} chroma bins, got {chroma.shape[0]}.")
	return chroma


def extract_bass_chroma_track(y: np.ndarray, sr: int, use_hpss: bool = True) -> np.ndarray:
	"""
	Low-register CQT chroma (same hop_length as main chord chroma) for bass/root hints only.
	"""
	waveform = np.asarray(y, dtype=float)
	if use_hpss:
		harmonic, _ = librosa.effects.hpss(waveform, margin=(3.05, 2.12))
		waveform = harmonic
	bass = librosa.feature.chroma_cqt(
		y=waveform,
		sr=sr,
		hop_length=HOP_LENGTH,
		norm=2,
		threshold=float(CHROMA_CQT_THRESHOLD),
		fmin=librosa.note_to_hz("C1"),
		n_octaves=5,
	)
	if bass.shape[0] != CHROMA_BINS:
		raise ValueError(f"Expected {CHROMA_BINS} bass chroma bins, got {bass.shape[0]}.")
	return bass


def _align_bass_chroma_to_track(bass_cc: np.ndarray, t_frames: int) -> np.ndarray | None:
	if bass_cc is None:
		return None
	bf = int(bass_cc.shape[1])
	if bf >= t_frames:
		return bass_cc[:, :t_frames].astype(np.float32, copy=False)
	if bf <= 0:
		return None
	pad = t_frames - bf
	out = np.pad(bass_cc.astype(float), ((0, 0), (0, pad)), mode="edge").astype(np.float32, copy=False)
	return out


def _temporal_smooth_chroma(chroma: np.ndarray, width: int) -> np.ndarray:
	"""Light temporal smoothing per pitch class; weakens short vocal/melody peaks vs sustained harmony."""
	if width <= 1 or chroma.shape[1] < 2:
		return chroma
	w = max(3, int(width))
	if w % 2 == 0:
		w += 1
	pad = w // 2
	padded = np.pad(chroma.astype(float), ((0, 0), (pad, pad)), mode="edge")
	out = np.zeros_like(chroma, dtype=float)
	for i in range(chroma.shape[1]):
		out[:, i] = np.mean(padded[:, i : i + w], axis=1)
	norms = np.linalg.norm(out, axis=0, keepdims=True) + 1e-12
	return (out / norms).astype(np.float32, copy=False)


def _chroma_hist_for_matching(chroma_hist: np.ndarray) -> np.ndarray:
	"""
	Sqrt compress non-negative chroma mass before L2 norm — tames one-bin vocal dominance a bit
	while keeping triad structure (heuristic, not robust source separation).
	"""
	v = np.maximum(np.asarray(chroma_hist, dtype=float).reshape(-1), 0.0)
	return np.sqrt(v)


def _aggregate_window_chroma_arpeggio(win_slice: np.ndarray) -> Tuple[np.ndarray, float]:
	"""
	Compress frame-wise chroma into a 12-bin histogram for cosine templates + temporal spread cue.

	arpeggio_support ≈ high when chord tones peek at different frames (spread) rather than collapsing
	to a sustained single-vector snapshot.
	"""
	if win_slice.size == 0 or win_slice.shape[1] < 1:
		h = aggregate_chroma(win_slice)
		return h, 0.0
	blended = blend_chroma_mean_max(win_slice, w_mean=CHROMA_BLEND_W_MEAN, w_max=CHROMA_BLEND_W_MAX)
	pc_mean = np.mean(win_slice.astype(float), axis=1)
	pc_max = np.max(win_slice.astype(float), axis=1)
	hist = 0.35 * blended + 0.32 * pc_mean + 0.33 * pc_max
	me = pc_mean / (np.linalg.norm(pc_mean) + 1e-12)
	xm = pc_max / (np.linalg.norm(pc_max) + 1e-12)
	align = float(np.clip(np.dot(me, xm), 0.0, 1.0))
	spread_bins = float(np.sum(pc_max > pc_mean * 1.38 + 1e-12))
	spread_frac = float(np.clip(spread_bins / 5.25, 0.0, 1.0))
	arp = float(np.clip(0.56 * (1.0 - align) + 0.44 * spread_frac, 0.0, 1.0))
	return hist.astype(np.float32, copy=False), arp


def _blend_histogram_for_analyze_slot(
	win_slice: np.ndarray,
	preset: ChordAnalysisPreset,
) -> Tuple[np.ndarray, float]:
	if win_slice.size == 0 or win_slice.shape[1] < 1:
		h = aggregate_chroma(win_slice)
		return h, 0.0
	if preset.use_arpeggio_aggregate:
		return _aggregate_window_chroma_arpeggio(win_slice)
	blended = blend_chroma_mean_max(win_slice, w_mean=CHROMA_BLEND_W_MEAN, w_max=CHROMA_BLEND_W_MAX)
	return blended.astype(np.float32, copy=False), 0.0


def _bass_root_hint_pc_for_window(b_win: np.ndarray) -> int | None:
	if b_win.shape[1] < 1:
		return None
	v = blend_chroma_mean_max(b_win, w_mean=0.55, w_max=0.45)
	v = np.maximum(np.asarray(v, dtype=float).reshape(-1), 0.0)
	sm = float(np.sum(v))
	if sm <= 1e-15:
		return None
	pc = int(np.argmax(v))
	if float(np.max(v)) / sm < 0.285:
		return None
	return pc


_EXOTIC_QUALITIES = frozenset({"dim", "aug", "sus2", "sus4"})
_SEVENTH_QUALITIES = frozenset({"7", "maj7", "min7"})


def _exotic_quality_thresholds(quality: str) -> Tuple[float, float, float]:
	"""`(min_cosine, gap_vs_best_simple_pair, gap_vs_second_exotic)`."""
	if quality == "aug":
		return (0.546, 0.066, 0.058)
	if quality == "dim":
		return (0.522, 0.050, 0.046)
	if quality == "sus2":
		return (0.510, 0.040, 0.036)
	if quality == "sus4":
		return (0.512, 0.042, 0.036)
	return (1.0, 10.0, 10.0)


def _seventh_quality_thresholds(quality: str) -> Tuple[float, float, float]:
	"""`(min_cosine, gap_vs_best_triad, gap_vs_second_seventh)`."""
	if quality == "7":
		return (0.502, 0.042, 0.032)
	if quality == "maj7":
		return (0.506, 0.044, 0.034)
	if quality == "min7":
		return (0.504, 0.043, 0.033)
	return (1.0, 10.0, 10.0)


def _vocal_gates(preset: ChordAnalysisPreset) -> Dict[str, float]:
	"""Theory uses stricter vocal hold thresholds; stable/experimental keep legacy values."""
	if preset.vocal_resistance:
		return {
			"peak_ratio": 0.50,
			"peak_ratio_loose": 0.46,
			"second_peak_max": 0.11,
			"entropy_max": 1.28,
			"max_strong_bins": 2,
			"switch_min_score": 0.56,
			"switch_min_margin": 0.048,
			"switch_min_strong_bins": 2.25,
			"triad_cover_min": 0.58,
			"conf_mult_single": 0.62,
			"conf_mult_multi": 0.72,
			"single_pc_conf_cap": 0.36,
		}
	return {
		"peak_ratio": CHROMA_VOCAL_PEAK_RATIO,
		"peak_ratio_loose": CHROMA_VOCAL_PEAK_RATIO_LOOSE,
		"second_peak_max": 0.0,
		"entropy_max": CHROMA_VOCAL_ENTROPY_MAX,
		"max_strong_bins": CHROMA_VOCAL_MAX_STRONG_BINS,
		"switch_min_score": VOCALChord_SWITCH_MIN_SCORE,
		"switch_min_margin": VOCALChord_SWITCH_MIN_MARGIN,
		"switch_min_strong_bins": float(VOCALChord_SWITCH_MIN_STRONG_BINS),
		"triad_cover_min": CHROMA_TRIAD_COVER_MIN,
		"conf_mult_single": 0.695,
		"conf_mult_multi": 0.780,
		"single_pc_conf_cap": 0.40,
	}


def _first_scored_quality(
	scored: List[Tuple[str, float]],
	qualities: frozenset[str],
) -> Tuple[str, float] | None:
	for n, s in scored:
		parts = n.split(":")
		if len(parts) != 2:
			continue
		if parts[1] in qualities:
			return n, float(s)
	return None


def _second_best_excluding(scored: List[Tuple[str, float]], name: str) -> float:
	alt = [s for n, s in scored if n != name]
	return float(max(alt)) if alt else 0.0


def _internal_root_pitch_class(internal: str) -> int | None:
	if not internal or internal == "N" or ":" not in internal:
		return None
	rs = internal.split(":")[0]
	try:
		return int(PITCH_NAMES_SHARP.index(rs))
	except ValueError:
		return None


def _format_key_label(raw: str) -> str:
	if raw.endswith(":maj"):
		return raw.replace(":maj", " major")
	return raw.replace(":min", " minor")


def _analysis_chord_templates(preset: ChordAnalysisPreset) -> Dict[str, np.ndarray]:
	if preset.vocabulary == "maj_min":
		return build_analyze_mvp_templates()
	if preset.include_sevenths:
		return build_analyze_theory_templates()
	return build_analyze_heuristic_templates()


def _sliding_win_hop_frames(sr: int, t_frames: int) -> Tuple[int, int]:
	win = max(2, min(int(round(CHORD_WINDOW_SEC * sr / HOP_LENGTH)), t_frames))
	hop = max(1, min(int(round(CHORD_HOP_SEC * sr / HOP_LENGTH)), max(1, win - 1)))
	return win, hop


def _chroma_pc_metrics(chroma_hist: np.ndarray) -> Dict[str, float]:
	h = _normalize_vector(_validate_chroma(chroma_hist))
	sorted_bins = np.sort(h)[::-1]
	second_peak = float(sorted_bins[1]) if sorted_bins.size > 1 else 0.0
	return {
		"n_strong": float(count_strong_chroma_bins(chroma_hist, threshold=0.2)),
		"entropy": float(chroma_hist_entropy_bits(chroma_hist)),
		"peak_ratio": float(np.max(h)),
		"second_peak_ratio": second_peak,
	}


def _triad_cover_on_unit_vector(chroma_unit: np.ndarray, template: np.ndarray) -> float:
	"""`chroma_unit` already L2-normalized 12-vector."""
	tpl = _normalize_vector(_validate_chroma(template))
	mask = tpl > 1e-6
	return float(np.sum(chroma_unit[mask]))


def _vocal_single_note_heuristic(metrics: Dict[str, float], gates: Dict[str, float]) -> bool:
	n_s = int(metrics["n_strong"])
	peak = float(metrics["peak_ratio"])
	ent = float(metrics["entropy"])
	second_peak = float(metrics.get("second_peak_ratio", 0.0))
	if n_s <= 1 and peak >= gates["peak_ratio"]:
		return True
	if (
		n_s <= 1
		and gates["second_peak_max"] > 0.0
		and peak >= 0.44
		and second_peak < gates["second_peak_max"]
	):
		return True
	if (
		n_s <= int(gates["max_strong_bins"])
		and peak >= gates["peak_ratio_loose"]
		and ent < gates["entropy_max"]
	):
		return True
	return False


def _audio_dots_sorted(chroma_hist: np.ndarray, templates: Dict[str, np.ndarray]) -> List[Tuple[str, float]]:
	chroma_vec = _normalize_vector(_validate_chroma(chroma_hist))
	scored: List[Tuple[str, float]] = []
	for name, template in templates.items():
		if name == "N":
			continue
		tpl = _normalize_vector(_validate_chroma(template))
		scored.append((name, float(np.dot(chroma_vec, tpl))))
	scored.sort(key=lambda x: x[1], reverse=True)
	return scored


def _best_analyze_slot(
	chroma_hist: np.ndarray,
	templates: Dict[str, np.ndarray],
	*,
	preset: ChordAnalysisPreset,
	prev_internal: str | None,
	arpeggio_support: float = 0.0,
	bass_root_pc: int | None = None,
) -> Tuple[str, str, float, float, float, bool, List[str]]:
	"""
	Direct template cosine path for /analyze (keeps labels simple; avoids heavy theory fusion on the grid).

	Returns: internal_name, display_label, best_dot, second_dot, confidence, vocal_interference, confidence_reasons
	"""
	reasons: List[str] = []
	raw = _validate_chroma(chroma_hist)
	comp = _chroma_hist_for_matching(raw)
	chroma_vec = _normalize_vector(comp)
	if float(np.linalg.norm(chroma_vec)) < 1e-12:
		return "N", "N", 0.0, 0.0, 0.0, False, ["empty_chroma"]

	metrics = _chroma_pc_metrics(comp)
	gates = _vocal_gates(preset)
	vocal = _vocal_single_note_heuristic(metrics, gates)
	if vocal:
		reasons.append("single_pc_or_sparse_chroma")

	scored = _audio_dots_sorted(comp, templates)
	if not scored:
		return "N", "N", 0.0, 0.0, 0.0, vocal, reasons + ["no_templates"]

	simple_pick = _first_scored_quality(scored, frozenset({"maj", "min"}))
	if simple_pick is None:
		simple_pick = scored[0]

	sn, sns = simple_pick
	best_name, bs = sn, sns

	if preset.include_sevenths:
		seventh_pick = _first_scored_quality(scored, _SEVENTH_QUALITIES)
		if seventh_pick is not None:
			sno, ses = seventh_pick
			sq = sno.split(":")[1]
			mc7, gap_triad, gap_7_self = _seventh_quality_thresholds(sq)
			sec7 = _second_best_excluding(scored, sno)
			sev_evidence_ok = (
				(not vocal)
				and (
					metrics["n_strong"] >= 3.0
					or (metrics["n_strong"] >= 2.5 and arpeggio_support >= 0.40)
					or (arpeggio_support >= 0.44 and (ses - sns) >= gap_triad + 0.028)
				)
			)
			if (
				sev_evidence_ok
				and ses >= mc7
				and (ses - sns) >= gap_triad
				and (ses - sec7) >= gap_7_self
				and sns + 1e-9 < 0.54
			):
				best_name = sno
				bs = ses
				reasons.append("seventh_quality_evidence")
			else:
				reasons.append("seventh_fallback_triad")

	exotic_pick = _first_scored_quality(scored, _EXOTIC_QUALITIES)
	if exotic_pick is not None:
		eno, es = exotic_pick
		exq = eno.split(":")[1]
		mc_min, gap_simple, gap_2_self = _exotic_quality_thresholds(exq)
		rel = float(preset.exotic_threshold_relief)
		if rel > 0.0:
			mc_min = max(0.12, mc_min - rel)
			gap_simple = max(0.018, gap_simple - rel * 0.62)
			gap_2_self = max(0.018, gap_2_self - rel * 0.56)
		sec_exo = _second_best_excluding(scored, eno)
		top_s = float(scored[0][1])
		sec_top = float(scored[1][1]) if len(scored) > 1 else 0.0
		list_strong_vocal = (top_s - sec_top) >= VOCAL_STRONG_SWITCH_MIN_MARGIN and top_s >= VOCAL_STRONG_SWITCH_MIN_BS

		evidence_ok = (
			(not vocal)
			or list_strong_vocal
			or metrics["n_strong"] >= 2.499
			or (arpeggio_support >= 0.38 and (es - sns) >= gap_simple + 0.026)
		)
		if evidence_ok and es >= mc_min and (es - sns) >= gap_simple and (es - sec_exo) >= gap_2_self:
			best_name = eno
			bs = es
			reasons.append("exotic_quality_high_conf")

	ss = _second_best_excluding(scored, best_name)

	if bs < CHORD_ANALYZE_MIN_AUDIO_DOT:
		return "N", "N", bs, ss, 0.0, vocal, reasons + ["weak_audio_dot"]

	if bass_root_pc is not None and preset.bass_template_dot_bonus > 1e-12:
		rpc = _internal_root_pitch_class(best_name)
		if rpc is not None and rpc == int(bass_root_pc) % 12:
			bs = float(min(1.0, bs + preset.bass_template_dot_bonus))
			reasons.append("bass_root_hint_agrees")

	tpl_best = templates.get(best_name)
	t_cover = _triad_cover_on_unit_vector(chroma_vec, tpl_best) if tpl_best is not None else 0.0

	strong_vocal_harmonic_change = (bs - ss) >= VOCAL_STRONG_SWITCH_MIN_MARGIN and bs >= VOCAL_STRONG_SWITCH_MIN_BS
	if strong_vocal_harmonic_change and vocal:
		reasons.append("vocal_frame_strong_template_margin")

	vocal_sw_score = float(gates["switch_min_score"] * (1.07 if metrics["n_strong"] <= 1.001 else 1.0))
	vocal_sw_margin = float(gates["switch_min_margin"] * (1.12 if metrics["n_strong"] <= 1.001 else 1.0))
	triad_cover_min = float(gates["triad_cover_min"])

	if (
		vocal
		and prev_internal
		and prev_internal != "N"
		and best_name != prev_internal
		and not strong_vocal_harmonic_change
		and (
			bs < vocal_sw_score
			or (bs - ss) < vocal_sw_margin
			or metrics["n_strong"] + 1e-9 < gates["switch_min_strong_bins"]
			or t_cover + 1e-9 < triad_cover_min
		)
	):
		best_name = prev_internal
		reasons.append("held_prev_weak_harmonic_under_vocal_heuristic")
		prev_t = templates.get(best_name)
		if prev_t is None:
			return "N", "N", bs, ss, 0.12, vocal, reasons + ["prev_missing_template"]
		bs = float(np.dot(chroma_vec, _normalize_vector(_validate_chroma(prev_t))))
		others = [s for n, s in scored if n != best_name]
		ss = float(max(others)) if others else 0.0

	label = format_internal_chord_label(best_name)
	raw_margin = (bs - ss) / (abs(bs) + 1e-9)
	raw_margin = float(max(0.0, min(1.0, raw_margin)))
	conf = (
		0.36 * raw_margin
		+ 0.30 * min(1.0, bs / 0.84)
		+ 0.18 * min(1.0, metrics["n_strong"] / 3.0)
		+ 0.16 * min(1.0, metrics["entropy"] / 2.485)
	)

	if vocal:
		conf *= gates["conf_mult_single"] if metrics["n_strong"] <= 1.001 else gates["conf_mult_multi"]
	if vocal and metrics["n_strong"] <= 1.001:
		conf = float(min(conf, gates["single_pc_conf_cap"]))
	if "held_prev_weak_harmonic_under_vocal_heuristic" in reasons:
		conf = min(conf, gates["single_pc_conf_cap"])
	if vocal and preset.vocal_resistance:
		reasons.append("vocal_interference_penalty")

	if (
		preset.arpeggio_conf_scale > 1e-12
		and arpeggio_support >= 0.32
		and t_cover >= 0.465
		and not (vocal and metrics["n_strong"] <= 1.001 and not strong_vocal_harmonic_change)
	):
		conf = float(
			min(
				1.0,
				conf * (1.0 + preset.arpeggio_conf_scale * min(1.0, float(arpeggio_support))),
			)
		)
		reasons.append("arpeggio_temporal_support")

	if bs < CHORD_WEAK_SCORE_CAP:
		conf = min(conf, (bs / CHORD_WEAK_SCORE_CAP) * 0.95)
	conf = float(max(0.0, min(1.0, conf)))

	return best_name, label, bs, ss, conf, vocal, reasons


def _best_template_full(
	chroma_hist: np.ndarray,
	templates: Dict[str, np.ndarray],
	*,
	key_raw: str | None = None,
	prev_internal: str | None = None,
) -> Tuple[str, str, float, float, float]:
	"""Reserved for debug / beat-grid paths; file analyze uses `_best_analyze_slot` on the sliding grid."""
	return pick_chord_with_theory(
		chroma_hist,
		templates,
		key_raw=key_raw,
		prev_internal=prev_internal,
		normalize_vector=_normalize_vector,
	)


def _apply_nearby_label_stability(confs: List[float], labels: List[str]) -> None:
	"""Re-weight confidences from local agreement of final sticky labels (cheap stability term)."""
	n = len(labels)
	if n < 2:
		return
	for k in range(n):
		ło = max(0, k - 2)
		hi = min(n, k + 3)
		lab = labels[k]
		agree = sum(1 for j in range(ło, hi) if labels[j] == lab) / float(hi - ło)
		confs[k] = float(min(1.0, confs[k] * (0.58 + 0.42 * agree)))


def _core_label_stats(chords: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
	stats: Dict[str, Dict[str, float]] = {}
	for c in chords:
		lab = str(c.get("label", "N"))
		if lab == "N":
			continue
		dur = float(c["end"]) - float(c["start"])
		row = stats.setdefault(lab, {"count": 0.0, "total_sec": 0.0})
		row["count"] += 1.0
		row["total_sec"] += dur
	return stats


def _annotate_core_eligibility_inplace(
	c: Dict[str, Any],
	stats: Dict[str, Dict[str, float]],
	*,
	low_conf_threshold: float = 0.16,
) -> None:
	"""Frontend may omit these segments when building the main progression (additive field).

	`low_conf_threshold` is calibrated per preset: Theory's 7th templates lower confidence
	scores so the threshold is lowered to avoid marking all Theory segments as non-core.
	The raw chord timeline (analyzeResult.chords) is unaffected — only core progression uses this.
	"""
	_ = stats
	dur = float(c["end"]) - float(c["start"])
	vox = bool(c.get("vocal_interference"))
	c["exclude_from_core"] = bool(
		c.get("low_confidence")
		or c.get("is_passing")
		or float(c.get("confidence", 0.0)) < low_conf_threshold
		or (vox and dur < 0.55)
	)


def _annotate_core_eligibility_all(
	chords: List[Dict[str, Any]],
	*,
	low_conf_threshold: float = 0.16,
) -> None:
	stats = _core_label_stats(chords)
	for c in chords:
		_annotate_core_eligibility_inplace(c, stats, low_conf_threshold=low_conf_threshold)


def _apply_long_segment_guardrail(
	chords: List[Dict[str, Any]],
	chords_pre_refine: List[Dict[str, Any]],
	*,
	max_sec: float,
	min_split_sec: float = 2.5,
) -> Tuple[List[Dict[str, Any]], int, int]:
	"""
	Safety-net after refine_chord_timeline: split any segment longer than max_sec.

	Uses pre-refine segments (output of chord_timeline_sliding, before snap/collapse) as
	split boundaries.  Only splits where a pre-refine segment has a *different* label AND
	enough duration to avoid re-creating tiny fragments (>= min_split_sec).

	Returns: (new_chords, guardrail_applied_count, split_count).
	"""
	if max_sec <= 0 or not chords_pre_refine:
		return chords, 0, 0

	out: List[Dict[str, Any]] = []
	applied = 0
	split_count = 0

	for seg in chords:
		dur = float(seg["end"]) - float(seg["start"])
		if dur <= max_sec:
			out.append(seg)
			continue

		t_start = float(seg["start"])
		t_end = float(seg["end"])
		seg_label = str(seg.get("label", "N"))

		# Find pre-refine sub-segments with a different label and minimum duration.
		candidates = sorted(
			[
				c for c in chords_pre_refine
				if (
					float(c["end"]) > t_start + 0.05
					and float(c["start"]) < t_end - 0.05
					and str(c.get("label", "N")) != seg_label
					and str(c.get("label", "N")) != "N"
					and float(c["end"]) - float(c["start"]) >= min_split_sec
				)
			],
			key=lambda c: float(c["start"]),
		)

		if not candidates:
			out.append(seg)
			continue

		applied += 1
		cursor = t_start
		pieces: List[Dict[str, Any]] = []

		for c in candidates:
			c_start = max(float(c["start"]), t_start)
			c_end = min(float(c["end"]), t_end)
			# Gap before candidate → original chord piece
			if c_start - cursor >= min_split_sec:
				piece = dict(seg)
				piece["start"] = round(cursor, 4)
				piece["end"] = round(c_start, 4)
				pieces.append(piece)
			# Candidate piece
			if c_end - c_start >= min_split_sec:
				piece_c = dict(c)
				piece_c["start"] = round(c_start, 4)
				piece_c["end"] = round(c_end, 4)
				pieces.append(piece_c)
				split_count += 1
			cursor = max(cursor, c_end)

		# Trailing original chord piece
		if t_end - cursor >= min_split_sec:
			piece = dict(seg)
			piece["start"] = round(cursor, 4)
			piece["end"] = round(t_end, 4)
			pieces.append(piece)

		out.extend(pieces if pieces else [seg])

	return _merge_adjacent_chord_labels(out), applied, split_count


def _simulate_core_fallback(chords_final: List[Dict[str, Any]]) -> Tuple[bool, str | None]:
	"""Simulate the frontend tiered fallback to report which tier would activate."""
	non_n = [c for c in chords_final if str(c.get("label", "N")) != "N"]
	if not non_n:
		return True, "empty_timeline"
	strict = [
		c for c in non_n
		if not c.get("exclude_from_core") and not c.get("is_passing") and not c.get("low_confidence")
	]
	if len(set(c.get("label") for c in strict)) >= 2:
		return False, None
	medium = [c for c in non_n if not c.get("is_passing")]
	if len(set(c.get("label") for c in medium)) >= 2:
		return True, "fallback_medium_non_passing"
	if len(set(c.get("label") for c in non_n)) >= 2:
		return True, "fallback_all_non_n"
	return True, "fallback_sparse_timeline"


def _filtering_debug_stats(
	chords_pre_refine: List[Dict[str, Any]],
	chords_final: List[Dict[str, Any]],
	raw_frame_count: int = 0,
	*,
	guardrail_applied: int = 0,
	guardrail_splits: int = 0,
	longest_before_guardrail: float = 0.0,
	longest_after_guardrail: float = 0.0,
	post_sticky_segment_count: int = 0,
	post_median_segment_count: int = 0,
) -> Dict[str, Any]:
	pre_labels = [str(c.get("label", "N")) for c in chords_pre_refine if str(c.get("label", "N")) != "N"]
	labels = [str(c.get("label", "N")) for c in chords_final if str(c.get("label", "N")) != "N"]
	core = [c for c in chords_final if not c.get("exclude_from_core")]
	core_labels = [str(c.get("label", "N")) for c in core if str(c.get("label", "N")) != "N"]

	# Longest segment stats
	longest_dur = 0.0
	longest_label = ""
	for c in chords_final:
		dur = float(c.get("end", 0)) - float(c.get("start", 0))
		if dur > longest_dur:
			longest_dur = dur
			longest_label = str(c.get("label", "N"))

	fallback_used, fallback_reason = _simulate_core_fallback(chords_final)
	return {
		"raw_chord_frame_count": raw_frame_count,
		"raw_chord_segment_count": len(chords_pre_refine),
		"post_sticky_segment_count": post_sticky_segment_count,
		"post_median_segment_count": post_median_segment_count,
		"returned_chord_segment_count": len(chords_final),
		"unique_raw_chords": len(set(pre_labels)),
		"unique_returned_chords": len(set(labels)),
		"core_candidate_count_before_filter": len(chords_final),
		"core_candidate_count_after_filter": len(core),
		"unique_core_chords": len(set(core_labels)),
		"excluded_from_core_count": sum(1 for c in chords_final if c.get("exclude_from_core")),
		"low_confidence_count": sum(1 for c in chords_final if c.get("low_confidence")),
		"vocal_interference_count": sum(1 for c in chords_final if c.get("vocal_interference")),
		"longest_segment_label": longest_label,
		"longest_segment_duration": round(longest_dur, 3),
		"longest_segment_before_guardrail": round(longest_before_guardrail, 3),
		"longest_segment_after_guardrail": round(longest_after_guardrail, 3),
		"final_long_segment_guardrail_applied_count": guardrail_applied,
		"final_long_segment_split_count": guardrail_splits,
		"core_progression_fallback_used": fallback_used,
		"core_progression_fallback_reason": fallback_reason,
	}


def _chord_quality_summary(chords: List[Dict[str, Any]]) -> Dict[str, int]:
	major = minor = seventh = low = 0
	for c in chords:
		if bool(c.get("low_confidence")):
			low += 1
		lab = str(c.get("label", "N"))
		if lab == "N":
			continue
		if lab.endswith("maj7") or (lab.endswith("7") and not lab.endswith("m7")):
			seventh += 1
		elif lab.endswith("m7"):
			seventh += 1
		elif "m" in lab[1:] or lab.endswith("m"):
			minor += 1
		else:
			major += 1
	return {
		"major_count": major,
		"minor_count": minor,
		"seventh_count": seventh,
		"low_confidence_count": low,
	}


def _merge_adjacent_chord_labels(chords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	out: List[Dict[str, Any]] = []
	for c in chords:
		if out and out[-1]["label"] == c["label"]:
			out[-1]["end"] = c["end"]
			out[-1]["confidence"] = float(max(out[-1].get("confidence", 0.0), c.get("confidence", 0.0)))
			out[-1]["low_confidence"] = bool(out[-1].get("low_confidence", False) or c.get("low_confidence", False))
			out[-1]["is_passing"] = False
			out[-1]["chord_role"] = None
			out[-1]["vocal_interference"] = bool(out[-1].get("vocal_interference") or c.get("vocal_interference"))
			pr = list(out[-1].get("confidence_reasons") or [])
			cr = list(c.get("confidence_reasons") or [])
			if cr or pr:
				merged = list(dict.fromkeys([*pr, *cr]))
				out[-1]["confidence_reasons"] = merged[:14]
			ts0 = out[-1].get("template_score")
			ts1 = c.get("template_score")
			if ts0 is not None and ts1 is not None:
				out[-1]["template_score"] = float(max(float(ts0), float(ts1)))
			tm0 = out[-1].get("template_margin")
			tm1 = c.get("template_margin")
			if tm0 is not None and tm1 is not None:
				out[-1]["template_margin"] = float(max(float(tm0), float(tm1)))
			ar0 = out[-1].get("arpeggio_support")
			ar1 = c.get("arpeggio_support")
			if ar0 is not None or ar1 is not None:
				out[-1]["arpeggio_support"] = round(float(max(float(ar0 or 0.0), float(ar1 or 0.0))), 4)
			bh0, bh1 = out[-1].get("bass_root_hint"), c.get("bass_root_hint")
			if bh1 is not None and (
				bh0 is None or float(c.get("confidence", 0.0)) >= float(out[-1].get("confidence", 0.0))
			):
				out[-1]["bass_root_hint"] = bh1
			elif bh0 is None:
				out[-1]["bass_root_hint"] = bh1
		else:
			row = dict(c)
			row.setdefault("is_passing", False)
			row.setdefault("chord_role", None)
			row.setdefault("vocal_interference", False)
			out.append(row)
	return out


def _collapse_short_chord_segments(chords: List[Dict[str, Any]], min_sec: float) -> List[Dict[str, Any]]:
	"""Absorb very short fragments into the previous span (prefers harmonic continuity)."""
	if len(chords) <= 1:
		return chords
	changed = True
	parts = [dict(c) for c in chords]
	while changed:
		changed = False
		new_parts: List[Dict[str, Any]] = []
		i = 0
		while i < len(parts):
			dur = parts[i]["end"] - parts[i]["start"]
			if dur + 1e-6 < min_sec and len(parts) > 1:
				if new_parts:
					new_parts[-1]["end"] = max(new_parts[-1]["end"], parts[i]["end"])
					new_parts[-1]["confidence"] = float(
						min(new_parts[-1].get("confidence", 1.0), parts[i].get("confidence", 1.0)),
					)
				elif i + 1 < len(parts):
					parts[i + 1]["start"] = parts[i]["start"]
				else:
					new_parts.append(parts[i])
				i += 1
				changed = True
				continue
			new_parts.append(parts[i])
			i += 1
		parts = _merge_adjacent_chord_labels(new_parts)
	return parts


def _remove_chord_spikes(chords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Sandwich outliers: A-B-A with tiny B → snap B to A."""
	if len(chords) < 3:
		return chords
	work = [dict(c) for c in chords]
	for i in range(1, len(work) - 1):
		d = work[i]["end"] - work[i]["start"]
		if (
			d <= SPIKE_ISOLATE_MAX_SEC
			and work[i]["label"] != work[i - 1]["label"]
			and work[i]["label"] != work[i + 1]["label"]
			and work[i - 1]["label"] == work[i + 1]["label"]
		):
			work[i]["label"] = work[i - 1]["label"]
			work[i]["confidence"] = float(min(work[i].get("confidence", 0.5), CHORD_LOW_CONF_CUTOFF))
			work[i]["low_confidence"] = True
	return _merge_adjacent_chord_labels(work)


def _snap_weak_chord_blips_to_prev(
	chords: List[Dict[str, Any]],
	chroma: np.ndarray,
	sr: int,
	*,
	snap_conf_threshold: float = SNAP_EXTRA_CONF_THRESHOLD,
) -> List[Dict[str, Any]]:
	"""
	If a short segment is low-confidence or very weakly scored but its chroma is very similar to the
	previous chord pocket, keep the prior chord (melody / passing tone — no source separation).

	`snap_conf_threshold` is calibrated per preset: Theory uses a lower value because 7th-chord
	template competition reduces confidence scores even for correct detections.
	"""
	if len(chords) < 2:
		return chords
	t_frames = chroma.shape[1]
	work = [dict(c) for c in chords]
	for i in range(1, len(work)):
		dur = float(work[i]["end"]) - float(work[i]["start"])
		if dur > SNAP_WEAK_MAX_SEC + 1e-6:
			continue
		weak = bool(
			work[i].get("low_confidence", False)
			or float(work[i].get("confidence", 1.0)) < snap_conf_threshold
		)
		if not weak:
			continue
		prev_l = work[i - 1].get("label")
		if not prev_l or prev_l == "N":
			continue
		f0a = max(0, int(float(work[i - 1]["start"]) * sr / HOP_LENGTH))
		f1a = min(int(np.ceil(float(work[i - 1]["end"]) * sr / HOP_LENGTH)), t_frames)
		f0b = max(0, int(float(work[i]["start"]) * sr / HOP_LENGTH))
		f1b = min(int(np.ceil(float(work[i]["end"]) * sr / HOP_LENGTH)), t_frames)
		if f1a <= f0a:
			f1a = min(f0a + 1, t_frames)
		if f1b <= f0b:
			f1b = min(f0b + 1, t_frames)
		va = aggregate_chroma(chroma[:, f0a:f1a])
		vb = aggregate_chroma(chroma[:, f0b:f1b])
		if float(np.dot(va, vb)) >= SNAP_WEAK_TO_PREV_CHROMA_SIM:
			work[i]["label"] = prev_l
			work[i]["confidence"] = float(min(work[i].get("confidence", 0.35), CHORD_LOW_CONF_CUTOFF))
			work[i]["low_confidence"] = True
	return _merge_adjacent_chord_labels(work)


def _merge_chroma_similar_neighbors(
	chords: List[Dict[str, Any]],
	chroma: np.ndarray,
	sr: int,
) -> List[Dict[str, Any]]:
	"""Merge adjacent intervals with different labels iff aggregate chroma nearly identical (same harmony)."""
	if len(chords) < 2:
		return chords
	t_frames = chroma.shape[1]
	vecs: List[np.ndarray] = []
	for c in chords:
		f0 = max(0, int(c["start"] * sr / HOP_LENGTH))
		f1 = min(int(np.ceil(c["end"] * sr / HOP_LENGTH)), t_frames)
		if f1 <= f0:
			f1 = min(f0 + 1, t_frames)
		h = aggregate_chroma(chroma[:, f0:f1])
		vecs.append(h)
	out: List[Dict[str, Any]] = [dict(chords[0])]
	out_vecs = [vecs[0]]
	for i in range(1, len(chords)):
		sim = float(np.dot(out_vecs[-1], vecs[i]))
		if sim >= CHORD_NEIGHBOR_MERGE_COS and chords[i]["label"] != out[-1]["label"]:
			pick = out[-1] if out[-1].get("confidence", 0) >= chords[i].get("confidence", 0) else chords[i]
			out[-1]["end"] = chords[i]["end"]
			out[-1]["label"] = pick["label"]
			out[-1]["confidence"] = float(max(out[-1].get("confidence", 0), chords[i].get("confidence", 0)))
			if "template_score" in pick:
				out[-1]["template_score"] = pick["template_score"]
			if "template_margin" in pick:
				out[-1]["template_margin"] = pick["template_margin"]
			bl = out_vecs[-1] + vecs[i]
			nrm = float(np.linalg.norm(bl))
			out_vecs[-1] = bl / nrm if nrm > 1e-12 else out_vecs[-1]
		else:
			out.append(dict(chords[i]))
			out_vecs.append(vecs[i])
	return out


def _nearest_beat_within(beat_times: List[float], t: float, max_delta: float) -> float | None:
	best: float | None = None
	bd = max_delta + 1.0
	for b in beat_times:
		d = abs(float(b) - t)
		if d < bd:
			bd = d
			best = float(b)
	if best is None or bd > max_delta + 1e-9:
		return None
	return best


def _harmonic_cusp_frame(chroma: np.ndarray, f_center: int, radius: int, half_win: int = 4) -> int:
	"""Shift split frame to maximize left/right chroma disagreement (cheap harmonic-change proxy)."""
	t_frames = chroma.shape[1]
	if t_frames < 2:
		return int(np.clip(f_center, 0, max(0, t_frames - 1)))
	f_center = int(np.clip(f_center, 0, t_frames - 1))
	best_f = f_center
	best_d = -1.0
	for o in range(-radius, radius + 1):
		fc = f_center + o
		if fc < half_win or fc + half_win > t_frames:
			continue
		L = aggregate_chroma(chroma[:, fc - half_win : fc])
		R = aggregate_chroma(chroma[:, fc : fc + half_win])
		d = 1.0 - float(np.dot(L, R))
		if d > best_d:
			best_d = d
			best_f = fc
	return int(best_f)


def _align_chord_segment_boundaries(
	chords: List[Dict[str, Any]],
	chroma: np.ndarray,
	sr: int,
	beat_times: List[float] | None,
	duration_sec: float,
) -> List[Dict[str, Any]]:
	"""Nudge interior boundaries toward local harmonic cusps; optional near-beat snap when consistent."""
	if len(chords) <= 1:
		return chords
	bt = sorted({round(float(t), 4) for t in (beat_times or []) if 0.0 < float(t) < float(duration_sec) - 1e-6})
	min_seg = 0.06
	out = [dict(c) for c in chords]
	for i in range(1, len(out)):
		t_edge = float(out[i]["start"])
		f_c = int(round(t_edge * sr / float(HOP_LENGTH)))
		f_n = _harmonic_cusp_frame(chroma, f_c, HARMONIC_CUSP_RADIUS_FRAMES)
		t_n = round(float(f_n * HOP_LENGTH / sr), 4)
		lo = float(out[i - 1]["start"]) + min_seg
		hi = float(out[i]["end"]) - min_seg
		if hi <= lo + 1e-6:
			continue
		t_n = max(lo, min(hi, t_n))
		if bt:
			b = _nearest_beat_within(bt, t_n, BEAT_SNAP_MAX_SEC)
			if b is not None and lo + 1e-6 < b < hi - 1e-6 and abs(b - t_n) <= BEAT_SNAP_MAX_SEC + 1e-6:
				t_n = round(b, 4)
		out[i - 1]["end"] = t_n
		out[i]["start"] = t_n
	if out:
		out[0]["start"] = round(max(0.0, float(out[0]["start"])), 4)
		out[-1]["end"] = round(min(float(duration_sec), float(out[-1]["end"])), 4)
	return out


def _annotate_passing_chords(chords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	if not chords:
		return chords
	out = [dict(c) for c in chords]
	cnt_by_lab: Dict[str, int] = {}
	tot_sec_by_lab: Dict[str, float] = defaultdict(float)
	for row in out:
		lab = str(row["label"])
		if lab == "N":
			continue
		cnt_by_lab[lab] = cnt_by_lab.get(lab, 0) + 1
		tot_sec_by_lab[lab] += float(row["end"]) - float(row["start"])
	structural_lab = {
		lab
		for lab, tot in tot_sec_by_lab.items()
		if cnt_by_lab.get(lab, 0) >= PASSING_REPEAT_MIN_COUNT and tot >= PASSING_REPEAT_MIN_TOTAL_DUR_SEC
	}

	for i, row in enumerate(out):
		if i == 0 or i == len(out) - 1:
			row["is_passing"] = False
			row["chord_role"] = None
			continue
		dur = float(row["end"]) - float(row["start"])
		prev_l = str(out[i - 1]["label"])
		next_l = str(out[i + 1]["label"])
		lab = str(row["label"])
		conf = float(row.get("confidence", 0.5))
		is_p = likely_passing_segment(
			dur,
			lab,
			prev_l,
			next_l,
			conf,
			max_dur=PASSING_MAX_SEGMENT_SEC,
		)
		if is_p and lab in structural_lab:
			is_p = False
		row["is_passing"] = bool(is_p)
		row["chord_role"] = "passing" if is_p else None
	return out


def _finalize_chord_confidence_flags(
	chords: List[Dict[str, Any]],
	*,
	low_conf_cutoff: float = CHORD_LOW_CONF_CUTOFF,
) -> List[Dict[str, Any]]:
	for c in chords:
		cf = float(c.get("confidence", 0.5))
		low = bool(c.get("low_confidence", False) or cf < low_conf_cutoff)
		c["low_confidence"] = low
		c["confidence"] = round(cf, 4)
		c.setdefault("is_passing", False)
		c.setdefault("chord_role", None)
		if "arpeggio_support" not in c:
			c["arpeggio_support"] = 0.0
	return chords


def _enrich_chord_segment(c: Dict[str, Any]) -> Dict[str, Any]:
	notes, hint = playable_triad_notes_and_hint(str(c.get("label", "N")))
	out = dict(c)
	out["notes"] = notes
	out["practice_hint"] = hint
	return out


def refine_chord_timeline(
	chords: List[Dict[str, Any]],
	chroma: np.ndarray,
	sr: int,
	*,
	beat_times: List[float] | None = None,
	duration_sec: float | None = None,
	snap_conf_threshold: float = SNAP_EXTRA_CONF_THRESHOLD,
	low_conf_cutoff: float = CHORD_LOW_CONF_CUTOFF,
) -> List[Dict[str, Any]]:
	"""Post-pass: spikes, fragments, chroma-merge, harmonic boundary nudge, passing tags.

	`snap_conf_threshold` and `low_conf_cutoff` are calibrated per preset so that Theory's
	smaller confidence scores (caused by 7th-chord template competition) do not over-filter
	the raw chord timeline.
	"""
	if not chords:
		return []
	dur = float(duration_sec) if duration_sec is not None else float(chords[-1]["end"])
	min_frag = max(MIN_CHORD_SEGMENT_SEC, MIN_STABLE_REGION_SEC)
	x = _remove_chord_spikes(chords)
	x = _snap_weak_chord_blips_to_prev(x, chroma, sr, snap_conf_threshold=snap_conf_threshold)
	x = _merge_chroma_similar_neighbors(x, chroma, sr)
	x = _collapse_short_chord_segments(x, min_frag)
	x = _merge_adjacent_chord_labels(x)
	if beat_times is not None:
		x = _align_chord_segment_boundaries(x, chroma, sr, beat_times, dur)
	x = _collapse_short_chord_segments(x, min_frag)
	x = _merge_adjacent_chord_labels(x)
	x = _annotate_passing_chords(x)
	return _finalize_chord_confidence_flags(x, low_conf_cutoff=low_conf_cutoff)


def _section_chroma_vector(chroma: np.ndarray, sr: int, t0: float, t1: float, duration_sec: float) -> np.ndarray:
	t_frames = chroma.shape[1]
	t0 = max(0.0, float(t0))
	t1 = min(float(t1), float(duration_sec))
	f0 = int(t0 * sr / HOP_LENGTH)
	f1 = min(int(np.ceil(t1 * sr / HOP_LENGTH)), t_frames)
	if f1 <= f0:
		f1 = min(f0 + 1, t_frames)
	return aggregate_chroma(chroma[:, f0:f1])


def _chord_label_at_time(chords: List[Dict[str, Any]], t: float, duration_sec: float) -> str:
	"""Which analyzed chord is active at time t (heuristic end for last segment)."""
	if not chords:
		return "N"
	for i, c in enumerate(chords):
		end = float(duration_sec) if i == len(chords) - 1 else float(c["end"])
		if float(c["start"]) <= t <= end + 1e-3:
			return str(c["label"])
	return str(chords[-1]["label"])


def _section_chord_fingerprint(
	chords: List[Dict[str, Any]],
	t0: float,
	t1: float,
	duration_sec: float,
	n_samples: int = 14,
) -> Tuple[str, ...]:
	"""Downsampled chord symbol sequence inside [t0, t1] for repetition matching across the song."""
	if not chords or t1 <= t0:
		return tuple()
	labels: List[str] = []
	for i in range(n_samples):
		u = (i + 0.5) / float(n_samples)
		t = t0 + u * (t1 - t0)
		labels.append(_chord_label_at_time(chords, t, duration_sec))
	return tuple(labels)


def _fingerprint_similarity(a: Tuple[str, ...], b: Tuple[str, ...]) -> float:
	if not a or not b:
		return 0.0
	n = max(len(a), len(b))
	matches = 0
	for i in range(n):
		ca = a[i] if i < len(a) else a[-1]
		cb = b[i] if i < len(b) else b[-1]
		if ca == cb:
			matches += 1
	return matches / float(n)


def merge_short_sections(
	sections: List[Dict[str, Any]],
	min_sec: float,
	chroma: np.ndarray,
	sr: int,
	duration_sec: float,
) -> List[Dict[str, Any]]:
	"""Absorb tiny section fragments into the more similar neighbor (global layout stays readable)."""
	if len(sections) <= 1:
		return sections
	parts: List[Dict[str, Any]] = [dict(s) for s in sections]

	def _cos_sim_interval(i: int, j: int) -> float:
		v_i = _section_chroma_vector(chroma, sr, parts[i]["start"], parts[i]["end"], duration_sec)
		v_j = _section_chroma_vector(chroma, sr, parts[j]["start"], parts[j]["end"], duration_sec)
		ni = float(np.linalg.norm(v_i)) + 1e-12
		nj = float(np.linalg.norm(v_j)) + 1e-12
		return float(np.dot(v_i / ni, v_j / nj))

	while True:
		short_idx = -1
		for i, p in enumerate(parts):
			if float(p["end"]) - float(p["start"]) < float(min_sec) - 1e-6:
				short_idx = i
				break
		if short_idx < 0:
			break
		i = short_idx
		if len(parts) <= 1:
			break
		if i == 0 and len(parts) > 1:
			parts[0]["end"] = parts[1]["end"]
			parts.pop(1)
			continue
		if i == len(parts) - 1:
			parts[-2]["end"] = parts[-1]["end"]
			parts.pop(-1)
			continue
		if _cos_sim_interval(i - 1, i) >= _cos_sim_interval(i, i + 1):
			parts[i - 1]["end"] = parts[i]["end"]
			parts.pop(i)
		else:
			parts[i]["end"] = parts[i + 1]["end"]
			parts.pop(i + 1)
	return parts


def split_long_sections(
	sections: List[Dict[str, Any]],
	beat_times: List[float],
	duration_sec: float,
	max_sec: float = MAX_SECTION_PREFERRED_SEC,
	min_sec: float = MIN_SECTION_DURATION_SEC,
	target_chunk: float = TARGET_PRACTICE_SECTION_SEC,
) -> List[Dict[str, Any]]:
	"""
	Subdivide very long sections near beat times so practice chunks typically fall in ~8–16s.
	Placeholder labels; relabel_sections_with_repetition runs after merge passes.
	"""
	if not sections:
		return sections
	be_sorted = sorted({float(t) for t in beat_times if 0.0 < float(t) < float(duration_sec)})
	out: List[Dict[str, Any]] = []
	for sec in sections:
		t0 = float(sec["start"])
		t1 = float(min(float(sec["end"]), float(duration_sec)))
		dur = t1 - t0
		if dur <= max_sec + 1e-6:
			out.append(dict(sec))
			continue
		cursor = t0
		bounds: List[float] = [t0]
		while t1 - cursor > max_sec + 1e-6:
			ideal = cursor + min(target_chunk, t1 - cursor - min_sec * 0.5)
			lo, hi = cursor + min_sec, t1 - min_sec
			if hi <= lo + 1e-6:
				break
			inside = [b for b in be_sorted if lo + 1e-6 < b < hi - 1e-6]
			if inside:
				split_at = float(min(inside, key=lambda b: abs(b - ideal)))
			else:
				split_at = float(max(lo, min(hi, ideal)))
			if split_at <= cursor + 1e-6 or split_at >= t1 - 1e-6:
				break
			bounds.append(round(split_at, 4))
			cursor = split_at
		if bounds[-1] < t1 - 1e-6:
			bounds.append(round(t1, 4))
		base_lbl = str(sec.get("label", "Section"))
		for i in range(len(bounds) - 1):
			a, b = bounds[i], bounds[i + 1]
			if b > a + 1e-9:
				out.append({"start": round(float(a), 4), "end": round(float(b), 4), "label": base_lbl})
	return out


def relabel_sections_with_repetition(
	sections: List[Dict[str, Any]],
	chroma: np.ndarray,
	sr: int,
	duration_sec: float,
	chords: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
	"""
	Assign Section A / B / A … when distant bodies share similar aggregate chroma and/or
	similar local chord fingerprint (repeated loops, choruses).
	Heuristic only—not structural analysis.
	"""
	if not sections:
		return sections
	prototypes: List[np.ndarray] = []
	proto_letters: List[str] = []
	prototype_fps: List[Tuple[str, ...]] = []
	letter_next = 0
	out: List[Dict[str, Any]] = []
	for sec in sections:
		v = _section_chroma_vector(chroma, sr, sec["start"], sec["end"], duration_sec)
		nv = v / (float(np.linalg.norm(v)) + 1e-12)
		fp_new = _section_chord_fingerprint(chords, sec["start"], sec["end"], duration_sec) if chords else tuple()

		best_k = -1
		best_score = -1.0
		for k, proto in enumerate(prototypes):
			pk = proto / (float(np.linalg.norm(proto)) + 1e-12)
			chroma_sim = float(np.dot(nv, pk))
			fp_sim = _fingerprint_similarity(fp_new, prototype_fps[k]) if chords else 0.0
			is_repeat = chroma_sim >= SECTION_REPEAT_SIM_THRESHOLD or (
				chroma_sim >= SECTION_REPEAT_CHROMA_LO and fp_sim >= SECTION_REPEAT_FP_SIM
			)
			if is_repeat:
				score = chroma_sim + 0.15 * fp_sim
				if score > best_score:
					best_score = score
					best_k = k

		if best_k < 0:
			prototypes.append(v.astype(float))
			proto_letters.append(chr(ord("A") + letter_next))
			prototype_fps.append(fp_new)
			letter = proto_letters[-1]
			letter_next += 1
			lbl = f"Section {letter}"
			rg = letter
		else:
			lbl = f"Section {proto_letters[best_k]}"
			rg = proto_letters[best_k]
		out.append({**sec, "label": lbl, "repeat_group": rg})
	return out


def _boundaries_from_beats(beat_times: List[float], duration_sec: float) -> List[float]:
	"""0 → interior beats → duration (aligned with chord beat path)."""
	interior = sorted({round(float(t), 4) for t in beat_times if 1e-4 < t < duration_sec - 1e-4})
	boundaries: List[float] = [0.0]
	for t in interior:
		if not boundaries or abs(t - boundaries[-1]) > 1e-4:
			boundaries.append(t)
	if not boundaries or abs(boundaries[-1] - duration_sec) > 1e-4:
		boundaries.append(round(float(duration_sec), 4))
	return boundaries


def _chroma_intervals_from_boundaries(
	chroma: np.ndarray,
	sr: int,
	boundaries: List[float],
) -> Tuple[List[Tuple[float, float]], List[np.ndarray]]:
	"""Inter-beat (or arbitrary) intervals with normalized aggregate chroma per interval."""
	t_frames = chroma.shape[1]
	intervals: List[Tuple[float, float]] = []
	vectors: List[np.ndarray] = []
	for i in range(len(boundaries) - 1):
		t0, t1 = boundaries[i], boundaries[i + 1]
		if t1 <= t0 + 1e-9:
			continue
		f0 = int(t0 * sr / HOP_LENGTH)
		f1 = min(int(np.ceil(t1 * sr / HOP_LENGTH)), t_frames)
		if f1 <= f0:
			continue
		hist = aggregate_chroma(chroma[:, f0:f1])
		intervals.append((t0, t1))
		vectors.append(hist)
	return intervals, vectors


def _merge_sections_by_chroma_similarity(
	intervals: List[Tuple[float, float]],
	vectors: List[np.ndarray],
	duration_sec: float,
) -> List[Dict[str, Any]]:
	"""Greedy merge of adjacent intervals when cosine(chroma_i, chroma_{i+1}) >= SECTION_MERGE_COS."""
	if not intervals:
		return [{"start": 0.0, "end": round(float(duration_sec), 4), "label": "Section 1"}]

	out: List[Dict[str, Any]] = []
	sec_idx = 1
	st, en = intervals[0][0], intervals[0][1]
	prev = vectors[0].astype(float)

	for j in range(1, len(intervals)):
		sim = float(np.dot(prev, vectors[j]))
		t0, t1 = intervals[j]
		if sim >= SECTION_MERGE_COS:
			en = t1
			blended = prev + vectors[j].astype(float)
			nrm = float(np.linalg.norm(blended))
			prev = blended / nrm if nrm > 1e-12 else prev
		else:
			out.append({"start": round(st, 4), "end": round(en, 4), "label": f"Section {sec_idx}"})
			sec_idx += 1
			st, en = t0, t1
			prev = vectors[j].astype(float)

	out.append({"start": round(st, 4), "end": round(en, 4), "label": f"Section {sec_idx}"})
	return out


def sections_with_indices(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Stable 0-based order for navigation and UI (Section 1 → index 0)."""
	return [{**s, "index": i} for i, s in enumerate(sections)]


def compute_rhythm_hints(
	beat_times: List[float],
	beats_per_bar: int = DEFAULT_BEATS_PER_BAR,
) -> Dict[str, Any]:
	"""
	Lightweight timing hints for practice: notional bar lines from beat_track.
	Treats every Nth beat as a bar start (first beat = downbeat). No meter detection.
	"""
	if beats_per_bar < 1:
		beats_per_bar = DEFAULT_BEATS_PER_BAR
	if not beat_times:
		return {"assumed_beats_per_bar": beats_per_bar, "bar_start_times": []}
	sorted_beats = sorted({round(float(t), 4) for t in beat_times})
	bar_starts = [sorted_beats[i] for i in range(0, len(sorted_beats), beats_per_bar)]
	return {
		"assumed_beats_per_bar": beats_per_bar,
		"bar_start_times": bar_starts,
	}


def detect_sections(
	chroma: np.ndarray,
	sr: int,
	beat_times: List[float],
	duration_sec: float,
) -> List[Dict[str, Any]]:
	"""
	First-pass sections: merge beat- or time-aligned windows when aggregate chroma is similar (repetition).
	Labels: Section 1, Section 2, ...
	"""
	if len(beat_times) >= 2:
		bounds = _boundaries_from_beats(beat_times, duration_sec)
		intervals, vectors = _chroma_intervals_from_boundaries(chroma, sr, bounds)
		if len(intervals) < 2:
			return _detect_sections_equal_time_windows(chroma, sr, duration_sec)
		return _merge_sections_by_chroma_similarity(intervals, vectors, duration_sec)
	return _detect_sections_equal_time_windows(chroma, sr, duration_sec)


def _detect_sections_equal_time_windows(
	chroma: np.ndarray,
	sr: int,
	duration_sec: float,
) -> List[Dict[str, Any]]:
	"""~10 s target windows when beats are unreliable; then same chroma merge."""
	n_parts = int(np.ceil(duration_sec / float(EQUAL_TIME_WINDOW_SEC)))
	n_parts = max(2, min(14, n_parts))
	chunk = duration_sec / float(n_parts)
	boundaries = [round(i * chunk, 4) for i in range(n_parts + 1)]
	boundaries[-1] = round(float(duration_sec), 4)
	intervals, vectors = _chroma_intervals_from_boundaries(chroma, sr, boundaries)
	if not intervals:
		return [{"start": 0.0, "end": round(float(duration_sec), 4), "label": "Section 1"}]
	return _merge_sections_by_chroma_similarity(intervals, vectors, duration_sec)


def _median_chord_labels(labels: List[str], width: int) -> List[str]:
	"""Majority vote in a sliding window across high-rate chord slots (suppresses single-frame noise)."""
	if width <= 1 or len(labels) <= 1:
		return labels
	w = max(3, int(width))
	if w % 2 == 0:
		w += 1
	half = w // 2
	out: List[str] = []
	for i in range(len(labels)):
		lo = max(0, i - half)
		hi = min(len(labels), i + half + 1)
		window = labels[lo:hi]
		out.append(Counter(window).most_common(1)[0][0])
	return out


def _sticky_post_median_slots(
	labels: List[str],
	best_scores: List[float],
	second_scores: List[float],
	confs: List[float],
	lows: List[bool],
	vocal_slots: List[bool] | None = None,
	vocal_sticky_mult: float = VOCAL_STICKY_MARGIN_MULT,
	min_raw_margin: float = STICKY_MIN_RAW_MARGIN,
	max_hold_slots: int = 0,
	forced_window_slots: int = 0,
) -> Tuple[List[str], List[float], List[bool]]:
	"""
	When a *change* is weakly supported vs the runner-up template, hold the previous chord.
	Approximates hysteresis so brief vocal chroma peaks flip harmony less often.

	`min_raw_margin` defaults to the global STICKY_MIN_RAW_MARGIN but can be overridden
	per-preset; Theory's 7th templates reduce margins, requiring a lower floor.

	`max_hold_slots` caps how many consecutive slots the gate may hold one chord before
	forcing a release — prevents a single chord from occupying a whole song when all frames
	have margins below the floor (common with 109-template Theory mode).  0 = unlimited.

	`forced_window_slots` sets how many slots to emit the forced chord after cap fires.
	Using a majority-vote look-ahead, this produces a segment long enough to survive the
	refine pipeline's snap/collapse steps (SNAP_WEAK_MAX_SEC=0.5s, min_frag=0.2s).
	A single-slot release (forced_window_slots=0) would be absorbed by those steps.
	"""
	if not labels:
		return labels, confs, lows
	out_labels = [labels[0]]
	out_confs = [confs[0]]
	out_lows = [lows[0]]
	hold_count = 0
	# Track consecutive slots where the CURRENT output label has been emitted (output age).
	# This is what the cap should be based on: wall-time held, not just rejected challenges.
	# (hold_count alone resets every time cand==prev, so a dominant chord is never capped.)
	current_label_age = 1
	forced_remaining = 0
	forced_label = ""
	for i in range(1, len(labels)):
		# --- Forced release window: emit the chosen chord without sticky ---
		if forced_remaining > 0:
			forced_remaining -= 1
			current_label_age += 1
			hold_count = 0
			out_labels.append(forced_label)
			out_confs.append(confs[i])
			out_lows.append(lows[i])
			continue

		cand = labels[i]
		prev = out_labels[-1]
		if cand == prev:
			hold_count = 0
			current_label_age += 1
			out_labels.append(cand)
			out_confs.append(confs[i])
			out_lows.append(lows[i])
			continue
		bs = best_scores[i]
		raw_m = bs - second_scores[i]
		mult = vocal_sticky_mult if vocal_slots and vocal_slots[i] else 1.0
		weak = bs < STICKY_MIN_BEST_SCORE or raw_m < min_raw_margin * mult

		# Cap: the current output label has been emitted for too many consecutive slots.
		# Use current_label_age (output wall-time) so a dominant chord cannot hold forever
		# even when it is re-detected naturally on same-label slots (which reset hold_count).
		if weak and max_hold_slots > 0 and current_label_age >= max_hold_slots:
			if forced_window_slots > 0:
				# Majority-vote on next forced_window_slots labels to pick best non-prev chord.
				look_end = min(len(labels), i + forced_window_slots)
				future = labels[i:look_end]
				non_prev = [l for l in future if l != prev and l != "N"]
				# Only force if at least 10 % of the window wants something different;
				# avoids injecting a phantom chord into a genuinely mono-chord passage.
				min_votes = max(1, int(forced_window_slots * 0.10))
				if len(non_prev) >= min_votes:
					ctr = Counter(non_prev)
					forced_label = ctr.most_common(1)[0][0]
					forced_remaining = forced_window_slots - 1
					hold_count = 0
					current_label_age = 1
					out_labels.append(forced_label)
					out_confs.append(confs[i])
					out_lows.append(lows[i])
					continue
			# Fallback: single-slot release (no window or not enough evidence).
			weak = False

		if weak:
			hold_count += 1
			current_label_age += 1
			out_labels.append(prev)
			out_confs.append(min(float(confs[i]), STICKY_CONF_CAP))
			out_lows.append(True)
		else:
			hold_count = 0
			current_label_age = 1
			out_labels.append(cand)
			out_confs.append(confs[i])
			out_lows.append(lows[i])
	return out_labels, out_confs, out_lows


def _median_bool_flags(flags: List[bool], width: int) -> List[bool]:
	"""Majority vote for bools in the same sliding window shape as `_median_chord_labels`."""
	if width <= 1 or len(flags) <= 1:
		return flags
	w = max(3, int(width))
	if w % 2 == 0:
		w += 1
	half = w // 2
	out: List[bool] = []
	for i in range(len(flags)):
		lo = max(0, i - half)
		hi = min(len(flags), i + half + 1)
		window = flags[lo:hi]
		out.append(sum(1 for x in window if x) * 2 > len(window))
	return out


def chord_timeline_sliding(
	chroma: np.ndarray,
	sr: int,
	key_raw: str | None = None,
	*,
	chord_preset: ChordAnalysisPreset,
	bass_chroma: Optional[np.ndarray] = None,
	slot_debug_out: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
	"""
	High-rate chord path: overlapping windows on a fine hop (sec-tuned), independent of beat boundaries.
	Median + sticky suppress vocal blips; segment edges land near harmonic evidence transitions.
	"""
	_ = key_raw
	templates = _analysis_chord_templates(chord_preset)
	t_frames = chroma.shape[1]
	if t_frames == 0:
		return []

	bcc: np.ndarray | None = None
	if chord_preset.use_bass_chroma and bass_chroma is not None:
		bcc = _align_bass_chroma_to_track(np.asarray(bass_chroma, dtype=float), t_frames)

	win, hop = _sliding_win_hop_frames(sr, t_frames)

	slot_starts: List[int] = []
	s = 0
	while s < t_frames:
		slot_starts.append(int(s))
		s += hop
	if slot_starts[-1] + win < t_frames:
		tail = max(0, t_frames - win)
		if tail not in slot_starts:
			slot_starts.append(int(tail))
	slot_starts = sorted(set(slot_starts))

	slot_labels: List[str] = []
	slot_conf: List[float] = []
	slot_low: List[bool] = []
	slot_best: List[float] = []
	slot_second: List[float] = []
	slot_vocal: List[bool] = []
	slot_reasons: List[List[str]] = []
	slot_arps: List[float] = []
	slot_bass_hints: List[int | None] = []
	prev_internal: str | None = None
	for s0 in slot_starts:
		e0 = min(t_frames, s0 + win)
		if e0 <= s0:
			e0 = min(s0 + 1, t_frames)
		win_slice = chroma[:, s0:e0]
		hist, arp = _blend_histogram_for_analyze_slot(win_slice, chord_preset)
		bass_hint: int | None = None
		if bcc is not None:
			b_win = bcc[:, s0:e0]
			bass_hint = _bass_root_hint_pc_for_window(b_win)
		slot_arps.append(float(arp))
		slot_bass_hints.append(bass_hint)
		_internal, label, _bs, _ss, conf, vocal, reasons = _best_analyze_slot(
			hist,
			templates,
			preset=chord_preset,
			prev_internal=prev_internal,
			arpeggio_support=arp,
			bass_root_pc=bass_hint,
		)
		if _internal != "N":
			prev_internal = _internal
		slot_labels.append(label)
		slot_conf.append(float(conf))
		slot_low.append(bool(conf < chord_preset.low_conf_cutoff or label == "N"))
		slot_best.append(float(_bs))
		slot_second.append(float(_ss))
		slot_vocal.append(bool(vocal))
		slot_reasons.append(list(reasons))

	med_w = CHORD_LABEL_MEDIAN_SLOTS
	filtered = _median_chord_labels(slot_labels, med_w)
	sticky_vocal = _median_bool_flags(slot_vocal, med_w)
	# Convert time-based sticky parameters to slot counts for this audio's hop size.
	_slot_sec = float(HOP_LENGTH * hop) / float(sr) if hop > 0 else 1.0
	_max_hold_slots = (
		max(1, int(chord_preset.max_sticky_hold_sec / _slot_sec))
		if chord_preset.max_sticky_hold_sec > 0.0
		else 0
	)
	_forced_window_slots = (
		max(1, int(chord_preset.sticky_forced_window_sec / _slot_sec))
		if chord_preset.sticky_forced_window_sec > 0.0
		else 0
	)
	stable_l, stable_c, stable_lo = _sticky_post_median_slots(
		filtered,
		slot_best,
		slot_second,
		slot_conf,
		slot_low,
		vocal_slots=sticky_vocal,
		vocal_sticky_mult=chord_preset.vocal_sticky_margin_mult,
		min_raw_margin=chord_preset.sticky_min_raw_margin,
		max_hold_slots=_max_hold_slots,
		forced_window_slots=_forced_window_slots,
	)
	_apply_nearby_label_stability(stable_c, stable_l)

	duration_sec = float(t_frames * HOP_LENGTH) / float(sr)

	def _mid_boundary_frame(left_slot_idx: int, right_slot_idx: int) -> int:
		left_edge = min(slot_starts[left_slot_idx] + win, t_frames)
		right_start = slot_starts[right_slot_idx]
		wb = float(CHORD_RUN_BOUNDARY_LEFT_BIAS)
		return int(round(wb * float(left_edge) + (1.0 - wb) * float(right_start)))

	runs: List[Tuple[int, int]] = []
	a_run = 0
	for k in range(1, len(stable_l) + 1):
		if k == len(stable_l) or stable_l[k] != stable_l[k - 1]:
			runs.append((a_run, k))
			a_run = k

	out: List[Dict[str, Any]] = []
	for ra, rb in runs:
		start_f = 0 if ra == 0 else _mid_boundary_frame(ra - 1, ra)
		end_f = t_frames if rb >= len(stable_l) else _mid_boundary_frame(rb - 1, rb)
		start_f = int(np.clip(start_f, 0, t_frames))
		end_f = int(np.clip(end_f, 0, t_frames))
		if end_f <= start_f:
			end_f = min(t_frames, start_f + 1)
		label = stable_l[ra]
		conf = float(max(stable_c[ra:rb]))
		# Use max aggregate confidence (not any-slot) so sticky-held slots don't poison the flag.
		low = (conf < chord_preset.low_conf_cutoff) or label == "N"
		t_sc = round(float(max(slot_best[ra:rb])), 4)
		br = range(ra, rb)
		t_mg = round(float(max(float(slot_best[i]) - float(slot_second[i]) for i in br)), 4)
		vocal_seg = any(slot_vocal[ra:rb])
		rs_union: List[str] = []
		for i in br:
			for r in slot_reasons[i]:
				if r not in rs_union:
					rs_union.append(r)
		t_arp = round(float(max(slot_arps[i] for i in br)), 4)
		bhints = [slot_bass_hints[i] for i in br if slot_bass_hints[i] is not None]
		row: Dict[str, Any] = {
			"start": round(float(start_f * HOP_LENGTH / sr), 4),
			"end": round(min(float(end_f * HOP_LENGTH / sr), duration_sec), 4),
			"label": label,
			"confidence": round(float(conf), 4),
			"low_confidence": bool(low),
			"template_score": t_sc,
			"template_margin": t_mg,
			"vocal_interference": bool(vocal_seg),
			"arpeggio_support": t_arp,
		}
		if bhints:
			cnt = Counter(bhints)
			row["bass_root_hint"] = int(cnt.most_common(1)[0][0])
		if rs_union:
			row["confidence_reasons"] = rs_union[:12]
		out.append(row)

	if slot_debug_out is not None:
		max_slots = 480
		n_slots = len(slot_starts)
		trunc = n_slots > max_slots
		sl = slice(0, min(n_slots, max_slots))
		bounds: List[float] = [0.0]
		for row in out:
			bounds.append(float(row["end"]))
		slot_debug_out.update(
			{
				"chord_engine": chord_preset.engine,
				"chord_analysis_preset": preset_for_debug(chord_preset),
				"chord_vocabulary_note": chord_vocab_description(
					chord_preset.vocabulary,
					include_sevenths=chord_preset.include_sevenths,
				),
				"CHORD_WINDOW_SEC": CHORD_WINDOW_SEC,
				"CHORD_HOP_SEC": CHORD_HOP_SEC,
				"BEAT_SNAP_MAX_SEC": BEAT_SNAP_MAX_SEC,
				"MIN_STABLE_REGION_SEC": MIN_STABLE_REGION_SEC,
				"CHORD_LABEL_MEDIAN_SLOTS": CHORD_LABEL_MEDIAN_SLOTS,
				"CHROMA_TIME_SMOOTH": chord_preset.chroma_time_smooth_frames,
				"CHORD_LOW_CONF_CUTOFF": CHORD_LOW_CONF_CUTOFF,
				"win_frames": win,
				"hop_frames": hop,
				"labels_raw_before_median": slot_labels[sl] if trunc else slot_labels,
				"labels_raw_template": slot_labels[sl] if trunc else slot_labels,
				"labels_after_median_before_sticky": filtered[sl] if trunc else filtered,
				"labels_after_median": filtered[sl] if trunc else filtered,
				"labels_after_sticky": stable_l[sl] if trunc else stable_l,
				"slot_start_times_sec": [round(float(s * HOP_LENGTH / sr), 4) for s in slot_starts][sl],
				"vocal_interference_slots": slot_vocal[sl] if trunc else slot_vocal,
				"arpeggio_support_slots": [round(float(x), 4) for x in slot_arps][sl] if trunc else [round(float(x), 4) for x in slot_arps],
				"bass_root_hint_slots": [x for x in slot_bass_hints][sl] if trunc else list(slot_bass_hints),
				"segment_boundary_times_sec": [round(float(b), 4) for b in sorted(set(bounds))],
				"slot_count": n_slots,
				"truncated": trunc,
				"slot_preprocess_note": (
					"histogram = mean/max blend + temporal max; arpeggio_support from PC spread; sqrt+L2 for template dots"
					if chord_preset.use_arpeggio_aggregate
					else "histogram = mean/max blend only (stable); arpeggio_support=0; sqrt+L2 for template dots"
				),
			},
		)

	return out


def chord_timeline(
	chroma: np.ndarray,
	sr: int,
	segment_seconds: float = SEGMENT_SECONDS,
	*,
	key_raw: str | None = None,
	chord_preset: ChordAnalysisPreset,
) -> List[Dict[str, Any]]:
	"""Non-overlapping segments; merge consecutive identical labels; HPSS chroma upstream."""
	templates = _analysis_chord_templates(chord_preset)
	t_frames = chroma.shape[1]
	if t_frames == 0:
		return []

	frames_per_seg = max(1, int(segment_seconds * sr / HOP_LENGTH))
	out: List[Dict[str, Any]] = []
	start_f = 0
	prev_internal = None
	while start_f < t_frames:
		end_f = min(start_f + frames_per_seg, t_frames)
		slice_c = chroma[:, start_f:end_f]
		hist, arp = _blend_histogram_for_analyze_slot(slice_c, chord_preset)
		_internal, label, bs, ss, conf, vocal, reasons = _best_analyze_slot(
			hist,
			templates,
			preset=chord_preset,
			prev_internal=prev_internal,
			arpeggio_support=float(arp),
			bass_root_pc=None,
		)
		if _internal != "N":
			prev_internal = _internal
		if label == "N":
			label = "N"
		t0 = start_f * HOP_LENGTH / sr
		t1 = end_f * HOP_LENGTH / sr
		low = conf < CHORD_LOW_CONF_CUTOFF or label == "N"
		row: Dict[str, Any] = {
			"start": round(t0, 4),
			"end": round(t1, 4),
			"label": label,
			"confidence": round(float(conf), 4),
			"low_confidence": bool(low),
			"template_score": round(float(bs), 4),
			"template_margin": round(float(bs - ss), 4),
			"vocal_interference": bool(vocal),
			"arpeggio_support": round(float(arp), 4),
		}
		if reasons:
			row["confidence_reasons"] = list(reasons)[:12]
		if out and out[-1]["label"] == label:
			out[-1]["end"] = row["end"]
			out[-1]["confidence"] = round(float(max(float(out[-1]["confidence"]), conf)), 4)
			out[-1]["low_confidence"] = bool(out[-1].get("low_confidence", False) or low)
			out[-1]["vocal_interference"] = bool(out[-1].get("vocal_interference") or vocal)
			prev_ts = float(out[-1].get("template_score", 0.0))
			prev_tm = float(out[-1].get("template_margin", 0.0))
			out[-1]["template_score"] = round(float(max(prev_ts, float(bs))), 4)
			out[-1]["template_margin"] = round(float(max(prev_tm, float(bs - ss))), 4)
			pa = float(out[-1].get("arpeggio_support", 0.0))
			out[-1]["arpeggio_support"] = round(float(max(pa, float(arp))), 4)
		else:
			out.append(row)
		start_f = end_f
	return out


def chord_timeline_beat_aligned(
	chroma: np.ndarray,
	sr: int,
	beat_times: List[float],
	duration_sec: float,
	*,
	key_raw: str | None = None,
	chord_preset: ChordAnalysisPreset,
) -> List[Dict[str, Any]]:
	"""
	One chord estimate per inter-beat interval; harmonic HPSS chroma only (percussion reduced).
	"""
	templates = _analysis_chord_templates(chord_preset)
	t_frames = chroma.shape[1]
	if t_frames == 0:
		return []

	boundaries = _boundaries_from_beats(beat_times, duration_sec)

	out: List[Dict[str, Any]] = []
	prev_internal = None
	for i in range(len(boundaries) - 1):
		t0, t1 = boundaries[i], boundaries[i + 1]
		if t1 <= t0 + 1e-9:
			continue
		f0 = int(t0 * sr / HOP_LENGTH)
		f1 = min(int(np.ceil(t1 * sr / HOP_LENGTH)), t_frames)
		if f1 <= f0:
			continue
		slice_c = chroma[:, f0:f1]
		hist, arp = _blend_histogram_for_analyze_slot(slice_c, chord_preset)
		_internal, label, bs, ss, conf, vocal, reasons = _best_analyze_slot(
			hist,
			templates,
			preset=chord_preset,
			prev_internal=prev_internal,
			arpeggio_support=float(arp),
			bass_root_pc=None,
		)
		if _internal != "N":
			prev_internal = _internal
		rt0, rt1 = round(t0, 4), round(t1, 4)
		low = conf < CHORD_LOW_CONF_CUTOFF or label == "N"
		row: Dict[str, Any] = {
			"start": rt0,
			"end": rt1,
			"label": label,
			"confidence": round(float(conf), 4),
			"low_confidence": bool(low),
			"template_score": round(float(bs), 4),
			"template_margin": round(float(bs - ss), 4),
			"vocal_interference": bool(vocal),
			"arpeggio_support": round(float(arp), 4),
		}
		if reasons:
			row["confidence_reasons"] = list(reasons)[:12]
		if out and out[-1]["label"] == label:
			out[-1]["end"] = rt1
			out[-1]["confidence"] = round(float(max(float(out[-1]["confidence"]), conf)), 4)
			out[-1]["low_confidence"] = bool(out[-1].get("low_confidence", False) or low)
			out[-1]["vocal_interference"] = bool(out[-1].get("vocal_interference") or vocal)
			prev_ts = float(out[-1].get("template_score", 0.0))
			prev_tm = float(out[-1].get("template_margin", 0.0))
			out[-1]["template_score"] = round(float(max(prev_ts, float(bs))), 4)
			out[-1]["template_margin"] = round(float(max(prev_tm, float(bs - ss))), 4)
			pa = float(out[-1].get("arpeggio_support", 0.0))
			out[-1]["arpeggio_support"] = round(float(max(pa, float(arp))), 4)
		else:
			out.append(row)
	return out


def _display_to_internal_rev(templates: Dict[str, np.ndarray]) -> Dict[str, str]:
	rev: Dict[str, str] = {}
	for k in templates.keys():
		if k == "N":
			continue
		rev[format_internal_chord_label(k)] = k
	return rev


def _match_pre_refine_segment(
	chords_pre: List[Dict[str, Any]],
	post: Dict[str, Any],
) -> tuple[Dict[str, Any] | None, float]:
	t0, t1 = float(post["start"]), float(post["end"])
	t_mid = 0.5 * (t0 + t1)
	best: Dict[str, Any] | None = None
	best_ov = -1.0
	for c in chords_pre:
		cs, ce = float(c["start"]), float(c["end"])
		ov = max(0.0, min(ce, t1) - max(cs, t0))
		if ov > best_ov + 1e-9:
			best_ov = ov
			best = c
		elif best is not None and abs(ov - best_ov) <= 1e-9:
			bcs, bce = float(best["start"]), float(best["end"])
			if cs <= t_mid <= ce and not (bcs <= t_mid <= bce):
				best = c
	if best is None or best_ov <= 1e-9:
		return None, 0.0
	return dict(best), float(best_ov)


def _refine_change_note(pre: Dict[str, Any] | None, post: Dict[str, Any]) -> str:
	if pre is None:
		return "no_pre_refine_overlap"
	pl, fl = str(pre.get("label")), str(post.get("label"))
	if pl == fl:
		if abs(float(pre["start"]) - float(post["start"])) > 0.02 or abs(float(pre["end"]) - float(post["end"])) > 0.02:
			return "label_unchanged_boundary_adjusted"
		return "unchanged"
	if bool(post.get("is_passing")):
		return "label_changed_marked_passing"
	return "label_changed_refinement"


def _analyze_segment_flags(post: Dict[str, Any]) -> List[str]:
	flags: List[str] = []
	if bool(post.get("low_confidence")):
		flags.append("low_confidence")
	if bool(post.get("is_passing")):
		flags.append("passing")
	if bool(post.get("vocal_interference")):
		flags.append("vocal_interference")
	if bool(post.get("exclude_from_core")):
		flags.append("exclude_from_core")
	cr = post.get("chord_role")
	if cr:
		flags.append(str(cr))
	cf = float(post.get("confidence", 0.5))
	if cf < 0.12:
		flags.append("unstable_confidence")
	reasons = post.get("confidence_reasons")
	if isinstance(reasons, list) and reasons:
		flags.append("confidence_reasons")
	return flags


def _build_analyze_debug(
	*,
	chords_pre_refine: List[Dict[str, Any]],
	chords_final: List[Dict[str, Any]],
	chroma: np.ndarray,
	sr: int,
	key_raw: str,
	chord_source: str,
	beat_times: List[float],
	templates: Dict[str, np.ndarray],
	chord_preset: ChordAnalysisPreset,
	chord_slot_granular: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
	label_to_internal = _display_to_internal_rev(templates)
	sample_n = 64
	bt_sample = [round(float(t), 4) for t in beat_times[:sample_n]]
	segments_dbg: List[Dict[str, Any]] = []
	prev_internal: str | None = None
	for post in chords_final:
		pre_match, overlap_sec = _match_pre_refine_segment(chords_pre_refine, post)
		change = _refine_change_note(pre_match, post)
		t0, t1 = float(post["start"]), float(post["end"])
		f0 = max(0, int(t0 * sr / HOP_LENGTH))
		f1 = min(int(np.ceil(t1 * sr / HOP_LENGTH)), chroma.shape[1])
		if f1 <= f0:
			f1 = min(f0 + 1, chroma.shape[1])
		slice_c = chroma[:, f0:f1]
		hist = (
			blend_chroma_mean_max(slice_c, w_mean=CHROMA_BLEND_W_MEAN, w_max=CHROMA_BLEND_W_MAX)
			if slice_c.shape[1] >= 1
			else aggregate_chroma(slice_c)
		)
		cands = chord_template_combined_candidates_debug(
			hist,
			templates,
			key_raw=key_raw,
			prev_internal=prev_internal,
			normalize_vector=_normalize_vector,
			top_k=8,
		)
		post_lab = str(post.get("label"))
		pi = label_to_internal.get(post_lab)
		if pi and pi != "N":
			prev_internal = pi
		pre_lab = None if pre_match is None else str(pre_match.get("label"))
		ts_v = post.get("template_score")
		tm_v = post.get("template_margin")
		segments_dbg.append(
			{
				"start": round(t0, 4),
				"end": round(t1, 4),
				"label_before_refinement": pre_lab,
				"label_after_refinement": post_lab,
				"refinement_change": change,
				"overlap_with_pre_sec": round(overlap_sec, 4),
				"confidence": round(float(post.get("confidence", 0.0)), 4),
				"template_score": round(float(ts_v), 4) if ts_v is not None else None,
				"template_margin": round(float(tm_v), 4) if tm_v is not None else None,
				"low_confidence": bool(post.get("low_confidence")),
				"is_passing": bool(post.get("is_passing")),
				"chord_role": post.get("chord_role"),
				"vocal_interference": bool(post.get("vocal_interference")),
				"confidence_reasons": post.get("confidence_reasons"),
				"exclude_from_core": bool(post.get("exclude_from_core")),
				"segment_flags": _analyze_segment_flags(post),
				"candidates_top": cands,
			},
		)

	vocab_note = chord_vocab_description(
		chord_preset.vocabulary,
		include_sevenths=chord_preset.include_sevenths,
	)
	return {
		"version": 3,
		"chord_engine": chord_preset.engine,
		"chord_analysis_preset": preset_for_debug(chord_preset),
		"chord_vocabulary_note": vocab_note,
		"chord_timeline_source": chord_source,
		"analysis_constants": {
			"CHORD_WINDOW_SEC": CHORD_WINDOW_SEC,
			"CHORD_HOP_SEC": CHORD_HOP_SEC,
			"CHORD_LABEL_MEDIAN_SLOTS": CHORD_LABEL_MEDIAN_SLOTS,
			"CHROMA_TIME_SMOOTH": chord_preset.chroma_time_smooth_frames,
			"MIN_CHORD_SEGMENT_SEC": MIN_CHORD_SEGMENT_SEC,
			"MIN_STABLE_REGION_SEC": MIN_STABLE_REGION_SEC,
			"BEAT_SNAP_MAX_SEC": BEAT_SNAP_MAX_SEC,
			"HARMONIC_CUSP_RADIUS_FRAMES": HARMONIC_CUSP_RADIUS_FRAMES,
			"CHROMA_BLEND_W_MEAN": CHROMA_BLEND_W_MEAN,
			"CHROMA_BLEND_W_MAX": CHROMA_BLEND_W_MAX,
			"CHORD_RUN_BOUNDARY_LEFT_BIAS": CHORD_RUN_BOUNDARY_LEFT_BIAS,
			"STICKY_MIN_BEST_SCORE": STICKY_MIN_BEST_SCORE,
			"STICKY_MIN_RAW_MARGIN": STICKY_MIN_RAW_MARGIN,
			"CHORD_LOW_CONF_CUTOFF": CHORD_LOW_CONF_CUTOFF,
			"TEMPO_PRACTICAL_MIN_BPM": TEMPO_PRACTICAL_MIN_BPM,
			"TEMPO_PRACTICAL_MAX_BPM": TEMPO_PRACTICAL_MAX_BPM,
			"TEMPO_HALF_IF_ABOVE_BPM": TEMPO_HALF_IF_ABOVE_BPM,
			"CORE_MIN_SEGMENT_SEC": CORE_MIN_SEGMENT_SEC,
			"CORE_MIN_CONFIDENCE": CORE_MIN_CONFIDENCE,
			"template_vocab": vocab_note,
			"chroma_match_preprocess": "per-slot sqrt compress + L2 norm before template dot",
			"hpss": "librosa.effects.hpss margin (harmonic, percussive) = (3.05, 2.12) in extract_chroma_track",
		},
		"key_selected_raw": key_raw,
		"key_candidates": key_ranked_candidates(aggregate_chroma(chroma), top_k=8),
		"beat_count": len(beat_times),
		"beat_times_sample": bt_sample,
		"beats_truncated": len(beat_times) > len(bt_sample),
		"segments_sliding_before_refine": [
			{
				"start": round(float(c["start"]), 4),
				"end": round(float(c["end"]), 4),
				"label": str(c.get("label")),
				"confidence": round(float(c.get("confidence", 0.0)), 4),
				"low_confidence": bool(c.get("low_confidence")),
				"vocal_interference": bool(c.get("vocal_interference")),
				"template_score": c.get("template_score"),
				"template_margin": c.get("template_margin"),
			}
			for c in chords_pre_refine
		],
		"segments_pre_refine_count": len(chords_pre_refine),
		"segments_after_refinement_count": len(chords_final),
		"refine_stages": [
			"spike_removal",
			"snap_weak_blips",
			"merge_chroma_neighbors",
			"collapse_short_merge",
			"beat_align_boundaries",
			"collapse_short_merge_again",
			"passing_annotate",
			"finalize_confidence_flags",
		],
		"segments": segments_dbg,
		"chord_slot_timeline": chord_slot_granular or {},
	}


def global_key_from_chroma(chroma: np.ndarray) -> Tuple[str, str, float]:
	hist = aggregate_chroma(chroma)
	raw, conf = estimate_key(hist)
	return _format_key_label(raw), raw, float(conf)


def _tempo_wave_from_stems(stems: StemBundle) -> np.ndarray:
	"""Prefer full mixture for rhythmic stability; add light drum emphasis once drum stem exists."""
	mix = np.asarray(stems.full_mix, dtype=np.float32).reshape(-1)
	if stems.drums is None:
		return mix
	dd = np.asarray(stems.drums, dtype=np.float32).reshape(-1)
	n = min(mix.shape[0], dd.shape[0])
	if n <= 0:
		return mix
	return np.clip(
		mix[:n].astype(np.float64) + 0.35 * dd[:n].astype(np.float64),
		-1.0,
		1.0,
	).astype(np.float32, copy=False)


def _mix_key_audio_from_stems(stems: StemBundle, chord_wave: np.ndarray) -> np.ndarray:
	"""
	Key profile from accompaniment + modest bass uplift when isolated bass stem exists.
	Reuses ``chord_wave`` buffer when no bass stem (avoids duplicate chroma extraction).
	"""
	if stems.bass is None:
		return chord_wave
	other = chord_wave
	bs = np.asarray(stems.bass, dtype=np.float32).reshape(-1)
	n = min(other.shape[0], bs.shape[0])
	if n <= 0:
		return chord_wave
	blended_head = np.clip(
		other[:n].astype(np.float64) + 0.45 * bs[:n].astype(np.float64),
		-1.0,
		1.0,
	).astype(np.float32, copy=False)
	if blended_head.shape[0] == other.shape[0]:
		return blended_head
	out = np.empty_like(other)
	out[: blended_head.shape[0]] = blended_head
	if blended_head.shape[0] < out.shape[0]:
		out[blended_head.shape[0] :] = other[blended_head.shape[0] :]
	return out


def _analyze_waveform(
	y: np.ndarray,
	sr: int,
	duration_sec: float,
	analysis_window: Dict[str, Any],
	*,
	debug: bool = False,
	use_source_separation: bool = False,
	engine: str | None = None,
	upload_size_bytes: int = 0,
) -> Dict[str, Any]:
	"""Run the full chord/tempo/key/section analysis on an already-decoded waveform.

	Separated from loading so both the bytes entry point (run_analysis) and the
	file-path entry point (run_analysis_from_path) can share this code without
	duplicating ~150 lines.
	"""
	separation_requested = bool(use_source_separation) or ENABLE_SOURCE_SEPARATION
	sep_result = separate_sources(y, sr, enabled=separation_requested)
	stems = sep_result.stems

	chord_wave = np.asarray(stems.other, dtype=np.float32).reshape(-1)
	tempo_wave = _tempo_wave_from_stems(stems)
	key_wave = _mix_key_audio_from_stems(stems, chord_wave)

	chord_preset = normalize_chord_engine(engine)
	smooth_frames = chord_preset.chroma_time_smooth_frames

	bpm, beat_times, tempo_meta = estimate_tempo_and_beats(tempo_wave, sr)

	chroma = extract_chroma_track(chord_wave, sr, use_hpss=True)
	chroma = _temporal_smooth_chroma(chroma, smooth_frames)
	bass_chroma: np.ndarray | None = None
	if chord_preset.use_bass_chroma:
		bass_chroma = extract_bass_chroma_track(chord_wave, sr, use_hpss=True)
		bass_chroma = _temporal_smooth_chroma(bass_chroma, smooth_frames)
	if key_wave is chord_wave:
		chroma_for_key = chroma
	else:
		chroma_raw_k = extract_chroma_track(key_wave, sr, use_hpss=True)
		chroma_for_key = _temporal_smooth_chroma(chroma_raw_k, smooth_frames)
	key_label, key_raw, key_conf = global_key_from_chroma(chroma_for_key)

	# --- Pitch / chord ML prelude (still no fusion) ---
	note_events = transcribe_pitch(stems.other, sr, enabled=ENABLE_PITCH_TRANSCRIPTION)
	ml_chords = predict_chords_ml(stems.other, sr, enabled=ENABLE_ML_CHORDS)
	_ = (note_events, ml_chords)

	chord_source = "sliding"
	slot_granular: Dict[str, Any] | None = {} if debug else None
	chords = chord_timeline_sliding(
		chroma,
		sr,
		key_raw=key_raw,
		chord_preset=chord_preset,
		bass_chroma=bass_chroma,
		slot_debug_out=slot_granular,
	)
	if not chords:
		chord_source = "equal_time_grid"
		chords = chord_timeline(
			chroma,
			sr,
			segment_seconds=SEGMENT_SECONDS,
			key_raw=key_raw,
			chord_preset=chord_preset,
		)
	need_pre_snapshot = debug or chord_preset.max_returned_segment_sec > 0
	chords_pre_snapshot = [dict(c) for c in chords] if need_pre_snapshot else []
	chords = refine_chord_timeline(
		chords,
		chroma,
		sr,
		beat_times=beat_times,
		duration_sec=duration_sec,
		snap_conf_threshold=chord_preset.snap_conf_threshold,
		low_conf_cutoff=chord_preset.low_conf_cutoff,
	)
	longest_before_guardrail = max(
		(float(c["end"]) - float(c["start"]) for c in chords), default=0.0
	)
	guardrail_applied = guardrail_splits = 0
	if chord_preset.max_returned_segment_sec > 0 and chords_pre_snapshot:
		chords, guardrail_applied, guardrail_splits = _apply_long_segment_guardrail(
			chords,
			chords_pre_snapshot,
			max_sec=chord_preset.max_returned_segment_sec,
			min_split_sec=2.5,
		)
	longest_after_guardrail = max(
		(float(c["end"]) - float(c["start"]) for c in chords), default=0.0
	)
	_annotate_core_eligibility_all(chords, low_conf_threshold=chord_preset.low_conf_cutoff)
	chords = [_enrich_chord_segment(c) for c in chords]

	beats_payload = [{"time": t} for t in beat_times]
	sections_raw = detect_sections(chroma, sr, beat_times, duration_sec)
	sections_raw = merge_short_sections(sections_raw, MIN_SECTION_DURATION_SEC, chroma, sr, duration_sec)
	sections_raw = split_long_sections(sections_raw, beat_times, duration_sec)
	sections_raw = merge_short_sections(sections_raw, MIN_SECTION_DURATION_SEC, chroma, sr, duration_sec)
	sections_labeled = relabel_sections_with_repetition(sections_raw, chroma, sr, duration_sec, chords)
	sections = sections_with_indices(sections_labeled)
	rhythm = compute_rhythm_hints(beat_times, DEFAULT_BEATS_PER_BAR)

	if log.isEnabledFor(logging.DEBUG):
		mean_cf = float(np.mean([float(c.get("confidence", 0.0)) for c in chords])) if chords else 0.0
		log.debug(
			"analyze: duration=%.2fs bpm=%.1f key=%r raw=%r kconf=%.2f segments=%d mean_conf=%.2f sep_backend=%s",
			duration_sec, bpm, key_label, key_raw, key_conf, len(chords), mean_cf, sep_result.backend,
		)

	simple_practice_prog, simple_debug = compute_simple_practice_progression(
		chords, duration_sec, debug=debug
	)

	payload: Dict[str, Any] = {
		"duration": round(duration_sec, 4),
		"tempo": round(bpm, 2),
		"key": {
			"label": key_label,
			"confidence": round(float(key_conf), 4),
		},
		"chord_engine": chord_preset.engine,
		"chords": chords,
		"beats": beats_payload,
		"sections": sections,
		"rhythm": rhythm,
		"simple_practice_progression": simple_practice_prog,
		"analysis_window": analysis_window,
	}
	if debug:
		dbg = _build_analyze_debug(
			chords_pre_refine=chords_pre_snapshot,
			chords_final=chords,
			chroma=chroma,
			sr=sr,
			key_raw=key_raw,
			chord_source=chord_source,
			beat_times=beat_times,
			templates=_analysis_chord_templates(chord_preset),
			chord_preset=chord_preset,
			chord_slot_granular=slot_granular,
		)
		dbg["source_separation_enabled"] = separation_requested
		dbg["source_separation_used"] = bool(sep_result.separation_used)
		dbg["source_separation_backend"] = sep_result.backend
		dbg["stem_available"] = dict(sep_result.stem_available)
		dbg["source_separation_warning"] = sep_result.warning
		dbg["source_separation_requested_client"] = bool(use_source_separation)
		dbg["source_separation_config_enabled"] = bool(ENABLE_SOURCE_SEPARATION)
		dbg["audio_routing_note"] = (
			"chords keyed off stems.other accompaniment path; tempo from full_mix+optional drums; "
			"global key blends other+bass stem when isolated bass exists."
		)
		if sep_result.meta:
			dbg["source_separation_meta"] = dict(sep_result.meta)
		dbg["raw_tempo_bpm"] = tempo_meta["raw_tempo_bpm"]
		dbg["normalized_tempo_bpm"] = tempo_meta["normalized_tempo_bpm"]
		dbg["tempo_reason"] = tempo_meta["tempo_reason"]
		dbg["tempo_candidates"] = tempo_meta.get("tempo_candidates", [])
		_sticky_labels = (slot_granular or {}).get("labels_after_sticky", [])
		_median_labels = (slot_granular or {}).get("labels_after_median", [])
		dbg.update(_filtering_debug_stats(
			chords_pre_snapshot, chords,
			raw_frame_count=int(chroma.shape[1]),
			guardrail_applied=guardrail_applied,
			guardrail_splits=guardrail_splits,
			longest_before_guardrail=longest_before_guardrail,
			longest_after_guardrail=longest_after_guardrail,
			post_sticky_segment_count=_count_label_runs(_sticky_labels),
			post_median_segment_count=_count_label_runs(_median_labels),
		))
		dbg["segments_pre_refine_count"] = len(chords_pre_snapshot)
		dbg["chord_quality_summary"] = _chord_quality_summary(chords)
		dbg["upload_size_bytes"] = upload_size_bytes
		dbg["beta_max_upload_size_mb"] = BETA_MAX_UPLOAD_SIZE_MB
		dbg["analysis_window"] = analysis_window
		dbg.update(simple_debug)
		payload["debug"] = dbg
	return payload


def run_analysis(
	audio_bytes: bytes,
	*,
	debug: bool = False,
	use_source_separation: bool = False,
	engine: str | None = None,
) -> Dict[str, Any]:
	"""Analyze audio from raw bytes.  Kept for backward compatibility with tests and scripts.

	For the production HTTP path, prefer run_analysis_from_path which streams the
	upload to disk and never holds the full file bytes in RAM.
	"""
	max_bytes = BETA_MAX_UPLOAD_SIZE_MB * 1024 * 1024
	if len(audio_bytes) > max_bytes:
		raise ValueError(
			f"file_too_large: uploaded file exceeds the {BETA_MAX_UPLOAD_SIZE_MB}MB beta limit."
		)

	y, sr = load_audio_bytes(audio_bytes, sr=ANALYSIS_SR, duration=BETA_ANALYSIS_DURATION_SEC)
	if y.size < sr * 0.2:
		raise ValueError("Audio is too short for analysis (need at least ~0.2s).")

	y_duration = float(len(y)) / float(sr)
	was_trimmed = y_duration >= BETA_ANALYSIS_DURATION_SEC - 0.5
	original_duration_sec = _probe_audio_duration_sec(audio_bytes) if was_trimmed else y_duration
	analysis_window: Dict[str, Any] = {
		"start": 0.0,
		"end": round(y_duration, 4),
		"duration_analyzed": round(y_duration, 4),
		"was_trimmed": was_trimmed,
		"original_duration": round(original_duration_sec, 4) if original_duration_sec is not None else None,
		"reason": "beta_duration_limit" if was_trimmed else None,
	}
	return _analyze_waveform(
		y, sr, y_duration, analysis_window,
		debug=debug,
		use_source_separation=use_source_separation,
		engine=engine,
		upload_size_bytes=len(audio_bytes),
	)


def run_analysis_from_path(
	audio_path: str,
	*,
	upload_size_bytes: int = 0,
	debug: bool = False,
	use_source_separation: bool = False,
	engine: str | None = None,
) -> Dict[str, Any]:
	"""Analyze audio from a file path — the memory-safe production entry point.

	Strategy (in order):
	1. If ffmpeg is available: pre-trim to BETA_ANALYSIS_DURATION_SEC seconds of mono WAV
	   at BETA_ANALYSIS_SAMPLE_RATE Hz.  librosa sees only that tiny WAV (~2 MB) and never
	   touches the full upload.
	2. Fallback: librosa duration cap — load directly from `audio_path` with duration= set.
	   Works for WAV/FLAC (soundfile seek) and MP3 (audioread progressive decode).

	Peak in-memory audio footprint with ffmpeg:  ~2 MB (60 s × 16 kHz float32).
	Peak in-memory audio footprint with fallback: ~6 MB (60 s × 22050 Hz float32).
	Either path is well within Render free-tier 512 MB.
	"""
	from app.audio.ffmpeg_trim import check_ffmpeg_available, ffmpeg_trim_to_wav, _safe_unlink

	log.info(
		"analyze: loading from temp file path=%s upload_bytes=%d beta_duration_sec=%.0f",
		audio_path, upload_size_bytes, BETA_ANALYSIS_DURATION_SEC,
	)

	trimmed_path: str | None = None
	try:
		if check_ffmpeg_available():
			try:
				trimmed_path = ffmpeg_trim_to_wav(
					audio_path,
					duration_sec=BETA_ANALYSIS_DURATION_SEC,
					sample_rate=BETA_ANALYSIS_SAMPLE_RATE,
				)
				analysis_src = trimmed_path
				use_duration_cap = False
				log.info("analyze: using ffmpeg-trimmed WAV path=%s", trimmed_path)
			except RuntimeError as exc:
				log.warning("ffmpeg trim failed (%s); falling back to librosa duration cap", exc)
				analysis_src = audio_path
				use_duration_cap = True
		else:
			log.info("ffmpeg not available; using librosa duration cap %.0f s", BETA_ANALYSIS_DURATION_SEC)
			analysis_src = audio_path
			use_duration_cap = True

		librosa_dur = BETA_ANALYSIS_DURATION_SEC if use_duration_cap else None
		y, sr = load_audio_from_path(analysis_src, sr=ANALYSIS_SR, duration=librosa_dur)
		if y.size < sr * 0.2:
			raise ValueError("Audio is too short for analysis (need at least ~0.2s).")

		y_duration = float(len(y)) / float(sr)
		was_trimmed = y_duration >= BETA_ANALYSIS_DURATION_SEC - 0.5
		original_duration_sec = _probe_audio_duration_from_path(audio_path) if was_trimmed else y_duration
		analysis_window: Dict[str, Any] = {
			"start": 0.0,
			"end": round(y_duration, 4),
			"duration_analyzed": round(y_duration, 4),
			"was_trimmed": was_trimmed,
			"original_duration": round(original_duration_sec, 4) if original_duration_sec is not None else None,
			"reason": "beta_duration_limit" if was_trimmed else None,
		}
		log.info(
			"analyze: waveform loaded duration=%.2fs was_trimmed=%s original_duration=%s ffmpeg=%s",
			y_duration, was_trimmed, original_duration_sec, not use_duration_cap,
		)
		return _analyze_waveform(
			y, sr, y_duration, analysis_window,
			debug=debug,
			use_source_separation=use_source_separation,
			engine=engine,
			upload_size_bytes=upload_size_bytes,
		)
	finally:
		_safe_unlink(trimmed_path)
