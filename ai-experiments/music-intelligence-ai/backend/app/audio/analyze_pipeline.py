"""
Offline /analyze pipeline (heuristic — not ML transcription):

1. **Load audio**: mono @ ANALYSIS_SR via librosa.
2. **Harmonic extraction**: HPSS harmonic stem feeds chord chroma; percussive energy is not used for harmony
   (rhythm uses the full waveform only for beat_track elsewhere).
   Future: optional `source_separation.separate_harmonic_stems()` could replace / blend this step.
3. **Time–frequency**: CQT chroma on harmonic stem + light temporal smoothing + mean/max blend per window
   (arpeggio-friendly).
4. **Beats**: librosa.beat_track on full `y` → boundaries for sections / optional timing snap (not mandatory chord grid).
5. **Chord candidates**: cosine vs sparse templates; **music_theory.pick_chord_with_theory** fuses audio + key + progression
   + melody sparsity penalty; extended qualities optional.
6. **Timeline**: overlapping sliding windows → median + sticky hysteresis → merged segments.
7. **Key**: whole-track Krumhansl estimate (global); biases scoring softly, never hard-forces.
8. **Sections**: beat- or time-grid chroma similarity merge + repetition fingerprint.
9. **Core progression** (frontend): derived from merged runs; excludes passing / low-confidence per client helpers.

Comments in code mark approximations (jazz/hip-hop honesty, etc.).
"""

from __future__ import annotations

import io
import logging
from collections import Counter
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np

from app.audio.chord_spellings import playable_triad_notes_and_hint
from app.audio.features import CHROMA_BINS, aggregate_chroma, estimate_key
from app.audio.music_theory import (
	blend_chroma_mean_max,
	likely_passing_segment,
	pick_chord_with_theory,
)
from app.models.chords import build_chord_templates

log = logging.getLogger(__name__)

# Chroma / segmentation (same hop as librosa CQT frames in this module).
HOP_LENGTH = 512
ANALYSIS_SR = 22050
# Sliding-window chord timeline: wider window + wider majority vote → fewer melody-led single-slot flips.
SLIDE_WIN_FRAMES = 10
SLIDE_HOP_FRAMES = 2
# Majority chord label over this many adjacent slots reduces vocal-led spikes before segment merge.
SLIDE_LABEL_MEDIAN = 9
# Moving average chroma along time (frames); dampens brief melodic energy that is not the accompaniment.
CHROMA_TIME_SMOOTH = 7
# Drop quiet CQT bins before chroma aggregation (reduces transient / broad-band bleed — heuristic gate).
CHROMA_CQT_THRESHOLD = 0.025
# Key diatonic bias moved to music_theory.key_fit_bonus.
STICKY_MIN_BEST_SCORE = 0.41
STICKY_MIN_RAW_MARGIN = 0.032
STICKY_CONF_CAP = 0.22
# Fallback when chroma is very short.
SEGMENT_SECONDS = 0.35
# Merge adjacent beat- or time-windows when mean-chroma cosine similarity >= this (fewer, longer sections).
SECTION_MERGE_COS = 0.66
# Match distant section bodies to same "Section A" when chroma prototypes agree (or chroma+chord pattern agree).
SECTION_REPEAT_SIM_THRESHOLD = 0.82
# If chroma alone is marginal, still repeat-label when chord fingerprints align.
SECTION_REPEAT_CHROMA_LO = 0.76
SECTION_REPEAT_FP_SIM = 0.58
# Drop / absorb chord segments shorter than this after refinement (vocal flutter).
MIN_CHORD_SEGMENT_SEC = 0.32
# Replace isolated one-beat-wonder labels shorter than this with neighbor harmony.
SPIKE_ISOLATE_MAX_SEC = 0.52
# Low-confidence segment shorter than this whose chroma matches previous chord → keep previous (melody note).
SNAP_WEAK_MAX_SEC = 0.62
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
# Snap segment edges to nearest beat when within this (s).
BEAT_SNAP_MAX_SEC = 0.09
# Search ± this many chroma frames around a boundary for strongest harmonic change.
HARMONIC_CUSP_RADIUS_FRAMES = 6
PASSING_MAX_SEGMENT_SEC = 0.48
# Heuristic grouping for practice UI: assume 4/4, first detected beat = bar downbeat (not aligned to real meter).
DEFAULT_BEATS_PER_BAR = 4


def load_audio_bytes(data: bytes, sr: int = ANALYSIS_SR) -> Tuple[np.ndarray, int]:
	"""Load WAV/MP3/etc. via librosa; mono, resampled to `sr`."""
	if not data:
		raise ValueError("audio_bytes is empty.")
	with io.BytesIO(data) as buf:
		y, _ = librosa.load(buf, sr=sr, mono=True)
	if y.size == 0:
		raise ValueError("Decoded waveform is empty.")
	return y.astype(np.float32, copy=False), sr


def estimate_tempo_and_beats(y: np.ndarray, sr: int) -> Tuple[float, List[float]]:
	"""
	Global tempo (BPM) + beat times (seconds) using librosa.beat.beat_track.
	Tempo may fall back via onset strength; beat times stay from the primary beat_track pass.
	"""
	tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
	beat_times_arr = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
	beat_list = [round(float(t), 4) for t in np.asarray(beat_times_arr).reshape(-1)]

	arr = np.asarray(tempo, dtype=float).reshape(-1)
	if arr.size == 0:
		bpm = 120.0
	else:
		bpm = float(arr[0])
	if not np.isfinite(bpm) or bpm < 30.0 or bpm > 320.0:
		onset_env = librosa.onset.onset_strength(y=y, sr=sr)
		t2 = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=np.median)
		t2a = np.asarray(t2, dtype=float).reshape(-1)
		if t2a.size > 0 and np.isfinite(t2a[0]) and 30.0 <= t2a[0] <= 320.0:
			bpm = float(t2a[0])
		else:
			bpm = 120.0
	return bpm, beat_list


def extract_chroma_track(y: np.ndarray, sr: int, use_hpss: bool = True) -> np.ndarray:
	"""
	Shape (12, T_frames). Prefer harmonic HPSS stem for steadier tonal features vs full mix.

	Future: optional `app.audio.source_separation.separate_harmonic_stems` could feed chroma
	instead of HPSS (not enabled in MVP).
	"""
	waveform = np.asarray(y, dtype=float)
	if use_hpss:
		# Wider harmonic margin → more sustained harmonic energy, less percussive chroma (approximate separation).
		harmonic, _ = librosa.effects.hpss(waveform, margin=(2.85, 2.05))
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


def _format_key_label(raw: str) -> str:
	if raw.endswith(":maj"):
		return raw.replace(":maj", " major")
	return raw.replace(":min", " minor")


def _analysis_chord_templates() -> Dict[str, np.ndarray]:
	"""Full template set for /analyze (triads + sevenths + dim/aug/sus/m7b5); stream/live unchanged elsewhere."""
	return build_chord_templates(include_sevenths=True, include_extended=True)


def _best_template_full(
	chroma_hist: np.ndarray,
	templates: Dict[str, np.ndarray],
	*,
	key_raw: str | None = None,
	prev_internal: str | None = None,
) -> Tuple[str, str, float, float, float]:
	"""Audio + lightweight theory (key / progression / melody sparsity / continuity)."""
	from app.models.chords import _normalize_vector

	return pick_chord_with_theory(
		chroma_hist,
		templates,
		key_raw=key_raw,
		prev_internal=prev_internal,
		normalize_vector=_normalize_vector,
	)


def _merge_adjacent_chord_labels(chords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	out: List[Dict[str, Any]] = []
	for c in chords:
		if out and out[-1]["label"] == c["label"]:
			out[-1]["end"] = c["end"]
			out[-1]["confidence"] = float(max(out[-1].get("confidence", 0.0), c.get("confidence", 0.0)))
			out[-1]["low_confidence"] = bool(out[-1].get("low_confidence", False) or c.get("low_confidence", False))
			out[-1]["is_passing"] = False
			out[-1]["chord_role"] = None
			ts0 = out[-1].get("template_score")
			ts1 = c.get("template_score")
			if ts0 is not None and ts1 is not None:
				out[-1]["template_score"] = float(max(float(ts0), float(ts1)))
			tm0 = out[-1].get("template_margin")
			tm1 = c.get("template_margin")
			if tm0 is not None and tm1 is not None:
				out[-1]["template_margin"] = float(max(float(tm0), float(tm1)))
		else:
			row = dict(c)
			row.setdefault("is_passing", False)
			row.setdefault("chord_role", None)
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
) -> List[Dict[str, Any]]:
	"""
	If a short segment is low-confidence or very weakly scored but its chroma is very similar to the
	previous chord pocket, keep the prior chord (melody / passing tone — no source separation).
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
			or float(work[i].get("confidence", 1.0)) < SNAP_EXTRA_CONF_THRESHOLD
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
		row["is_passing"] = bool(is_p)
		row["chord_role"] = "passing" if is_p else None
	return out


def _finalize_chord_confidence_flags(chords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	for c in chords:
		cf = float(c.get("confidence", 0.5))
		low = bool(c.get("low_confidence", False) or cf < CHORD_LOW_CONF_CUTOFF)
		c["low_confidence"] = low
		c["confidence"] = round(cf, 4)
		c.setdefault("is_passing", False)
		c.setdefault("chord_role", None)
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
) -> List[Dict[str, Any]]:
	"""Post-pass: spikes, fragments, chroma-merge, harmonic boundary nudge, passing tags."""
	if not chords:
		return []
	dur = float(duration_sec) if duration_sec is not None else float(chords[-1]["end"])
	x = _remove_chord_spikes(chords)
	x = _snap_weak_chord_blips_to_prev(x, chroma, sr)
	x = _merge_chroma_similar_neighbors(x, chroma, sr)
	x = _collapse_short_chord_segments(x, MIN_CHORD_SEGMENT_SEC)
	x = _merge_adjacent_chord_labels(x)
	if beat_times is not None:
		x = _align_chord_segment_boundaries(x, chroma, sr, beat_times, dur)
	x = _collapse_short_chord_segments(x, MIN_CHORD_SEGMENT_SEC)
	x = _merge_adjacent_chord_labels(x)
	x = _annotate_passing_chords(x)
	return _finalize_chord_confidence_flags(x)


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
) -> Tuple[List[str], List[float], List[bool]]:
	"""
	When a *change* is weakly supported vs the runner-up template, hold the previous chord.
	Approximates hysteresis so brief vocal chroma peaks flip harmony less often.
	"""
	if not labels:
		return labels, confs, lows
	out_labels = [labels[0]]
	out_confs = [confs[0]]
	out_lows = [lows[0]]
	for i in range(1, len(labels)):
		cand = labels[i]
		prev = out_labels[-1]
		if cand == prev:
			out_labels.append(cand)
			out_confs.append(confs[i])
			out_lows.append(lows[i])
			continue
		bs = best_scores[i]
		raw_m = bs - second_scores[i]
		weak = bs < STICKY_MIN_BEST_SCORE or raw_m < STICKY_MIN_RAW_MARGIN
		if weak:
			out_labels.append(prev)
			out_confs.append(min(float(confs[i]), STICKY_CONF_CAP))
			out_lows.append(True)
		else:
			out_labels.append(cand)
			out_confs.append(confs[i])
			out_lows.append(lows[i])
	return out_labels, out_confs, out_lows


def chord_timeline_sliding(chroma: np.ndarray, sr: int, key_raw: str | None = None) -> List[Dict[str, Any]]:
	"""
	High-rate chord path: short CQT windows on a small hop (overlap), independent of beat boundaries.
	Median + sticky label passes suppress vocal blips before segment merge.
	"""
	templates = _analysis_chord_templates()
	t_frames = chroma.shape[1]
	if t_frames == 0:
		return []

	win = max(2, min(SLIDE_WIN_FRAMES, t_frames))
	hop = max(1, min(SLIDE_HOP_FRAMES, max(1, win - 1)))

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
	prev_internal: str | None = None
	for s0 in slot_starts:
		e0 = min(t_frames, s0 + win)
		if e0 <= s0:
			e0 = min(s0 + 1, t_frames)
		win_slice = chroma[:, s0:e0]
		hist = blend_chroma_mean_max(win_slice) if win_slice.shape[1] >= 1 else aggregate_chroma(win_slice)
		_internal, label, _bs, _ss, conf = _best_template_full(
			hist,
			templates,
			key_raw=key_raw,
			prev_internal=prev_internal,
		)
		if _internal != "N":
			prev_internal = _internal
		slot_labels.append(label)
		slot_conf.append(float(conf))
		slot_low.append(bool(conf < CHORD_LOW_CONF_CUTOFF or label == "N"))
		slot_best.append(float(_bs))
		slot_second.append(float(_ss))

	filtered = _median_chord_labels(slot_labels, SLIDE_LABEL_MEDIAN)
	stable_l, stable_c, stable_lo = _sticky_post_median_slots(
		filtered,
		slot_best,
		slot_second,
		slot_conf,
		slot_low,
	)

	out: List[Dict[str, Any]] = []
	for k in range(len(slot_starts)):
		t0 = slot_starts[k] * float(HOP_LENGTH) / float(sr)
		next_start = slot_starts[k + 1] if k + 1 < len(slot_starts) else t_frames
		t1 = next_start * float(HOP_LENGTH) / float(sr)
		label = stable_l[k]
		conf = stable_c[k]
		low = bool(stable_lo[k] or label == "N")
		t_sc = round(float(slot_best[k]), 4)
		t_mg = round(float(slot_best[k] - slot_second[k]), 4)
		row = {
			"start": round(float(t0), 4),
			"end": round(float(t1), 4),
			"label": label,
			"confidence": round(float(conf), 4),
			"low_confidence": low,
			"template_score": t_sc,
			"template_margin": t_mg,
		}
		if out and out[-1]["label"] == label:
			out[-1]["end"] = row["end"]
			out[-1]["confidence"] = round(float(max(float(out[-1]["confidence"]), conf)), 4)
			out[-1]["low_confidence"] = bool(out[-1]["low_confidence"] or low)
			prev_ts = float(out[-1].get("template_score", 0.0))
			prev_tm = float(out[-1].get("template_margin", 0.0))
			out[-1]["template_score"] = round(float(max(prev_ts, t_sc)), 4)
			out[-1]["template_margin"] = round(float(max(prev_tm, t_mg)), 4)
		else:
			out.append(row)
	return out


def chord_timeline(
	chroma: np.ndarray,
	sr: int,
	segment_seconds: float = SEGMENT_SECONDS,
	*,
	key_raw: str | None = None,
) -> List[Dict[str, Any]]:
	"""Non-overlapping segments; merge consecutive identical labels; HPSS chroma upstream."""
	templates = _analysis_chord_templates()
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
		hist = blend_chroma_mean_max(slice_c) if slice_c.shape[1] >= 1 else aggregate_chroma(slice_c)
		_internal, label, bs, ss, conf = _best_template_full(
			hist,
			templates,
			key_raw=key_raw,
			prev_internal=prev_internal,
		)
		if _internal != "N":
			prev_internal = _internal
		if label == "N":
			label = "N"
		t0 = start_f * HOP_LENGTH / sr
		t1 = end_f * HOP_LENGTH / sr
		low = conf < CHORD_LOW_CONF_CUTOFF or label == "N"
		row = {
			"start": round(t0, 4),
			"end": round(t1, 4),
			"label": label,
			"confidence": round(float(conf), 4),
			"low_confidence": bool(low),
			"template_score": round(float(bs), 4),
			"template_margin": round(float(bs - ss), 4),
		}
		if out and out[-1]["label"] == label:
			out[-1]["end"] = row["end"]
			out[-1]["confidence"] = round(float(max(float(out[-1]["confidence"]), conf)), 4)
			prev_ts = float(out[-1].get("template_score", 0.0))
			prev_tm = float(out[-1].get("template_margin", 0.0))
			out[-1]["template_score"] = round(float(max(prev_ts, float(bs))), 4)
			out[-1]["template_margin"] = round(float(max(prev_tm, float(bs - ss))), 4)
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
) -> List[Dict[str, Any]]:
	"""
	One chord estimate per inter-beat interval; harmonic HPSS chroma only (percussion reduced).
	"""
	templates = _analysis_chord_templates()
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
		hist = blend_chroma_mean_max(slice_c) if slice_c.shape[1] >= 1 else aggregate_chroma(slice_c)
		_internal, label, bs, ss, conf = _best_template_full(
			hist,
			templates,
			key_raw=key_raw,
			prev_internal=prev_internal,
		)
		if _internal != "N":
			prev_internal = _internal
		rt0, rt1 = round(t0, 4), round(t1, 4)
		low = conf < CHORD_LOW_CONF_CUTOFF or label == "N"
		row = {
			"start": rt0,
			"end": rt1,
			"label": label,
			"confidence": round(float(conf), 4),
			"low_confidence": bool(low),
			"template_score": round(float(bs), 4),
			"template_margin": round(float(bs - ss), 4),
		}
		if out and out[-1]["label"] == label:
			out[-1]["end"] = rt1
			out[-1]["confidence"] = round(float(max(float(out[-1]["confidence"]), conf)), 4)
			prev_ts = float(out[-1].get("template_score", 0.0))
			prev_tm = float(out[-1].get("template_margin", 0.0))
			out[-1]["template_score"] = round(float(max(prev_ts, float(bs))), 4)
			out[-1]["template_margin"] = round(float(max(prev_tm, float(bs - ss))), 4)
		else:
			out.append(row)
	return out


def global_key_from_chroma(chroma: np.ndarray) -> Tuple[str, str, float]:
	hist = aggregate_chroma(chroma)
	raw, conf = estimate_key(hist)
	return _format_key_label(raw), raw, float(conf)


def run_analysis(audio_bytes: bytes) -> Dict[str, Any]:
	y, sr = load_audio_bytes(audio_bytes, sr=ANALYSIS_SR)
	if y.size < sr * 0.2:
		raise ValueError("Audio is too short for analysis (need at least ~0.2s).")

	duration_sec = float(len(y)) / float(sr)
	bpm, beat_times = estimate_tempo_and_beats(y, sr)
	chroma = extract_chroma_track(y, sr, use_hpss=True)
	chroma = _temporal_smooth_chroma(chroma, CHROMA_TIME_SMOOTH)
	key_label, key_raw, key_conf = global_key_from_chroma(chroma)

	chords = chord_timeline_sliding(chroma, sr, key_raw=key_raw)
	if not chords:
		chords = chord_timeline(chroma, sr, segment_seconds=SEGMENT_SECONDS, key_raw=key_raw)

	chords = refine_chord_timeline(chords, chroma, sr, beat_times=beat_times, duration_sec=duration_sec)
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
			"analyze: duration=%.2fs bpm=%.1f key=%r raw=%r kconf=%.2f segments=%d mean_conf=%.2f",
			duration_sec,
			bpm,
			key_label,
			key_raw,
			key_conf,
			len(chords),
			mean_cf,
		)

	return {
		"duration": round(duration_sec, 4),
		"tempo": round(bpm, 2),
		"key": {
			"label": key_label,
			"confidence": round(float(key_conf), 4),
		},
		"chords": chords,
		"beats": beats_payload,
		"sections": sections,
		"rhythm": rhythm,
	}
