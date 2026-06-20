#!/usr/bin/env python3
"""
Compare Analyze File chord engines against hand-labeled benchmark clips.

Run from backend/ (venv active):

    python scripts/evaluate_chord_accuracy.py

Add clips under benchmarks/audio/ and matching JSON under benchmarks/annotations/.
See benchmarks/README.md.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Tuple

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
	sys.path.insert(0, str(BACKEND_ROOT))

import librosa
import numpy as np
import soundfile as sf

from app.audio.analyze_pipeline import run_analysis  # noqa: E402

BENCHMARKS_ROOT = BACKEND_ROOT / "benchmarks"
AUDIO_DIR = BENCHMARKS_ROOT / "audio"
ANNOTATIONS_DIR = BENCHMARKS_ROOT / "annotations"
RESULTS_DIR = BENCHMARKS_ROOT / "results"
DEFAULT_RESULTS_PATH = RESULTS_DIR / "latest.json"

DEFAULT_ENGINES = ("stable", "theory", "experimental")
DEFAULT_BIN_SEC = 0.05
EXAMPLE_AUDIO = "synthetic_c_g_am_f.wav"
EXAMPLE_JSON = "synthetic_c_g_am_f.json"
SYNTH_SR = 22050


def write_example_clip() -> None:
	"""Create a small synthetic progression WAV + annotation for smoke-testing the benchmark."""

	def hz(note: str) -> float:
		return float(librosa.note_to_hz(note))

	def triad(n1: str, n2: str, n3: str, seconds: float) -> np.ndarray:
		t = np.linspace(0.0, seconds, int(SYNTH_SR * seconds), endpoint=False, dtype=np.float64)
		sig = (
			np.sin(2 * np.pi * hz(n1) * t)
			+ np.sin(2 * np.pi * hz(n2) * t)
			+ np.sin(2 * np.pi * hz(n3) * t)
		) * 0.32
		return sig.astype(np.float32)

	parts = [
		(triad("C4", "E4", "G4", 1.1), "C", 0.0),
		(triad("G3", "B3", "D4", 1.1), "G", 1.1),
		(triad("A3", "C4", "E4", 1.1), "Am", 2.2),
		(triad("F3", "A3", "C4", 1.1), "F", 3.3),
	]
	y = np.concatenate([p[0] for p in parts])
	AUDIO_DIR.mkdir(parents=True, exist_ok=True)
	ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
	wav_path = AUDIO_DIR / EXAMPLE_AUDIO
	buf = io.BytesIO()
	sf.write(buf, y, SYNTH_SR, format="WAV", subtype="PCM_16")
	wav_path.write_bytes(buf.getvalue())
	chords = [{"start": start, "end": start + 1.1, "label": lab} for _, lab, start in parts]
	ann = {
		"title": "Synthetic C–G–Am–F",
		"audio_file": EXAMPLE_AUDIO,
		"key": "C major",
		"chords": chords,
	}
	(ANNOTATIONS_DIR / EXAMPLE_JSON).write_text(
		json.dumps(ann, indent=2) + "\n",
		encoding="utf-8",
	)
	print(f"Wrote {wav_path}")
	print(f"Wrote {ANNOTATIONS_DIR / EXAMPLE_JSON}")


@dataclass(frozen=True)
class Segment:
	start: float
	end: float
	label: str


@dataclass
class ClipAnnotation:
	title: str
	audio_file: str
	audio_path: Path
	key: str
	chords: List[Segment]
	source_json: str


@dataclass
class EngineClipMetrics:
	engine: str
	key_predicted: str
	key_match: bool
	accuracy_weighted_pct: float
	accuracy_unweighted_pct: float
	annotated_seconds: float
	matched_seconds: float
	mean_start_error_sec: float | None
	mean_end_error_sec: float | None
	boundary_pairs: int
	confusion_seconds: Dict[str, Dict[str, float]] = field(default_factory=dict)
	segment_count_predicted: int = 0


def normalize_chord_label(label: str) -> str:
	t = (label or "").strip()
	if not t or t.upper() == "N":
		return "N"
	return t.replace(" ", "")


def normalize_key_label(label: str) -> str:
	"""Loose string compare for 'C major' vs 'c  major'."""
	return " ".join((label or "").strip().lower().split())


def parse_segments(raw: Iterable[dict[str, Any]]) -> List[Segment]:
	out: List[Segment] = []
	for row in raw:
		start = float(row["start"])
		end = float(row["end"])
		if end <= start:
			continue
		out.append(
			Segment(
				start=start,
				end=end,
				label=normalize_chord_label(str(row.get("label", "N"))),
			),
		)
	return sorted(out, key=lambda s: s.start)


def label_at_time(segments: List[Segment], t: float) -> str:
	for seg in segments:
		if seg.start <= t < seg.end - 1e-9:
			return seg.label
	# Last sample at end boundary
	if segments and abs(t - segments[-1].end) < 1e-6:
		return segments[-1].label
	return "N"


def timeline_span(segments: List[Segment]) -> Tuple[float, float]:
	if not segments:
		return 0.0, 0.0
	return float(segments[0].start), float(max(s.end for s in segments))


def overlap_duration(a0: float, a1: float, b0: float, b1: float) -> float:
	return max(0.0, min(a1, b1) - max(a0, b0))


def score_timeline(
	gt: List[Segment],
	pred: List[Segment],
	*,
	t0: float,
	t1: float,
	bin_sec: float,
) -> Tuple[float, float, float, float, DefaultDict[Tuple[str, str], float]]:
	"""Returns matched_sec, annotated_sec, unweighted_hits, unweighted_total, confusion (gt,pred)->sec."""

	matched = 0.0
	annotated = 0.0
	hits = 0
	total = 0
	confusion: DefaultDict[Tuple[str, str], float] = defaultdict(float)

	t = t0
	while t < t1 - 1e-12:
		t_end = min(t + bin_sec, t1)
		dur = t_end - t
		mid = t + 0.5 * dur
		gl = label_at_time(gt, mid)
		if gl == "N":
			t = t_end
			continue
		pl = label_at_time(pred, mid)
		annotated += dur
		confusion[(gl, pl)] += dur
		if gl == pl:
			matched += dur
			hits += 1
		total += 1
		t = t_end

	return matched, annotated, float(hits), float(total), confusion


def timing_errors(gt: List[Segment], pred: List[Segment]) -> Tuple[float | None, float | None, int]:
	"""Mean |Δstart| and |Δend| when a predicted segment overlaps GT with the same label."""

	start_errs: List[float] = []
	end_errs: List[float] = []
	pairs = 0

	for g in gt:
		if g.label == "N":
			continue
		best: Segment | None = None
		best_ov = 0.0
		for p in pred:
			if p.label != g.label:
				continue
			ov = overlap_duration(g.start, g.end, p.start, p.end)
			if ov > best_ov + 1e-9:
				best_ov = ov
				best = p
		if best is None or best_ov < 0.05:
			continue
		pairs += 1
		start_errs.append(abs(best.start - g.start))
		end_errs.append(abs(best.end - g.end))

	if pairs == 0:
		return None, None, 0
	return (
		sum(start_errs) / float(pairs),
		sum(end_errs) / float(pairs),
		pairs,
	)


def confusion_to_nested(conf: DefaultDict[Tuple[str, str], float]) -> Dict[str, Dict[str, float]]:
	out: Dict[str, Dict[str, float]] = {}
	for (g, p), sec in sorted(conf.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])):
		if sec < 1e-6:
			continue
		out.setdefault(g, {})[p] = round(float(sec), 4)
	return out


def evaluate_clip_engine(
	annotation: ClipAnnotation,
	engine: str,
	*,
	bin_sec: float,
) -> EngineClipMetrics:
	raw_bytes = annotation.audio_path.read_bytes()
	analysis = run_analysis(raw_bytes, debug=False, engine=engine)

	pred_key = str((analysis.get("key") or {}).get("label", ""))
	key_match = normalize_key_label(pred_key) == normalize_key_label(annotation.key)

	pred_segments = parse_segments(analysis.get("chords") or [])

	gt_t0, gt_t1 = timeline_span(annotation.chords)
	if gt_t1 <= gt_t0:
		return EngineClipMetrics(
			engine=engine,
			key_predicted=pred_key,
			key_match=key_match,
			accuracy_weighted_pct=0.0,
			accuracy_unweighted_pct=0.0,
			annotated_seconds=0.0,
			matched_seconds=0.0,
			mean_start_error_sec=None,
			mean_end_error_sec=None,
			boundary_pairs=0,
			segment_count_predicted=len(pred_segments),
		)

	matched, annotated, hits, total, confusion = score_timeline(
		annotation.chords,
		pred_segments,
		t0=gt_t0,
		t1=gt_t1,
		bin_sec=bin_sec,
	)

	w_pct = 100.0 * matched / annotated if annotated > 1e-9 else 0.0
	u_pct = 100.0 * hits / total if total > 1e-9 else 0.0
	ms, me, pairs = timing_errors(annotation.chords, pred_segments)

	return EngineClipMetrics(
		engine=engine,
		key_predicted=pred_key,
		key_match=key_match,
		accuracy_weighted_pct=round(w_pct, 2),
		accuracy_unweighted_pct=round(u_pct, 2),
		annotated_seconds=round(annotated, 4),
		matched_seconds=round(matched, 4),
		mean_start_error_sec=round(ms, 4) if ms is not None else None,
		mean_end_error_sec=round(me, 4) if me is not None else None,
		boundary_pairs=pairs,
		confusion_seconds=confusion_to_nested(confusion),
		segment_count_predicted=len(pred_segments),
	)


def load_annotations() -> List[ClipAnnotation]:
	if not ANNOTATIONS_DIR.is_dir():
		return []

	clips: List[ClipAnnotation] = []
	for path in sorted(ANNOTATIONS_DIR.glob("*.json")):
		if path.name.startswith("_"):
			continue
		data = json.loads(path.read_text(encoding="utf-8"))
		audio_name = str(data.get("audio_file", "")).strip()
		if not audio_name:
			print(f"skip {path.name}: missing audio_file", file=sys.stderr)
			continue
		audio_path = AUDIO_DIR / audio_name
		if not audio_path.is_file():
			print(f"skip {path.name}: audio not found at {audio_path}", file=sys.stderr)
			continue
		chords_raw = data.get("chords")
		if not isinstance(chords_raw, list) or not chords_raw:
			print(f"skip {path.name}: chords must be a non-empty list", file=sys.stderr)
			continue
		clips.append(
			ClipAnnotation(
				title=str(data.get("title") or path.stem),
				audio_file=audio_name,
				audio_path=audio_path,
				key=str(data.get("key", "")),
				chords=parse_segments(chords_raw),
				source_json=path.name,
			),
		)
	return clips


def aggregate_engine_summary(
	clip_results: List[Tuple[ClipAnnotation, EngineClipMetrics]],
) -> Dict[str, Any]:
	by_engine: DefaultDict[str, List[EngineClipMetrics]] = defaultdict(list)
	for _clip, m in clip_results:
		by_engine[m.engine].append(m)

	summary: Dict[str, Any] = {}
	for engine, rows in sorted(by_engine.items()):
		n = len(rows)
		w_acc = sum(r.accuracy_weighted_pct for r in rows) / float(n)
		u_acc = sum(r.accuracy_unweighted_pct for r in rows) / float(n)
		key_rate = 100.0 * sum(1 for r in rows if r.key_match) / float(n)
		starts = [r.mean_start_error_sec for r in rows if r.mean_start_error_sec is not None]
		ends = [r.mean_end_error_sec for r in rows if r.mean_end_error_sec is not None]
		summary[engine] = {
			"clips": n,
			"mean_accuracy_weighted_pct": round(w_acc, 2),
			"mean_accuracy_unweighted_pct": round(u_acc, 2),
			"key_match_rate_pct": round(key_rate, 2),
			"mean_start_error_sec": round(sum(starts) / len(starts), 4) if starts else None,
			"mean_end_error_sec": round(sum(ends) / len(ends), 4) if ends else None,
		}
	return summary


def print_report(
	clips: List[ClipAnnotation],
	all_metrics: List[Tuple[ClipAnnotation, List[EngineClipMetrics]]],
	*,
	engines: Tuple[str, ...],
	bin_sec: float,
) -> None:
	print()
	print("Chord benchmark — Analyze File engines")
	print(f"  clips: {len(clips)}   engines: {', '.join(engines)}   bin: {bin_sec}s")
	print()

	# Per-clip table
	header = f"{'Clip':<28}" + "".join(f"{e:>12}" for e in engines) + f"{'key GT':>14}"
	print(header)
	print("-" * len(header))
	for clip, per_engine in all_metrics:
		cells = []
		for eng in engines:
			row = next((m for m in per_engine if m.engine == eng), None)
			cells.append(f"{row.accuracy_weighted_pct:>11.1f}%" if row else "       —")
		print(f"{clip.title[:28]:<28}" + "".join(cells) + f"{clip.key:>14}")

	print()
	print("Summary (mean weighted accuracy %, key match rate %)")
	sum_header = f"{'Engine':<14}{'Wtd acc %':>12}{'Unwt acc %':>12}{'Key match':>12}{'Δstart s':>10}{'Δend s':>10}"
	print(sum_header)
	print("-" * len(sum_header))

	flat: List[Tuple[ClipAnnotation, EngineClipMetrics]] = []
	for clip, per_engine in all_metrics:
		for m in per_engine:
			flat.append((clip, m))
	agg = aggregate_engine_summary(flat)
	for eng in engines:
		s = agg.get(eng)
		if not s:
			continue
		ds = s["mean_start_error_sec"]
		de = s["mean_end_error_sec"]
		print(
			f"{eng:<14}"
			f"{s['mean_accuracy_weighted_pct']:>12.2f}"
			f"{s['mean_accuracy_unweighted_pct']:>12.2f}"
			f"{s['key_match_rate_pct']:>11.1f}%"
			f"{ds if ds is not None else '—':>10}"
			f"{de if de is not None else '—':>10}"
		)

	print()
	print("Per-clip detail (weighted %, key, segments, top confusions)")
	for clip, per_engine in all_metrics:
		print(f"\n  {clip.title} ({clip.audio_file})  GT key: {clip.key}")
		for m in per_engine:
			kflag = "✓" if m.key_match else "✗"
			print(
				f"    {m.engine:<12}  wtd {m.accuracy_weighted_pct:5.1f}%  "
				f"unwt {m.accuracy_unweighted_pct:5.1f}%  key {kflag} ({m.key_predicted})  "
				f"pred segs {m.segment_count_predicted}"
			)
			if m.confusion_seconds:
				mis = []
				for gt_lab, preds in m.confusion_seconds.items():
					for pred_lab, sec in preds.items():
						if gt_lab != pred_lab and sec >= 0.25:
							mis.append(f"{gt_lab}→{pred_lab} {sec:.1f}s")
				if mis:
					print(f"      confusion: {', '.join(mis[:8])}")


def build_results_payload(
	clips: List[ClipAnnotation],
	all_metrics: List[Tuple[ClipAnnotation, List[EngineClipMetrics]]],
	*,
	engines: Tuple[str, ...],
	bin_sec: float,
) -> Dict[str, Any]:
	clip_payloads: List[Dict[str, Any]] = []
	flat: List[Tuple[ClipAnnotation, EngineClipMetrics]] = []

	for clip, per_engine in all_metrics:
		eng_map: Dict[str, Any] = {}
		for m in per_engine:
			flat.append((clip, m))
			eng_map[m.engine] = {
				"key_predicted": m.key_predicted,
				"key_match": m.key_match,
				"accuracy_weighted_pct": m.accuracy_weighted_pct,
				"accuracy_unweighted_pct": m.accuracy_unweighted_pct,
				"annotated_seconds": m.annotated_seconds,
				"matched_seconds": m.matched_seconds,
				"mean_start_error_sec": m.mean_start_error_sec,
				"mean_end_error_sec": m.mean_end_error_sec,
				"boundary_pairs": m.boundary_pairs,
				"segment_count_predicted": m.segment_count_predicted,
				"confusion_seconds": m.confusion_seconds,
			}
		clip_payloads.append(
			{
				"title": clip.title,
				"audio_file": clip.audio_file,
				"annotation_file": clip.source_json,
				"ground_truth": {
					"key": clip.key,
					"chord_segment_count": len(clip.chords),
					"chords": [
						{"start": s.start, "end": s.end, "label": s.label}
						for s in clip.chords
					],
				},
				"engines": eng_map,
			},
		)

	return {
		"generated_at": datetime.now(timezone.utc).isoformat(),
		"bin_sec": bin_sec,
		"engines": list(engines),
		"clip_count": len(clips),
		"clips": clip_payloads,
		"summary_by_engine": aggregate_engine_summary(flat),
	}


def main() -> int:
	parser = argparse.ArgumentParser(description="Evaluate Analyze File chord engines against benchmarks.")
	parser.add_argument(
		"--engines",
		default=",".join(DEFAULT_ENGINES),
		help="Comma-separated engines: stable,theory,experimental",
	)
	parser.add_argument(
		"--results",
		type=Path,
		default=DEFAULT_RESULTS_PATH,
		help="Output JSON path (default: benchmarks/results/latest.json)",
	)
	parser.add_argument(
		"--bin-sec",
		type=float,
		default=DEFAULT_BIN_SEC,
		help="Timeline sample step in seconds (default: 0.05)",
	)
	parser.add_argument(
		"--write-example",
		action="store_true",
		help="Write synthetic_c_g_am_f.wav + JSON example clip, then exit",
	)
	args = parser.parse_args()

	if args.write_example:
		write_example_clip()
		return 0

	engines = tuple(e.strip() for e in args.engines.split(",") if e.strip())
	if not engines:
		print("No engines specified.", file=sys.stderr)
		return 1

	bin_sec = max(0.01, float(args.bin_sec))
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)

	clips = load_annotations()
	if not clips:
		print(
			"No benchmark clips found.\n"
			f"  Add audio under: {AUDIO_DIR}\n"
			f"  Add JSON under:  {ANNOTATIONS_DIR}  (see _template.json)\n"
			f"  Docs:            {BENCHMARKS_ROOT / 'README.md'}",
			file=sys.stderr,
		)
		return 1

	all_metrics: List[Tuple[ClipAnnotation, List[EngineClipMetrics]]] = []
	for clip in clips:
		print(f"Analyzing: {clip.title} ({clip.audio_file}) …", flush=True)
		per_engine: List[EngineClipMetrics] = []
		for engine in engines:
			print(f"  engine={engine} …", flush=True)
			per_engine.append(evaluate_clip_engine(clip, engine, bin_sec=bin_sec))
		all_metrics.append((clip, per_engine))

	payload = build_results_payload(clips, all_metrics, engines=engines, bin_sec=bin_sec)
	args.results.parent.mkdir(parents=True, exist_ok=True)
	args.results.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

	print_report(clips, all_metrics, engines=engines, bin_sec=bin_sec)
	print(f"\nWrote {args.results}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
