"""
Build POST /live-transcribe JSON from a full `run_analysis` dict.

Heuristic goals for *live* song windows (not instant /stream):
- Prefer stable, diatonic-friendly summaries; passing chords de-emphasized in core loop.
- Current chord = harmony near the *end* of the window (where "now" is for rolling capture).
- Keep logic explainable — complements music_theory / analyze_pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _runs_from_segments(chords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Merge adjacent same-label segments → chord runs with merged flags."""
	if not chords:
		return []
	out: List[Dict[str, Any]] = []
	for c in chords:
		label = str(c.get("label", "N"))
		if label == "N":
			continue
		row = {
			"label": label,
			"notes": list(c.get("notes") or []),
			"any_low": bool(c.get("low_confidence", False)),
			"is_passing": bool(c.get("is_passing", False)),
			"practice_hint": str(c.get("practice_hint") or ""),
		}
		if out and out[-1]["label"] == label:
			out[-1]["any_low"] = out[-1]["any_low"] or row["any_low"]
			out[-1]["is_passing"] = out[-1]["is_passing"] and row["is_passing"]
			if len(row["notes"]) > len(out[-1]["notes"]):
				out[-1]["notes"] = row["notes"]
			if row["practice_hint"] and len(row["practice_hint"]) > len(out[-1].get("practice_hint", "")):
				out[-1]["practice_hint"] = row["practice_hint"]
		else:
			out.append(row)
	return out


def _runs_for_core_progression(runs: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], str]:
	structural = [r for r in runs if not r["is_passing"]]
	base = structural if structural else runs
	parts: list[str] = []
	if structural:
		parts.append("non_passing_runs")
	else:
		parts.append("all_passing_fallback_full_runs")
	hi = [r for r in base if not r["any_low"]]
	if len(hi) >= 2:
		parts.append("use_high_confidence_runs_len_ge_2")
		return hi, ";".join(parts)
	if len(hi) == 1:
		parts.append("single_hi_span_use_base_non_N")
	else:
		parts.append("no_hi_span_use_base_non_N")
	return [r for r in base if r["label"] != "N"], ";".join(parts)


def _unique_in_order(labels: List[str]) -> List[str]:
	seen: set[str] = set()
	out: List[str] = []
	for l in labels:
		if l in seen:
			continue
		seen.add(l)
		out.append(l)
	return out


def _core_entries_from_work(work: List[Dict[str, Any]], max_unique: int) -> List[Dict[str, Any]]:
	if not work:
		return []
	labels = [r["label"] for r in work]
	unique = _unique_in_order(labels)[:max_unique]
	entries: List[Dict[str, Any]] = []
	for lab in unique:
		r = next((x for x in work if x["label"] == lab), None)
		entries.append(
			{
				"label": lab,
				"notes": list(r["notes"]) if r else [],
			},
		)
	return entries


def build_core_progression_entries(runs: List[Dict[str, Any]], max_unique: int = 10) -> List[Dict[str, Any]]:
	work, _ = _runs_for_core_progression(runs)
	return _core_entries_from_work(work, max_unique)


def _fallback_core_from_raw_chords(chords_raw: List[Dict[str, Any]], *, max_unique: int = 8) -> List[Dict[str, Any]]:
	"""
	Best-effort progression when strict core path yields nothing:
	time order, skip N, optionally skip tiny/unreliable blips, collapse consecutive duplicates,
	first-seen unique labels (cap).
	"""
	rows = [
		c
		for c in chords_raw
		if str(c.get("label", "N")).strip() not in ("N", "", "n")
	]
	if not rows:
		return []
	rows.sort(key=lambda c: float(c["start"]))
	seq_labels: List[str] = []
	for c in rows:
		lab = str(c["label"])
		dur = float(c["end"]) - float(c["start"])
		if dur < 0.05:
			continue
		low = bool(c.get("low_confidence", False))
		conf = float(c.get("confidence", 0.5))
		if low and conf < 0.12 and dur < 0.35:
			continue
		if not seq_labels or seq_labels[-1] != lab:
			seq_labels.append(lab)
	unique = _unique_in_order(seq_labels)[:max_unique]
	entries: List[Dict[str, Any]] = []
	for lab in unique:
		r = next((x for x in rows if str(x.get("label")) == lab), None)
		entries.append(
			{
				"label": lab,
				"notes": list(r.get("notes", [])) if r else [],
			},
		)
	return entries


def _raw_chord_stats(chords_raw: List[Dict[str, Any]]) -> Tuple[int, int, int]:
	"""counts: total segments, non-N segments, non-N with low_confidence."""
	total = len(chords_raw)
	non_n = 0
	low_non_n = 0
	for c in chords_raw:
		lab = str(c.get("label", "N")).strip()
		if lab in ("N", "", "n"):
			continue
		non_n += 1
		if bool(c.get("low_confidence", False)):
			low_non_n += 1
	return total, non_n, low_non_n


def _choose_progression_and_meta(
	chords_raw: List[Dict[str, Any]],
	runs: List[Dict[str, Any]],
	work: List[Dict[str, Any]],
	work_strategy: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
	"""
	Prefer stable core; else fallback from raw segments; set progression_meta for UI + debug.
	"""
	meta: Dict[str, Any] = {
		"source": "none",
		"quality": "still_listening",
		"empty_reason": None,
		"work_strategy": work_strategy,
	}

	core_stable = _core_entries_from_work(work, max_unique=10)
	if core_stable:
		meta["source"] = "stable_core"
		if "use_high_confidence_runs_len_ge_2" in work_strategy:
			meta["quality"] = "likely"
		elif len(core_stable) >= 4:
			meta["quality"] = "stabilizing"
		else:
			meta["quality"] = "rough"
		return core_stable, meta

	fallback = _fallback_core_from_raw_chords(chords_raw, max_unique=8)
	if fallback:
		meta["source"] = "fallback_time_order"
		meta["quality"] = "rough"
		meta["empty_reason"] = None
		return fallback, meta

	total, non_n, low_nn = _raw_chord_stats(chords_raw)
	if total == 0:
		meta["empty_reason"] = "no_chords"
	elif non_n == 0:
		meta["empty_reason"] = "all_invalid"
	elif non_n > 0 and low_nn >= non_n:
		meta["empty_reason"] = "all_low_confidence"
	else:
		meta["empty_reason"] = "not_enough_harmony"
	return [], meta


def _explain_core_empty(runs: List[Dict[str, Any]], work: List[Dict[str, Any]], core: List[Dict[str, Any]]) -> str | None:
	"""Legacy debug helper when strict core builder alone is empty."""
	if core:
		return None
	if not runs:
		return "no_chord_runs_all_segments_were_N"
	if not work:
		return "core_filter_removed_everything"
	non_n = [r for r in work if str(r.get("label", "N")) != "N"]
	if not non_n:
		return "work_list_has_only_placeholders"
	return "unexpected_empty_core"


def _build_live_transcribe_debug(
	analysis: Dict[str, Any],
	*,
	window_start: float,
	window_duration_sec: float,
	chords_raw: List[Dict[str, Any]],
	runs: List[Dict[str, Any]],
	work: List[Dict[str, Any]],
	work_strategy: str,
	core: List[Dict[str, Any]],
	merged_timeline_seg_count: int | None,
	progression_meta: Dict[str, Any],
) -> Dict[str, Any]:
	ad = analysis.get("debug") if isinstance(analysis.get("debug"), dict) else {}
	key_obj = analysis.get("key") or {}
	total_seg = len(chords_raw)
	stable = sum(1 for c in chords_raw if not bool(c.get("is_passing", False)))
	low_c = sum(1 for c in chords_raw if bool(c.get("low_confidence", False)))
	passing_c = sum(1 for c in chords_raw if bool(c.get("is_passing", False)))
	cand_labels = [e["label"] for e in core] if core else [r["label"] for r in work]
	ptotal, pnon_n, plow = _raw_chord_stats(chords_raw)
	dbg_empty = progression_meta.get("empty_reason")
	core_dbg = _explain_core_empty(runs, work, core)
	dbg: Dict[str, Any] = {
		"version": 1,
		"window_duration_sec": round(float(window_duration_sec), 4),
		"window_start": round(float(window_start), 4),
		"window_end": round(float(window_start + window_duration_sec), 4),
		"segment_count": total_seg,
		"stable_segment_count": stable,
		"low_confidence_segment_count": low_c,
		"passing_segment_count": passing_c,
		"runs_count": len(runs),
		"runs_for_core_count": len(work),
		"runs_for_core_strategy": work_strategy,
		"core_progression_candidates": cand_labels,
		"core_empty_reason": core_dbg,
		"progression_source": progression_meta.get("source"),
		"progression_quality": progression_meta.get("quality"),
		"progression_empty_reason": dbg_empty,
		"raw_non_n_segment_count": pnon_n,
		"raw_low_conf_non_n_count": plow,
		"key_candidates": ad.get("key_candidates"),
		"analysis_key": {
			"label": key_obj.get("label"),
			"confidence": key_obj.get("confidence"),
		},
		"client_timeline_merged_seg_count": merged_timeline_seg_count,
	}
	if not core and not dbg_empty:
		dbg["progression_note"] = "filtered_by_merge_or_internal_inconsistency"
	return dbg


def _current_chord_at_end(chords: List[Dict[str, Any]], duration_sec: float) -> str:
	"""Chord active in the last ~300ms of the window (melody-resistant vs very end sample)."""
	if not chords:
		return "—"
	t = max(0.0, float(duration_sec) - 0.3)
	for i, c in enumerate(chords):
		end = float(duration_sec) if i == len(chords) - 1 else float(c["end"])
		if float(c["start"]) <= t <= end + 1e-3:
			lab = str(c.get("label", "N"))
			return "—" if lab == "N" else lab
	last = str(chords[-1].get("label", "N"))
	return "—" if last == "N" else last


def enrich_chords_for_response(chords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	out: List[Dict[str, Any]] = []
	for c in chords:
		label = str(c.get("label", "N"))
		low = bool(c.get("low_confidence", False))
		out.append(
			{
				"start": round(float(c["start"]), 4),
				"end": round(float(c["end"]), 4),
				"label": label,
				"confidence": round(float(c.get("confidence", 0.0)), 4),
				"notes": list(c.get("notes") or []),
				"practice_hint": str(c.get("practice_hint") or ""),
				"low_confidence": low,
				"is_passing": bool(c.get("is_passing", False)),
				"chord_role": c.get("chord_role"),
			}
		)
	return out


def build_summary(key_label: str, key_conf: float, core: List[Dict[str, Any]], tempo: float) -> str:
	chain = " → ".join(c["label"] for c in core[:8]) if core else "(still learning)"
	conf_word = "fair" if key_conf >= 0.45 else "tentative"
	return (
		f"Likely key ({conf_word}): {key_label}. "
		f"Main chords spotted in this window: {chain}. "
		f"Rough tempo ~{tempo:.0f} BPM (for reference only). "
		"Use full file analysis for the cleanest chart."
	)


def build_live_transcribe_from_analysis(
	analysis: Dict[str, Any],
	*,
	window_start: float,
	session_id: str | None,
	include_debug: bool = False,
	merged_timeline_seg_count: int | None = None,
) -> Dict[str, Any]:
	"""Shape `run_analysis` output into /live-transcribe contract."""
	duration = float(analysis.get("duration", 0.0))
	key_obj = analysis.get("key") or {"label": "—", "confidence": 0.0}
	key_label = str(key_obj.get("label", "—"))
	key_conf = float(key_obj.get("confidence", 0.0))

	chords_raw: List[Dict[str, Any]] = list(analysis.get("chords") or [])
	chords_out = enrich_chords_for_response(chords_raw)
	runs = _runs_from_segments(chords_raw)
	work, work_strategy = _runs_for_core_progression(runs)
	core, progression_meta = _choose_progression_and_meta(chords_raw, runs, work, work_strategy)

	current = _current_chord_at_end(chords_raw, duration)
	if current == "—" and core:
		current = core[0]["label"]

	tempo = float(analysis.get("tempo", 0.0))
	summary = build_summary(key_label, key_conf, core, tempo)

	status = "ready"
	if progression_meta.get("source") == "none" and progression_meta.get("empty_reason"):
		status = "listening"

	out: Dict[str, Any] = {
		"window_start": round(float(window_start), 4),
		"window_end": round(float(window_start + duration), 4),
		"session_id": session_id,
		"key": {"label": key_label, "confidence": round(key_conf, 4)},
		"current_chord": current,
		"chords": chords_out,
		"core_progression": core,
		"progression_meta": {
			"source": progression_meta["source"],
			"quality": progression_meta["quality"],
			"empty_reason": progression_meta.get("empty_reason"),
		},
		"summary": summary,
		"status": status,
		"tempo_bpm": round(tempo, 2),
	}
	if include_debug:
		out["debug"] = _build_live_transcribe_debug(
			analysis,
			window_start=window_start,
			window_duration_sec=duration,
			chords_raw=chords_raw,
			runs=runs,
			work=work,
			work_strategy=work_strategy,
			core=core,
			merged_timeline_seg_count=merged_timeline_seg_count,
			progression_meta=progression_meta,
		)
	return out
