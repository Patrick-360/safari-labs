"""
Build POST /live-transcribe JSON from a full `run_analysis` dict.

Heuristic goals for *live* song windows (not instant /stream):
- Prefer stable, diatonic-friendly summaries; passing chords de-emphasized in core loop.
- Current chord = harmony near the *end* of the window (where "now" is for rolling capture).
- Keep logic explainable — complements music_theory / analyze_pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from app.audio.live_thresholds import LIVE_ROUTE_LIVE_TRANSCRIPTION, SEMANTIC_LIVE_TRANSCRIPTION


def _qualifying_live_segments(
	chords_raw: List[Dict[str, Any]],
	*,
	conf_floor: float,
) -> tuple[int, float]:
	"""Chunks that look like tonal evidence (templates were confident enough on /analyze heuristic path)."""

	n_ok = 0
	sec_ok = 0.0
	for c in chords_raw:
		lab = str(c.get("label", "N")).strip()
		if lab in ("", "N", "n"):
			continue
		if bool(c.get("low_confidence", False)):
			continue
		if float(c.get("confidence", 0.0)) + 1e-9 < conf_floor:
			continue
		n_ok += 1
		sec_ok += float(c["end"]) - float(c["start"])
	return n_ok, sec_ok


def _live_progression_publishable(entries: List[Dict[str, Any]], chords_raw: List[Dict[str, Any]]) -> tuple[bool, str | None]:
	"""
	Block single-noise-chip “progressions”: need ≥2 harmonic symbols unless many confident segments/time back one label.
	Raises empty_reason codes consumed by masked UI summaries.
	"""
	if len(entries) < 1:
		return False, None
	if len(entries) >= 2:
		return True, None
	conf_floor = 0.24
	ok_n, ok_sec = _qualifying_live_segments(chords_raw, conf_floor=conf_floor)
	# Sustain / drone: plenty of tonal time but only one harmonic label printed.
	if ok_sec >= 4.9:
		return True, None
	if ok_n >= 4 and ok_sec >= 3.2:
		return True, None
	if ok_n < 2:
		return False, "not_enough_harmonic_segment_count"
	return False, "need_second_progression_symbol"


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
		pub_ok, er = _live_progression_publishable(core_stable, chords_raw)
		if pub_ok:
			meta["source"] = "stable_core"
			if "use_high_confidence_runs_len_ge_2" in work_strategy:
				meta["quality"] = "likely"
			elif len(core_stable) >= 4:
				meta["quality"] = "stabilizing"
			else:
				meta["quality"] = "rough"
			return core_stable, meta
		meta["source"] = "none"
		meta["quality"] = "still_listening"
		meta["empty_reason"] = er or "still_listening"
		return [], meta

	fallback = _fallback_core_from_raw_chords(chords_raw, max_unique=8)
	if fallback:
		pub_ok, er = _live_progression_publishable(fallback, chords_raw)
		if pub_ok:
			meta["source"] = "fallback_time_order"
			meta["quality"] = "rough"
			meta["empty_reason"] = None
			return fallback, meta
		meta["source"] = "none"
		meta["quality"] = "still_listening"
		meta["empty_reason"] = er or "still_listening"
		return [], meta

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
	preflight_snapshot: Dict[str, Any] | None = None,
	listen_masked: bool = False,
	confident_live_segment_approx: int = 0,
	confident_live_duration_sec_approx: float = 0.0,
	live_transcription_preset_id: str | None = None,
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
		"version": 2,
		"window_duration_sec": round(float(window_duration_sec), 4),
		"window_start": round(float(window_start), 4),
		"window_end": round(float(window_start + window_duration_sec), 4),
		"live_transcription_preset_id": live_transcription_preset_id,
		"live_preset_semantic": SEMANTIC_LIVE_TRANSCRIPTION,
		"preflight_metrics": dict(preflight_snapshot) if isinstance(preflight_snapshot, dict) else {},
		"harmonic_listen_masked": bool(listen_masked),
		"confident_live_segment_approx": confident_live_segment_approx,
		"confident_live_duration_sec_approx": confident_live_duration_sec_approx,
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


_SUMMARY_HINTS_LISTEN_ONLY: Dict[str, str] = {
	"waiting_for_more_audio": "Need a slightly longer buffered slice.",
	"input_too_quiet": "Input is quiet — move closer, raise volume, or increase input boost.",
	"not_enough_harmonic_signal": "Not enough steady harmonic tone in this slice (noise / speech / silence).",
	"not_enough_confident_chord_time": "Heard tonal hints but chord evidence is too thin for a progression sketch.",
	"need_second_progression_symbol": "Still listening for a stable progression…",
	"not_enough_harmonic_segment_count": "Still listening for a stable progression…",
}


def build_live_listen_only_payload(
	window_start: float,
	window_end: float,
	*,
	session_id: str | None,
	reason_code: str,
	summary_hint: str | None = None,
	include_debug: bool = False,
	preflight_metrics: Dict[str, Any] | None = None,
	progression_was_updated: bool = False,
	core_empty_explanation: str | None = None,
) -> Dict[str, Any]:
	"""Response when skipping or discarding harmonic output (live mic safety)."""

	summary_text = (
		summary_hint
		if summary_hint is not None
		else _SUMMARY_HINTS_LISTEN_ONLY.get(
			str(reason_code),
			"Still listening for a stable progression…",
		)
	)
	out_dict: Dict[str, Any] = {
		"window_start": round(float(window_start), 4),
		"window_end": round(float(window_end), 4),
		"session_id": session_id,
		"key": {"label": "—", "confidence": 0.0},
		"current_chord": "—",
		"chords": [],
		"core_progression": [],
		"progression_meta": {
			"source": "none",
			"quality": "still_listening",
			"empty_reason": str(reason_code) if reason_code else None,
		},
		"summary": summary_text,
		"status": "listening",
		"tempo_bpm": 0.0,
	}
	if include_debug:
		out_dict["debug"] = {
			"version": 2,
			"live_preset_semantic": SEMANTIC_LIVE_TRANSCRIPTION,
			"live_route": LIVE_ROUTE_LIVE_TRANSCRIPTION,
			"listen_only": True,
			"listening_reason": reason_code,
			"rejection_reason": str(reason_code) if reason_code else "",
			"key_updated_this_window": False,
			"progression_updated_this_window": bool(progression_was_updated),
			"why_progression_empty": core_empty_explanation or reason_code,
			"preflight_metrics": preflight_metrics,
			"final_current_chord": "—",
		}
	return out_dict


def suppress_noisy_live_analysis(analysis: Dict[str, Any]) -> Tuple[bool, str]:
	"""
	Post-hoc suppression when `run_analysis` returns labels but tonal evidence spans too little time
	(typical noisy/silent mic captures after template loosening).
	"""
	duration = float(analysis.get("duration", 0.0))
	conf_floor = 0.247
	chords_raw: List[Dict[str, Any]] = list(analysis.get("chords") or [])
	evidence_sec = 0.0
	confident_segments = 0
	for c in chords_raw:
		lab = str(c.get("label", "N")).strip()
		if lab in ("N", "", "n"):
			continue
		if bool(c.get("low_confidence", False)):
			continue
		conf = float(c.get("confidence", 0.0))
		if conf < conf_floor:
			continue
		confident_segments += 1
		evidence_sec += float(c["end"]) - float(c["start"])

	if confident_segments <= 0:
		return True, "not_enough_harmonic_signal"
	if confident_segments < 2:
		return True, "not_enough_harmonic_signal"
	if duration + 1e-9 >= 14.5 and evidence_sec < 5.9:
		return True, "not_enough_confident_chord_time"
	if duration >= 9.25 and evidence_sec < 4.95:
		return True, "not_enough_confident_chord_time"
	if duration <= 9.24 and duration >= 7.05 and evidence_sec < 3.75:
		return True, "not_enough_confident_chord_time"
	if duration <= 7.05 and evidence_sec < 2.2:
		return True, "not_enough_confident_chord_time"
	return False, ""


def _apply_live_listen_mask(out: Dict[str, Any], *, empty_reason: str | None) -> None:
	rc = str(empty_reason or "")
	out["key"] = {"label": "—", "confidence": 0.0}
	out["current_chord"] = "—"
	out["chords"] = []
	out["core_progression"] = []
	out["tempo_bpm"] = 0.0
	out["summary"] = _SUMMARY_HINTS_LISTEN_ONLY.get(rc, "Still listening for a stable progression…")


def build_live_transcribe_from_analysis(
	analysis: Dict[str, Any],
	*,
	window_start: float,
	session_id: str | None,
	include_debug: bool = False,
	merged_timeline_seg_count: int | None = None,
	preflight_metrics: Dict[str, Any] | None = None,
	live_transcription_preset_id: str | None = None,
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
	qual_n, qual_sec = _qualifying_live_segments(chords_raw, conf_floor=0.24)

	current = _current_chord_at_end(chords_raw, duration)
	if current == "—" and core:
		current = core[0]["label"]

	tempo = float(analysis.get("tempo", 0.0))
	summary = build_summary(key_label, key_conf, core, tempo)

	core_nonempty = len(core) > 0
	status = "ready" if core_nonempty else "listening"
	listen_masked = not core_nonempty

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
	if listen_masked:
		_apply_live_listen_mask(out, empty_reason=str(progression_meta.get("empty_reason") or ""))
	if include_debug:
		dbg_pf = dict(preflight_metrics or {})
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
			preflight_snapshot=dbg_pf,
			listen_masked=listen_masked,
			confident_live_segment_approx=qual_n,
			confident_live_duration_sec_approx=round(qual_sec, 4),
			live_transcription_preset_id=live_transcription_preset_id,
		)
		pg_empty = progression_meta.get("empty_reason")
		explain = pg_empty if not core_nonempty else None
		if not explain and status == "listening":
			explain = "core_progression_empty_or_filtered"
		out["debug"]["live_route_active"] = LIVE_ROUTE_LIVE_TRANSCRIPTION
		out["debug"]["live_preset_semantic"] = SEMANTIC_LIVE_TRANSCRIPTION
		out["debug"]["final_current_chord"] = str(out["current_chord"])
		out["debug"]["listen_only"] = bool(listen_masked)
		out["debug"]["key_updated_this_window"] = bool(not listen_masked)
		out["debug"]["progression_updated_this_window"] = bool(core_nonempty and not listen_masked)
		out["debug"]["harmonic_listen_masked"] = bool(listen_masked)
		out["debug"]["why_progression_empty"] = explain
	return out
