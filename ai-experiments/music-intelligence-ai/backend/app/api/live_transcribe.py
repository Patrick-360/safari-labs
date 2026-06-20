"""
POST /live-transcribe — rolling-window analysis for *learn the song* live listening.

Reuses `run_analysis` (Analyze File-equivalent heavyweight path) once the rolling buffer passes
cheap **live-only** RMS / harmonic energy gates (`live_window_energy`).

Those gates are intentional: microphones often upload silence/noise that should not reuse the
same permissive tolerances Analyze File earns from full-context listening.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.audio.analyze_pipeline import run_analysis
from app.audio.features import load_audio_bytes_wav
from app.audio.live_thresholds import live_transcription_preset
from app.audio.live_transcribe_build import (
	build_live_listen_only_payload,
	build_live_transcribe_from_analysis,
	suppress_noisy_live_analysis,
)
from app.audio.live_window_energy import evaluate_live_transcription_window
from app.schemas.live_transcribe import LiveTranscribeKey, LiveTranscribeProgressionMeta, LiveTranscribeResponse

router = APIRouter()


@router.post("/live-transcribe", response_model=LiveTranscribeResponse)
async def live_transcribe(
	file: UploadFile = File(...),
	session_id: str | None = Query(default=None, description="Opaque client session id (echoed back)"),
	window_start: float = Query(default=0.0, ge=0.0, description="Client timeline offset for this window (seconds)"),
	mode: str = Query(default="song", description="Reserved for future tuning (e.g. song)"),
	debug: bool = Query(
		default=False,
		description="Include extended debug (forwards debug flag into run_analysis when analysis runs)",
	),
	client_timeline_seg_count: int | None = Query(
		default=None,
		ge=0,
		description="Optional: merged segment count from client rolling timeline for correlation",
	),
) -> LiveTranscribeResponse:
	"""Rolling-window harmonic sketch with live-energy preflight (never loosen /analyze internals)."""

	try:
		audio_bytes = await file.read()
		if not audio_bytes:
			raise HTTPException(status_code=400, detail="Empty upload.")
	except HTTPException:
		raise
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	try:
		y, sr = load_audio_bytes_wav(audio_bytes)
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	y = np.asarray(y, dtype=float).reshape(-1)
	audio_duration_sec = float(len(y)) / float(sr if sr > 0 else 1)
	window_start_r = round(float(window_start), 4)
	window_end_r = round(float(window_start + audio_duration_sec), 4)

	lt_preset = live_transcription_preset(mode)
	preflight_bundle = dict(lt_preset.__dict__)

	def _pf_dbg(metrics: Dict[str, Any]) -> Dict[str, Any]:
		out = dict(metrics)
		out.setdefault("live_transcription_preset_id", lt_preset.preset_id)
		for pk, pv in preflight_bundle.items():
			out.setdefault(f"preset_snapshot_{pk}", pv)
		return out

	pf = evaluate_live_transcription_window(
		y,
		sr,
		lt_preset,
		audio_duration_sec=audio_duration_sec,
	)
	if not pf.ok:
		kr = pf.reason_code
		if kr == "waiting_for_more_audio":
			return LiveTranscribeResponse(
				window_start=window_start_r,
				window_end=window_start_r,
				session_id=session_id,
				key=LiveTranscribeKey(label="—", confidence=0.0),
				current_chord="—",
				chords=[],
				core_progression=[],
				progression_meta=LiveTranscribeProgressionMeta(
					source="none",
					quality="still_listening",
					empty_reason="waiting_for_more_audio",
				),
				summary="Still listening — need a longer slice of audio before analysis.",
				status="listening",
				tempo_bpm=0.0,
				debug={"preflight_metrics": _pf_dbg(dict(pf.detail))} if debug else None,
			)
		payload = build_live_listen_only_payload(
			window_start_r,
			window_end_r,
			session_id=session_id,
			reason_code=kr,
			include_debug=debug,
			preflight_metrics=_pf_dbg(dict(pf.detail)),
		)
		return LiveTranscribeResponse.model_validate(payload)

	try:
		raw = run_analysis(audio_bytes, debug=debug)
	except ValueError:
		return LiveTranscribeResponse(
			window_start=window_start_r,
			window_end=window_start_r,
			session_id=session_id,
			key=LiveTranscribeKey(label="—", confidence=0.0),
			current_chord="—",
			chords=[],
			core_progression=[],
			progression_meta=LiveTranscribeProgressionMeta(
				source="none",
				quality="still_listening",
				empty_reason="waiting_for_more_audio",
			),
			summary="Still listening — need a longer slice of audio before analysis.",
			status="listening",
			tempo_bpm=0.0,
			debug={"preflight_metrics": _pf_dbg(dict(pf.detail))} if debug else None,
		)
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	bl, reason = suppress_noisy_live_analysis(raw)
	if bl:
		dbg_extra = _pf_dbg(dict(pf.detail))
		payload = build_live_listen_only_payload(
			window_start_r,
			window_end_r,
			session_id=session_id,
			reason_code=reason,
			include_debug=debug,
			preflight_metrics=dbg_extra,
			core_empty_explanation=reason,
		)
		if debug and isinstance(payload.get("debug"), dict):
			payload["debug"]["post_listen_suppress_reason"] = reason
			payload["debug"]["suppress_segment_summaries"] = {
				"duration_sec": raw.get("duration"),
				"chord_segments": len((raw.get("chords") or [])),
			}
		return LiveTranscribeResponse.model_validate(payload)

	payload = build_live_transcribe_from_analysis(
		raw,
		window_start=window_start,
		session_id=session_id,
		include_debug=debug,
		merged_timeline_seg_count=client_timeline_seg_count,
		preflight_metrics=_pf_dbg(dict(pf.detail)),
		live_transcription_preset_id=lt_preset.preset_id,
	)
	return LiveTranscribeResponse.model_validate(payload)
