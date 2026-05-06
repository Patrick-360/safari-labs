"""
POST /live-transcribe — rolling-window analysis for *learn the song* live listening.

Reuses `run_analysis` (same code path as /analyze) on each uploaded WAV window.
Not optimized for instant feedback; tuned for longer audio context.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.audio.analyze_pipeline import run_analysis
from app.audio.live_transcribe_build import build_live_transcribe_from_analysis
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
		description="Include extended debug (and forward to /analyze-equivalent pipeline when True)",
	),
	client_timeline_seg_count: int | None = Query(
		default=None,
		ge=0,
		description="Optional: merged segment count from client rolling timeline for correlation",
	),
) -> LiveTranscribeResponse:
	_ = mode  # MVP: same analysis path; mode reserved for future presets
	try:
		audio_bytes = await file.read()
		if not audio_bytes:
			raise HTTPException(status_code=400, detail="Empty upload.")
	except HTTPException:
		raise
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	try:
		raw = run_analysis(audio_bytes, debug=debug)
	except ValueError:
		# Too short / unusable — tell client to keep buffering
		return LiveTranscribeResponse(
			window_start=round(window_start, 4),
			window_end=round(window_start, 4),
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
		)
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	payload = build_live_transcribe_from_analysis(
		raw,
		window_start=window_start,
		session_id=session_id,
		include_debug=debug,
		merged_timeline_seg_count=client_timeline_seg_count,
	)
	return LiveTranscribeResponse.model_validate(payload)
