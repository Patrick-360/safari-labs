from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import ValidationError

from app.audio.analyze_pipeline import run_analysis
from app.audio.chord_analysis_preset import DEFAULT_CHORD_ENGINE
from app.schemas.analyze import AnalyzeResponse

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_track(
	file: UploadFile = File(...),
	debug: bool = Query(
		default=False,
		description="When true, include a structured `debug` object for accuracy evaluation",
	),
	use_source_separation: bool = Query(
		default=False,
		description="When true, attempt optional accompaniment separation for chord/key paths (fallback safe if deps missing)",
	),
	engine: str | None = Query(
		default=None,
		description=(
			"Chord analysis engine preset: stable | theory | experimental. "
			f"Omit for default `{DEFAULT_CHORD_ENGINE}`."
		),
	),
) -> AnalyzeResponse:
	"""
	Full-song analysis: duration, tempo (BPM), global key + confidence, chord segments.
	Accepts WAV, MP3, and other formats supported by librosa/audioread.
	"""
	try:
		audio_bytes = await file.read()
		if not audio_bytes:
			raise HTTPException(
				status_code=400,
				detail={
					"error": "empty_upload",
					"message": "No audio bytes were received. Use form field name 'file', or ensure the file is non-empty.",
				},
			)
		payload = run_analysis(
			audio_bytes,
			debug=debug,
			use_source_separation=use_source_separation,
			engine=engine,
		)
		try:
			return AnalyzeResponse.model_validate(payload)
		except ValidationError as exc:
			raise HTTPException(
				status_code=500,
				detail={
					"error": "analyze_response_schema_mismatch",
					"message": "Analysis produced data that failed API schema validation (server bug).",
					"validation_errors": exc.errors(),
				},
			) from exc
	except HTTPException:
		raise
	except ValueError as exc:
		raise HTTPException(
			status_code=400,
			detail={"error": "invalid_audio", "message": str(exc)},
		) from exc
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(
			status_code=400,
			detail={
				"error": "analyze_failed",
				"message": str(exc),
				"exception_type": type(exc).__name__,
			},
		) from exc
