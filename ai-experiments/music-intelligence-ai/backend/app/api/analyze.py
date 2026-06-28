from __future__ import annotations

import logging
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import ValidationError

from app.audio.analyze_pipeline import run_analysis_from_path
from app.audio.chord_analysis_preset import DEFAULT_CHORD_ENGINE
from app.core.config import BETA_MAX_UPLOAD_SIZE_MB
from app.schemas.analyze import AnalyzeResponse

log = logging.getLogger(__name__)
router = APIRouter()

_MAX_UPLOAD_BYTES = BETA_MAX_UPLOAD_SIZE_MB * 1024 * 1024
_CHUNK_SIZE = 1024 * 1024  # 1 MB per read call


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

	The upload is streamed to a temporary file on disk in 1 MB chunks so the full
	file bytes are never held in RAM.  Only the first 90 seconds are decoded and
	analyzed, keeping peak memory usage well within Render 512 MB limits.
	"""
	tmp_path: str | None = None
	try:
		# --- Stream upload to a temp file, counting bytes to enforce the size limit ---
		original_name = file.filename or "upload"
		ext = os.path.splitext(original_name)[1].lower() or ".audio"
		try:
			tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="analyze_")
			total_bytes = 0
			with os.fdopen(tmp_fd, "wb") as tmp_file:
				while True:
					chunk = await file.read(_CHUNK_SIZE)
					if not chunk:
						break
					total_bytes += len(chunk)
					if total_bytes > _MAX_UPLOAD_BYTES:
						# Discard the rest and reject — file is already too large.
						raise HTTPException(
							status_code=400,
							detail={
								"error": "file_too_large",
								"message": (
									f"File exceeds the {BETA_MAX_UPLOAD_SIZE_MB}MB beta limit. "
									"Try an MP3 under 30MB or a shorter clip."
								),
							},
						)
					tmp_file.write(chunk)
		except HTTPException:
			raise
		except OSError as exc:
			raise HTTPException(
				status_code=500,
				detail={
					"error": "temp_file_error",
					"message": "Could not save upload to disk for analysis.",
				},
			) from exc

		if total_bytes == 0:
			raise HTTPException(
				status_code=400,
				detail={
					"error": "empty_upload",
					"message": "No audio bytes were received. Use form field name 'file', or ensure the file is non-empty.",
				},
			)

		log.info(
			"analyze: upload spooled to temp file bytes=%d path=%s",
			total_bytes, tmp_path,
		)

		# --- Decode only the first 90 s and run analysis ---
		payload = run_analysis_from_path(
			tmp_path,
			upload_size_bytes=total_bytes,
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
	finally:
		# Always delete the temp file, even on error.
		if tmp_path and os.path.exists(tmp_path):
			try:
				os.unlink(tmp_path)
				log.info("analyze: temp file deleted path=%s", tmp_path)
			except OSError:
				pass
