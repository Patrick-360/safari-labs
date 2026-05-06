from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.audio.analyze_pipeline import run_analysis
from app.schemas.analyze import AnalyzeResponse

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_track(file: UploadFile = File(...)) -> AnalyzeResponse:
	"""
	Full-song analysis: duration, tempo (BPM), global key + confidence, chord segments.
	Accepts WAV, MP3, and other formats supported by librosa/audioread.
	"""
	try:
		audio_bytes = await file.read()
		if not audio_bytes:
			raise HTTPException(status_code=400, detail="Empty upload.")
		payload = run_analysis(audio_bytes)
		return AnalyzeResponse.model_validate(payload)
	except HTTPException:
		raise
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=400, detail=str(exc)) from exc
