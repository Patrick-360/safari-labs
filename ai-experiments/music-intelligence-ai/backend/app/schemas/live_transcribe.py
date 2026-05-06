"""JSON contract for POST /live-transcribe (rolling-window live song help)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class LiveTranscribeKey(BaseModel):
	model_config = ConfigDict(extra="forbid")
	label: str
	confidence: float = Field(0.0, ge=0.0, le=1.0)


class LiveTranscribeChordSeg(BaseModel):
	model_config = ConfigDict(extra="forbid")
	start: float
	end: float
	label: str
	confidence: float = 0.0
	notes: list[str] = Field(default_factory=list)
	practice_hint: str = ""
	low_confidence: bool = False
	is_passing: bool = False
	chord_role: str | None = None


class LiveTranscribeCoreEntry(BaseModel):
	model_config = ConfigDict(extra="forbid")
	label: str
	notes: list[str] = Field(default_factory=list)


class LiveTranscribeProgressionMeta(BaseModel):
	"""How `core_progression` was produced; always present for Live Song Transcription UX."""

	model_config = ConfigDict(extra="forbid")

	source: Literal["stable_core", "fallback_time_order", "none"] = "none"
	quality: Literal["likely", "rough", "stabilizing", "still_listening"] = "still_listening"
	empty_reason: str | None = Field(
		default=None,
		description="When source=none: no_chords | all_invalid | all_low_confidence | not_enough_harmony",
	)


class LiveTranscribeResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	window_start: float
	window_end: float
	session_id: str | None = None
	key: LiveTranscribeKey
	current_chord: str
	chords: list[LiveTranscribeChordSeg]
	core_progression: list[LiveTranscribeCoreEntry]
	progression_meta: LiveTranscribeProgressionMeta = Field(default_factory=LiveTranscribeProgressionMeta)
	summary: str
	status: Literal["listening", "analyzing", "ready"] = "ready"
	tempo_bpm: float = 0.0
	debug: dict[str, Any] | None = Field(
		default=None,
		description="Extended diagnostics when query debug=true",
	)
