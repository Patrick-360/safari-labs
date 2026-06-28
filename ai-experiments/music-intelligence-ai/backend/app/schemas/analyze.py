"""Typed JSON contract for POST /analyze."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class KeyInfo(BaseModel):
	model_config = ConfigDict(extra="forbid")

	label: str = Field(..., description="Global key, e.g. 'C major'")
	confidence: float = Field(0.0, ge=0.0, le=1.0, description="Key estimate confidence")


class ChordSegment(BaseModel):
	model_config = ConfigDict(extra="forbid")

	start: float = Field(..., description="Start time in seconds")
	end: float = Field(..., description="End time in seconds")
	label: str = Field(..., description="Chord symbol, e.g. 'Am', 'C'")
	notes: list[str] = Field(
		default_factory=list,
		description="Heuristic close-position triad spellings (not voicing/inversion)",
	)
	practice_hint: str = Field(
		"",
		description="Short practice sentence; simplified theory—not exact transcription",
	)
	confidence: float = Field(
		0.5,
		ge=0.0,
		le=1.0,
		description="Relative certainty from template separation (marginal)",
	)
	low_confidence: bool = Field(
		False,
		description="True when estimate is weak — interpret label with caution",
	)
	# Optional debugging / review fields (additive; safe for older clients to ignore).
	template_score: float | None = Field(
		None,
		description="Best cosine similarity vs chord templates at segment chroma (0–1 scale)",
	)
	template_margin: float | None = Field(
		None,
		description="Raw separation best_score − second_score before UI confidence shaping",
	)
	is_passing: bool = Field(
		False,
		description="Heuristic: very short sandwich segment between two equal stable chords — likely passing harmony",
	)
	chord_role: str | None = Field(
		None,
		description="Optional role tag, e.g. 'passing' — beginner UI may ignore",
	)
	vocal_interference: bool = Field(
		False,
		description="Heuristic: chroma looked single-pitch heavy — treat chord change with caution",
	)
	exclude_from_core: bool = Field(
		False,
		description="When true, client may omit this segment from compact 'main progression' UI",
	)
	confidence_reasons: list[str] | None = Field(
		None,
		description="Optional short codes explaining weak confidence (debug / transparency)",
	)
	arpeggio_support: float | None = Field(
		None,
		description="Temporal spread of chroma peaks in the segment (0–1); higher suggests arpeggiated/split chord tones",
	)
	bass_root_hint: int | None = Field(
		None,
		ge=0,
		le=11,
		description="Heuristic bass pitch-class index (0=C … 11=B) from low-register chroma; small tie-break only",
	)


class BeatTime(BaseModel):
	model_config = ConfigDict(extra="forbid")

	time: float = Field(..., ge=0.0, description="Beat time in seconds from start of track")


class SectionSpan(BaseModel):
	model_config = ConfigDict(extra="forbid")

	index: int = Field(..., ge=0, description="0-based section order in the track")
	start: float = Field(..., ge=0.0, description="Section start in seconds")
	end: float = Field(..., description="Section end in seconds")
	label: str = Field(..., description="Section label, e.g. 'Section A' when repeats are merged")
	repeat_group: str | None = Field(
		None,
		description="Letter shared by musically similar blocks (heuristic repetition)",
	)


class RhythmHint(BaseModel):
	model_config = ConfigDict(extra="forbid")

	assumed_beats_per_bar: int = Field(
		4,
		ge=1,
		le=12,
		description="Heuristic beats per measure for grouping (default 4/4 assumption)",
	)
	bar_start_times: list[float] = Field(
		default_factory=list,
		description="Times (s) of assumed downbeats: every Nth beat from beat_track",
	)


class SimplePracticeChord(BaseModel):
	model_config = ConfigDict(extra="forbid")

	label: str = Field(..., description="Simplified beginner-friendly chord symbol")
	source_labels: list[str] = Field(
		default_factory=list,
		description="Original detected labels merged/simplified into this chord",
	)
	total_duration: float = Field(
		0.0,
		description="Total seconds this chord covers across the track",
	)
	count: int = Field(
		0,
		description="Number of chord segments in the timeline contributing to this entry",
	)
	reason: str | None = Field(
		None,
		description="Simplification applied, e.g. 'simplified_major_seventh'; null when label was unchanged",
	)


class AnalysisWindow(BaseModel):
	model_config = ConfigDict(extra="forbid")

	start: float = Field(0.0, description="Analysis start time in seconds (always 0 for beta)")
	end: float = Field(..., description="Analysis end time in seconds")
	duration_analyzed: float = Field(..., description="Seconds of audio that were actually analyzed")
	was_trimmed: bool = Field(
		False,
		description="True when the uploaded file was longer than the analysis window",
	)
	original_duration: float | None = Field(
		None,
		description="Full file duration in seconds; None when the format does not support header-only probing (e.g. MP3)",
	)
	reason: str | None = Field(
		None,
		description="Why the window was limited, e.g. 'beta_duration_limit'; null for short files",
	)


class AnalyzeResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	duration: float = Field(..., description="Audio duration in seconds")
	tempo: float = Field(..., description="Estimated tempo in BPM")
	key: KeyInfo
	chord_engine: str = Field(
		default="theory",
		description="Chord analysis preset used for this response: stable | theory | experimental",
	)
	chords: list[ChordSegment] = Field(default_factory=list)
	beats: list[BeatTime] = Field(
		default_factory=list,
		description="Detected beat onsets (librosa beat_track), seconds",
	)
	sections: list[SectionSpan] = Field(
		default_factory=list,
		description="Coarse sections from chroma similarity over beat- or time-aligned windows",
	)
	rhythm: RhythmHint = Field(
		default_factory=RhythmHint,
		description="Heuristic bar grouping from detected beats (not true meter detection)",
	)
	simple_practice_progression: list[SimplePracticeChord] = Field(
		default_factory=list,
		description=(
			"Beginner-friendly simplified chord progression for practice roadmap. "
			"Distinct from the full chord timeline — passing chords, short diminished, "
			"and color tones are filtered or simplified. Use chords[] for playback."
		),
	)
	analysis_window: AnalysisWindow | None = Field(
		default=None,
		description=(
			"Describes which portion of the audio was analyzed. For beta deployments with "
			"memory constraints, long files are trimmed to the first 90 seconds. "
			"was_trimmed=true signals the frontend to show a trimming notice."
		),
	)
	debug: dict[str, Any] | None = Field(
		default=None,
		description="Optional evaluation payload when debug=true was requested",
	)
