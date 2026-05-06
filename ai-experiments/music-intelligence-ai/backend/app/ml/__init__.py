"""
Future ML backends for transcription (source separation, notes, chords).

All entry points expose ``enabled=False`` fallbacks so `/analyze` stays fast and deterministic
until optional heavy dependencies are added.
"""

from app.ml.chord_ml import ChordPrediction, predict_chords_ml
from app.ml.pitch_transcription import NoteEvent, transcribe_pitch
from app.ml.source_separation import SeparationResult, StemBundle, separate_sources

__all__ = [
	"ChordPrediction",
	"NoteEvent",
	"SeparationResult",
	"StemBundle",
	"predict_chords_ml",
	"separate_sources",
	"transcribe_pitch",
]
