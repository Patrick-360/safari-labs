"""
Central feature flags for optional ML / deep-learning stages.

Heavy models (Demucs, MT3/Basic Pitch, custom chord nets) plug in behind `app/ml/*`;
defaults keep the backend lightweight with no downloads or GPU load.
"""

# When True alongside client opt-in: `/analyze` also attempts neural separation (`use_source_separation` query).
# Leave False unless you intentionally force separation server-wide during development.
ENABLE_SOURCE_SEPARATION = False

# When True: run note-level pitch transcription (polyphonic/monophonic backends TBD).
# Fallback (False): no note streams; heuristic chord chroma unchanged.
ENABLE_PITCH_TRANSCRIPTION = False

# When True: run learned chord predictor (CNN/CRNN/Transformer backends TBD).
# Fallback (False): template chroma path only; richer qualities (dim/aug/7ths) deferred to future fusion.
ENABLE_ML_CHORDS = False
