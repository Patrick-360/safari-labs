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

# --- Beta resource limits ---
# Reject uploads above this size while streaming (before any decode).
BETA_MAX_UPLOAD_SIZE_MB: int = 30
# Analyze only the first N seconds.  60 s keeps peak RAM well under 512 MB on
# Render free tier.  Lower to 30 s if memory is still tight.
BETA_ANALYSIS_DURATION_SEC: float = 60.0
# ffmpeg output sample rate before librosa resamples to ANALYSIS_SR (22050).
# 16 kHz mono produces a ~1.9 MB WAV for 60 s — tiny for librosa to load.
BETA_ANALYSIS_SAMPLE_RATE: int = 16000
