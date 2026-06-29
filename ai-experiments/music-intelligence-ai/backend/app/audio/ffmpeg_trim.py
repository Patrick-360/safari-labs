"""Memory-safe audio pre-trim via ffmpeg.

ffmpeg reads the source file as a byte stream and writes only the first N seconds
of mono WAV to a new temp file.  librosa then loads only that small WAV, so peak
in-memory audio footprint is bounded to ~2 MB (60 s × 16 kHz float32) rather than
the full decoded song.

Falls back gracefully: callers check check_ffmpeg_available() first and use librosa
duration cap when ffmpeg is not installed.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile

log = logging.getLogger(__name__)


def check_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ffmpeg_trim_to_wav(
    input_path: str,
    duration_sec: float,
    sample_rate: int = 16000,
) -> str:
    """Trim `input_path` to `duration_sec` seconds, output as mono WAV at `sample_rate` Hz.

    Returns the path to a new temp WAV file.  The caller is responsible for deleting it
    (use _safe_unlink in a finally block).

    Raises RuntimeError on ffmpeg failure or timeout.
    """
    tmp_fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="trimmed_")
    os.close(tmp_fd)
    cmd = [
        "ffmpeg",
        "-y",                        # overwrite output without prompting
        "-i", input_path,
        "-t", str(duration_sec),     # stop after this many seconds
        "-ac", "1",                  # mono
        "-ar", str(sample_rate),     # resample to target rate
        "-f", "wav",
        out_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=45,
        )
    except subprocess.TimeoutExpired as exc:
        _safe_unlink(out_path)
        raise RuntimeError("ffmpeg timed out after 45 s") from exc

    if result.returncode != 0:
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")[-400:]
        _safe_unlink(out_path)
        raise RuntimeError(f"ffmpeg exited {result.returncode}: {stderr}")

    log.info(
        "ffmpeg_trim: wrote %.0f s WAV to %s (sr=%d)",
        duration_sec, out_path, sample_rate,
    )
    return out_path


def _safe_unlink(path: str | None) -> None:
    if path:
        try:
            os.unlink(path)
        except OSError:
            pass
