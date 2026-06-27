"""
Debug script for Analyze File pipeline.

Usage (from the backend/ directory):
    python scripts/debug_analyze_file.py --file path/to/audio.wav --engine theory
    python scripts/debug_analyze_file.py --file path/to/audio.mp3 --engine stable
    python scripts/debug_analyze_file.py --file path/to/audio.wav  # defaults to theory
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root or the backend/ directory.
_here = Path(__file__).resolve().parent
for _candidate in (_here.parent, _here.parent.parent):
    if (_candidate / "app").is_dir():
        sys.path.insert(0, str(_candidate))
        break


def _run(audio_path: str, engine: str) -> None:
    from app.audio.analyze_pipeline import run_analysis

    raw = Path(audio_path).read_bytes()
    print(f"\n=== debug_analyze_file: {audio_path} | engine={engine} ===\n")

    payload = run_analysis(raw, debug=True, engine=engine)
    dbg = payload.get("debug") or {}
    chords = payload.get("chords", [])

    # --- Basic facts ---
    print(f"duration          : {payload.get('duration', 0):.2f}s")
    print(f"tempo             : {payload.get('tempo', 0):.1f} BPM")
    key = payload.get("key", {})
    print(f"key               : {key.get('label', '?')} (conf={key.get('confidence', 0):.3f})")
    print(f"chord_engine      : {payload.get('chord_engine', '?')}")

    # --- Stage segment counts ---
    print()
    print(f"raw_chord_frame_count      : {dbg.get('raw_chord_frame_count', '?')}")
    print(f"post_median_segment_count  : {dbg.get('post_median_segment_count', '?')}")
    print(f"post_sticky_segment_count  : {dbg.get('post_sticky_segment_count', '?')}")
    print(f"raw_chord_segment_count    : {dbg.get('raw_chord_segment_count', '?')}  (pre-refine)")
    print(f"returned_chord_segments    : {dbg.get('returned_chord_segment_count', len(chords))}  (post-refine)")
    print(f"unique_raw_chords          : {dbg.get('unique_raw_chords', '?')}")
    print(f"unique_returned_chords     : {dbg.get('unique_returned_chords', '?')}")

    # --- Filtering stats ---
    print()
    print(f"low_confidence_count      : {dbg.get('low_confidence_count', '?')}")
    print(f"vocal_interference_count  : {dbg.get('vocal_interference_count', '?')}")
    print(f"excluded_from_core_count  : {dbg.get('excluded_from_core_count', '?')}")
    print(f"core_after_filter         : {dbg.get('core_candidate_count_after_filter', '?')}")
    print(f"unique_core_chords        : {dbg.get('unique_core_chords', '?')}")

    # --- Longest segment (before/after guardrail) ---
    longest_label = dbg.get("longest_segment_label", "?")
    longest_dur = dbg.get("longest_segment_duration", 0)
    before_gr = dbg.get("longest_segment_before_guardrail", longest_dur)
    after_gr = dbg.get("longest_segment_after_guardrail", longest_dur)
    gr_applied = dbg.get("final_long_segment_guardrail_applied_count", 0)
    gr_splits = dbg.get("final_long_segment_split_count", 0)
    print()
    print(f"longest_segment_label          : {longest_label}")
    print(f"longest_segment_duration       : {longest_dur:.3f}s")
    print(f"longest_before_guardrail       : {before_gr:.3f}s")
    print(f"longest_after_guardrail        : {after_gr:.3f}s")
    print(f"guardrail_applied_count        : {gr_applied}")
    print(f"guardrail_split_count          : {gr_splits}")

    # --- Core progression fallback ---
    fb_used = dbg.get("core_progression_fallback_used")
    fb_reason = dbg.get("core_progression_fallback_reason")
    print()
    print(f"core_progression_fallback_used   : {fb_used}")
    print(f"core_progression_fallback_reason : {fb_reason}")

    # --- Warnings ---
    warnings: list[str] = []
    n_unique = dbg.get("unique_returned_chords")
    n_segs = dbg.get("returned_chord_segment_count", len(chords))
    n_raw_unique = dbg.get("unique_raw_chords")

    if n_unique is not None and int(n_unique) <= 1 and int(n_segs) > 0:
        warnings.append("TIMELINE COLLAPSED — only 1 unique chord in returned timeline")
    if (
        n_unique is not None and n_raw_unique is not None
        and int(n_unique) <= 3 and int(n_raw_unique) >= 6
    ):
        warnings.append(
            f"TIMELINE COMPRESSED — raw had {n_raw_unique} unique chords "
            f"but returned has only {n_unique}"
        )
    if fb_used and n_unique is not None and int(n_unique) >= 2:
        warnings.append(
            f"CORE PROGRESSION FALLBACK ACTIVE ({fb_reason}) — "
            "strict filter excluded everything despite multiple returned chords"
        )
    core_after = dbg.get("core_candidate_count_after_filter")
    if core_after is not None and int(core_after) == 0 and n_unique is not None and int(n_unique) >= 2:
        warnings.append(
            "WARNING: core progression empty despite multiple returned chords — "
            "frontend fallback will activate"
        )
    for c in chords:
        dur = float(c.get("end", 0)) - float(c.get("start", 0))
        if dur > 60:
            warnings.append(
                f"UNUSUALLY LONG SEGMENT: {c.get('label','?')} "
                f"({c.get('start',0):.1f}–{c.get('end',0):.1f}s = {dur:.1f}s)"
            )

    if warnings:
        print()
        for w in warnings:
            print(f"*** {w} ***")

    # --- Core progression (backend simulation) ---
    core_segs = [c for c in chords if not c.get("exclude_from_core") and c.get("label") != "N"]
    core_labels_ordered: list[str] = []
    seen: set[str] = set()
    for c in core_segs:
        lab = str(c.get("label", "N"))
        if lab not in seen:
            seen.add(lab)
            core_labels_ordered.append(lab)

    medium_segs = [c for c in chords if not c.get("is_passing") and c.get("label") != "N"]
    medium_labels: list[str] = []
    seen2: set[str] = set()
    for c in medium_segs:
        lab = str(c.get("label", "N"))
        if lab not in seen2:
            seen2.add(lab)
            medium_labels.append(lab)

    print()
    print(f"core progression (strict)  : {' → '.join(core_labels_ordered[:12]) or '(empty)'}")
    print(f"core progression (medium)  : {' → '.join(medium_labels[:12]) or '(empty)'}")

    # --- First 25 chord segments ---
    print()
    print(f"First {min(25, len(chords))} chord segments:")
    hdr = f"  {'#':>3}  {'start':>7}  {'end':>7}  {'label':<10}  {'conf':>6}  {'low':>5}  {'vocal':>7}  {'excl':>5}  {'dur':>7}"
    print(hdr)
    for i, c in enumerate(chords[:25]):
        label = str(c.get("label", "N"))
        start = float(c.get("start", 0))
        end = float(c.get("end", 0))
        dur = end - start
        conf = float(c.get("confidence", 0))
        low = bool(c.get("low_confidence", False))
        vocal = bool(c.get("vocal_interference", False))
        excl = bool(c.get("exclude_from_core", False))
        print(f"  {i:>3}  {start:>7.3f}  {end:>7.3f}  {label:<10}  {conf:>6.3f}  {str(low):>5}  {str(vocal):>7}  {str(excl):>5}  {dur:>7.2f}s")

    if len(chords) > 25:
        print(f"  ... ({len(chords) - 25} more segments)")

    # --- Preset calibration values ---
    preset_dbg = dbg.get("chord_analysis_preset") or {}
    if preset_dbg:
        print()
        print("Preset calibration:")
        for key_name in (
            "sticky_min_raw_margin",
            "low_conf_cutoff",
            "snap_conf_threshold",
            "max_sticky_hold_sec",
            "sticky_forced_window_sec",
            "max_returned_segment_sec",
        ):
            print(f"  {key_name}: {preset_dbg.get(key_name, '?')}")

    print()
    print("--- done ---")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug Analyze File pipeline for a single audio file.")
    parser.add_argument("--file", required=True, help="Path to audio file (WAV, MP3, …)")
    parser.add_argument("--engine", default="theory", choices=["stable", "theory", "experimental"],
                        help="Chord engine preset (default: theory)")
    args = parser.parse_args()
    _run(args.file, args.engine)


if __name__ == "__main__":
    main()
