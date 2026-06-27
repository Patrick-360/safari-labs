"""
Simplify the raw detected chord timeline into a beginner-friendly practice progression.

This module is purely additive — it does NOT modify the detailed chord timeline.
The simplified output lives alongside the original in the response.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Label simplification
# ---------------------------------------------------------------------------

_ROOT_RE = re.compile(r"^([A-G][#b]?)(.*)")


def _parse_chord(label: str) -> Tuple[str, str] | None:
    """Return (root, quality) or None for unparseable / 'N' labels."""
    if not label or label == "N":
        return None
    m = _ROOT_RE.match(label)
    if not m:
        return None
    return m.group(1), m.group(2)


def simplify_chord_label(label: str) -> Tuple[str, str | None]:
    """
    Map an advanced chord label to a beginner-friendly version.

    Returns ``(simplified_label, reason)`` where *reason* is ``None`` when
    no simplification was applied.

    Rules (in priority order):
    1. Diminished (dim, dim7, °, m7b5) → kept as ``Xdim`` — caller decides
       whether to filter it or promote to minor.
    2. Minor variants (m7, min7, mM7, m9…) → stripped to plain minor ``Xm``.
    3. Major-seventh variants (maj7, maj9…, M7) → stripped to plain major ``X``.
    4. Dominant-seventh variants (7, 9, 11, 13…) → plain major ``X``.
    5. Augmented (aug, +) → plain major ``X``.
    6. Suspended (sus2, sus4, sus) → plain major ``X``.
    7. Power chord (5), added tones (add9, 6…) → plain major ``X``.
    8. Plain major (``""``) and plain minor (``"m"``) → unchanged.
    9. Unrecognised quality → returned as-is with reason ``"unknown_quality"``.
    """
    parsed = _parse_chord(label)
    if parsed is None:
        return label, None
    root, quality = parsed

    # --- Diminished -------------------------------------------------------
    if quality in ("dim", "dim7", "°", "°7", "m7b5", "ø7", "ø"):
        return f"{root}dim", "diminished"

    # --- Minor extensions (keep minor) ------------------------------------
    if quality in ("m7", "min7", "mM7", "m9", "m11", "m13", "m6"):
        return f"{root}m", "simplified_minor_seventh"

    # --- Major-seventh extensions (strip to major) ------------------------
    if quality in ("maj7", "maj9", "maj11", "maj13", "M7", "Δ7", "Δ"):
        return f"{root}", "simplified_major_seventh"

    # --- Dominant-seventh / extensions (strip to major) -------------------
    if quality in ("7", "9", "11", "13", "7sus4", "7sus2"):
        return f"{root}", "simplified_dominant_seventh"

    # --- Augmented --------------------------------------------------------
    if quality in ("aug", "aug7", "+", "+7"):
        return f"{root}", "simplified_augmented"

    # --- Suspended --------------------------------------------------------
    if quality in ("sus4", "sus2", "sus"):
        return f"{root}", "simplified_suspended"

    # --- Power chord / added tones ----------------------------------------
    if quality in ("5", "add9", "add11", "add13", "6", "6/9", "2"):
        return f"{root}", "simplified_added_tone"

    # --- Plain major / minor (no change) ----------------------------------
    if quality == "":
        return f"{root}", None
    if quality == "m":
        return f"{root}m", None

    # --- Longer-prefix match (e.g. "m7b5" → "m") -------------------------
    priority = [
        ("dim", f"{root}dim", "diminished"),
        ("m7", f"{root}m", "simplified_minor_seventh"),
        ("maj7", f"{root}", "simplified_major_seventh"),
        ("m", f"{root}m", None),
    ]
    for prefix, simplified, reason in priority:
        if quality.startswith(prefix):
            return simplified, reason

    return label, "unknown_quality"


# ---------------------------------------------------------------------------
# Progression building
# ---------------------------------------------------------------------------

_MIN_DIM_TOTAL_SEC = 2.0
_MIN_DIM_COUNT = 2
_MIN_SINGLE_OCC_SEC = 3.0
_MIN_TOTAL_RATIO = 0.008   # 0.8 % of song duration
_MIN_TOTAL_ABS_SEC = 1.5   # absolute floor


def compute_simple_practice_progression(
    chords: List[Dict[str, Any]],
    duration_sec: float,
    *,
    debug: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Derive a beginner-friendly practice progression from the full chord timeline.

    Does **not** modify ``chords`` — returns a new list alongside debug info.

    Each item in the returned list::

        {
            "label":          str,        # simplified chord symbol
            "source_labels":  list[str],  # original detected labels merged here
            "total_duration": float,      # seconds covered across the whole track
            "count":          int,        # segment count contributing to this chord
            "reason":         str | None, # simplification reason (first one found)
        }

    The list is ordered by first-appearance time in the track.
    """
    # --- Pass 1: per-original-label stats ---------------------------------
    orig_stats: Dict[str, Dict[str, Any]] = {}
    simplified_label_map: Dict[str, str] = {}

    for seg in chords:
        orig = str(seg.get("label", "N"))
        if orig == "N":
            continue
        dur = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
        if dur <= 0:
            continue

        if orig not in orig_stats:
            simp, reason = simplify_chord_label(orig)
            orig_stats[orig] = {
                "simplified": simp,
                "reason": reason,
                "total_duration": 0.0,
                "count": 0,
                "first_time": float(seg.get("start", 0.0)),
            }
            simplified_label_map[orig] = simp

        orig_stats[orig]["total_duration"] += dur
        orig_stats[orig]["count"] += 1

    # --- Pass 2: group by simplified label --------------------------------
    groups: Dict[str, Dict[str, Any]] = {}

    for orig, info in orig_stats.items():
        simp = info["simplified"]
        if simp not in groups:
            groups[simp] = {
                "source_labels": [],
                "total_duration": 0.0,
                "count": 0,
                "first_time": info["first_time"],
                "reasons": [],
            }
        g = groups[simp]
        g["source_labels"].append(orig)
        g["total_duration"] += info["total_duration"]
        g["count"] += info["count"]
        g["first_time"] = min(g["first_time"], info["first_time"])
        if info["reason"] and info["reason"] not in g["reasons"]:
            g["reasons"].append(info["reason"])

    # --- Pass 3: filter ---------------------------------------------------
    min_total = max(_MIN_TOTAL_ABS_SEC, duration_sec * _MIN_TOTAL_RATIO)

    kept: Dict[str, Dict[str, Any]] = {}
    removed_dim = 0
    removed_passing = 0

    for simp, g in groups.items():
        is_dim = "dim" in simp or "°" in simp

        if is_dim:
            if g["total_duration"] >= _MIN_DIM_TOTAL_SEC and g["count"] >= _MIN_DIM_COUNT:
                kept[simp] = g
            else:
                removed_dim += 1
            continue

        # Single-occurrence, short → treat as passing
        if g["count"] == 1 and g["total_duration"] < _MIN_SINGLE_OCC_SEC:
            removed_passing += 1
            continue

        # Below total-duration threshold
        if g["total_duration"] < min_total:
            removed_passing += 1
            continue

        kept[simp] = g

    # --- Fallback if too aggressive ---------------------------------------
    fallback_used = False
    fallback_reason: str | None = None

    if len(kept) < 2:
        fallback_used = True
        # Try non-dim chords with at least 1.0 s
        candidate = {
            s: g for s, g in groups.items()
            if ("dim" not in s and "°" not in s) and g["total_duration"] >= 1.0
        }
        if len(candidate) >= 2:
            kept = candidate
            fallback_reason = "simplification_removed_too_much"
        else:
            # Accept everything non-dim
            candidate = {s: g for s, g in groups.items() if "dim" not in s and "°" not in s}
            if len(candidate) >= 1:
                kept = candidate
                fallback_reason = "fallback_all_non_dim"
            else:
                kept = dict(groups)
                fallback_reason = "fallback_all_chords"

    # --- Build ordered output ---------------------------------------------
    ordered = sorted(kept.items(), key=lambda kv: kv[1]["first_time"])

    progression = [
        {
            "label": simp,
            "source_labels": sorted(g["source_labels"]),
            "total_duration": round(g["total_duration"], 2),
            "count": g["count"],
            "reason": g["reasons"][0] if g["reasons"] else None,
        }
        for simp, g in ordered
    ]

    # --- Debug payload ----------------------------------------------------
    debug_info: Dict[str, Any] = {}
    if debug:
        debug_info = {
            "simple_progression_source_count": len(groups),
            "simple_progression_filtered_count": len(groups) - len(kept),
            "simplified_label_map": simplified_label_map,
            "removed_passing_chord_count": removed_passing,
            "removed_diminished_count": removed_dim,
            "simple_progression_fallback_used": fallback_used,
            "simple_progression_fallback_reason": fallback_reason,
        }

    return progression, debug_info
