/**
 * Heuristic practice copy for Analyze File mode — no API calls, uses existing analysis only.
 */

import type { CoreChordEntry } from "./coreProgression";
import type { PracticePart } from "./practiceSections";

/** Matches AnalyzeChordSeg fields used here */
export type GuidableChordSeg = {
  start: number;
  end: number;
  label: string;
  notes?: string[];
  practice_hint?: string;
  low_confidence?: boolean;
};

/** Shown near piano guidance — honest scope for beginners */
export const PIANO_GUIDANCE_DISCLAIMER =
  "Simplified piano guide: roots and chord tones from the analysis — not exact octaves, inversions, or voicings.";

/**
 * Heuristic chord root from a chord symbol (e.g. Am7 → A, Bb → Bb). Slash chords use the left side only.
 */
export function chordRootFromLabel(label: string): string | null {
  const t = label.trim();
  if (!t || t === "N") return null;
  const head = (t.split("/")[0] ?? t).trim();
  const m = head.match(/^([A-Ga-g])([#b]?)/u);
  if (!m) return null;
  return m[1].toUpperCase() + (m[2] ?? "");
}

/**
 * Beginner piano layout: LH root (from symbol if possible), RH from notes[] or practice_hint fallback.
 */
export function getSimplePianoHands(seg: GuidableChordSeg | null | undefined): {
  lh: string;
  rh: string;
  oneLine: string;
} {
  if (!seg) {
    return { lh: "—", rh: "—", oneLine: "—" };
  }
  const rootFromSymbol = chordRootFromLabel(seg.label);
  const lh =
    rootFromSymbol ??
    (seg.notes?.length ? String(seg.notes[0]) : undefined) ??
    "—";
  let rh = "—";
  if (seg.notes?.length) {
    rh = seg.notes.join(" - ");
  } else if (seg.practice_hint?.trim()) {
    rh = seg.practice_hint.trim();
  }
  const oneLine = rh !== "—" ? `LH: ${lh} | RH: ${rh}` : `LH: ${lh}`;
  return { lh, rh, oneLine };
}

export function buildPianoPracticeStepsForPart(
  part: PracticePart,
  chords: GuidableChordSeg[],
  durationSec: number,
): string[] {
  const seq = chordSequenceForPart(part, chords, durationSec);
  const steps: string[] = [
    "Practice the left-hand roots first (one note per chord).",
    "Add the right-hand chord tones.",
  ];

  if (seq.length > 1) {
    const block = seq
      .map((row) => {
        const lh = chordRootFromLabel(row.label) ?? "?";
        const rhRaw = row.toneLine ? row.toneLine.replace(/\s*·\s*/g, "-") : "";
        const rh = rhRaw || "—";
        return `${row.label}: LH ${lh} | RH ${rh}`;
      })
      .join("\n");
    steps.push(`Chord map (simplified):\n${block}`);
  }

  steps.push("Practice the changes slowly with the track.", "Loop this section until it feels smooth.");

  if (seq.some((s) => s.low_confidence)) {
    steps.push("If something sounds off, check it by ear — the chart is a guide, not a full transcription.");
  }

  return steps;
}

export function buildLearnThisSongSummary(params: {
  keyLabel: string;
  tempoBpm: number;
  coreEntries: CoreChordEntry[];
  practicePartCount: number;
}): { mainProgressionArrow: string; suggestion: string } {
  const labels = params.coreEntries.map((e) => e.label).filter((l) => l !== "N");
  const arrow = labels.join(" → ");
  const progPhrase = arrow ? arrow : "the main chord changes";

  let suggestion: string;
  if (params.practicePartCount <= 0) {
    suggestion = `Listen through once, then work through ${progPhrase} slowly with the track.`;
  } else if (params.practicePartCount === 1) {
    suggestion = `Start in Part 1: loop it and practice ${progPhrase} slowly until the changes feel easy.`;
  } else {
    suggestion = `Start by looping Part 1 and practicing ${progPhrase} slowly. When Part 1 feels solid, move on to the next part.`;
  }

  return { mainProgressionArrow: arrow, suggestion };
}

export type PartChordRow = {
  label: string;
  toneLine: string;
  low_confidence?: boolean;
};

/**
 * Chords that overlap a practice part (time order, consecutive duplicate labels collapsed).
 */
export function chordSequenceForPart(
  part: PracticePart,
  chords: GuidableChordSeg[],
  durationSec: number,
): PartChordRow[] {
  const t0 = part.start;
  const t1 = Math.min(part.end, durationSec);
  const out: PartChordRow[] = [];

  for (const c of chords) {
    if (c.label === "N") continue;
    if (c.end <= t0 + 1e-4 || c.start >= t1 - 1e-4) continue;
    const prev = out[out.length - 1];
    if (prev && prev.label === c.label) continue;

    const toneLine = chordToneLine(c);
    out.push({
      label: c.label,
      toneLine,
      low_confidence: c.low_confidence,
    });
  }
  return out;
}

function chordToneLine(c: GuidableChordSeg): string {
  if (c.practice_hint?.trim()) return c.practice_hint.trim();
  if (c.notes?.length) return c.notes.join("-");
  return "";
}

export function buildPracticeStepsForPart(
  part: PracticePart,
  chords: GuidableChordSeg[],
  durationSec: number,
): string[] {
  const seq = chordSequenceForPart(part, chords, durationSec);
  if (!seq.length) {
    return [
      "Listen through this section once without playing.",
      "Hum or tap the beat, then add chords when you are ready.",
      "Loop this section until it feels comfortable.",
    ];
  }

  if (seq.length === 1) {
    const s = seq[0];
    const steps = [
      "Listen once while you follow this chord on the chart.",
      `Stay on ${s.label} and focus on rhythm and timing with the track.`,
    ];
    if (s.toneLine) {
      steps.push(`Chord tones: ${s.toneLine.replace(/\s*·\s*/g, "-")}.`);
    } else {
      steps.push("Use a voicing that feels easy to hold steady.");
    }
    steps.push("Loop until this section feels locked in.");
    if (s.low_confidence) {
      steps.push("Check this one by ear if it does not quite match the recording.");
    }
    return steps;
  }

  const arrow = seq.map((s) => s.label).join(" → ");
  const toneParts = seq
    .map((s) => (s.toneLine ? `${s.label} (${s.toneLine.replace(/\s*·\s*/g, "-")})` : null))
    .filter((x): x is string => Boolean(x));

  const steps = [
    "Listen through once without playing, just to hear the changes.",
    `Practice the moves slowly: ${arrow}.`,
  ];
  if (toneParts.length) {
    steps.push(`Chord tones: ${toneParts.join(", ")}.`);
  }
  steps.push("Loop this section until the changes feel smooth.");
  if (seq.some((s) => s.low_confidence)) {
    steps.push("If something sounds off, check it by ear.");
  }
  return steps;
}

/** Musician-facing line, e.g. "Play: C - E - G" */
export function formatPlayHint(c: GuidableChordSeg | null | undefined): string {
  if (!c) return "";
  if (c.notes?.length) {
    return `Play: ${c.notes.join(" - ")}`;
  }
  if (c.practice_hint?.trim()) {
    return `Play: ${c.practice_hint.trim()}`;
  }
  return "";
}
