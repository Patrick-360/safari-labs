/**
 * Derive a short "main progression" for learning UI from full chord runs.
 * Not the same as the beat-by-beat timeline — this is the reusable harmonic loop
 * or a compact ordered set of chords.
 */

export type CoreChordEntry = {
  label: string;
  notesLine: string;
  anyLowConfidence: boolean;
};

/** Mirror ChordRun from page — avoid importing React page here. */
export type ChordRunLike = {
  label: string;
  notesLine: string;
  anyLowConfidence: boolean;
  /** When true, segment was tagged as likely passing harmony — omit from core progression when possible */
  isPassing?: boolean;
};

const CORE_PROGRESSION_MAX_UNIQUE = 12;
const REPEAT_MIN_PERIODS = 2;
const REPEAT_MIN_PATTERN_LEN = 2;
const REPEAT_MAX_PATTERN_LEN = 8;

function runsWithoutN(runs: ChordRunLike[]): ChordRunLike[] {
  return runs.filter((r) => r.label !== "N");
}

/**
 * If the whole sequence equals `pattern` repeated (partial tail allowed), return pattern length * min periods.
 */
function findSmallestRepeatingPatternLength(labels: string[]): number | null {
  const n = labels.length;
  if (n < REPEAT_MIN_PERIODS * REPEAT_MIN_PATTERN_LEN) {
    return null;
  }
  for (
    let p = REPEAT_MIN_PATTERN_LEN;
    p <= Math.min(REPEAT_MAX_PATTERN_LEN, Math.floor(n / REPEAT_MIN_PERIODS));
    p += 1
  ) {
    let ok = true;
    for (let i = 0; i < n; i += 1) {
      if (labels[i] !== labels[i % p]) {
        ok = false;
        break;
      }
    }
    if (ok) {
      return p;
    }
  }
  return null;
}

function uniqueInOrder(labels: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const l of labels) {
    if (seen.has(l)) {
      continue;
    }
    seen.add(l);
    out.push(l);
  }
  return out;
}

function entryForLabel(runs: ChordRunLike[], label: string): CoreChordEntry {
  const run = runs.find((r) => r.label === label);
  return {
    label,
    notesLine: run?.notesLine ?? "—",
    anyLowConfidence: run?.anyLowConfidence ?? false,
  };
}

/**
 * Prefer high-confidence runs when building the main progression so brief / uncertain
 * fragments do not become "the loop" (timeline playback still uses full `chords`).
 */
function runsForCoreProgression(runs: ChordRunLike[]): ChordRunLike[] {
  const structural = runs.filter((r) => !r.isPassing);
  const base = structural.length ? structural : runs;
  const hi = base.filter((r) => r.label !== "N" && !r.anyLowConfidence);
  if (hi.length >= 2) {
    return hi;
  }
  return runsWithoutN(base);
}

/**
 * Build core progression from consecutive chord runs (already merged identical neighbors).
 */
export function deriveCoreProgression(runs: ChordRunLike[]): CoreChordEntry[] {
  const work = runsForCoreProgression(runs);
  if (!work.length) {
    return [];
  }
  const labels = work.map((r) => r.label);

  const periodLen = findSmallestRepeatingPatternLength(labels);
  if (periodLen !== null) {
    return work.slice(0, periodLen).map((r) => ({
      label: r.label,
      notesLine: r.notesLine,
      anyLowConfidence: r.anyLowConfidence,
    }));
  }

  const uniqueLabels = uniqueInOrder(labels);
  const capped = uniqueLabels.slice(0, CORE_PROGRESSION_MAX_UNIQUE);
  return capped.map((lab) => entryForLabel(work, lab));
}

export function firstChordTimeForLabel(
  chords: { start: number; label: string }[],
  label: string,
): number | null {
  const hit = chords.find((c) => c.label === label);
  return hit ? hit.start : null;
}
