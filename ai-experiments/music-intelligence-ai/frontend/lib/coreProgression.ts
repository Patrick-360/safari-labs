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

export type CoreProgressionResult = {
  entries: CoreChordEntry[];
  fallbackUsed: boolean;
  fallbackReason: string | null;
};

/** Mirror ChordRun from page — avoid importing React page here. */
export type ChordRunLike = {
  label: string;
  notesLine: string;
  anyLowConfidence: boolean;
  /** When true, segment was tagged as likely passing harmony — omit from core progression when possible */
  isPassing?: boolean;
  /** When true, omit from "main progression" heuristics (additive backend flag). */
  excludeFromCore?: boolean;
};

const CORE_PROGRESSION_MAX_UNIQUE = 12;
const REPEAT_MIN_PERIODS = 2;
const REPEAT_MIN_PATTERN_LEN = 2;
const REPEAT_MAX_PATTERN_LEN = 8;

function runsWithoutN(runs: ChordRunLike[]): ChordRunLike[] {
  return runs.filter((r) => r.label !== "N");
}

function uniqueLabelCount(runs: ChordRunLike[]): number {
  return new Set(runs.filter((r) => r.label !== "N").map((r) => r.label)).size;
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

function buildEntriesFromRuns(work: ChordRunLike[]): CoreChordEntry[] {
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

/**
 * Strict filter: prefer high-confidence, non-passing runs not flagged excludeFromCore.
 */
function runsStrictCore(runs: ChordRunLike[]): ChordRunLike[] {
  const noCoreOptOut = runs.filter((r) => !r.excludeFromCore);
  const structural = noCoreOptOut.filter((r) => !r.isPassing);
  const base = structural.length ? structural : noCoreOptOut;
  const hi = base.filter((r) => r.label !== "N" && !r.anyLowConfidence);
  if (hi.length >= 2) {
    return hi;
  }
  return runsWithoutN(base);
}

/**
 * Medium filter: ignore excludeFromCore; drop passing only.
 */
function runsMediumCore(runs: ChordRunLike[]): ChordRunLike[] {
  return runsWithoutN(runs.filter((r) => !r.isPassing));
}

/**
 * First distinct label changes in timeline order (readable fallback).
 */
function runsFirstRawChanges(runs: ChordRunLike[], max = 8): ChordRunLike[] {
  const noN = runsWithoutN(runs);
  const out: ChordRunLike[] = [];
  let last = "";
  for (const r of noN) {
    if (r.label !== last) {
      out.push(r);
      last = r.label;
    }
    if (out.length >= max) {
      break;
    }
  }
  return out.length ? out : noN.slice(0, Math.min(4, noN.length));
}

type SelectedRuns = { runs: ChordRunLike[]; fallbackReason: string | null };

/**
 * Pick runs for core progression with tiered fallback so filtering never collapses
 * a normal song to one chord when the raw timeline has more harmony.
 */
function selectRunsForCoreProgression(runs: ChordRunLike[]): SelectedRuns {
  const noN = runsWithoutN(runs);
  if (!noN.length) {
    return { runs: [], fallbackReason: "empty_timeline" };
  }

  const strict = runsStrictCore(runs);
  if (uniqueLabelCount(strict) >= 2 && strict.length >= 2) {
    return { runs: strict, fallbackReason: null };
  }

  const medium = runsMediumCore(runs);
  if (uniqueLabelCount(medium) >= 2 && medium.length >= 2) {
    return { runs: medium, fallbackReason: "fallback_medium_non_passing" };
  }

  if (uniqueLabelCount(noN) >= 2 && noN.length >= 2) {
    return { runs: noN, fallbackReason: "fallback_all_non_n" };
  }

  if (noN.length >= 4) {
    return { runs: runsFirstRawChanges(runs), fallbackReason: "fallback_first_raw_changes" };
  }

  return { runs: noN, fallbackReason: "fallback_sparse_timeline" };
}

/**
 * Build core progression from consecutive chord runs (already merged identical neighbors).
 * Timeline playback still uses full `chords` — this is the cleaned summary only.
 */
export function deriveCoreProgression(runs: ChordRunLike[]): CoreProgressionResult {
  const { runs: work, fallbackReason } = selectRunsForCoreProgression(runs);
  return {
    entries: buildEntriesFromRuns(work),
    fallbackUsed: fallbackReason !== null,
    fallbackReason,
  };
}

export function firstChordTimeForLabel(
  chords: { start: number; label: string }[],
  label: string,
): number | null {
  const hit = chords.find((c) => c.label === label);
  return hit ? hit.start : null;
}
