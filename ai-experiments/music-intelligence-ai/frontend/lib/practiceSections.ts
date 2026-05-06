/**
 * practiceParts: simplified “Part 1 / Part 2 …” chunks for Analyze File — learning-oriented,
 * not a 1:1 mirror of backend harmonic sections. Built only on the client from raw /analyze sections.
 */

export type RawSectionLike = {
  start: number;
  end: number;
  label: string;
  index?: number;
  repeat_group?: string | null;
};

export type PracticePart = {
  partIndex: number;
  /** User-facing label, e.g. "Part 1" */
  label: string;
  start: number;
  end: number;
  /** Indices into the original `analyzeResult.sections` array */
  rawIndices: number[];
};

const MIN_PART_SEC = 16;
const MAX_PART_SEC = 60;
/** Target thickness when the song is long enough to split */
const TARGET_PART_WIDTH_MIN = 24;

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

/**
 * How many practice parts we aim for by length (cap 6, prefer 3–6 on typical songs).
 */
function targetPartCount(durationSec: number): number {
  if (!Number.isFinite(durationSec) || durationSec <= 0) return 1;
  if (durationSec < 45) return 1;
  if (durationSec < 85) return 2;
  if (durationSec < 130) return 3;
  if (durationSec < 175) return 4;
  if (durationSec < 230) return 5;
  return 6;
}

/**
 * Merge raw segments (in time order) into fewer, larger loops for practice UI.
 */
export function buildPracticeParts(
  sections: RawSectionLike[] | null | undefined,
  durationSec: number,
): PracticePart[] {
  const dur = Math.max(0, durationSec);
  if (!sections?.length) {
    return [
      {
        partIndex: 0,
        label: "Part 1",
        start: 0,
        end: round4(dur),
        rawIndices: [],
      },
    ];
  }

  const order = [...sections.keys()].sort((a, b) => sections[a].start - sections[b].start);
  const goal = targetPartCount(dur);
  let targetW = dur / Math.max(1, goal);
  targetW = Math.max(TARGET_PART_WIDTH_MIN, Math.min(MAX_PART_SEC - 4, targetW));

  type Acc = { start: number; end: number; rawIndices: number[] };
  const buckets: Acc[] = [];

  let acc: Acc = {
    start: sections[order[0]].start,
    end: Math.min(sections[order[0]].end, dur),
    rawIndices: [order[0]],
  };

  for (let k = 1; k < order.length; k++) {
    const ri = order[k];
    const s = sections[ri];
    const mergedEnd = Math.max(acc.end, Math.min(s.end, dur));
    const spanIfMerged = mergedEnd - acc.start;
    const curSpan = acc.end - acc.start;

    if (spanIfMerged > MAX_PART_SEC + 1e-3) {
      buckets.push(acc);
      acc = {
        start: s.start,
        end: Math.min(s.end, dur),
        rawIndices: [ri],
      };
      continue;
    }

    if (curSpan >= targetW - 1e-3 && curSpan >= MIN_PART_SEC * 0.75 && k < order.length) {
      buckets.push(acc);
      acc = {
        start: s.start,
        end: Math.min(s.end, dur),
        rawIndices: [ri],
      };
      continue;
    }

    acc.end = mergedEnd;
    acc.rawIndices.push(ri);
  }
  buckets.push(acc);

  // Absorb parts shorter than MIN into the previous bucket when possible (don’t exceed MAX).
  const merged: Acc[] = [];
  for (const b of buckets) {
    const span = b.end - b.start;
    if (
      merged.length > 0 &&
      span < MIN_PART_SEC - 1e-3 &&
      b.end - merged[merged.length - 1].start <= MAX_PART_SEC + 1e-3
    ) {
      const prev = merged[merged.length - 1];
      prev.end = Math.max(prev.end, b.end);
      prev.rawIndices.push(...b.rawIndices);
    } else {
      merged.push({ ...b, rawIndices: [...b.rawIndices] });
    }
  }

  // If we still have more than 6 parts, repeatedly merge the two adjacent parts with the smallest combined span.
  while (merged.length > 6) {
    let bestI = 0;
    let bestCombined = Infinity;
    for (let i = 0; i < merged.length - 1; i++) {
      const comb = merged[i + 1].end - merged[i].start;
      if (comb < bestCombined) {
        bestCombined = comb;
        bestI = i;
      }
    }
    const a = merged[bestI];
    const b = merged[bestI + 1];
    a.end = b.end;
    a.rawIndices.push(...b.rawIndices);
    merged.splice(bestI + 1, 1);
  }

  return merged.map((m, i) => ({
    partIndex: i,
    label: `Part ${i + 1}`,
    start: round4(Math.max(0, m.start)),
    end: round4(Math.min(m.end, dur)),
    rawIndices: m.rawIndices,
  }));
}

/** Active practice part index for playhead `t`. */
export function practicePartIndexAtTime(
  t: number,
  parts: PracticePart[],
  durationSec: number,
): number {
  if (!parts.length || !Number.isFinite(t) || !Number.isFinite(durationSec)) {
    return -1;
  }
  const tClamped = Math.max(0, Math.min(t, durationSec + 0.02));
  for (let i = 0; i < parts.length; i++) {
    const p = parts[i];
    const isLast = i === parts.length - 1;
    const end = isLast ? durationSec : p.end;
    if (tClamped + 1e-4 >= p.start && tClamped <= end + 1e-3) {
      return i;
    }
  }
  return parts.length - 1;
}
