export type LiveTranscribeKey = { label: string; confidence: number };

/** Prefer stable key: switch only if new estimate is clearly stronger or matches refinement. */
export function mergeLiveTranscribeKey(prev: LiveTranscribeKey | null, next: LiveTranscribeKey): LiveTranscribeKey {
  if (!prev || prev.label === "—" || next.label === "—") {
    return next.label !== "—" ? next : prev ?? next;
  }
  if (next.label === prev.label) {
    return { label: prev.label, confidence: Math.max(prev.confidence, next.confidence) };
  }
  if (next.confidence >= prev.confidence + 0.12) {
    return next;
  }
  return prev;
}

/**
 * One harmonic span on the client session clock after resolving overlaps.
 * Newer analysis (`sourceWindowEnd`) wins when windows disagree on the same instant.
 */
export type TimelineSeg = {
  t0: number;
  t1: number;
  label: string;
  lowConfidence?: boolean;
  confidence?: number;
  notes?: string[];
  /** Absolute client timeline end of the window that produced this segment; higher = fresher. */
  sourceWindowEnd: number;
};

export type TranscribeChordInput = {
  start: number;
  end: number;
  label: string;
  low_confidence?: boolean;
  confidence?: number;
  notes?: string[];
};

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

function coerceSeg(s: TimelineSeg): TimelineSeg {
  return {
    ...s,
    sourceWindowEnd: typeof s.sourceWindowEnd === "number" ? s.sourceWindowEnd : 0,
  };
}

/**
 * Merge a new analysis window into the rolling timeline:
 * - Clip to [horizonEnd - keepSeconds, horizonEnd]
 * - Resolve micro-second conflicts: prefer segment from the analysis with larger `sourceWindowEnd`
 * - Merge adjacent same-label atoms into longer holds
 */
export function mergeTranscribeTimeline(
  prev: TimelineSeg[],
  windowStart: number,
  windowEnd: number,
  chords: TranscribeChordInput[],
  keepSeconds: number,
): TimelineSeg[] {
  const horizonEnd = round4(windowEnd);
  const cutoff = round4(horizonEnd - keepSeconds);

  const fresh: TimelineSeg[] = chords
    .filter((c) => c.label !== "N" && c.label.trim() !== "")
    .map((c) => ({
      t0: round4(windowStart + c.start),
      t1: round4(windowStart + c.end),
      label: c.label,
      lowConfidence: c.low_confidence === true,
      confidence: typeof c.confidence === "number" ? c.confidence : undefined,
      notes: c.notes?.length ? [...c.notes] : undefined,
      sourceWindowEnd: horizonEnd,
    }))
    .filter((s) => s.t1 > s.t0 + 1e-5);

  const legacy = prev.map(coerceSeg);
  const allRaw: TimelineSeg[] = [];
  for (const s of legacy) {
    const a = round4(Math.max(s.t0, cutoff));
    const b = round4(Math.min(s.t1, horizonEnd));
    if (b > a + 1e-5 && s.label !== "N") {
      allRaw.push({
        ...s,
        t0: a,
        t1: b,
      });
    }
  }
  for (const s of fresh) {
    const a = round4(Math.max(s.t0, cutoff));
    const b = round4(Math.min(s.t1, horizonEnd));
    if (b > a + 1e-5) {
      allRaw.push({ ...s, t0: a, t1: b });
    }
  }

  if (allRaw.length === 0) {
    return [];
  }

  const breaks = new Set<number>();
  breaks.add(cutoff);
  breaks.add(horizonEnd);
  for (const s of allRaw) {
    breaks.add(round4(s.t0));
    breaks.add(round4(s.t1));
  }
  const br = [...breaks].sort((x, y) => x - y);

  const atomic: TimelineSeg[] = [];
  for (let i = 0; i < br.length - 1; i++) {
    const t0 = br[i];
    const t1 = br[i + 1];
    if (t1 <= t0 + 1e-5) {
      continue;
    }

    let winner: TimelineSeg | null = null;
    for (const s of allRaw) {
      if (s.t0 < t1 - 1e-5 && s.t1 > t0 + 1e-5) {
        const s0 = Math.max(s.t0, cutoff);
        const s1 = Math.min(s.t1, horizonEnd);
        if (!(s0 < s1 - 1e-5)) {
          continue;
        }
        if (s1 <= t0 || s0 >= t1) {
          continue;
        }
        if (
          !winner ||
          s.sourceWindowEnd > winner.sourceWindowEnd ||
          (s.sourceWindowEnd === winner.sourceWindowEnd &&
            (s.confidence ?? 0) > (winner.confidence ?? 0))
        ) {
          winner = s;
        }
      }
    }

    if (!winner || winner.label === "N" || isBadLabel(winner.label)) {
      continue;
    }

    atomic.push({
      t0,
      t1,
      label: winner.label,
      lowConfidence: winner.lowConfidence,
      confidence: winner.confidence,
      notes: winner.notes?.length ? [...winner.notes] : undefined,
      sourceWindowEnd: winner.sourceWindowEnd,
    });
  }

  const merged: TimelineSeg[] = [];
  for (const a of atomic) {
    const last = merged[merged.length - 1];
    if (
      last &&
      last.label === a.label &&
      Math.abs(last.t1 - a.t0) < 2e-3 &&
      last.sourceWindowEnd <= a.sourceWindowEnd + 1e-5
    ) {
      last.t1 = a.t1;
      last.lowConfidence = Boolean(last.lowConfidence) || Boolean(a.lowConfidence);
      last.confidence =
        last.confidence != null || a.confidence != null
          ? Math.min(last.confidence ?? 1, a.confidence ?? 1)
          : undefined;
      if ((!last.notes || last.notes.length === 0) && a.notes?.length) {
        last.notes = [...a.notes];
      }
      last.sourceWindowEnd = Math.max(last.sourceWindowEnd, a.sourceWindowEnd);
    } else {
      merged.push({ ...a });
    }
  }

  return merged.filter((s) => s.t1 > s.t0 + 1e-5 && s.t1 > cutoff + 1e-5);
}

function isBadLabel(lab: string): boolean {
  const t = lab.trim();
  return !t || t === "N" || t === "n";
}
