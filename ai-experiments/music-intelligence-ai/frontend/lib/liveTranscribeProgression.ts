/**
 * Derive a short, learnable “main progression” from merged live transcription segments.
 * Heuristic only: favors repeated harmonic loops and de-emphasizes one-off noisy slices.
 *
 * Note: `mergeTranscribeTimeline` produces many short atomic spans. A strict per-segment
 * duration floor (~0.34s) filters almost everything — we coalesce same-label neighbors first
 * and use a lenient fallback tier so rough progressions still appear.
 */

import type { TimelineSeg } from "@/lib/liveTranscribeMerge";

export type LiveProgressionQualityId = "still_listening" | "rough" | "stabilizing" | "likely";

export type LiveDerivedProgression = {
  /** Primary progression to show (2–8 typical; up to 12 fallback). */
  labels: string[];
  /** Same as labels; kept for a clear API if we later trim chips differently. */
  chipLabels: string[];
  isLikelyLoop: boolean;
  qualityId: LiveProgressionQualityId;
  /** Short quality line for UI */
  qualityLabel: string;
  /** Show “still listening for the pattern…” helper */
  showPatternHint: boolean;
  /** True when we used the looser second-pass filter (tiny slices / softer low-conf gate). */
  usedLenientFallback: boolean;
};

type HarmonicBlock = {
  label: string;
  dur: number;
  lowConfidence: boolean;
  conf: number;
};

/** After coalescing, blocks shorter than this are dropped in the strict pass. */
const MIN_BLOCK_DURATION_STRICT = 0.14;
/** Lenient pass: keep very short holds if same label was glued. */
const MIN_BLOCK_DURATION_LENIENT = 0.06;
/** Below this model confidence, treat as weak (when low_confidence is also true, drop in strict pass). */
const LOW_CONF_DROP_STRICT = 0.28;
const LOW_CONF_DROP_LENIENT = 0.12;

const TARGET_MIN = 2;
const TARGET_MAX = 8;
const HARD_CAP = 12;

const QUALITY_LABELS: Record<LiveProgressionQualityId, string> = {
  still_listening: "Still listening",
  rough: "Rough progression",
  stabilizing: "Pattern stabilizing",
  likely: "Likely progression",
};

function segmentDuration(s: TimelineSeg): number {
  return Math.max(0, s.t1 - s.t0);
}

function isBadLabel(lab: string): boolean {
  const t = lab.trim();
  return !t || t === "N" || t === "n";
}

/**
 * Merge adjacent spans with the same label when separated by a tiny gap (window stitching).
 */
function coalesceSameLabelGaps(segments: TimelineSeg[], maxGapSec: number): TimelineSeg[] {
  const sorted = [...segments]
    .filter((s) => !isBadLabel(s.label))
    .sort((a, b) => a.t0 - b.t0 || b.sourceWindowEnd - a.sourceWindowEnd);
  const out: TimelineSeg[] = [];
  for (const s of sorted) {
    const last = out[out.length - 1];
    if (last && last.label === s.label && s.t0 - last.t1 <= maxGapSec + 1e-4) {
      last.t1 = Math.max(last.t1, s.t1);
      last.lowConfidence = Boolean(last.lowConfidence) || Boolean(s.lowConfidence);
      last.confidence =
        last.confidence != null || s.confidence != null
          ? Math.min(last.confidence ?? 1, s.confidence ?? 1)
          : undefined;
      last.sourceWindowEnd = Math.max(last.sourceWindowEnd, s.sourceWindowEnd);
    } else {
      out.push({ ...s });
    }
  }
  return out;
}

/** Drop blocks that are too short or clearly weak (strict). */
function filterBlocks(blocks: HarmonicBlock[], minDur: number, lowConfDrop: number): HarmonicBlock[] {
  return blocks.filter((b) => {
    if (b.dur < minDur - 1e-4) {
      return false;
    }
    const low = Boolean(b.lowConfidence);
    const conf = typeof b.conf === "number" ? b.conf : 1;
    if (low && conf < lowConfDrop) {
      return false;
    }
    return true;
  });
}

/** Merge adjacent same-label segments in time order (duration-weighted). */
function toHarmonicBlocks(sorted: TimelineSeg[]): HarmonicBlock[] {
  const blocks: HarmonicBlock[] = [];
  for (const s of sorted) {
    const dur = segmentDuration(s);
    const low = Boolean(s.lowConfidence);
    const conf = typeof s.confidence === "number" ? s.confidence : 1;
    const last = blocks[blocks.length - 1];
    if (last && last.label === s.label) {
      last.dur += dur;
      last.lowConfidence = last.lowConfidence || low;
      last.conf = Math.min(last.conf, conf);
    } else {
      blocks.push({
        label: s.label,
        dur,
        lowConfidence: low,
        conf,
      });
    }
  }
  return blocks;
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

function trailingRepeatLoop(collapsed: string[]): string[] | null {
  const n = collapsed.length;
  for (let len = Math.min(8, Math.floor(n / 2)); len >= 2; len--) {
    let repeats = 1;
    let idx = n - len;
    while (idx >= len) {
      const cur = collapsed.slice(idx, idx + len);
      const prev = collapsed.slice(idx - len, idx);
      if (cur.every((v, i) => v === prev[i])) {
        repeats++;
        idx -= len;
      } else {
        break;
      }
    }
    if (repeats >= 2) {
      return collapsed.slice(n - len, n);
    }
  }
  return null;
}

function interiorRepeatLoop(collapsed: string[]): string[] | null {
  const n = collapsed.length;
  for (let len = Math.min(8, Math.floor(n / 2)); len >= 2; len--) {
    for (let start = 0; start <= n - 2 * len; start++) {
      const a = collapsed.slice(start, start + len);
      const b = collapsed.slice(start + len, start + 2 * len);
      if (a.every((v, i) => v === b[i])) {
        return [...a];
      }
    }
  }
  return null;
}

function stableOrderedCore(blocks: HarmonicBlock[]): string[] {
  const collapsed = blocks.map((b) => b.label);
  const order = uniqueInOrder(collapsed);

  const freq = new Map<string, number>();
  const durBy = new Map<string, number>();
  for (const b of blocks) {
    freq.set(b.label, (freq.get(b.label) ?? 0) + 1);
    durBy.set(b.label, (durBy.get(b.label) ?? 0) + b.dur);
  }

  const filtered = order.filter((lab) => {
    const f = freq.get(lab) ?? 0;
    const d = durBy.get(lab) ?? 0;
    if (f >= 2) {
      return true;
    }
    if (d >= 0.85) {
      return true;
    }
    return false;
  });

  let out = filtered.length >= TARGET_MIN ? filtered.slice(0, TARGET_MAX) : [];

  if (out.length < TARGET_MIN) {
    out = order.slice(0, Math.min(TARGET_MAX, order.length));
  }
  if (out.length === 0 && order.length > 0) {
    out = order.slice(0, Math.min(HARD_CAP, order.length));
  }

  return out.slice(0, HARD_CAP);
}

function lowConfidenceMassShare(blocks: HarmonicBlock[]): number {
  let mass = 0;
  let lowMass = 0;
  for (const b of blocks) {
    mass += b.dur;
    if (b.lowConfidence) {
      lowMass += b.dur;
    }
  }
  return mass > 1e-6 ? lowMass / mass : 0;
}

function inferQuality(
  ctx: { analysisCount: number; bufferSec: number },
  collapsedLen: number,
  isLoop: boolean,
  outLen: number,
  lowShare: number,
  lenient: boolean,
): LiveProgressionQualityId {
  if (ctx.analysisCount < 2 || ctx.bufferSec < 5 || collapsedLen < 2) {
    return lenient && outLen >= 2 ? "rough" : "still_listening";
  }
  if (isLoop && outLen >= 2 && lowShare < 0.55) {
    return "likely";
  }
  if (ctx.analysisCount >= 5 && outLen >= 3 && lowShare < 0.58) {
    return "stabilizing";
  }
  if (ctx.analysisCount >= 2 && outLen >= 2) {
    return "rough";
  }
  return "still_listening";
}

function buildFromBlocks(
  blocks: HarmonicBlock[],
  ctx: { analysisCount: number; bufferSec: number },
  lenient: boolean,
): LiveDerivedProgression {
  const empty: LiveDerivedProgression = {
    labels: [],
    chipLabels: [],
    isLikelyLoop: false,
    qualityId: "still_listening",
    qualityLabel: QUALITY_LABELS.still_listening,
    showPatternHint: true,
    usedLenientFallback: lenient,
  };

  if (!blocks.length) {
    return empty;
  }

  const collapsed = blocks.map((b) => b.label);

  if (collapsed.length < 2) {
    const out = uniqueInOrder(collapsed).slice(0, HARD_CAP);
    const lowShare = lowConfidenceMassShare(blocks);
    const qualityId = inferQuality(ctx, collapsed.length, false, out.length, lowShare, lenient);
    const showPatternHint = out.length > 0 && out.length < TARGET_MIN && qualityId !== "likely";
    return {
      labels: out,
      chipLabels: out,
      isLikelyLoop: false,
      qualityId,
      qualityLabel: lenient && out.length >= 1 ? "Rough progression so far" : QUALITY_LABELS[qualityId],
      showPatternHint,
      usedLenientFallback: lenient,
    };
  }

  const loopTail = trailingRepeatLoop(collapsed);
  const loopInner = loopTail ?? interiorRepeatLoop(collapsed);
  const isLikelyLoop = loopInner != null && loopInner.length >= 2 && loopInner.length <= 8;

  let labels: string[];
  if (loopInner) {
    labels = [...loopInner];
    if (labels.length > TARGET_MAX) {
      labels = labels.slice(0, TARGET_MAX);
    }
  } else {
    labels = stableOrderedCore(blocks);
  }

  labels = labels.filter((l) => !isBadLabel(l)).slice(0, HARD_CAP);

  const lowShare = lowConfidenceMassShare(blocks);
  const qualityId = inferQuality(ctx, collapsed.length, Boolean(isLikelyLoop), labels.length, lowShare, lenient);
  let qualityLabel = QUALITY_LABELS[qualityId];
  if (lenient && labels.length >= 2 && qualityId !== "likely") {
    qualityLabel = "Rough progression so far";
  }

  const showPatternHint = labels.length > 0 && labels.length < TARGET_MIN && qualityId !== "likely";

  return {
    labels,
    chipLabels: labels,
    isLikelyLoop,
    qualityId,
    qualityLabel,
    showPatternHint,
    usedLenientFallback: lenient,
  };
}

/**
 * Build chip-ready progression + loop detection + simple quality tier.
 */
export function deriveLiveStableProgression(
  timeline: TimelineSeg[],
  ctx: { analysisCount: number; bufferSec: number },
): LiveDerivedProgression {
  const empty: LiveDerivedProgression = {
    labels: [],
    chipLabels: [],
    isLikelyLoop: false,
    qualityId: "still_listening",
    qualityLabel: QUALITY_LABELS.still_listening,
    showPatternHint: true,
    usedLenientFallback: false,
  };

  if (!timeline.length) {
    return empty;
  }

  const coalesced = coalesceSameLabelGaps(timeline, 0.05);
  const sorted = [...coalesced].sort((a, b) => a.t0 - b.t0 || b.sourceWindowEnd - a.sourceWindowEnd);
  const blocksAll = toHarmonicBlocks(sorted);
  let blocks = filterBlocks(blocksAll, MIN_BLOCK_DURATION_STRICT, LOW_CONF_DROP_STRICT);
  let result = buildFromBlocks(blocks, ctx, false);

  if (!result.labels.length && blocksAll.length) {
    blocks = filterBlocks(blocksAll, MIN_BLOCK_DURATION_LENIENT, LOW_CONF_DROP_LENIENT);
    result = buildFromBlocks(blocks, ctx, true);
  }

  return result;
}

/**
 * When the server returned chords but the merged timeline is empty (e.g. first paint),
 * derive a rough progression from the latest window chord list alone.
 */
export function deriveFallbackProgressionFromWindowChords(
  chords: { label: string; low_confidence?: boolean; confidence?: number; start: number; end: number }[],
  max = 8,
): string[] {
  const usable = chords.filter((c) => !isBadLabel(c.label));
  if (!usable.length) {
    return [];
  }
  const sorted = [...usable].sort((a, b) => a.start - b.start);
  const seq: string[] = [];
  for (const c of sorted) {
    const dur = Math.max(0, c.end - c.start);
    if (dur < 0.04) {
      continue;
    }
    const low = Boolean(c.low_confidence);
    const conf = typeof c.confidence === "number" ? c.confidence : 0.5;
    if (low && conf < 0.1 && dur < 0.25) {
      continue;
    }
    if (!seq.length || seq[seq.length - 1] !== c.label) {
      seq.push(c.label);
    }
  }
  return uniqueInOrder(seq).slice(0, max);
}
