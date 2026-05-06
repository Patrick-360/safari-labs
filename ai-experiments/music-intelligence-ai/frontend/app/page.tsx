"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { deriveCoreProgression, firstChordTimeForLabel } from "@/lib/coreProgression";
import {
  buildLearnThisSongSummary,
  buildPianoPracticeStepsForPart,
  buildPracticeStepsForPart,
  chordSequenceForPart,
  formatPlayHint,
  getSimplePianoHands,
  PIANO_GUIDANCE_DISCLAIMER,
} from "@/lib/practiceGuidance";
import { buildPracticeParts, practicePartIndexAtTime, type PracticePart } from "@/lib/practiceSections";
import { startMicWavChunks } from "@/lib/micWavChunks";

const CHUNK_SECONDS = 1.0;
const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

/** Console + optional panel: env or add <code>?liveDebug=1</code> to the URL. */
const LIVE_MIC_DEBUG = process.env.NEXT_PUBLIC_LIVE_MIC_DEBUG === "1";

type LiveDebugSnapshot = {
  micPermission: "pending" | "granted" | "denied";
  audioContextState: string;
  chunksCreated: number;
  chunksSent: number;
  lastChunkSize: number;
  lastSamplePeak: number;
  lastSampleRms: number;
  lastUploadStatus: string;
  lastBackendChord: string;
  lastBackendRaw: string;
  lastBackendError: string;
  ignoredResponseReason: string;
};

const LIVE_DEBUG_INITIAL: LiveDebugSnapshot = {
  micPermission: "pending",
  audioContextState: "—",
  chunksCreated: 0,
  chunksSent: 0,
  lastChunkSize: 0,
  lastSamplePeak: 0,
  lastSampleRms: 0,
  lastUploadStatus: "—",
  lastBackendChord: "—",
  lastBackendRaw: "—",
  lastBackendError: "—",
  ignoredResponseReason: "—",
};

const CHORD_HISTORY_MAX = 12;
const CHORD_PLACEHOLDER_IDLE = "--";
const CHORD_PLACEHOLDER_LISTENING = "Listening...";
const POST_STOP_FADE_MS = 5000;
const POST_STOP_CLEAR_MS = 8000;

type StreamResponse = {
  chord: string;
  confidence: number;
  key: string;
  key_confidence: number;
  timestamp: number;
  debug?: { raw_chord?: string; scores_top3?: [string, number][] };
};

type AnalyzeRhythm = {
  assumed_beats_per_bar: number;
  bar_start_times: number[];
};

type AnalyzeChordSeg = {
  start: number;
  end: number;
  label: string;
  notes?: string[];
  practice_hint?: string;
  confidence?: number;
  low_confidence?: boolean;
  is_passing?: boolean;
  chord_role?: string | null;
  /** Debug: best template cosine at segment (optional from API) */
  template_score?: number | null;
  template_margin?: number | null;
};

type AnalyzeApiResponse = {
  duration: number;
  tempo: number;
  key: { label: string; confidence: number };
  chords: AnalyzeChordSeg[];
  beats: { time: number }[];
  sections: { index: number; start: number; end: number; label: string; repeat_group?: string | null }[];
  rhythm?: AnalyzeRhythm;
};

type ChordRun = {
  startSeg: number;
  endSeg: number;
  label: string;
  start: number;
  end: number;
  notesLine: string;
  anyLowConfidence: boolean;
  isPassing?: boolean;
  repeatCount: number;
};

function formatTimeSec(t: number): string {
  if (!Number.isFinite(t)) return "0:00";
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function chordNotesLine(seg: AnalyzeChordSeg | null | undefined): string {
  if (!seg) return "—";
  if (seg.practice_hint && seg.practice_hint.trim()) return seg.practice_hint.trim();
  if (seg.notes?.length) return seg.notes.join(" · ");
  return "—";
}

/** Playback time → chord segment; tolerates end-of-track and boundary floats. */
function chordSegmentIndexAtTime(t: number, chords: AnalyzeChordSeg[], durationSec: number): number {
  if (!chords.length || !Number.isFinite(t) || !Number.isFinite(durationSec)) {
    return -1;
  }
  const tClamped = Math.max(0, Math.min(t, durationSec + 0.02));
  for (let i = 0; i < chords.length; i++) {
    const c = chords[i];
    const isLast = i === chords.length - 1;
    const segEnd = isLast ? durationSec : c.end;
    if (tClamped + 1e-4 >= c.start && tClamped <= segEnd + 1e-3) {
      if (
        !isLast &&
        i + 1 < chords.length &&
        tClamped >= c.end - 1e-4 &&
        tClamped >= chords[i + 1].start - 1e-4
      ) {
        continue;
      }
      return i;
    }
  }
  return chords.length - 1;
}

function chordLabelAtTime(t: number, chords: AnalyzeChordSeg[], durationSec: number): string {
  const idx = chordSegmentIndexAtTime(t, chords, durationSec);
  if (idx < 0) return "—";
  return chords[idx].label;
}

type AnalyzeSectionForLabel = {
  label: string;
  repeat_group?: string | null;
  index?: number;
  start: number;
  end: number;
};

const SECTION_LETTER_RE = /^[A-Za-z]$/;

/** Collapse “Section Section …” prefixes from the backend. */
function collapseDuplicateSectionPrefixes(s: string): string {
  let t = s.trim();
  let guard = 0;
  while (guard++ < 24 && /^Section\s+Section(\s|$)/i.test(t)) {
    t = t.replace(/^Section\s+Section(\s*)/i, "Section$1");
  }
  return t.trim();
}

/**
 * If the entire label is the same phrase repeated (e.g. “Section A Section A”), keep one copy.
 */
function collapseRepeatedFullPhrase(text: string): string {
  const words = text.trim().split(/\s+/);
  if (words.length < 2) {
    return text.trim();
  }
  for (let unitLen = Math.floor(words.length / 2); unitLen >= 1; unitLen--) {
    if (words.length % unitLen !== 0) {
      continue;
    }
    const reps = words.length / unitLen;
    if (reps < 2) {
      continue;
    }
    const unit = words.slice(0, unitLen).join(" ");
    let ok = true;
    for (let r = 1; r < reps; r++) {
      if (words.slice(r * unitLen, (r + 1) * unitLen).join(" ") !== unit) {
        ok = false;
        break;
      }
    }
    if (ok) {
      return unit;
    }
  }
  return text.trim();
}

/**
 * Single user-facing section title for Analyze File (readout, ribbon, loop copy, pill title text).
 * Same logic everywhere — no duplicated “Section” wording.
 *
 * Expected examples (local sanity reference):
 * - ("Section A", "A") → "Section A"
 * - ("Section Section A", "A") → "Section A"  (repeat_group wins)
 * - ("Section Section A", null) → "Section A"
 * - ("Section 3", null) → "Section 3"
 * - ("A", "A") → "Section A"
 * - ("A", null) → "Section A"
 */
function normalizeAnalyzeSectionLabel(
  rawLabel: string,
  repeatGroup?: string | null,
  fallbackIndex?: number,
): string {
  const rg = repeatGroup?.trim();
  if (rg && SECTION_LETTER_RE.test(rg)) {
    return `Section ${rg.toUpperCase()}`;
  }
  let s = (rawLabel ?? "").trim();
  s = collapseDuplicateSectionPrefixes(s);
  s = collapseRepeatedFullPhrase(s);
  if (SECTION_LETTER_RE.test(s)) {
    return `Section ${s.toUpperCase()}`;
  }
  if (!s) {
    return `Section ${(fallbackIndex ?? 0) + 1}`;
  }
  return s;
}

/** Dropdown and tooltips: one clean name plus a single time range (en dash). */
function formatSectionDropdownLabel(s: AnalyzeSectionForLabel, durationSec: number, listIndex: number): string {
  const idx = s.index !== undefined && s.index !== null ? s.index : listIndex;
  const name = normalizeAnalyzeSectionLabel(s.label, s.repeat_group, idx);
  const endDisplay =
    s.end >= durationSec - 0.08 ? Math.min(s.end, durationSec) : s.end;
  return `${name} (${formatTimeSec(s.start)}–${formatTimeSec(endDisplay)})`;
}

/** Practice Part (merged) — dropdown / tooltips with one time range. */
function formatPracticePartDropdown(p: PracticePart, durationSec: number): string {
  const endDisplay = p.end >= durationSec - 0.08 ? Math.min(p.end, durationSec) : p.end;
  return `${p.label} (${formatTimeSec(p.start)}–${formatTimeSec(endDisplay)})`;
}

/** Countdown until next chord (seconds); under 10 → one decimal, else whole seconds. */
function formatCountdown(seconds: number): string {
  const s = Math.max(0, Number(seconds));
  if (!Number.isFinite(s)) {
    return "—";
  }
  if (s < 10) {
    return `${s.toFixed(1)}s`;
  }
  return `${Math.round(s)}s`;
}

const LOOP_WRAP_EPS_SEC = 0.06;

/**
 * Next chord change using detailed segment timings only (not core progression).
 * When looping a section, prefer the next change inside the window; near the loop end,
 * count down to the loop boundary and show the first chord change after restart inside the loop when possible.
 */
function computeNextChordChange(
  t: number,
  chords: AnalyzeChordSeg[],
  durationSec: number,
  loop: { s0: number; s1: number } | null,
): { label: string; seconds: number } | null {
  if (!chords.length || !Number.isFinite(t) || !Number.isFinite(durationSec) || durationSec <= 0) {
    return null;
  }
  const tClamped = Math.max(0, Math.min(t, durationSec));
  const idx = chordSegmentIndexAtTime(tClamped, chords, durationSec);
  if (idx < 0) {
    return null;
  }
  let j = idx + 1;
  const curLabel = chords[idx].label;
  while (j < chords.length && chords[j].label === curLabel) {
    j += 1;
  }
  if (j >= chords.length) {
    return null;
  }
  const inLoop =
    loop !== null &&
    tClamped + 1e-4 >= loop.s0 &&
    tClamped < loop.s1 - 1e-4;
  const loopEndEff = loop ? Math.min(loop.s1, durationSec) : durationSec;

  if (!inLoop || !loop) {
    const sec = Math.max(0, chords[j].start - tClamped);
    return { label: chords[j].label, seconds: sec };
  }

  if (chords[j].start < loopEndEff - LOOP_WRAP_EPS_SEC) {
    const sec = Math.max(0, chords[j].start - tClamped);
    return { label: chords[j].label, seconds: sec };
  }

  const wrapSec = Math.max(0, loopEndEff - tClamped);
  const i0 = chordSegmentIndexAtTime(loop.s0 + 0.02, chords, durationSec);
  let nextInside: string | null = null;
  if (i0 >= 0) {
    for (let k = i0 + 1; k < chords.length && chords[k].start < loopEndEff - 1e-3; k++) {
      if (chords[k].label !== chords[i0].label) {
        nextInside = chords[k].label;
        break;
      }
    }
  }
  const label =
    nextInside ??
    (i0 >= 0 ? chords[i0].label : chords[j].label);
  return { label, seconds: wrapSec };
}

function groupChordRuns(chords: AnalyzeChordSeg[]): ChordRun[] {
  if (!chords.length) return [];
  const runs: ChordRun[] = [];
  let startSeg = 0;
  for (let i = 1; i <= chords.length; i++) {
    if (i === chords.length || chords[i].label !== chords[startSeg].label) {
      const endSeg = i - 1;
      const c0 = chords[startSeg];
      const c1 = chords[endSeg];
      const slice = chords.slice(startSeg, endSeg + 1);
      runs.push({
        startSeg,
        endSeg,
        label: c0.label,
        start: c0.start,
        end: c1.end,
        notesLine: chordNotesLine(c0),
        anyLowConfidence: slice.some((c) => c.low_confidence),
        isPassing: slice.some((c) => c.is_passing === true),
        repeatCount: endSeg - startSeg + 1,
      });
      startSeg = i;
    }
  }
  return runs;
}

function inferRhythmFromBeats(beats: { time: number }[]): AnalyzeRhythm {
  const sorted = [...beats].map((b) => b.time).sort((a, b) => a - b);
  const bpb = 4;
  const bar_start_times = sorted.filter((_, i) => i % bpb === 0);
  return { assumed_beats_per_bar: bpb, bar_start_times };
}

/** ~Bar n · beat m of N — assumes first detected beat is downbeat (heuristic). */
function formatApproxMeterPosition(
  t: number,
  beats: { time: number }[],
  bpb: number,
): string {
  if (!beats.length || bpb < 1) return "—";
  const times = [...beats].map((b) => b.time).sort((a, b) => a - b);
  let idx = -1;
  for (let i = 0; i < times.length; i++) {
    if (times[i] <= t + 1e-4) {
      idx = i;
    } else {
      break;
    }
  }
  if (idx < 0) {
    return "Before first beat";
  }
  const barNum = Math.floor(idx / bpb) + 1;
  const beatInBar = (idx % bpb) + 1;
  return `~Bar ${barNum} · beat ${beatInBar} of ${bpb}`;
}

/** Backend margin scores are in [0, 1]. */
function confidenceLevel(value: number): "Low" | "Medium" | "High" {
  if (value >= 0.5) return "High";
  if (value >= 0.2) return "Medium";
  return "Low";
}

/** Illustrative stages while POST /analyze runs — not timed to real server progress. */
const ANALYZE_STAGE_MESSAGES = [
  {
    title: "Loading audio",
    detail: "Reading your file and sending it for analysis.",
  },
  {
    title: "Rhythm & tempo",
    detail: "Detecting beats and tempo (may overlap with other steps on the server).",
  },
  {
    title: "Harmony",
    detail: "Estimating key, chord changes, and practice sections.",
  },
  {
    title: "Finishing",
    detail: "Preparing the timeline and progression for practice.",
  },
] as const;

const ANALYZE_STAGE_ROTATE_MS = 2300;

const ANALYZE_PLAYBACK_SPEEDS = [0.5, 0.75, 1, 1.25] as const;
type AnalyzePlaybackSpeed = (typeof ANALYZE_PLAYBACK_SPEEDS)[number];
/** Pause at loop boundary before seeking back (count-in style). */
const PRACTICE_LOOP_COUNTIN_MS = 420;

const PRACTICE_CHECKLIST_STEP_LABELS = [
  "Listen to the part once",
  "Practice the left-hand/root notes",
  "Practice the chord changes slowly",
  "Try playing with the recording",
  "Increase speed when comfortable",
] as const;

const PRACTICE_CHECKLIST_TOTAL = PRACTICE_CHECKLIST_STEP_LABELS.length;

function practicePartSessionKey(p: { partIndex: number; start: number; end: number }): string {
  return `${p.partIndex}-${p.start}-${p.end}`;
}

function speedPracticeTip(rate: AnalyzePlaybackSpeed): string {
  switch (rate) {
    case 0.5:
      return "Good for learning notes";
    case 0.75:
      return "Good for chord transitions";
    case 1:
      return "Try full-speed practice";
    case 1.25:
      return "Challenge speed";
    default:
      return "";
  }
}

export default function Home() {
  const [appMode, setAppMode] = useState<"live" | "file">("live");

  const [recording, setRecording] = useState(false);
  const [chord, setChord] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [key, setKey] = useState("—");
  const [keyConfidence, setKeyConfidence] = useState<number | null>(null);
  const [chordHistory, setChordHistory] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [postStopFade, setPostStopFade] = useState(false);

  const [liveTraceUI] = useState(() => {
    if (typeof window === "undefined") {
      return LIVE_MIC_DEBUG;
    }
    return LIVE_MIC_DEBUG || new URLSearchParams(window.location.search).get("liveDebug") === "1";
  });
  const [liveDebug, setLiveDebug] = useState<LiveDebugSnapshot>(LIVE_DEBUG_INITIAL);
  const liveTelemetryThrottleRef = useRef(0);
  /** Mic capture context: created synchronously on Start click, resumed before any await; reused across sessions. */
  const liveMicCtxRef = useRef<AudioContext | null>(null);

  const [analyzeFile, setAnalyzeFile] = useState<File | null>(null);
  const [analyzeFileName, setAnalyzeFileName] = useState<string | null>(null);
  const [analyzeResult, setAnalyzeResult] = useState<AnalyzeApiResponse | null>(null);
  const [analyzeLoading, setAnalyzeLoading] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [analyzeAudioUrl, setAnalyzeAudioUrl] = useState<string | null>(null);
  const [analyzePlaybackTime, setAnalyzePlaybackTime] = useState(0);
  const [analyzeMediaDuration, setAnalyzeMediaDuration] = useState(0);
  const [loopSectionEnabled, setLoopSectionEnabled] = useState(false);
  const [loopSectionIndex, setLoopSectionIndex] = useState<number | null>(null);
  const [analyzeStageIndex, setAnalyzeStageIndex] = useState(0);
  const [analyzePracticeView, setAnalyzePracticeView] = useState<"chords" | "piano">("chords");
  const [analyzePlaybackRate, setAnalyzePlaybackRate] = useState<AnalyzePlaybackSpeed>(1);
  const [analyzePlaying, setAnalyzePlaying] = useState(false);
  const [practiceCountInEnabled, setPracticeCountInEnabled] = useState(false);
  const [loopRestarting, setLoopRestarting] = useState(false);
  /** Session-only: keyed by practice part identity; resets on new file/analysis */
  const [sessionPartChecklist, setSessionPartChecklist] = useState<Record<string, boolean[]>>({});

  const analyzeAudioRef = useRef<HTMLAudioElement | null>(null);
  const progressionScrollRef = useRef<HTMLDivElement | null>(null);
  const loopRestartInProgressRef = useRef(false);

  const analyzePlaybackDuration = useMemo(() => {
    if (!analyzeResult) return 1;
    return analyzeResult.duration > 0
      ? analyzeResult.duration
      : analyzeMediaDuration > 0
        ? analyzeMediaDuration
        : 1;
  }, [analyzeResult, analyzeMediaDuration]);

  /**
   * practiceParts: fewer, larger “Part 1 / Part 2 …” chunks derived from raw /analyze sections
   * (client-only merge for practice — not the same as backend segmentation).
   */
  const practiceParts = useMemo((): PracticePart[] => {
    if (!analyzeResult) {
      return [];
    }
    return buildPracticeParts(analyzeResult.sections ?? [], analyzePlaybackDuration);
  }, [analyzeResult, analyzePlaybackDuration]);

  const activePracticePartIdx = useMemo(() => {
    if (!practiceParts.length) {
      return -1;
    }
    return practicePartIndexAtTime(analyzePlaybackTime, practiceParts, analyzePlaybackDuration);
  }, [practiceParts, analyzePlaybackTime, analyzePlaybackDuration]);

  useEffect(() => {
    if (loopSectionIndex === null || !practiceParts.length) {
      return;
    }
    if (loopSectionIndex >= practiceParts.length) {
      setLoopSectionIndex(Math.max(0, practiceParts.length - 1));
    }
  }, [practiceParts, loopSectionIndex]);

  const activeAnalyzeChordSeg = useMemo(() => {
    if (!analyzeResult) return -1;
    return chordSegmentIndexAtTime(analyzePlaybackTime, analyzeResult.chords, analyzePlaybackDuration);
  }, [analyzeResult, analyzePlaybackTime, analyzePlaybackDuration]);

  useEffect(() => {
    if (!analyzeLoading) {
      setAnalyzeStageIndex(0);
      return;
    }
    setAnalyzeStageIndex(0);
    const id = setInterval(() => {
      setAnalyzeStageIndex((i) => (i + 1) % ANALYZE_STAGE_MESSAGES.length);
    }, ANALYZE_STAGE_ROTATE_MS);
    return () => clearInterval(id);
  }, [analyzeLoading]);

  const chordRuns = useMemo(
    () => groupChordRuns(analyzeResult?.chords ?? []),
    [analyzeResult?.chords],
  );

  const coreProgression = useMemo(() => deriveCoreProgression(chordRuns), [chordRuns]);

  const currentChordLabelForHighlight = useMemo(() => {
    if (!analyzeResult) return "";
    return chordLabelAtTime(analyzePlaybackTime, analyzeResult.chords, analyzePlaybackDuration);
  }, [analyzeResult, analyzePlaybackTime, analyzePlaybackDuration]);

  const activeChordRunIndex = useMemo(() => {
    if (!chordRuns.length || activeAnalyzeChordSeg < 0) return -1;
    return chordRuns.findIndex(
      (r) => activeAnalyzeChordSeg >= r.startSeg && activeAnalyzeChordSeg <= r.endSeg,
    );
  }, [chordRuns, activeAnalyzeChordSeg]);

  useEffect(() => {
    if (!analyzeResult || activeChordRunIndex < 0) return;
    const el = progressionScrollRef.current?.querySelector(
      `[data-prog-run="${activeChordRunIndex}"]`,
    );
    el?.scrollIntoView({ block: "nearest", inline: "center" });
  }, [activeChordRunIndex, analyzeResult]);

  const nextRunDisplay = useMemo(() => {
    if (!analyzeResult || activeChordRunIndex < 0) return { label: "—", notesLine: "—" };
    const nr = chordRuns[activeChordRunIndex + 1];
    if (!nr) return { label: "End of chart", notesLine: "" };
    return { label: nr.label, notesLine: nr.notesLine };
  }, [analyzeResult, chordRuns, activeChordRunIndex]);

  const selectedPracticePart = useMemo((): PracticePart | null => {
    if (loopSectionIndex === null || !practiceParts.length) {
      return null;
    }
    return practiceParts[loopSectionIndex] ?? null;
  }, [practiceParts, loopSectionIndex]);

  const practiceLoopStatusLine = useMemo(() => {
    if (!loopSectionEnabled) return "Loop: off";
    if (selectedPracticePart) return `Looping: ${selectedPracticePart.label}`;
    return "Loop on — pick a part below";
  }, [loopSectionEnabled, selectedPracticePart]);

  /** When loop wrap uses count-in, avoid re-entrant wrap handling */
  useEffect(() => {
    if (!loopSectionEnabled) {
      loopRestartInProgressRef.current = false;
      setLoopRestarting(false);
    }
  }, [loopSectionEnabled]);

  useEffect(() => {
    const el = analyzeAudioRef.current;
    if (el) {
      el.playbackRate = analyzePlaybackRate;
    }
  }, [analyzePlaybackRate, analyzeAudioUrl]);
  const practiceLoopWindow = useMemo((): { s0: number; s1: number } | null => {
    if (!loopSectionEnabled || loopSectionIndex === null || !practiceParts.length) {
      return null;
    }
    const p = practiceParts[loopSectionIndex];
    if (!p) return null;
    const s0 = p.start;
    const s1 = Math.min(p.end, analyzePlaybackDuration);
    const t = analyzePlaybackTime;
    if (t < s0 - 1e-3 || t >= s1 - 1e-3) {
      return null;
    }
    return { s0, s1 };
  }, [
    loopSectionEnabled,
    loopSectionIndex,
    practiceParts,
    analyzePlaybackDuration,
    analyzePlaybackTime,
  ]);

  const nextChordCountdown = useMemo(() => {
    if (!analyzeResult?.chords?.length) {
      return null;
    }
    return computeNextChordChange(
      analyzePlaybackTime,
      analyzeResult.chords,
      analyzePlaybackDuration,
      practiceLoopWindow,
    );
  }, [analyzeResult?.chords, analyzePlaybackTime, analyzePlaybackDuration, practiceLoopWindow]);

  const analyzeRhythmEffective = useMemo((): AnalyzeRhythm => {
    if (!analyzeResult) return { assumed_beats_per_bar: 4, bar_start_times: [] };
    if (analyzeResult.rhythm) return analyzeResult.rhythm;
    return inferRhythmFromBeats(analyzeResult.beats ?? []);
  }, [analyzeResult]);

  const beatDisplayList = useMemo(() => {
    if (!analyzeResult) return [];
    const times = [...(analyzeResult.beats ?? [])].map((b) => b.time).sort((a, b) => a - b);
    const bpb = analyzeRhythmEffective.assumed_beats_per_bar;
    return times.map((time, i) => ({ time, isBarStart: i % bpb === 0 }));
  }, [analyzeResult, analyzeRhythmEffective.assumed_beats_per_bar]);

  const currentAnalyzeChord = useMemo((): AnalyzeChordSeg | null => {
    if (!analyzeResult || activeAnalyzeChordSeg < 0) return null;
    return analyzeResult.chords[activeAnalyzeChordSeg] ?? null;
  }, [analyzeResult, activeAnalyzeChordSeg]);

  const learnThisSongSummary = useMemo(() => {
    if (!analyzeResult) return null;
    return buildLearnThisSongSummary({
      keyLabel: analyzeResult.key.label,
      tempoBpm: analyzeResult.tempo,
      coreEntries: coreProgression,
      practicePartCount: practiceParts.length,
    });
  }, [analyzeResult, coreProgression, practiceParts.length]);

  const nextChordSegAfterCurrent = useMemo((): AnalyzeChordSeg | null => {
    if (!analyzeResult?.chords?.length || activeAnalyzeChordSeg < 0) return null;
    const ch = analyzeResult.chords;
    let j = activeAnalyzeChordSeg + 1;
    const cur = ch[activeAnalyzeChordSeg].label;
    while (j < ch.length && ch[j].label === cur) {
      j += 1;
    }
    return j < ch.length ? ch[j] : null;
  }, [analyzeResult, activeAnalyzeChordSeg]);

  const displayedNextChordSeg = useMemo((): AnalyzeChordSeg | null => {
    if (!analyzeResult) return null;
    if (nextChordCountdown) {
      const label = nextChordCountdown.label;
      if (nextChordSegAfterCurrent?.label === label) {
        return nextChordSegAfterCurrent;
      }
      return analyzeResult.chords.find((c) => c.label === label) ?? nextChordSegAfterCurrent;
    }
    if (activeChordRunIndex >= 0 && chordRuns[activeChordRunIndex + 1]) {
      const nr = chordRuns[activeChordRunIndex + 1];
      return analyzeResult.chords[nr.startSeg] ?? null;
    }
    return null;
  }, [analyzeResult, nextChordCountdown, nextChordSegAfterCurrent, chordRuns, activeChordRunIndex]);

  const focusPracticePart = useMemo((): PracticePart | null => {
    if (!practiceParts.length) return null;
    const idx =
      loopSectionIndex !== null
        ? loopSectionIndex
        : activePracticePartIdx >= 0
          ? activePracticePartIdx
          : 0;
    return practiceParts[idx] ?? null;
  }, [practiceParts, loopSectionIndex, activePracticePartIdx]);

  const focusPartChordSeq = useMemo(() => {
    if (!analyzeResult || !focusPracticePart) return [];
    return chordSequenceForPart(focusPracticePart, analyzeResult.chords, analyzePlaybackDuration);
  }, [analyzeResult, focusPracticePart, analyzePlaybackDuration]);

  const focusPracticePartSteps = useMemo(() => {
    if (!analyzeResult || !focusPracticePart) return [];
    return buildPracticeStepsForPart(focusPracticePart, analyzeResult.chords, analyzePlaybackDuration);
  }, [analyzeResult, focusPracticePart, analyzePlaybackDuration]);

  const focusPianoPartSteps = useMemo(() => {
    if (!analyzeResult || !focusPracticePart) return [];
    return buildPianoPracticeStepsForPart(focusPracticePart, analyzeResult.chords, analyzePlaybackDuration);
  }, [analyzeResult, focusPracticePart, analyzePlaybackDuration]);

  const focusPartChecklistChecks = useMemo((): boolean[] | null => {
    if (!focusPracticePart) return null;
    const k = practicePartSessionKey(focusPracticePart);
    const raw = sessionPartChecklist[k];
    if (!raw?.length) {
      return PRACTICE_CHECKLIST_STEP_LABELS.map(() => false);
    }
    return PRACTICE_CHECKLIST_STEP_LABELS.map((_, i) => Boolean(raw[i]));
  }, [focusPracticePart, sessionPartChecklist]);

  const focusChecklistProgress = useMemo(() => {
    if (!focusPartChecklistChecks) {
      return { done: 0, total: PRACTICE_CHECKLIST_TOTAL, all: false };
    }
    const done = focusPartChecklistChecks.filter(Boolean).length;
    return {
      done,
      total: PRACTICE_CHECKLIST_TOTAL,
      all: done >= PRACTICE_CHECKLIST_TOTAL,
    };
  }, [focusPartChecklistChecks]);

  const togglePracticeChecklistItem = useCallback((part: PracticePart, index: number) => {
    const k = practicePartSessionKey(part);
    setSessionPartChecklist((prev) => {
      const prevRow = prev[k];
      const base: boolean[] =
        prevRow && prevRow.length === PRACTICE_CHECKLIST_TOTAL
          ? [...prevRow]
          : PRACTICE_CHECKLIST_STEP_LABELS.map((_, i) => Boolean(prevRow?.[i]));
      base[index] = !base[index];
      return { ...prev, [k]: base };
    });
  }, []);

  const approxMeterReadout = useMemo(() => {
    if (!analyzeResult) return "—";
    return formatApproxMeterPosition(
      analyzePlaybackTime,
      analyzeResult.beats ?? [],
      analyzeRhythmEffective.assumed_beats_per_bar,
    );
  }, [analyzeResult, analyzePlaybackTime, analyzeRhythmEffective.assumed_beats_per_bar]);

  const seekToSectionIndex = useCallback(
    (idx: number) => {
      if (!practiceParts.length) return;
      const p = practiceParts[idx];
      const el = analyzeAudioRef.current;
      if (!p || !el) return;
      el.currentTime = p.start;
      setAnalyzePlaybackTime(p.start);
      setLoopSectionIndex(idx);
    },
    [practiceParts],
  );

  const goPrevSection = useCallback(() => {
    if (!practiceParts.length) return;
    const idx = activePracticePartIdx >= 0 ? activePracticePartIdx : 0;
    seekToSectionIndex(Math.max(0, idx - 1));
  }, [practiceParts, activePracticePartIdx, seekToSectionIndex]);

  const goNextSection = useCallback(() => {
    if (!practiceParts.length) return;
    const n = practiceParts.length;
    const idx = activePracticePartIdx >= 0 ? activePracticePartIdx : 0;
    seekToSectionIndex(Math.min(n - 1, idx + 1));
  }, [practiceParts, activePracticePartIdx, seekToSectionIndex]);

  const sessionRef = useRef<{ stop: () => Promise<void> } | null>(null);
  const liveEpochRef = useRef(0);
  const recordingRef = useRef(false);
  const postStopTimersRef = useRef<{ fade?: ReturnType<typeof setTimeout>; clear?: ReturnType<typeof setTimeout> }>(
    {},
  );

  const clearPostStopTimers = useCallback(() => {
    const t = postStopTimersRef.current;
    if (t.fade !== undefined) clearTimeout(t.fade);
    if (t.clear !== undefined) clearTimeout(t.clear);
    postStopTimersRef.current = {};
    setPostStopFade(false);
  }, []);

  useEffect(() => {
    if (recording || chord === null) {
      clearPostStopTimers();
      return;
    }
    postStopTimersRef.current.fade = setTimeout(() => setPostStopFade(true), POST_STOP_FADE_MS);
    postStopTimersRef.current.clear = setTimeout(() => {
      setChord(null);
      setPostStopFade(false);
    }, POST_STOP_CLEAR_MS);
    return () => {
      const t = postStopTimersRef.current;
      if (t.fade !== undefined) clearTimeout(t.fade);
      if (t.clear !== undefined) clearTimeout(t.clear);
      postStopTimersRef.current = {};
    };
  }, [recording, chord, clearPostStopTimers]);

  useEffect(() => {
    if (!analyzeFile) {
      setAnalyzeAudioUrl(null);
      setAnalyzePlaybackTime(0);
      setAnalyzeMediaDuration(0);
      return;
    }
    const url = URL.createObjectURL(analyzeFile);
    setAnalyzeAudioUrl(url);
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [analyzeFile]);

  const applyResponse = useCallback((data: StreamResponse) => {
    setConfidence(data.confidence);
    setKey(data.key);
    setKeyConfidence(data.key_confidence);
    if (data.chord !== "N") {
      setChord(data.chord);
      setChordHistory((prev) => {
        if (prev[0] === data.chord) {
          return prev;
        }
        return [data.chord, ...prev].slice(0, CHORD_HISTORY_MAX);
      });
    }
  }, []);

  const sendWav = useCallback(
    async (blob: Blob) => {
      const epoch = liveEpochRef.current;
      if (LIVE_MIC_DEBUG) {
        console.info("[live] upload chunk", { bytes: blob.size, epoch });
      }

      const form = new FormData();
      form.append("file", blob, "chunk.wav");

      let res: Response;
      try {
        res = await fetch(`${API_BASE}/stream`, {
          method: "POST",
          body: form,
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        if (liveTraceUI) {
          setLiveDebug((p) => ({
            ...p,
            lastBackendError: msg,
            lastUploadStatus: "fetch failed",
          }));
        }
        if (LIVE_MIC_DEBUG) {
          console.warn("[live] /stream fetch failed", e);
        }
        throw e;
      }

      if (!res.ok) {
        const text = await res.text();
        if (liveTraceUI) {
          setLiveDebug((p) => ({
            ...p,
            lastBackendError: `${res.status}: ${text.slice(0, 200)}`,
            lastUploadStatus: `http ${res.status}`,
          }));
        }
        throw new Error(`${res.status} ${res.statusText}: ${text}`);
      }

      const data = (await res.json()) as StreamResponse;

      if (epoch !== liveEpochRef.current) {
        if (liveTraceUI) {
          setLiveDebug((p) => ({
            ...p,
            lastBackendChord: data.chord,
            lastBackendRaw: data.debug?.raw_chord ?? "—",
            lastUploadStatus: `ok ${res.status} (ignored)`,
            ignoredResponseReason: "stale_epoch",
          }));
        }
        if (LIVE_MIC_DEBUG) {
          console.info("[live] ignore response (stale epoch)", {
            epoch,
            current: liveEpochRef.current,
            chord: data.chord,
          });
        }
        return;
      }
      if (!recordingRef.current) {
        if (liveTraceUI) {
          setLiveDebug((p) => ({
            ...p,
            lastBackendChord: data.chord,
            lastBackendRaw: data.debug?.raw_chord ?? "—",
            lastUploadStatus: `ok ${res.status} (ignored)`,
            ignoredResponseReason: "recording_stopped",
          }));
        }
        if (LIVE_MIC_DEBUG) {
          console.info("[live] ignore response (recording stopped)", { chord: data.chord });
        }
        return;
      }

      if (liveTraceUI) {
        setLiveDebug((p) => ({
          ...p,
          chunksSent: p.chunksSent + 1,
          lastUploadStatus: `ok ${res.status}`,
          lastBackendChord: data.chord,
          lastBackendRaw: data.debug?.raw_chord ?? "—",
          lastBackendError: "—",
          ignoredResponseReason: "—",
        }));
      }
      if (LIVE_MIC_DEBUG) {
        console.info("[live] /stream ok", {
          chord: data.chord,
          confidence: data.confidence,
          raw: data.debug?.raw_chord,
        });
      }
      applyResponse(data);
    },
    [applyResponse, liveTraceUI],
  );

  const startRecording = useCallback(async () => {
    liveEpochRef.current += 1;
    setError(null);
    setStatus(null);
    clearPostStopTimers();
    setChord(null);
    setChordHistory([]);
    setLiveDebug({
      ...LIVE_DEBUG_INITIAL,
      micPermission: "pending",
    });

    if (sessionRef.current) {
      if (LIVE_MIC_DEBUG) {
        console.info("[live] stopping previous mic session before restart");
      }
      try {
        await sessionRef.current.stop();
      } catch {
        /* ignore */
      }
      sessionRef.current = null;
    }

    try {
      const AC =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      let sharedCtx = liveMicCtxRef.current;
      if (!sharedCtx || sharedCtx.state === "closed") {
        sharedCtx = new AC();
        liveMicCtxRef.current = sharedCtx;
      }
      void sharedCtx.resume();
      if (liveTraceUI) {
        setLiveDebug((p) => ({ ...p, audioContextState: sharedCtx.state }));
      }

      recordingRef.current = true;
      const session = await startMicWavChunks({
        audioContext: sharedCtx,
        chunkSeconds: CHUNK_SECONDS,
        tailMinSeconds: 0.2,
        onDebug:
          LIVE_MIC_DEBUG || liveTraceUI
            ? (message, detail) => {
                if (LIVE_MIC_DEBUG) {
                  console.info("[live:mic]", message, detail ?? "");
                }
                if (message === "get_user_media_ok") {
                  setLiveDebug((p) => ({ ...p, micPermission: "granted" }));
                }
              }
            : undefined,
        onTelemetry:
          LIVE_MIC_DEBUG || liveTraceUI
            ? (info) => {
                const now = Date.now();
                if (now - liveTelemetryThrottleRef.current < 220) {
                  return;
                }
                liveTelemetryThrottleRef.current = now;
                setLiveDebug((p) => ({
                  ...p,
                  audioContextState: info.audioContextState,
                  lastSamplePeak: info.inputPeak,
                  lastSampleRms: info.inputRms,
                }));
              }
            : undefined,
        onChunk: ({ blob }) => {
          if (liveTraceUI) {
            setLiveDebug((p) => ({
              ...p,
              chunksCreated: p.chunksCreated + 1,
              lastChunkSize: blob.size,
            }));
          }
          sendWav(blob).catch((e) => {
            const message = e instanceof Error ? e.message : String(e);
            setError(message);
            if (LIVE_MIC_DEBUG) {
              console.warn("[live] sendWav error", message);
            }
          });
        },
        onError: (err) => {
          setError(err.message);
          if (LIVE_MIC_DEBUG) {
            console.warn("[live:mic] onError", err);
          }
        },
      });
      sessionRef.current = session;
      setRecording(true);
      if (liveTraceUI) {
        setLiveDebug((p) => ({ ...p, audioContextState: sharedCtx.state }));
      }
    } catch (e) {
      recordingRef.current = false;
      const message = e instanceof Error ? e.message : String(e);
      setError(message);
      setLiveDebug((p) => ({ ...p, micPermission: "denied" }));
      if (LIVE_MIC_DEBUG) {
        console.warn("[live] startRecording failed", message);
      }
    }
  }, [sendWav, clearPostStopTimers, liveTraceUI]);

  const stopRecording = useCallback(async () => {
    recordingRef.current = false;
    liveEpochRef.current += 1;
    const session = sessionRef.current;
    sessionRef.current = null;
    if (session) {
      await session.stop();
    }
    setRecording(false);
    setStatus("Stopped.");
    if (liveTraceUI) {
      const ctx = liveMicCtxRef.current;
      setLiveDebug((p) => ({ ...p, audioContextState: ctx && ctx.state !== "closed" ? ctx.state : "—" }));
    }
  }, [liveTraceUI]);

  const onAnalyzeFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    setAnalyzeFile(f);
    setAnalyzeFileName(f?.name ?? null);
    setAnalyzeResult(null);
    setAnalyzeError(null);
    setLoopSectionEnabled(false);
    setLoopSectionIndex(null);
    setAnalyzePlaybackRate(1);
    setSessionPartChecklist({});
  }, []);

  const runAnalyze = useCallback(async () => {
    if (!analyzeFile) return;
    setAnalyzeLoading(true);
    setAnalyzeError(null);
    try {
      const fd = new FormData();
      fd.append("file", analyzeFile);
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: fd,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`${res.status} ${res.statusText}: ${text}`);
      }
      const data = (await res.json()) as AnalyzeApiResponse;
      setAnalyzeResult(data);
      setAnalyzeFileName(analyzeFile.name);
      setLoopSectionEnabled(false);
      setLoopSectionIndex(null);
      setAnalyzePlaybackRate(1);
      setSessionPartChecklist({});
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setAnalyzeError(message);
    } finally {
      setAnalyzeLoading(false);
    }
  }, [analyzeFile]);

  const syncAnalyzePlaybackFromElement = useCallback(
    (el: HTMLAudioElement) => {
      let ct = el.currentTime;
      if (loopSectionEnabled && loopSectionIndex !== null && practiceParts.length) {
        const p = practiceParts[loopSectionIndex];
        if (p && ct >= p.end - 0.05) {
          if (loopRestartInProgressRef.current) {
            setAnalyzePlaybackTime(ct);
            return;
          }
          if (practiceCountInEnabled) {
            loopRestartInProgressRef.current = true;
            el.pause();
            setLoopRestarting(true);
            const start = p.start;
            const savedRate = el.playbackRate;
            window.setTimeout(() => {
              const a = analyzeAudioRef.current;
              if (!a) {
                loopRestartInProgressRef.current = false;
                setLoopRestarting(false);
                return;
              }
              a.playbackRate = savedRate;
              a.currentTime = start;
              setAnalyzePlaybackTime(start);
              void a.play().then(
                () => {
                  loopRestartInProgressRef.current = false;
                  setLoopRestarting(false);
                },
                () => {
                  loopRestartInProgressRef.current = false;
                  setLoopRestarting(false);
                },
              );
            }, PRACTICE_LOOP_COUNTIN_MS);
            setAnalyzePlaybackTime(ct);
            return;
          }
          el.currentTime = p.start;
          ct = p.start;
          setLoopRestarting(true);
          window.setTimeout(() => setLoopRestarting(false), 750);
        }
      }
      setAnalyzePlaybackTime(ct);
    },
    [loopSectionEnabled, loopSectionIndex, practiceParts, practiceCountInEnabled],
  );

  const handleAnalyzeAudioTime = useCallback(
    (e: React.SyntheticEvent<HTMLAudioElement>) => {
      syncAnalyzePlaybackFromElement(e.currentTarget);
    },
    [syncAnalyzePlaybackFromElement],
  );

  const handleAnalyzeAudioSeeked = useCallback(
    (e: React.SyntheticEvent<HTMLAudioElement>) => {
      syncAnalyzePlaybackFromElement(e.currentTarget);
    },
    [syncAnalyzePlaybackFromElement],
  );

  const handleAnalyzeLoadedMetadata = useCallback(
    (e: React.SyntheticEvent<HTMLAudioElement>) => {
      const el = e.currentTarget;
      setAnalyzeMediaDuration(el.duration || 0);
      el.playbackRate = analyzePlaybackRate;
    },
    [analyzePlaybackRate],
  );

  /** Smooth playhead while the file is playing (timeupdate is too coarse on many browsers). */
  useEffect(() => {
    if (appMode !== "file" || !analyzeAudioUrl) return;
    const el = analyzeAudioRef.current;
    if (!el) return;
    let raf = 0;
    const tick = () => {
      syncAnalyzePlaybackFromElement(el);
      if (!el.paused) {
        raf = requestAnimationFrame(tick);
      }
    };
    const onPlay = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(tick);
    };
    const onPause = () => cancelAnimationFrame(raf);
    el.addEventListener("play", onPlay);
    el.addEventListener("pause", onPause);
    if (!el.paused) {
      onPlay();
    }
    return () => {
      cancelAnimationFrame(raf);
      el.removeEventListener("play", onPlay);
      el.removeEventListener("pause", onPause);
    };
  }, [appMode, analyzeAudioUrl, analyzeResult, syncAnalyzePlaybackFromElement]);

  const handleTimelineSeek = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const audioEl = analyzeAudioRef.current;
      if (!audioEl || !Number.isFinite(analyzePlaybackDuration) || analyzePlaybackDuration <= 0) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const pct = Math.max(0, Math.min(1, x / rect.width));
      const t = pct * analyzePlaybackDuration;
      audioEl.currentTime = t;
      syncAnalyzePlaybackFromElement(audioEl);
    },
    [analyzePlaybackDuration, syncAnalyzePlaybackFromElement],
  );

  const chordDisplay =
    chord !== null
      ? {
          text: chord,
          placeholder: false,
          fade: !recording && postStopFade,
        }
      : recording
        ? { text: CHORD_PLACEHOLDER_LISTENING, placeholder: true, fade: false }
        : { text: CHORD_PLACEHOLDER_IDLE, placeholder: true, fade: false };

  const chordValueClass = chordDisplay.placeholder
    ? "chord-stage-value chord-stage-placeholder"
    : chordDisplay.fade
      ? "chord-stage-value chord-stage-fading"
      : "chord-stage-value";

  return (
    <main className={`demo${appMode === "file" ? " demo--file" : ""}`}>
      <header className="hero">
        <h1>Chord lab</h1>
        <p className="hero-sub">Live microphone or full-track analysis</p>
        <div className="mode-toggle" role="tablist" aria-label="Mode">
          <button
            type="button"
            role="tab"
            aria-selected={appMode === "live"}
            className={appMode === "live" ? "active" : ""}
            onClick={() => setAppMode("live")}
          >
            Live microphone
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={appMode === "file"}
            className={appMode === "file" ? "active" : ""}
            onClick={() => setAppMode("file")}
          >
            Analyze file
          </button>
        </div>
      </header>

      {appMode === "live" ? (
        <>
          <div className="controls">
            <button type="button" onClick={() => void startRecording()} disabled={recording}>
              Start Recording
            </button>
            <button type="button" onClick={() => void stopRecording()} disabled={!recording}>
              Stop Recording
            </button>
          </div>

          {status ? (
            <p className="status-line" role="status">
              {status}
            </p>
          ) : null}
          {error ? (
            <p className="error" role="alert">
              {error}
            </p>
          ) : null}

          <section className="chord-stage" aria-live="polite" aria-atomic="true">
            <p className="chord-stage-label">Current chord</p>
            <p className={chordValueClass}>{chordDisplay.text}</p>
            {confidence !== null && chord !== null ? (
              <p className="chord-stage-confidence chord-stage-confidence--soft">
                {confidenceLevel(confidence)} confidence in this chord
              </p>
            ) : null}
          </section>

          <section className="details details--live" aria-label="Key">
            <div className="detail-grid">
              <div className="detail-block">
                <span className="detail-label">Song key</span>
                <span className="detail-value">{key}</span>
              </div>
              {keyConfidence !== null && key !== "—" ? (
                <div className="detail-block">
                  <span className="detail-label">Key (best guess)</span>
                  <span className="detail-value">{confidenceLevel(keyConfidence)}</span>
                </div>
              ) : null}
            </div>
          </section>

          <section className="history-section" aria-label="Chord history">
            <h2 className="section-title">Chord history</h2>
            {chordHistory.length === 0 ? (
              <p className="history-empty">Recent chords appear here as they change.</p>
            ) : (
              <ol className="history-list" aria-label="Recent chords, newest first">
                {chordHistory.map((c, i) => (
                  <li key={`${c}-${i}`}>{c}</li>
                ))}
              </ol>
            )}
          </section>

          {liveTraceUI ? (
            <details className="live-debug-panel">
              <summary>Live debug (remove ?liveDebug=1 or env flag for production)</summary>
              <dl className="live-debug-dl">
                <dt>API</dt>
                <dd>
                  <code>{API_BASE}</code>/stream
                </dd>
                <dt>Mic permission</dt>
                <dd>{liveDebug.micPermission}</dd>
                <dt>AudioContext</dt>
                <dd>{liveDebug.audioContextState}</dd>
                <dt>Chunks created (WAV)</dt>
                <dd>{liveDebug.chunksCreated}</dd>
                <dt>Responses applied</dt>
                <dd>{liveDebug.chunksSent}</dd>
                <dt>Last chunk size</dt>
                <dd>{liveDebug.lastChunkSize} B</dd>
                <dt>Input peak / RMS (throttled)</dt>
                <dd>
                  {liveDebug.lastSamplePeak.toFixed(4)} / {liveDebug.lastSampleRms.toFixed(5)}
                </dd>
                <dt>Last upload</dt>
                <dd>{liveDebug.lastUploadStatus}</dd>
                <dt>Backend chord / raw</dt>
                <dd>
                  {liveDebug.lastBackendChord} / {liveDebug.lastBackendRaw}
                </dd>
                <dt>Ignored reason</dt>
                <dd>{liveDebug.ignoredResponseReason}</dd>
                <dt>Backend error</dt>
                <dd>{liveDebug.lastBackendError}</dd>
              </dl>
            </details>
          ) : null}
        </>
      ) : (
        <section className="analyze-panel" aria-label="File analysis">
          <header className="analyze-panel-header">
            <h2>Practice with a recording</h2>
            <p className="analyze-lead">
              Load a track, follow the big chords, then use practice controls to loop and slow down until it sticks.
            </p>
          </header>

          <div className="analyze-upload-toolbar">
            <label className="analyze-file-input-label">
              <span className="analyze-file-input-btn">Choose audio file</span>
              <input
                type="file"
                accept="audio/*,.wav,.mp3"
                onChange={onAnalyzeFileChange}
                className="analyze-file-input-native"
              />
            </label>
            <button
              type="button"
              className="analyze-run-btn"
              onClick={() => void runAnalyze()}
              disabled={!analyzeFile || analyzeLoading}
            >
              {analyzeLoading ? "Analyzing…" : "Run analysis"}
            </button>
          </div>

          {analyzeLoading ? (
            <div className="analyze-processing" role="status" aria-live="polite" aria-busy="true">
              <div className="analyze-spinner" aria-hidden />
              <div className="analyze-processing-body">
                <strong className="analyze-processing-title">Working on your track</strong>
                <p className="analyze-processing-detail">
                  Longer files take longer. There is no percentage bar because the server does not stream fine-grained
                  progress—the steps below are <strong>illustrative</strong> and may overlap in time.
                </p>
                <ol className="analyze-stage-list" aria-label="Analysis stages (illustrative)">
                  {ANALYZE_STAGE_MESSAGES.map((step, i) => (
                    <li
                      key={step.title}
                      className={
                        i === analyzeStageIndex
                          ? "analyze-stage-item analyze-stage-item--current"
                          : "analyze-stage-item"
                      }
                      aria-current={i === analyzeStageIndex ? "step" : undefined}
                    >
                      <span className="analyze-stage-title">{step.title}</span>
                      <span className="analyze-stage-detail">{step.detail}</span>
                    </li>
                  ))}
                </ol>
              </div>
            </div>
          ) : null}

          {analyzeError ? (
            <p className="error" role="alert">
              {analyzeError}
            </p>
          ) : null}

          {analyzeFile && !analyzeResult ? (
            <div className="analyze-file-card">
              <h3 className="analyze-subhead">File</h3>
              <p className="analyze-file-name">{analyzeFileName ?? analyzeFile.name}</p>
              <p className="analyze-file-meta">{formatBytes(analyzeFile.size)}</p>
            </div>
          ) : null}

          {analyzeResult ? (
            <>
            <div className="analyze-song-summary" aria-label="Song summary">
                <h3 className="analyze-song-summary-title">Your track</h3>
                <p className="analyze-song-summary-name">{analyzeFileName ?? "Song"}</p>
                <div className="analyze-song-summary-stats">
                  <div className="analyze-song-stat">
                    <span className="analyze-song-stat-label">Key</span>
                    <span className="analyze-song-stat-value">{analyzeResult.key.label}</span>
                  </div>
                  <div className="analyze-song-stat">
                    <span className="analyze-song-stat-label">Tempo</span>
                    <span className="analyze-song-stat-value">{Math.round(analyzeResult.tempo)} BPM</span>
                  </div>
                  <div className="analyze-song-stat">
                    <span className="analyze-song-stat-label">Length</span>
                    <span className="analyze-song-stat-value">{formatTimeSec(analyzePlaybackDuration)}</span>
                  </div>
                </div>
                {learnThisSongSummary ? (
                  <div className="analyze-song-suggestion">
                    <p className="analyze-song-suggestion-text">{learnThisSongSummary.suggestion}</p>
                    <p className="analyze-song-parts-meta">
                      {practiceParts.length === 1 ? "1 practice section" : `${practiceParts.length} practice sections`}{" "}
                      for looping below.
                    </p>
                  </div>
                ) : null}
              </div>

              <div className="analyze-chord-rail-block analyze-chord-rail-block--core">
                <h2 className="analyze-learning-heading">Main chord progression</h2>
                <p className="analyze-learning-lead">
                  Big-picture harmony—tap a chord to jump to where it first appears in this track.
                </p>
                <div className="analyze-core-row" aria-label="Core chord progression">
                  {coreProgression.length === 0 ? (
                    <p className="analyze-core-empty">No chord summary available for this track.</p>
                  ) : (
                    coreProgression.map((entry, i) => {
                      const isActive = entry.label === currentChordLabelForHighlight && entry.label !== "N";
                      return (
                        <div className="analyze-core-slot" key={`${entry.label}-core-${i}`}>
                          {i > 0 ? (
                            <span className="analyze-core-arrow" aria-hidden>
                              →
                            </span>
                          ) : null}
                          <button
                            type="button"
                            className={`analyze-core-chord${isActive ? " analyze-core-chord--active" : ""}${
                              entry.anyLowConfidence ? " analyze-core-chord--low-conf" : ""
                            }`}
                            onClick={() => {
                              const el = analyzeAudioRef.current;
                              if (!el || !analyzeResult) return;
                              const t = firstChordTimeForLabel(analyzeResult.chords, entry.label);
                              if (t === null) return;
                              el.currentTime = t;
                              setAnalyzePlaybackTime(t);
                            }}
                            title={`Jump to first ${entry.label}`}
                          >
                            {isActive ? <span className="analyze-core-now">Now</span> : null}
                            <span className="analyze-core-symbol">{entry.label}</span>
                            <span className="analyze-core-notes">{entry.notesLine}</span>
                          </button>
                        </div>
                      );
                    })
                  )}
                </div>
                {analyzeResult.chords.some((c) => c.low_confidence) ? (
                  <p className="analyze-muted-foot">
                    Some chords are best-effort guesses—use your ears when something sounds off.
                  </p>
                ) : null}
              </div>

              {analyzeAudioUrl ? (
                <div className="analyze-practice-controls-card">
                  <h3 className="analyze-subhead">Practice setup</h3>
                  <p className="analyze-practice-controls-lead">
                    Playback, tempo, and looping—get these set, then use <strong>Right now</strong> to follow the chart.
                  </p>
                  <div className="analyze-practice-transport" role="group" aria-label="Practice playback">
                    <button
                      type="button"
                      className="analyze-transport-btn analyze-transport-btn--primary"
                      onClick={() => {
                        const el = analyzeAudioRef.current;
                        if (!el) return;
                        if (el.paused) void el.play();
                        else el.pause();
                      }}
                      aria-label={analyzePlaying ? "Pause" : "Play"}
                    >
                      {analyzePlaying ? "Pause" : "Play"}
                    </button>
                    <div className="analyze-speed-group" role="group" aria-label="Playback speed">
                      {ANALYZE_PLAYBACK_SPEEDS.map((s) => (
                        <button
                          key={s}
                          type="button"
                          className={
                            analyzePlaybackRate === s
                              ? "analyze-speed-btn analyze-speed-btn--active"
                              : "analyze-speed-btn"
                          }
                          onClick={() => setAnalyzePlaybackRate(s)}
                          aria-pressed={analyzePlaybackRate === s}
                          aria-label={`${s === 1 ? "Normal speed" : `${s} times speed`}`}
                        >
                          {s === 1 ? "1×" : `${s}×`}
                        </button>
                      ))}
                    </div>
                  </div>
                  <p className="analyze-speed-tip">{speedPracticeTip(analyzePlaybackRate)}</p>
                  {loopRestarting ? (
                    <p className="analyze-loop-restarting" role="status" aria-live="polite">
                      Loop restarting…
                    </p>
                  ) : null}
                  <audio
                    ref={analyzeAudioRef}
                    src={analyzeAudioUrl}
                    controls
                    onTimeUpdate={handleAnalyzeAudioTime}
                    onSeeked={handleAnalyzeAudioSeeked}
                    onPlay={() => setAnalyzePlaying(true)}
                    onPause={() => setAnalyzePlaying(false)}
                    onEnded={() => setAnalyzePlaying(false)}
                    onLoadedMetadata={handleAnalyzeLoadedMetadata}
                  />
                  <div className="analyze-loop-inline analyze-loop-inline--controls">
                    <label className="analyze-loop-checkbox">
                      <input
                        type="checkbox"
                        checked={loopSectionEnabled}
                        onChange={(e) => setLoopSectionEnabled(e.target.checked)}
                      />
                      <span>Loop a section</span>
                    </label>
                    <select
                      className="analyze-loop-select"
                      value={loopSectionIndex === null ? "" : String(loopSectionIndex)}
                      onChange={(e) => {
                        const v = e.target.value;
                        setLoopSectionIndex(v === "" ? null : Number.parseInt(v, 10));
                      }}
                      aria-label="Section to loop"
                    >
                      <option value="">Choose section…</option>
                      {practiceParts.map((p, i) => (
                        <option key={`part-${p.partIndex}-${p.start}`} value={String(i)}>
                          {formatPracticePartDropdown(p, analyzePlaybackDuration)}
                        </option>
                      ))}
                    </select>
                    <button
                      type="button"
                      className="analyze-loop-jump analyze-loop-jump--inline"
                      title="Jump to the start of the focused practice section (below)"
                      onClick={() => {
                        const el = analyzeAudioRef.current;
                        if (!focusPracticePart || !el) return;
                        el.currentTime = focusPracticePart.start;
                        setAnalyzePlaybackTime(focusPracticePart.start);
                      }}
                    >
                      Jump to section start
                    </button>
                  </div>
                  <label className="analyze-countin-label">
                    <input
                      type="checkbox"
                      checked={practiceCountInEnabled}
                      onChange={(e) => setPracticeCountInEnabled(e.target.checked)}
                    />
                    <span>Pause briefly when a loop repeats (count-in)</span>
                  </label>
                  <p className="analyze-countin-hint analyze-countin-hint--tight">
                    Simple gap before the loop jumps back—not a metronome.
                  </p>
                  <p className="analyze-loop-mode-line" role="status">
                    {practiceLoopStatusLine}
                  </p>
                </div>
              ) : null}

              <div className="analyze-practice-view-bar" role="group" aria-label="Practice view">
                <span className="analyze-practice-view-label">How to read chords</span>
                <div className="analyze-practice-view-toggle">
                  <button
                    type="button"
                    className={
                      analyzePracticeView === "chords"
                        ? "analyze-view-btn analyze-view-btn--active"
                        : "analyze-view-btn"
                    }
                    onClick={() => setAnalyzePracticeView("chords")}
                    aria-pressed={analyzePracticeView === "chords"}
                  >
                    Chords
                  </button>
                  <button
                    type="button"
                    className={
                      analyzePracticeView === "piano"
                        ? "analyze-view-btn analyze-view-btn--active"
                        : "analyze-view-btn"
                    }
                    onClick={() => setAnalyzePracticeView("piano")}
                    aria-pressed={analyzePracticeView === "piano"}
                  >
                    Piano basics
                  </button>
                </div>
              </div>

              <div className="analyze-practice-panel" aria-label="Right now">
                <h3 className="analyze-subhead">Right now in the track</h3>
                <div className="analyze-practice-grid analyze-practice-grid--two">
                  <div className="analyze-practice-cell">
                    <span className="analyze-practice-eyebrow">Current chord</span>
                    <p className="analyze-practice-chord" aria-live="polite">
                      {chordLabelAtTime(analyzePlaybackTime, analyzeResult.chords, analyzePlaybackDuration)}
                    </p>
                    {analyzePracticeView === "piano" ? (
                      (() => {
                        const h = getSimplePianoHands(currentAnalyzeChord);
                        return (
                          <>
                            <p className="analyze-practice-piano-combo">
                              <span className="analyze-practice-piano-k">LH</span> {h.lh}
                              <span className="analyze-practice-piano-sep" aria-hidden>
                                {" "}
                                |{" "}
                              </span>
                              <span className="analyze-practice-piano-k">RH</span> {h.rh}
                            </p>
                            {currentAnalyzeChord?.low_confidence ? (
                              <p className="analyze-practice-ear">Check this one by ear</p>
                            ) : null}
                          </>
                        );
                      })()
                    ) : formatPlayHint(currentAnalyzeChord) ? (
                      <p className="analyze-practice-playhint">{formatPlayHint(currentAnalyzeChord)}</p>
                    ) : chordNotesLine(currentAnalyzeChord) !== "—" ? (
                      <p className="analyze-practice-tones">{chordNotesLine(currentAnalyzeChord)}</p>
                    ) : null}
                    {analyzePracticeView === "chords" && currentAnalyzeChord?.low_confidence ? (
                      <p className="analyze-practice-ear">Check this one by ear</p>
                    ) : null}
                  </div>
                  <div className="analyze-practice-cell">
                    <span className="analyze-practice-eyebrow">Next chord</span>
                    <p className="analyze-practice-chord analyze-practice-chord--secondary" aria-live="polite">
                      {nextChordCountdown ? nextChordCountdown.label : nextRunDisplay.label}
                    </p>
                    {analyzePracticeView === "piano" && nextRunDisplay.label !== "End of chart" ? (
                      (() => {
                        const h = getSimplePianoHands(displayedNextChordSeg);
                        return (
                          <>
                            <p className="analyze-practice-piano-combo">
                              <span className="analyze-practice-piano-k">LH</span> {h.lh}
                              <span className="analyze-practice-piano-sep" aria-hidden>
                                {" "}
                                |{" "}
                              </span>
                              <span className="analyze-practice-piano-k">RH</span> {h.rh}
                            </p>
                            {displayedNextChordSeg?.low_confidence ? (
                              <p className="analyze-practice-ear">Check this one by ear</p>
                            ) : null}
                          </>
                        );
                      })()
                    ) : analyzePracticeView === "chords" ? (
                      <>
                        {displayedNextChordSeg ? (
                          formatPlayHint(displayedNextChordSeg) ? (
                            <p className="analyze-practice-playhint">{formatPlayHint(displayedNextChordSeg)}</p>
                          ) : chordNotesLine(displayedNextChordSeg) !== "—" ? (
                            <p className="analyze-practice-tones">{chordNotesLine(displayedNextChordSeg)}</p>
                          ) : null
                        ) : !nextChordCountdown && nextRunDisplay.notesLine && nextRunDisplay.notesLine !== "—" ? (
                          <p className="analyze-practice-tones">{nextRunDisplay.notesLine}</p>
                        ) : null}
                      </>
                    ) : null}
                    {analyzePracticeView === "chords" && displayedNextChordSeg?.low_confidence ? (
                      <p className="analyze-practice-ear">Check this one by ear</p>
                    ) : null}
                    <p className="analyze-practice-countdown" aria-live="polite">
                      {nextChordCountdown ? (
                        <>
                          Next change in <strong>{formatCountdown(nextChordCountdown.seconds)}</strong>
                        </>
                      ) : (
                        <span className="analyze-practice-countdown--end">End of progression</span>
                      )}
                    </p>
                  </div>
                </div>
                {analyzePracticeView === "piano" ? (
                  <p className="analyze-piano-disclaimer">{PIANO_GUIDANCE_DISCLAIMER}</p>
                ) : null}
              </div>

              {focusPracticePart ? (
                <div className="analyze-practice-part-card" aria-label="Practice part">
                  <h3 className="analyze-subhead">Practice this part</h3>
                  <p className="analyze-part-card-title">
                    <strong>{focusPracticePart.label}</strong>
                    <span className="analyze-part-card-time">
                      {" "}
                      {formatTimeSec(focusPracticePart.start)}–
                      {formatTimeSec(
                        focusPracticePart.end >= analyzePlaybackDuration - 0.08
                          ? Math.min(focusPracticePart.end, analyzePlaybackDuration)
                          : focusPracticePart.end,
                      )}
                    </span>
                  </p>
                  <p className="analyze-part-card-chords">
                    {focusPartChordSeq.length ? (
                      <>
                        Chords in this section:{" "}
                        <span className="analyze-part-card-chord-seq">
                          {focusPartChordSeq.map((r, i) => (
                            <span key={`${r.label}-${i}`}>
                              {i > 0 ? " → " : null}
                              {r.label}
                            </span>
                          ))}
                        </span>
                      </>
                    ) : (
                      "No chord segments in range — follow the chart as you listen."
                    )}
                  </p>
                  <div className="analyze-practice-checklist" aria-label="Practice checklist">
                    <h4 className="analyze-checklist-heading">Checklist</h4>
                    <p className="analyze-checklist-progress" role="status">
                      {focusChecklistProgress.all ? (
                        <span className="analyze-checklist-celebrate">Part complete — try the next part</span>
                      ) : (
                        <>
                          {focusChecklistProgress.done}/{focusChecklistProgress.total} complete
                        </>
                      )}
                    </p>
                    <ul className="analyze-checklist-list">
                      {PRACTICE_CHECKLIST_STEP_LABELS.map((label, i) => (
                        <li key={`chk-${label}`} className="analyze-checklist-item">
                          <label className="analyze-checklist-label">
                            <input
                              type="checkbox"
                              checked={focusPartChecklistChecks?.[i] ?? false}
                              onChange={() => togglePracticeChecklistItem(focusPracticePart, i)}
                            />
                            <span
                              className={
                                focusPartChecklistChecks?.[i]
                                  ? "analyze-checklist-text analyze-checklist-text--done"
                                  : "analyze-checklist-text"
                              }
                            >
                              {label}
                            </span>
                          </label>
                        </li>
                      ))}
                    </ul>
                  </div>
                  <h4 className="analyze-guided-steps-heading">Guided steps</h4>
                  <ol
                    className={
                      analyzePracticeView === "piano"
                        ? "analyze-part-steps analyze-part-steps--piano"
                        : "analyze-part-steps"
                    }
                  >
                    {(analyzePracticeView === "piano" ? focusPianoPartSteps : focusPracticePartSteps).map(
                      (step, i) => (
                        <li
                          key={`step-${i}`}
                          className={step.includes("\n") ? "analyze-part-step--pre" : undefined}
                        >
                          {step}
                        </li>
                      ),
                    )}
                  </ol>
                  <p className="analyze-part-card-pointer">
                    Play, speed, loop, and count-in live in <strong>Practice setup</strong> above.
                  </p>
                </div>
              ) : null}

              <div className="analyze-section-flow analyze-section-flow--compact">
                <h3 className="analyze-subhead">Parts</h3>
                <p className="analyze-section-flow-hint analyze-section-flow-hint--short">
                  Tap a section to jump—same chunks as in <strong>Practice setup</strong>.
                </p>
                <div className="analyze-section-nav-buttons">
                  <button
                    type="button"
                    className="analyze-sec-nav-btn"
                    onClick={goPrevSection}
                    disabled={!practiceParts.length}
                  >
                    ← Previous
                  </button>
                  <button
                    type="button"
                    className="analyze-sec-nav-btn"
                    onClick={goNextSection}
                    disabled={!practiceParts.length}
                  >
                    Next →
                  </button>
                </div>
                <div className="analyze-section-pills" role="list">
                  {practiceParts.map((p, i) => (
                    <button
                      key={`part-${p.partIndex}-${p.start}-${i}`}
                      type="button"
                      role="listitem"
                      className={
                        activePracticePartIdx === i
                          ? "analyze-section-pill analyze-section-pill--active"
                          : loopSectionIndex === i
                            ? "analyze-section-pill analyze-section-pill--loop-target"
                            : "analyze-section-pill"
                      }
                      onClick={() => seekToSectionIndex(i)}
                      title={formatPracticePartDropdown(p, analyzePlaybackDuration)}
                    >
                      <span className="analyze-section-pill-name">{p.label}</span>
                      <span className="analyze-section-pill-time">
                        {formatTimeSec(p.start)}–
                        {formatTimeSec(
                          p.end >= analyzePlaybackDuration - 0.08
                            ? Math.min(p.end, analyzePlaybackDuration)
                            : p.end,
                        )}
                      </span>
                    </button>
                  ))}
                </div>
              </div>

              <details className="analyze-advanced-drawer">
                <summary>More detail — song map, full timeline, raw analysis</summary>
                <div className="analyze-advanced-inner">
                  <section className="analyze-advanced-chunk" aria-label="Song map">
                    <h4 className="analyze-advanced-h">Song map</h4>
                    <div className="analyze-timeline-block analyze-timeline-block--secondary">
                      <p className="analyze-legend analyze-legend--muted">
                        Optional overview — click the bar to seek. For daily practice, the main progression and{" "}
                        <strong>Right now</strong> panels above are usually enough.
                      </p>
                <div className="analyze-section-ribbon" aria-label="Practice parts" role="presentation">
                  {practiceParts.map((p, i) => {
                    const dur = analyzePlaybackDuration;
                    const t0 = p.start;
                    const t1 = Math.min(p.end, dur);
                    const w = dur > 0 ? Math.max(0, ((t1 - t0) / dur) * 100) : 0;
                    const left = dur > 0 ? Math.min(100, (t0 / dur) * 100) : 0;
                    const active = activePracticePartIdx === i;
                    return (
                      <div
                        key={`rib-${p.label}-${p.start}-${i}`}
                        className={
                          active
                            ? "analyze-section-ribbon-seg analyze-section-ribbon-seg--active"
                            : "analyze-section-ribbon-seg"
                        }
                        style={{ left: `${left}%`, width: `${w}%` }}
                        title={formatPracticePartDropdown(p, analyzePlaybackDuration)}
                      >
                        <span className="analyze-section-ribbon-label">{p.label}</span>
                      </div>
                    );
                  })}
                </div>
                <div className="analyze-scrub-wrap analyze-scrub-wrap--learning">
                  <div
                    className="analyze-scrub-track"
                    role="slider"
                    aria-label="Seek in song"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={Math.round(
                      analyzePlaybackDuration > 0
                        ? Math.min(100, (analyzePlaybackTime / analyzePlaybackDuration) * 100)
                        : 0,
                    )}
                    tabIndex={0}
                    onClick={handleTimelineSeek}
                    onKeyDown={(e) => {
                      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
                      e.preventDefault();
                      const audioEl = analyzeAudioRef.current;
                      if (!audioEl || analyzePlaybackDuration <= 0) return;
                      const delta = (e.key === "ArrowLeft" ? -1 : 1) * Math.max(2, analyzePlaybackDuration * 0.02);
                      const next = Math.max(0, Math.min(analyzePlaybackDuration, audioEl.currentTime + delta));
                      audioEl.currentTime = next;
                      syncAnalyzePlaybackFromElement(audioEl);
                    }}
                  >
                    <div className="analyze-chord-lane" aria-hidden>
                      {analyzeResult.chords.map((c, i) => {
                        const w = ((c.end - c.start) / analyzePlaybackDuration) * 100;
                        const active = activeAnalyzeChordSeg === i;
                        return (
                          <div
                            key={`${c.start}-${c.end}-${i}`}
                            className={`analyze-chord-seg${active ? " analyze-chord-active" : ""}${
                              c.low_confidence ? " analyze-chord-seg--low-conf" : ""
                            }`}
                            style={{ width: `${w}%` }}
                            title={`${c.label} ${formatTimeSec(c.start)}–${formatTimeSec(c.end)}`}
                          >
                            {c.label}
                          </div>
                        );
                      })}
                    </div>
                    {beatDisplayList.map((b, i) => (
                      <div
                        key={`beat-${i}-${b.time}`}
                        className={
                          b.isBarStart ? "analyze-beat-tick analyze-beat-tick--bar" : "analyze-beat-tick analyze-beat-tick--offbeat"
                        }
                        style={{ left: `${Math.min(100, (b.time / analyzePlaybackDuration) * 100)}%` }}
                      />
                    ))}
                    {practiceParts.map((p, i) => (
                      <div
                        key={`part-edge-${i}-${p.start}`}
                        className="analyze-section-edge"
                        style={{ left: `${Math.min(100, (p.start / analyzePlaybackDuration) * 100)}%` }}
                        title={`${p.label} · starts`}
                      />
                    ))}
                    <div
                      className="analyze-playhead"
                      style={{
                        left: `${Math.min(
                          100,
                          Math.max(0, (analyzePlaybackTime / analyzePlaybackDuration) * 100),
                        )}%`,
                      }}
                    />
                  </div>
                </div>
              </div>
                  </section>

                  <section className="analyze-advanced-chunk" aria-label="Every chord in order">
                    <h4 className="analyze-advanced-h">Every chord in order</h4>
                    <p className="analyze-more-p">
                      Full timeline with start and end times — for fine placement, not a replacement for the main
                      progression above.
                    </p>
                <div
                  className="analyze-progression-scroll analyze-chord-rail analyze-chord-rail--detail"
                  ref={progressionScrollRef}
                >
                  {chordRuns.map((run, runIdx) => {
                    const isActive = runIdx === activeChordRunIndex;
                    const isPast = activeChordRunIndex >= 0 && runIdx < activeChordRunIndex;
                    const isNext = activeChordRunIndex >= 0 && runIdx === activeChordRunIndex + 1;
                    const holdSec = Math.max(0, run.end - run.start);
                    return (
                      <button
                        key={`run-${run.start}-${run.end}-${run.label}-${runIdx}`}
                        type="button"
                        data-prog-run={runIdx}
                        className={`analyze-prog-chord${isActive ? " analyze-prog-chord--active" : ""}${
                          isPast ? " analyze-prog-chord--past" : ""
                        }${!isPast && !isActive ? " analyze-prog-chord--upcoming" : ""}`}
                        onClick={() => {
                          const el = analyzeAudioRef.current;
                          if (!el) return;
                          el.currentTime = run.start;
                          setAnalyzePlaybackTime(run.start);
                        }}
                        title={`${run.label} at ${formatTimeSec(run.start)} (${holdSec.toFixed(0)}s)`}
                      >
                        {isActive ? <span className="analyze-prog-badge">Now</span> : null}
                        {isNext ? <span className="analyze-prog-badge analyze-prog-badge--next">Next</span> : null}
                        <span className="analyze-prog-symbol">
                          {run.label}
                          {run.repeatCount > 1 ? (
                            <span className="analyze-prog-repeat"> ×{run.repeatCount}</span>
                          ) : null}
                        </span>
                        <span className="analyze-prog-time">
                          {formatTimeSec(run.start)} – {formatTimeSec(run.end)}
                        </span>
                        <span className="analyze-prog-notes">{run.notesLine}</span>
                      </button>
                    );
                  })}
                </div>
                  </section>

                  <section className="analyze-advanced-chunk">
                    <h4 className="analyze-advanced-h">Harmonic segments (from analysis)</h4>
                <p className="analyze-more-p">
                  Raw Section A / B-style cuts returned by the server. The main UI uses merged <strong>Part 1 / Part 2</strong>{" "}
                  practice chunks above—open this only if you want to compare with the automatic segmentation.
                </p>
                <ul className="analyze-raw-sections-list">
                  {(analyzeResult.sections ?? []).map((s, i) => (
                    <li key={`raw-sec-${s.index ?? i}-${s.start}`}>
                      {formatSectionDropdownLabel(s, analyzePlaybackDuration, i)}
                    </li>
                  ))}
                </ul>
                  </section>

                  <section className="analyze-advanced-chunk">
                    <h4 className="analyze-advanced-h">Meter / beats (estimate)</h4>
                <p className="analyze-more-p">
                  Approximate beat: {approxMeterReadout}. Uses {analyzeRhythmEffective.assumed_beats_per_bar} beats per
                  bar (estimate only).
                </p>
                  </section>
                </div>
              </details>
            </>
          ) : analyzeAudioUrl ? (
            <div className="analyze-playback-block">
              <h3 className="analyze-subhead">Playback</h3>
              <div className="analyze-practice-transport" role="group" aria-label="Practice playback">
                <button
                  type="button"
                  className="analyze-transport-btn analyze-transport-btn--primary"
                  onClick={() => {
                    const el = analyzeAudioRef.current;
                    if (!el) return;
                    if (el.paused) void el.play();
                    else el.pause();
                  }}
                  aria-label={analyzePlaying ? "Pause" : "Play"}
                >
                  {analyzePlaying ? "Pause" : "Play"}
                </button>
                <div className="analyze-speed-group" role="group" aria-label="Playback speed">
                  {ANALYZE_PLAYBACK_SPEEDS.map((s) => (
                    <button
                      key={s}
                      type="button"
                      className={
                        analyzePlaybackRate === s
                          ? "analyze-speed-btn analyze-speed-btn--active"
                          : "analyze-speed-btn"
                      }
                      onClick={() => setAnalyzePlaybackRate(s)}
                      aria-pressed={analyzePlaybackRate === s}
                      aria-label={`${s === 1 ? "Normal speed" : `${s} times speed`}`}
                    >
                      {s === 1 ? "1×" : `${s}×`}
                    </button>
                  ))}
                </div>
              </div>
              <p className="analyze-speed-tip">{speedPracticeTip(analyzePlaybackRate)}</p>
              <audio
                ref={analyzeAudioRef}
                src={analyzeAudioUrl}
                controls
                onTimeUpdate={handleAnalyzeAudioTime}
                onSeeked={handleAnalyzeAudioSeeked}
                onPlay={() => setAnalyzePlaying(true)}
                onPause={() => setAnalyzePlaying(false)}
                onEnded={() => setAnalyzePlaying(false)}
                onLoadedMetadata={handleAnalyzeLoadedMetadata}
              />
            </div>
          ) : null}

          {!analyzeResult && !analyzeLoading ? (
            <p className="analyze-empty-hint">Choose an audio file, then run analysis to see chords and sections.</p>
          ) : null}
        </section>
      )}

      <p className="meta-footer">
        API: <code>{API_BASE}</code> — set <code>NEXT_PUBLIC_API_URL</code> to override. Live diagnostics:{" "}
        <code>?liveDebug=1</code> or <code>NEXT_PUBLIC_LIVE_MIC_DEBUG=1</code>.
      </p>
    </main>
  );
}
