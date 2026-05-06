"use client";

import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";

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
import { LiveTranscribeRing } from "@/lib/liveTranscribeRing";
import {
  mergeLiveTranscribeKey,
  mergeTranscribeTimeline,
  type LiveTranscribeKey,
  type TimelineSeg,
} from "@/lib/liveTranscribeMerge";
import { deriveFallbackProgressionFromWindowChords, deriveLiveStableProgression } from "@/lib/liveTranscribeProgression";
import {
  buildLiveTranscribeSnapshot,
  copyTextToClipboard,
  downloadLiveTranscribeSnapshotJson,
} from "@/lib/liveTranscribeSnapshot";
import { encodeFloat32MonoToWav, startMicWavChunks, type MicTrackDebugSnapshot } from "@/lib/micWavChunks";
import {
  liveTriadNoteNamesFromLabel,
  transposeChordLabel,
  transposeChordSegment,
  transposeChordToneLine,
  transposeNotes,
} from "@/lib/transpose";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

/** Verbose `[live]` console logs — optional via env or <code>?liveDebug=1</code>. */
const LIVE_MIC_CONSOLE = process.env.NEXT_PUBLIC_LIVE_MIC_DEBUG === "1";

/** Sent as <code>?mode=…</code> on every POST /stream (live sensitivity preset). */
type LiveInputMode = "instrument" | "song" | "debug";

const LIVE_INPUT_MODE_OPTIONS: { value: LiveInputMode; label: string; hint: string }[] = [
  {
    value: "instrument",
    label: "Instrument",
    hint: "Best when you are playing piano or guitar into the mic. Ignores most room noise and chatter.",
  },
  {
    value: "song",
    label: "Speaker / room",
    hint: "For a track playing on speakers or a phone in the room. Snappier, but messier than a file upload.",
  },
  {
    value: "debug",
    label: "Diagnostics",
    hint: "Looser listening for testing only — not for serious practice.",
  },
];

/** Default mic analysis gain when the user has not manually chosen a boost level. */
const LIVE_BOOST_DEFAULTS: Record<LiveInputMode, number> = {
  instrument: 1,
  song: 4,
  debug: 2,
};

const LIVE_INPUT_BOOST_OPTIONS: readonly number[] = [1, 2, 4, 8];

/** WAV chunk length for live mic; shorter in song mode for lower latency (still valid WAV for /stream). */
const LIVE_CHUNK_SECONDS: Record<LiveInputMode, number> = {
  instrument: 1.0,
  song: 0.45,
  debug: 1.0,
};

/** Top-level live product: instant /stream vs rolling-window transcription. */
type LiveExperienceMode = "instant" | "transcribe";

/**
 * Live song transcription timing (tune here).
 * - FIRST: wait until this much audio is buffered before the first POST (tradeoff: lower = snappier but noisier).
 * - WINDOW: each request sends up to this many seconds (harmonic context; longer = more stable).
 * - INTERVAL: how often follow-up POSTs run (tradeoff: lower = fresher UI but more server load; keep < WINDOW so windows overlap).
 */
const FIRST_TRANSCRIBE_AFTER_SEC = 5;
/** Max seconds of audio per analysis request (rolling tail). */
const TRANSCRIBE_WINDOW_SEC = 12;
/** Seconds between follow-up analyses while listening. */
const TRANSCRIBE_INTERVAL_SEC = 8;
/** Reject analysis if the sliced WAV is shorter than this (avoids random guesses on tiny slices). */
const MIN_TRANSCRIBE_AUDIO_SEC = 4;

const FIRST_TRANSCRIBE_DELAY_MS = FIRST_TRANSCRIBE_AFTER_SEC * 1000;
const TRANSCRIBE_INTERVAL_MS = TRANSCRIBE_INTERVAL_SEC * 1000;

/** Ring must hold at least WINDOW; small headroom for scheduling jitter. */
const TRANSCRIBE_RING_MAX_SEC = 16;
const TRANSCRIBE_TIMELINE_KEEP_SEC = 32;

type LiveTranscribeApiResponse = {
  window_start: number;
  window_end: number;
  session_id?: string | null;
  key: { label: string; confidence: number };
  current_chord: string;
  chords?: {
    start: number;
    end: number;
    label: string;
    confidence: number;
    notes: string[];
    practice_hint: string;
    low_confidence: boolean;
    is_passing?: boolean;
    chord_role?: string | null;
  }[];
  core_progression: { label: string; notes: string[] }[];
  summary: string;
  status: "listening" | "analyzing" | "ready";
  tempo_bpm?: number;
  progression_meta?: {
    source?: string;
    quality?: string;
    empty_reason?: string | null;
  };
  debug?: Record<string, unknown>;
};

type TranscribeDebugSnapshot = {
  ringBufferedSec: number;
  lastWindowSec: number;
  lastRawPeak: number;
  lastRawRms: number;
  lastRequestStatus: string;
  lastKeyLabel: string;
  lastProgression: string;
  lastError: string;
  analysisCount: number;
  /** Round-trip time for last completed /live-transcribe request (ms), excluding client prep. */
  lastRequestDurationMs: number | null;
  /** Last applied JSON `status` from server (listening / analyzing / ready). */
  lastAnalysisStatus: string;
  /** From last response `debug.core_empty_reason` when debug=true */
  lastLtCoreEmptyReason: string;
  /** From last response `debug.runs_for_core_strategy` */
  lastLtRunsForCoreStrategy: string;
  /** Compact segment counts: total/stable/low from server debug */
  lastLtServerSegmentSummary: string;
  /** Echo of optional query client_timeline_seg_count when debug=true */
  lastLtClientTimelineSegEcho: string;
  lastProgressionSource: string;
  lastProgressionQuality: string;
};

const TRANSCRIBE_DEBUG_INITIAL: TranscribeDebugSnapshot = {
  ringBufferedSec: 0,
  lastWindowSec: 0,
  lastRawPeak: 0,
  lastRawRms: 0,
  lastRequestStatus: "—",
  lastKeyLabel: "—",
  lastProgression: "—",
  lastError: "—",
  analysisCount: 0,
  lastRequestDurationMs: null,
  lastAnalysisStatus: "—",
  lastLtCoreEmptyReason: "—",
  lastLtRunsForCoreStrategy: "—",
  lastLtServerSegmentSummary: "—",
  lastLtClientTimelineSegEcho: "—",
  lastProgressionSource: "—",
  lastProgressionQuality: "—",
};

type LiveDebugSnapshot = {
  /** Tracks latest getUserMedia outcome for this app session */
  micPermission: "pending" | "granted" | "denied";
  /** `navigator.permissions` for microphone when supported (may differ until you press Start) */
  browserMicPermission: string;
  audioContextState: string;
  /** Total ScriptProcessor callbacks since mic session started */
  audioProcessCallbacks: number;
  /** How long ago `onaudioprocess` last ran (ms), refreshed while recording */
  msSinceLastAudioProcess: number;
  chunksCreated: number;
  /** Successful HTTP responses from POST /stream (body read) */
  chunksPosted: number;
  /** Responses that passed epoch + recording gate and called applyResponse */
  responsesApplied: number;
  lastChunkSize: number;
  /** Pre-boost mono peak (same downmix as WAV source, before gain). */
  lastRawSamplePeak: number;
  lastRawSampleRms: number;
  /** After input boost + clamp [-1,1] — levels fed to WAV/PCM. */
  lastBoostedSamplePeak: number;
  lastBoostedSampleRms: number;
  /** Linear gain applied in mic capture (WAV path only; speakers unchanged). */
  liveInputBoost: number;
  /** Fraction of samples in the last telemetry buffer that hit the clamp (pre-WAV). */
  lastBoostClipFraction: number;
  /** Channels in ScriptProcessor inputBuffer for last telemetry tick */
  lastInputBufferChannels: number;
  /** WAV chunk duration for current Live input mode (seconds). */
  liveChunkSeconds: number;
  /** Ms since a WAV chunk was queued for upload (while recording; refreshed ~250ms). */
  msSinceLastChunkSent: number | null;
  /** Ms since last /stream JSON received (while recording; refreshed ~250ms). */
  msSinceLastStreamResponse: number | null;
  /** Last API debug: weak_confirm_chunks */
  lastPresetWeakConfirmChunks: number | null;
  /** MediaStreamTrack snapshot (see Live debug) */
  trackCount: number;
  trackLabel: string;
  trackEnabled: boolean;
  trackMuted: boolean;
  trackReadyState: string;
  trackDeviceId: string;
  trackSampleRate: string;
  trackChannelCount: string;
  trackEchoCancellation: string;
  trackNoiseSuppression: string;
  trackAutoGainControl: string;
  lastUploadStatus: string;
  lastHttpStatus: number | null;
  lastBackendChord: string;
  lastBackendRaw: string;
  /** Latest `debug.rejection_reason` from POST /stream (silence, weak_signal, …) */
  lastStreamRejectionReason: string;
  lastBackendWaveformRms: number | null;
  lastBackendWaveformPeak: number | null;
  lastBackendBestScore: number | null;
  lastBackendSecondScore: number | null;
  /** Backend template winner margin (best vs runner-up), not aggregate `confidence` field */
  lastBackendTemplateMargin: number | null;
  lastBackendFinalChord: string;
  lastBackendAccepted: boolean | null;
  lastBackendClearDisplay: boolean | null;
  lastBackendChromaEntropy: number | null;
  lastBackendChromaStability: number | null;
  lastBackendStrongChromaBins: number | null;
  lastBackendSilenceFlag: boolean | null;
  /** Last response <code>debug.input_mode</code> (instrument / song / debug) */
  lastStreamInputMode: string;
  /** Last response <code>debug.preset_name</code> — human-readable preset label */
  lastStreamPresetName: string;
  /** Last API: consecutive chunks sub silence RMS before backend clears held chord */
  lastPresetSilenceStreakClear: number | null;
  /** Last API: consecutive gate rejects before backend forces clear_display */
  lastPresetInvalidStreakClear: number | null;
  /** Instant Krumhansl key on this chunk (`debug.instant_key_raw`) */
  lastStreamInstantKeyRaw: string;
  lastStreamInstantKeyConf: number | null;
  /** Sticky internal key state (`debug.smoothed_key_raw_internal`) */
  lastStreamSmoothedKeyRaw: string;
  lastStreamKeyDisplaySource: string;
  lastStreamHeldLastValid: boolean | null;
  /** Last <code>debug.chord_commit_kind</code> (immediate vs confirmed vs gate, …) */
  lastStreamChordCommitKind: string;
  /** Last <code>debug.displayed_chord</code> (what the JSON <code>chord</code> field carried) */
  lastStreamDisplayedChord: string;
  lastFetchError: string;
  lastUiError: string;
  ignoredResponseReason: string;
};

const LIVE_DEBUG_INITIAL: LiveDebugSnapshot = {
  micPermission: "pending",
  browserMicPermission: "—",
  audioContextState: "—",
  audioProcessCallbacks: 0,
  msSinceLastAudioProcess: 0,
  chunksCreated: 0,
  chunksPosted: 0,
  responsesApplied: 0,
  lastChunkSize: 0,
  lastRawSamplePeak: 0,
  lastRawSampleRms: 0,
  lastBoostedSamplePeak: 0,
  lastBoostedSampleRms: 0,
  liveInputBoost: 1,
  lastBoostClipFraction: 0,
  lastInputBufferChannels: 0,
  liveChunkSeconds: 1,
  msSinceLastChunkSent: null,
  msSinceLastStreamResponse: null,
  lastPresetWeakConfirmChunks: null,
  trackCount: 0,
  trackLabel: "—",
  trackEnabled: false,
  trackMuted: false,
  trackReadyState: "—",
  trackDeviceId: "—",
  trackSampleRate: "—",
  trackChannelCount: "—",
  trackEchoCancellation: "—",
  trackNoiseSuppression: "—",
  trackAutoGainControl: "—",
  lastUploadStatus: "—",
  lastHttpStatus: null,
  lastBackendChord: "—",
  lastBackendRaw: "—",
  lastStreamRejectionReason: "—",
  lastBackendWaveformRms: null,
  lastBackendWaveformPeak: null,
  lastBackendBestScore: null,
  lastBackendSecondScore: null,
  lastBackendTemplateMargin: null,
  lastBackendFinalChord: "—",
  lastBackendAccepted: null,
  lastBackendClearDisplay: null,
  lastBackendChromaEntropy: null,
  lastBackendChromaStability: null,
  lastBackendStrongChromaBins: null,
  lastBackendSilenceFlag: null,
  lastStreamInputMode: "—",
  lastStreamPresetName: "—",
  lastPresetSilenceStreakClear: null,
  lastPresetInvalidStreakClear: null,
  lastStreamInstantKeyRaw: "—",
  lastStreamInstantKeyConf: null,
  lastStreamSmoothedKeyRaw: "—",
  lastStreamKeyDisplaySource: "—",
  lastStreamHeldLastValid: null,
  lastStreamChordCommitKind: "—",
  lastStreamDisplayedChord: "—",
  lastFetchError: "—",
  lastUiError: "—",
  ignoredResponseReason: "—",
};

const CHORD_HISTORY_MAX = 12;
const CHORD_PLACEHOLDER_IDLE = "--";
const CHORD_PLACEHOLDER_LISTENING = "Listening...";
const POST_STOP_FADE_MS = 5000;
const POST_STOP_CLEAR_MS = 8000;

type StreamDebug = {
  raw_chord?: string;
  final_chord?: string;
  rejection_reason?: string;
  accepted?: boolean;
  clear_display?: boolean;
  scores_top3?: [string, number][];
  waveform_rms?: number;
  waveform_peak?: number;
  best_score?: number;
  second_score?: number;
  /** Template margin (same as backend `debug.confidence`); distinct from response `confidence` */
  confidence?: number;
  chroma_entropy?: number;
  chroma_stability?: number;
  strong_chroma_bins?: number;
  silence?: boolean;
  input_mode?: string;
  preset_name?: string;
  preset_weak_confirm_chunks?: number;
  preset_silence_streak_clear?: number;
  preset_invalid_streak_clear?: number;
  preset_strong_best?: number;
  preset_strong_margin?: number;
  preset_medium_fast_best?: number;
  preset_medium_fast_margin?: number;
  held_last_valid_chord?: boolean;
  key_display_source?: string;
  instant_key_raw?: string | null;
  instant_key_confidence?: number | null;
  smoothed_key_raw_internal?: string;
  chord_commit_kind?: string;
  displayed_chord?: string;
};

type StreamResponse = {
  chord: string;
  confidence: number;
  key: string;
  key_confidence: number;
  timestamp: number;
  debug?: StreamDebug;
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
  /** Backend: sparse-chroma / vocal heuristics flagged this window */
  vocal_interference?: boolean;
  confidence_reasons?: string[];
  exclude_from_core?: boolean;
};

type AnalyzeApiResponse = {
  duration: number;
  tempo: number;
  key: { label: string; confidence: number };
  chords: AnalyzeChordSeg[];
  beats: { time: number }[];
  sections: { index: number; start: number; end: number; label: string; repeat_group?: string | null }[];
  rhythm?: AnalyzeRhythm;
  debug?: Record<string, unknown>;
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
  excludeFromCore?: boolean;
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

/** More decimals when levels are very small so quiet / leaky signal is visible. */
function formatLivePeakRms(peak: number, rms: number): string {
  const pDec = peak > 0 && peak < 0.001 ? 8 : peak < 0.01 ? 6 : 4;
  const rDec = rms > 0 && rms < 0.0001 ? 10 : rms < 0.001 ? 8 : 5;
  return `${peak.toFixed(pDec)} / ${rms.toFixed(rDec)}`;
}

function formatOptionalFloat(n: number | null | undefined, digits = 4): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

function formatMsOptional(ms: number | null): string {
  if (ms == null || !Number.isFinite(ms)) return "—";
  return `${Math.round(ms)} ms`;
}

function applyTrackSnapshotToDebug(
  snap: MicTrackDebugSnapshot,
): Pick<
  LiveDebugSnapshot,
  | "trackCount"
  | "trackLabel"
  | "trackEnabled"
  | "trackMuted"
  | "trackReadyState"
  | "trackDeviceId"
  | "trackSampleRate"
  | "trackChannelCount"
  | "trackEchoCancellation"
  | "trackNoiseSuppression"
  | "trackAutoGainControl"
> {
  return {
    trackCount: snap.trackCount,
    trackLabel: snap.label || "—",
    trackEnabled: snap.enabled,
    trackMuted: snap.muted,
    trackReadyState: snap.readyState || "—",
    trackDeviceId: snap.settingsDeviceId ?? "—",
    trackSampleRate: snap.settingsSampleRate != null ? String(snap.settingsSampleRate) : "—",
    trackChannelCount: snap.settingsChannelCount != null ? String(snap.settingsChannelCount) : "—",
    trackEchoCancellation:
      snap.settingsEchoCancellation !== undefined ? String(snap.settingsEchoCancellation) : "—",
    trackNoiseSuppression:
      snap.settingsNoiseSuppression !== undefined ? String(snap.settingsNoiseSuppression) : "—",
    trackAutoGainControl:
      snap.settingsAutoGainControl !== undefined ? String(snap.settingsAutoGainControl) : "—",
  };
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
        excludeFromCore: slice.some((c) => c.exclude_from_core === true),
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

/** Backend margin scores are in [0, 1] — tier labels for technical / snapshot use. */
function confidenceLevel(value: number): "Low" | "Medium" | "High" {
  if (value >= 0.5) return "High";
  if (value >= 0.2) return "Medium";
  return "Low";
}

/** Friendlier readout for main UI (avoid harsh “low confidence” wording). */
function readStrengthLabel(value: number): string {
  if (value >= 0.5) return "Strong";
  if (value >= 0.2) return "Moderate";
  return "Light";
}

/** Illustrative stages while POST /analyze runs — not timed to real server progress. */
const ANALYZE_STAGE_MESSAGES = [
  {
    title: "Loading audio",
    detail: "Sending your track for analysis.",
  },
  {
    title: "Rhythm & tempo",
    detail: "Finding the pulse and tempo (steps may overlap on the server).",
  },
  {
    title: "Harmony",
    detail: "Mapping key, chords, and practice sections.",
  },
  {
    title: "Almost ready",
    detail: "Building the chart you will practice with.",
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

/** Map backend "N" to a non-technical placeholder (never show raw N in live transcription UI). */
function transcribeChordDisplay(label: string | undefined | null): string {
  const t = (label ?? "").trim();
  if (!t || t === "N" || t === "n") {
    return "—";
  }
  return t;
}

function transcribePickCurrentNotes(data: LiveTranscribeApiResponse): string[] {
  const chords = data.chords ?? [];
  const curRaw = data.current_chord?.trim();
  const cur = curRaw && curRaw !== "N" ? curRaw : "";
  const pickForLabel = (lab: string) => {
    for (let i = chords.length - 1; i >= 0; i--) {
      const c = chords[i];
      if (c.label === lab && c.label !== "N") {
        return c.notes ?? [];
      }
    }
    return [];
  };
  if (cur) {
    const n = pickForLabel(cur);
    if (n.length) {
      return n;
    }
  }
  for (let i = chords.length - 1; i >= 0; i--) {
    const c = chords[i];
    if (c.label && c.label !== "N") {
      return c.notes ?? [];
    }
  }
  return [];
}

/** Client timeline rounding (query param). */
function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

function formDataLiveTranscribeWav(blob: Blob): FormData {
  const form = new FormData();
  form.append("file", blob, "window.wav");
  return form;
}

/** Shared −6…+6 display transpose; does not affect detection or stored analysis. */
function TransposeDisplayControl(props: { id: string; value: number; onChange: (v: number) => void }) {
  const options: { v: number; label: string }[] = [];
  for (let s = -6; s <= 6; s++) {
    options.push({
      v: s,
      label: s === 0 ? "Original" : s > 0 ? `+${s} semitones` : `${s} semitones`,
    });
  }
  return (
    <div className="transpose-display-row" role="group" aria-label="Transpose displayed chords">
      <span className="transpose-display-heading">Transpose</span>
      <label className="transpose-display-sr-only" htmlFor={props.id}>
        Transpose displayed chords by semitones
      </label>
      <select
        id={props.id}
        className="transpose-display-select"
        value={props.value}
        onChange={(e) => props.onChange(Number.parseInt(e.target.value, 10))}
      >
        {options.map((o) => (
          <option key={o.v} value={o.v}>
            {o.label}
          </option>
        ))}
      </select>
      <span className="transpose-display-note">Display only — audio unchanged</span>
    </div>
  );
}

export default function Home() {
  const [appMode, setAppMode] = useState<"live" | "file">("live");
  /** Practice display: concert pitch vs written transposition; Analyze + Live transcription share this. */
  const [displayTransposeSemitones, setDisplayTransposeSemitones] = useState(0);

  const [recording, setRecording] = useState(false);
  const [chord, setChord] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [key, setKey] = useState("—");
  const [keyConfidence, setKeyConfidence] = useState<number | null>(null);
  const [chordHistory, setChordHistory] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [postStopFade, setPostStopFade] = useState(false);

  const [liveConsoleDebug] = useState(() => {
    if (typeof window === "undefined") {
      return LIVE_MIC_CONSOLE;
    }
    return LIVE_MIC_CONSOLE || new URLSearchParams(window.location.search).get("liveDebug") === "1";
  });
  const [liveDebug, setLiveDebug] = useState<LiveDebugSnapshot>(LIVE_DEBUG_INITIAL);
  const liveTelemetryThrottleRef = useRef(0);
  const liveProcessTickRef = useRef({ index: 0, at: 0 });
  const lastLiveChunkSentAtRef = useRef<number | null>(null);
  const lastLiveStreamResponseAtRef = useRef<number | null>(null);
  /** Live mode: optional mic picker (labels fill after permission). */
  const [liveAudioInputs, setLiveAudioInputs] = useState<{ deviceId: string; label: string }[]>([]);
  const [liveMicDeviceId, setLiveMicDeviceId] = useState("");
  /** Live /stream sensitivity preset (query <code>mode</code>). */
  const [liveInputMode, setLiveInputMode] = useState<LiveInputMode>("instrument");
  /** Software gain on mic samples sent to /stream only (not speaker playback). */
  const [liveInputBoost, setLiveInputBoost] = useState(1);
  const liveBoostUserTouchedRef = useRef(false);
  /** Mic capture context: created synchronously on Start click, resumed before any await; reused across sessions. */
  const liveMicCtxRef = useRef<AudioContext | null>(null);

  /** Instant /stream vs rolling-window live transcription. */
  const [liveExperienceMode, setLiveExperienceMode] = useState<LiveExperienceMode>("instant");
  const liveExperienceModeRef = useRef<LiveExperienceMode>("instant");
  useEffect(() => {
    liveExperienceModeRef.current = liveExperienceMode;
  }, [liveExperienceMode]);

  const [transcribeSessionId, setTranscribeSessionId] = useState("");
  const transcribeSessionIdRef = useRef("");
  useEffect(() => {
    transcribeSessionIdRef.current = transcribeSessionId;
  }, [transcribeSessionId]);

  const [transcribeKey, setTranscribeKey] = useState<LiveTranscribeKey | null>(null);
  const [transcribeTimeline, setTranscribeTimeline] = useState<TimelineSeg[]>([]);
  const transcribeTimelineRef = useRef<TimelineSeg[]>([]);
  useEffect(() => {
    transcribeTimelineRef.current = transcribeTimeline;
  }, [transcribeTimeline]);
  const [transcribeServerDebug, setTranscribeServerDebug] = useState(false);
  /** Best-effort progression from latest /live-transcribe JSON (stable + server fallback). */
  const [liveServerCoreLabels, setLiveServerCoreLabels] = useState<string[]>([]);
  const [liveProgressionMeta, setLiveProgressionMeta] = useState<LiveTranscribeApiResponse["progression_meta"] | null>(
    null,
  );
  const [liveLastWindowChords, setLiveLastWindowChords] = useState<
    { label: string; start: number; end: number; low_confidence?: boolean; confidence?: number }[]
  >([]);
  const [transcribeCurrentChord, setTranscribeCurrentChord] = useState("—");
  const [transcribeCurrentNotes, setTranscribeCurrentNotes] = useState<string[]>([]);
  const [transcribeSummary, setTranscribeSummary] = useState("");
  const [transcribePhase, setTranscribePhase] = useState<"idle" | "listening" | "analyzing" | "ready">("idle");
  const [transcribeDebug, setTranscribeDebug] = useState<TranscribeDebugSnapshot>(TRANSCRIBE_DEBUG_INITIAL);
  const [transcribeCopyFlash, setTranscribeCopyFlash] = useState(false);
  const [transcribeBufferDisplaySec, setTranscribeBufferDisplaySec] = useState(0);
  const [transcribeRequestInFlight, setTranscribeRequestInFlight] = useState(false);
  const transcribeRingRef = useRef<LiveTranscribeRing | null>(null);
  const transcribeTimerRef = useRef<number | null>(null);
  const transcribeSessionStartPerfRef = useRef(0);
  const transcribeEpochRef = useRef(0);
  const transcribeKickoffRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const transcribeRequestInFlightRef = useRef(false);
  const transcribeCopyDoneTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const liveDebugRef = useRef<LiveDebugSnapshot>(LIVE_DEBUG_INITIAL);
  useEffect(() => {
    liveDebugRef.current = liveDebug;
  }, [liveDebug]);

  useEffect(() => {
    return () => {
      if (transcribeCopyDoneTimerRef.current != null) {
        clearTimeout(transcribeCopyDoneTimerRef.current);
      }
    };
  }, []);

  const liveChunkSeconds = useMemo(() => LIVE_CHUNK_SECONDS[liveInputMode], [liveInputMode]);

  const [analyzeFile, setAnalyzeFile] = useState<File | null>(null);
  const [analyzeFileName, setAnalyzeFileName] = useState<string | null>(null);
  const [analyzeResult, setAnalyzeResult] = useState<AnalyzeApiResponse | null>(null);
  const [analyzeLoading, setAnalyzeLoading] = useState(false);
  const [analyzeQueryDebug, setAnalyzeQueryDebug] = useState(false);
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

  /** Browser `Permissions` API state for microphone (when supported). */
  useEffect(() => {
    if (appMode !== "live") return;
    if (typeof navigator === "undefined" || !navigator.permissions?.query) {
      setLiveDebug((p) => ({ ...p, browserMicPermission: "unsupported" }));
      return;
    }
    let perm: PermissionStatus | null = null;
    const sync = () => {
      if (perm) {
        setLiveDebug((p) => ({ ...p, browserMicPermission: perm!.state }));
      }
    };
    navigator.permissions
      .query({ name: "microphone" as PermissionName })
      .then((p) => {
        perm = p;
        sync();
        p.addEventListener("change", sync);
      })
      .catch(() => {
        setLiveDebug((p) => ({ ...p, browserMicPermission: "query_failed" }));
      });
    return () => {
      perm?.removeEventListener("change", sync);
    };
  }, [appMode]);

  /** Mirror UI error string into Live debug. */
  useEffect(() => {
    if (appMode !== "live") return;
    setLiveDebug((p) => ({ ...p, lastUiError: error ?? "—" }));
  }, [error, appMode]);

  /** Default boost per input type until the user picks a level manually. */
  useEffect(() => {
    if (liveBoostUserTouchedRef.current) return;
    setLiveInputBoost(LIVE_BOOST_DEFAULTS[liveInputMode]);
  }, [liveInputMode]);

  /** Live song transcription listens like phone/speaker capture — pin to Song sensitivity when entering that mode. */
  useEffect(() => {
    if (liveExperienceMode !== "transcribe") return;
    setLiveInputMode("song");
  }, [liveExperienceMode]);

  /** Keep debug snapshot in sync with WAV chunk length when input mode changes. */
  useEffect(() => {
    if (appMode !== "live") return;
    setLiveDebug((p) => ({ ...p, liveChunkSeconds: LIVE_CHUNK_SECONDS[liveInputMode] }));
  }, [liveInputMode, appMode]);

  /** Refresh ScriptProcessor age / count while recording (onaudioprocess itself is not a React event). */
  useEffect(() => {
    if (!recording || appMode !== "live") return;
    const id = window.setInterval(() => {
      const { index, at } = liveProcessTickRef.current;
      const msSince = at ? Math.max(0, Date.now() - at) : 0;
      const now = Date.now();
      const cAt = lastLiveChunkSentAtRef.current;
      const rAt = lastLiveStreamResponseAtRef.current;
      setLiveDebug((p) => ({
        ...p,
        audioProcessCallbacks: index,
        msSinceLastAudioProcess: msSince,
        msSinceLastChunkSent: cAt != null ? now - cAt : null,
        msSinceLastStreamResponse: rAt != null ? now - rAt : null,
      }));
    }, 250);
    return () => clearInterval(id);
  }, [recording, appMode]);

  /** Live transcription: sample rolling ring length for “gathering audio” UI (no technical copy in main status). */
  useEffect(() => {
    if (!recording || appMode !== "live" || liveExperienceMode !== "transcribe") {
      return;
    }
    const tick = () => {
      setTranscribeBufferDisplaySec(transcribeRingRef.current?.bufferedSeconds() ?? 0);
    };
    tick();
    const id = window.setInterval(tick, 200);
    return () => clearInterval(id);
  }, [recording, appMode, liveExperienceMode]);

  /** Refresh audio input device list (labels often empty until mic permission granted once). */
  useEffect(() => {
    if (appMode !== "live" || typeof navigator === "undefined" || !navigator.mediaDevices?.enumerateDevices) {
      return;
    }
    let cancelled = false;
    const run = async () => {
      try {
        const list = await navigator.mediaDevices.enumerateDevices();
        if (cancelled) return;
        const inputs = list
          .filter((d) => d.kind === "audioinput")
          .map((d) => ({
            deviceId: d.deviceId,
            label:
              d.label?.trim() ||
              (d.deviceId ? `Microphone (${d.deviceId.slice(0, 6)}…)` : "Audio input"),
          }));
        setLiveAudioInputs(inputs);
      } catch {
        /* ignore */
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [appMode, recording]);

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

  const coreProgressionDisplay = useMemo(() => {
    const n = displayTransposeSemitones;
    if (n === 0) return coreProgression;
    return coreProgression.map((e) => ({
      ...e,
      label: transposeChordLabel(e.label, n),
      notesLine: transposeChordToneLine(e.notesLine, n),
    }));
  }, [coreProgression, displayTransposeSemitones]);

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

  const nextRunDisplayForUi = useMemo(() => {
    if (!analyzeResult || activeChordRunIndex < 0) return { label: "—", notesLine: "—" };
    const nr = chordRuns[activeChordRunIndex + 1];
    if (!nr) return { label: "End of chart", notesLine: "" };
    const n = displayTransposeSemitones;
    if (n === 0) return { label: nr.label, notesLine: nr.notesLine };
    const segAt = analyzeResult.chords[nr.startSeg];
    const notesLine = segAt
      ? chordNotesLine(transposeChordSegment(segAt, n)!)
      : transposeChordToneLine(nr.notesLine, n);
    return { label: transposeChordLabel(nr.label, n), notesLine };
  }, [analyzeResult, chordRuns, activeChordRunIndex, displayTransposeSemitones]);

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
    const coreForSummary = displayTransposeSemitones === 0 ? coreProgression : coreProgressionDisplay;
    return buildLearnThisSongSummary({
      keyLabel: analyzeResult.key.label,
      tempoBpm: analyzeResult.tempo,
      coreEntries: coreForSummary,
      practicePartCount: practiceParts.length,
    });
  }, [analyzeResult, coreProgression, coreProgressionDisplay, displayTransposeSemitones, practiceParts.length]);

  const analyzeChordsPracticeDisplay = useMemo((): AnalyzeChordSeg[] => {
    if (!analyzeResult?.chords?.length) return [];
    if (displayTransposeSemitones === 0) return analyzeResult.chords;
    return analyzeResult.chords.map((c) => transposeChordSegment(c, displayTransposeSemitones)!);
  }, [analyzeResult?.chords, displayTransposeSemitones]);

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

  const displayedNextChordSegForUi = useMemo(
    () => transposeChordSegment(displayedNextChordSeg, displayTransposeSemitones),
    [displayedNextChordSeg, displayTransposeSemitones],
  );

  const currentAnalyzeChordForUi = useMemo(
    () => transposeChordSegment(currentAnalyzeChord, displayTransposeSemitones),
    [currentAnalyzeChord, displayTransposeSemitones],
  );

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
    return chordSequenceForPart(focusPracticePart, analyzeChordsPracticeDisplay, analyzePlaybackDuration);
  }, [analyzeResult, focusPracticePart, analyzeChordsPracticeDisplay, analyzePlaybackDuration]);

  const focusPracticePartSteps = useMemo(() => {
    if (!analyzeResult || !focusPracticePart) return [];
    return buildPracticeStepsForPart(focusPracticePart, analyzeChordsPracticeDisplay, analyzePlaybackDuration);
  }, [analyzeResult, focusPracticePart, analyzeChordsPracticeDisplay, analyzePlaybackDuration]);

  const focusPianoPartSteps = useMemo(() => {
    if (!analyzeResult || !focusPracticePart) return [];
    return buildPianoPracticeStepsForPart(focusPracticePart, analyzeChordsPracticeDisplay, analyzePlaybackDuration);
  }, [analyzeResult, focusPracticePart, analyzeChordsPracticeDisplay, analyzePlaybackDuration]);

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
    const dbg = data.debug;
    const accepted = dbg?.accepted === true;
    const clearDisplay = dbg?.clear_display === true;

    setConfidence(data.confidence);
    setKey(data.key);
    setKeyConfidence(data.key_confidence);

    if (clearDisplay) {
      setChord(null);
      return;
    }
    if (data.chord === "N") {
      setChord(null);
      return;
    }

    setChord(data.chord);

    if (accepted) {
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
      if (liveConsoleDebug) {
        console.info("[live] upload chunk", { bytes: blob.size, epoch });
      }

      const form = new FormData();
      form.append("file", blob, "chunk.wav");

      let res: Response;
      try {
        res = await fetch(`${API_BASE}/stream?mode=${encodeURIComponent(liveInputMode)}`, {
          method: "POST",
          body: form,
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setLiveDebug((p) => ({
          ...p,
          lastFetchError: msg,
          lastUploadStatus: "fetch failed (network / CORS / wrong API URL / offline)",
          lastHttpStatus: null,
        }));
        if (liveConsoleDebug) {
          console.warn("[live] /stream fetch failed", e);
        }
        throw e;
      }

      if (!res.ok) {
        const text = await res.text();
        setLiveDebug((p) => ({
          ...p,
          lastFetchError: `${res.status}: ${text.slice(0, 240)}`,
          lastUploadStatus: `HTTP ${res.status} ${res.statusText}`,
          lastHttpStatus: res.status,
        }));
        throw new Error(`${res.status} ${res.statusText}: ${text}`);
      }

      let data: StreamResponse;
      try {
        data = (await res.json()) as StreamResponse;
        lastLiveStreamResponseAtRef.current = Date.now();
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setLiveDebug((p) => ({
          ...p,
          lastFetchError: `response not JSON: ${msg}`,
          lastUploadStatus: `HTTP ${res.status} (invalid body)`,
          lastHttpStatus: res.status,
        }));
        throw e;
      }

      const stale = epoch !== liveEpochRef.current;
      const stopped = !recordingRef.current;
      let appliedInc = 0;
      let ignored: string;
      if (stale) {
        ignored = "ignored: stale_epoch (Stop/Start changed session before this response)";
      } else if (stopped) {
        ignored = "ignored: recording_stopped (response arrived after Stop)";
      } else {
        appliedInc = 1;
        if (data.debug?.clear_display) {
          ignored = "applied: clear_display (backend cleared held chord)";
        } else if (data.chord === "N") {
          ignored = "applied: chord N — UI shows Listening…; history unchanged";
        } else if (data.debug?.accepted === false) {
          ignored = `applied: hold/update display without new history (${data.debug?.rejection_reason ?? "not accepted"})`;
        } else {
          ignored = "—";
        }
      }

      const d = data.debug;
      setLiveDebug((p) => ({
        ...p,
        chunksPosted: p.chunksPosted + 1,
        responsesApplied: p.responsesApplied + appliedInc,
        lastHttpStatus: res.status,
        lastUploadStatus:
          stale || stopped ? `HTTP ${res.status} OK (response not applied to UI)` : `HTTP ${res.status} OK`,
        lastBackendChord: data.chord,
        lastBackendRaw: d?.raw_chord ?? "—",
        lastStreamRejectionReason: d?.rejection_reason ?? "—",
        lastBackendWaveformRms: d?.waveform_rms ?? null,
        lastBackendWaveformPeak: d?.waveform_peak ?? null,
        lastBackendBestScore: d?.best_score ?? null,
        lastBackendSecondScore: d?.second_score ?? null,
        lastBackendTemplateMargin: d?.confidence ?? null,
        lastBackendFinalChord: d?.final_chord ?? "—",
        lastBackendAccepted: d?.accepted ?? null,
        lastBackendClearDisplay: d?.clear_display ?? null,
        lastBackendChromaEntropy: d?.chroma_entropy ?? null,
        lastBackendChromaStability: d?.chroma_stability ?? null,
        lastBackendStrongChromaBins: d?.strong_chroma_bins ?? null,
        lastBackendSilenceFlag: d?.silence ?? null,
        lastStreamInputMode: d?.input_mode ?? "—",
        lastStreamPresetName: d?.preset_name ?? "—",
        lastPresetWeakConfirmChunks: d?.preset_weak_confirm_chunks ?? null,
        lastPresetSilenceStreakClear: d?.preset_silence_streak_clear ?? null,
        lastPresetInvalidStreakClear: d?.preset_invalid_streak_clear ?? null,
        lastStreamInstantKeyRaw:
          d?.instant_key_raw != null && d.instant_key_raw !== "" ? String(d.instant_key_raw) : "—",
        lastStreamInstantKeyConf:
          typeof d?.instant_key_confidence === "number" ? d.instant_key_confidence : null,
        lastStreamSmoothedKeyRaw:
          d?.smoothed_key_raw_internal != null && String(d.smoothed_key_raw_internal) !== ""
            ? String(d.smoothed_key_raw_internal)
            : "—",
        lastStreamKeyDisplaySource: d?.key_display_source != null ? String(d.key_display_source) : "—",
        lastStreamHeldLastValid: typeof d?.held_last_valid_chord === "boolean" ? d.held_last_valid_chord : null,
        lastStreamChordCommitKind:
          d?.chord_commit_kind != null && String(d.chord_commit_kind) !== "" ? String(d.chord_commit_kind) : "—",
        lastStreamDisplayedChord:
          d?.displayed_chord != null && String(d.displayed_chord) !== "" ? String(d.displayed_chord) : "—",
        lastFetchError: "—",
        ignoredResponseReason: ignored,
      }));

      if (stale) {
        if (liveConsoleDebug) {
          console.info("[live] ignore response (stale epoch)", {
            epoch,
            current: liveEpochRef.current,
            chord: data.chord,
          });
        }
        return;
      }
      if (stopped) {
        if (liveConsoleDebug) {
          console.info("[live] ignore response (recording stopped)", { chord: data.chord });
        }
        return;
      }

      if (liveConsoleDebug) {
        console.info("[live] /stream ok", {
          chord: data.chord,
          confidence: data.confidence,
          raw: data.debug?.raw_chord,
        });
      }
      applyResponse(data);
    },
    [applyResponse, liveConsoleDebug, liveInputMode],
  );

  const clearTranscribeSession = useCallback(() => {
    transcribeEpochRef.current += 1;
    transcribeRequestInFlightRef.current = false;
    setTranscribeRequestInFlight(false);
    if (transcribeCopyDoneTimerRef.current != null) {
      clearTimeout(transcribeCopyDoneTimerRef.current);
      transcribeCopyDoneTimerRef.current = null;
    }
    setTranscribeCopyFlash(false);
    if (transcribeKickoffRef.current != null) {
      clearTimeout(transcribeKickoffRef.current);
      transcribeKickoffRef.current = null;
    }
    transcribeRingRef.current?.clear();
    setTranscribeBufferDisplaySec(0);
    setTranscribeKey(null);
    setTranscribeTimeline([]);
    setTranscribeCurrentChord("—");
    setTranscribeCurrentNotes([]);
    setTranscribeSummary("");
    setTranscribeDebug(TRANSCRIBE_DEBUG_INITIAL);
    setLiveServerCoreLabels([]);
    setLiveProgressionMeta(null);
    setLiveLastWindowChords([]);
    const sid = `lt_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 9)}`;
    setTranscribeSessionId(sid);
    transcribeSessionIdRef.current = sid;
    transcribeSessionStartPerfRef.current = performance.now();
    setTranscribePhase(recordingRef.current ? "listening" : "idle");
  }, []);

  const runTranscribeCycle = useCallback(async () => {
    if (liveExperienceModeRef.current !== "transcribe" || !recordingRef.current) {
      return;
    }
    if (transcribeRequestInFlightRef.current) {
      return;
    }
    const epoch = transcribeEpochRef.current;
    const ring = transcribeRingRef.current;
    if (!ring) {
      return;
    }

    const bufSec = ring.bufferedSeconds();
    const dbgSnap = liveDebugRef.current;

    setTranscribeDebug((p) => ({
      ...p,
      ringBufferedSec: bufSec,
      lastRawPeak: dbgSnap.lastRawSamplePeak,
      lastRawRms: dbgSnap.lastRawSampleRms,
    }));

    if (bufSec < FIRST_TRANSCRIBE_AFTER_SEC) {
      setTranscribePhase("listening");
      return;
    }

    const sr = ring.getSampleRate();
    const mono = ring.sliceLastSeconds(TRANSCRIBE_WINDOW_SEC);
    const durActual = sr > 0 ? mono.length / sr : 0;
    if (mono.length === 0 || durActual < MIN_TRANSCRIBE_AUDIO_SEC) {
      setTranscribePhase("listening");
      return;
    }

    transcribeRequestInFlightRef.current = true;
    setTranscribeRequestInFlight(true);
    setTranscribePhase("analyzing");

    const tReq0 = performance.now();
    const nowSec = (performance.now() - transcribeSessionStartPerfRef.current) / 1000;
    const windowEnd = nowSec;
    const windowStart = Math.max(0, windowEnd - durActual);

    const blob = encodeFloat32MonoToWav(mono, sr);
    const sessionId = transcribeSessionIdRef.current;

    try {
      let res: Response;
      try {
        const qs = new URLSearchParams({
          window_start: String(round4(windowStart)),
          mode: "song",
        });
        if (sessionId) {
          qs.set("session_id", sessionId);
        }
        if (transcribeServerDebug) {
          qs.set("debug", "true");
          qs.set("client_timeline_seg_count", String(transcribeTimelineRef.current.length));
        }
        res = await fetch(`${API_BASE}/live-transcribe?${qs.toString()}`, {
          method: "POST",
          body: formDataLiveTranscribeWav(blob),
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        if (transcribeEpochRef.current !== epoch || !recordingRef.current) {
          return;
        }
        setTranscribeDebug((p) => ({
          ...p,
          lastRequestStatus: "network error",
          lastError: msg,
          lastWindowSec: durActual,
          lastRequestDurationMs: Math.round(performance.now() - tReq0),
          lastAnalysisStatus: "—",
        }));
        setTranscribePhase("listening");
        return;
      }

      if (transcribeEpochRef.current !== epoch || !recordingRef.current) {
        return;
      }

      const bodyText = await res.text();
      const requestMs = Math.round(performance.now() - tReq0);

      if (!res.ok) {
        setTranscribeDebug((p) => ({
          ...p,
          lastRequestStatus: `HTTP ${res.status}`,
          lastError: bodyText.slice(0, 480),
          lastWindowSec: durActual,
          lastRequestDurationMs: requestMs,
          lastAnalysisStatus: "—",
        }));
        setTranscribePhase("listening");
        return;
      }

      let data: LiveTranscribeApiResponse;
      try {
        data = JSON.parse(bodyText) as LiveTranscribeApiResponse;
      } catch {
        setTranscribeDebug((p) => ({
          ...p,
          lastRequestStatus: `HTTP ${res.status} (bad JSON)`,
          lastError: bodyText.slice(0, 200),
          lastWindowSec: durActual,
          lastRequestDurationMs: requestMs,
          lastAnalysisStatus: "—",
        }));
        setTranscribePhase("listening");
        return;
      }

      if (transcribeEpochRef.current !== epoch || !recordingRef.current) {
        return;
      }

      if (liveConsoleDebug) {
        console.info("[live-transcribe] ok", {
          window: [data.window_start, data.window_end],
          key: data.key,
          core: data.core_progression?.map((c) => c.label),
          status: data.status,
          requestMs,
        });
      }

      const dbg = data.debug;
      setTranscribeDebug((p) => ({
        ...p,
        lastRequestStatus: `HTTP ${res.status} OK`,
        lastError: "—",
        analysisCount: p.analysisCount + 1,
        lastWindowSec: durActual,
        lastKeyLabel: data.key.label,
        lastProgression: (data.core_progression ?? []).map((c) => c.label).join(" → ") || "—",
        lastRequestDurationMs: requestMs,
        lastAnalysisStatus: data.status ?? "—",
        lastLtCoreEmptyReason:
          dbg && typeof dbg.progression_empty_reason === "string"
            ? dbg.progression_empty_reason
            : typeof data.progression_meta?.empty_reason === "string"
              ? data.progression_meta.empty_reason
              : "—",
        lastLtRunsForCoreStrategy:
          dbg && typeof dbg.runs_for_core_strategy === "string" ? dbg.runs_for_core_strategy : "—",
        lastLtServerSegmentSummary:
          dbg &&
          typeof dbg.segment_count === "number" &&
          typeof dbg.stable_segment_count === "number" &&
          typeof dbg.low_confidence_segment_count === "number"
            ? `${dbg.segment_count} total / ${dbg.stable_segment_count} stable / ${dbg.low_confidence_segment_count} low-conf`
            : "—",
        lastLtClientTimelineSegEcho:
          dbg && typeof dbg.client_timeline_merged_seg_count === "number"
            ? String(dbg.client_timeline_merged_seg_count)
            : "—",
        lastProgressionSource: data.progression_meta?.source ?? "—",
        lastProgressionQuality: data.progression_meta?.quality ?? "—",
      }));

      setLiveServerCoreLabels((data.core_progression ?? []).map((c) => c.label));
      setLiveProgressionMeta(data.progression_meta ?? null);
      setLiveLastWindowChords(
        (data.chords ?? []).map((c) => ({
          label: c.label,
          start: c.start,
          end: c.end,
          low_confidence: c.low_confidence,
          confidence: c.confidence,
        })),
      );

      setTranscribeKey((prev) =>
        mergeLiveTranscribeKey(prev, {
          label: data.key.label,
          confidence: data.key.confidence,
        }),
      );
      setTranscribeTimeline((prev) =>
        mergeTranscribeTimeline(
          prev,
          data.window_start,
          data.window_end,
          (data.chords ?? []).filter((c) => c.label !== "N"),
          TRANSCRIBE_TIMELINE_KEEP_SEC,
        ),
      );
      setTranscribeCurrentChord(transcribeChordDisplay(data.current_chord));
      setTranscribeCurrentNotes(transcribePickCurrentNotes(data));
      setTranscribeSummary(data.summary ?? "");
      setTranscribePhase(data.status === "ready" ? "ready" : "listening");
    } finally {
      transcribeRequestInFlightRef.current = false;
      setTranscribeRequestInFlight(false);
    }
  }, [liveConsoleDebug, transcribeServerDebug]);

  const startRecording = useCallback(async () => {
    liveEpochRef.current += 1;
    setError(null);
    setStatus(null);
    clearPostStopTimers();
    setChord(null);
    setChordHistory([]);
    liveProcessTickRef.current = { index: 0, at: 0 };
    lastLiveChunkSentAtRef.current = null;
    lastLiveStreamResponseAtRef.current = null;
    const isTranscribe = liveExperienceMode === "transcribe";

    if (isTranscribe) {
      if (transcribeKickoffRef.current != null) {
        clearTimeout(transcribeKickoffRef.current);
        transcribeKickoffRef.current = null;
      }
      transcribeEpochRef.current += 1;
      if (transcribeTimerRef.current != null) {
        clearInterval(transcribeTimerRef.current);
        transcribeTimerRef.current = null;
      }
      transcribeRequestInFlightRef.current = false;
      setTranscribeRequestInFlight(false);
      setTranscribeBufferDisplaySec(0);
      transcribeRingRef.current = new LiveTranscribeRing(TRANSCRIBE_RING_MAX_SEC);
      transcribeSessionStartPerfRef.current = performance.now();
      const sid = `lt_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 9)}`;
      setTranscribeSessionId(sid);
      transcribeSessionIdRef.current = sid;
      setTranscribeKey(null);
      setTranscribeTimeline([]);
      setTranscribeCurrentChord("—");
      setTranscribeCurrentNotes([]);
      setTranscribeSummary("");
      setTranscribePhase("listening");
      setTranscribeDebug(TRANSCRIBE_DEBUG_INITIAL);
      setLiveServerCoreLabels([]);
      setLiveProgressionMeta(null);
      setLiveLastWindowChords([]);
    }

    setLiveDebug((prev) => ({
      ...LIVE_DEBUG_INITIAL,
      liveInputBoost,
      liveChunkSeconds,
      browserMicPermission: prev.browserMicPermission,
      micPermission: "pending",
    }));

    if (sessionRef.current) {
      if (liveConsoleDebug) {
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
      setLiveDebug((p) => ({ ...p, audioContextState: sharedCtx.state }));

      recordingRef.current = true;
      const session = await startMicWavChunks({
        audioContext: sharedCtx,
        deviceId: liveMicDeviceId || undefined,
        chunkSeconds: isTranscribe ? 1 : liveChunkSeconds,
        tailMinSeconds: 0.2,
        inputBoost: liveInputBoost,
        streamChunks: !isTranscribe,
        onDebug: (message, detail) => {
          if (liveConsoleDebug) {
            console.info("[live:mic]", message, detail ?? "");
          }
          if (message === "get_user_media_ok") {
            setLiveDebug((p) => ({ ...p, micPermission: "granted" }));
          }
        },
        onTrackSnapshot: (snap) => {
          setLiveDebug((p) => ({ ...p, ...applyTrackSnapshotToDebug(snap) }));
        },
        onTelemetry: (info) => {
          const now = Date.now();
          if (now - liveTelemetryThrottleRef.current < 220) {
            return;
          }
          liveTelemetryThrottleRef.current = now;
          setLiveDebug((p) => ({
            ...p,
            audioContextState: info.audioContextState,
            lastRawSamplePeak: info.inputPeak,
            lastRawSampleRms: info.inputRms,
            lastBoostedSamplePeak: info.boostedPeak,
            lastBoostedSampleRms: info.boostedRms,
            liveInputBoost: info.inputBoost,
            lastBoostClipFraction: info.clippedFractionInBuffer,
            lastInputBufferChannels: info.inputChannels,
          }));
        },
        onProcessTick: ({ callbackIndex }) => {
          liveProcessTickRef.current = { index: callbackIndex, at: Date.now() };
        },
        onMonoFrames:
          isTranscribe
            ? (mono, sr) => {
                transcribeRingRef.current?.push(mono, sr);
              }
            : undefined,
        onChunk: isTranscribe
          ? undefined
          : ({ blob }) => {
              lastLiveChunkSentAtRef.current = Date.now();
              setLiveDebug((p) => ({
                ...p,
                chunksCreated: p.chunksCreated + 1,
                lastChunkSize: blob.size,
              }));
              sendWav(blob).catch((e) => {
                const message = e instanceof Error ? e.message : String(e);
                setError(message);
                if (liveConsoleDebug) {
                  console.warn("[live] sendWav error", message);
                }
              });
            },
        onError: (err) => {
          setError(err.message);
          if (liveConsoleDebug) {
            console.warn("[live:mic] onError", err);
          }
        },
      });
      sessionRef.current = session;
      setRecording(true);
      setLiveDebug((p) => ({ ...p, audioContextState: sharedCtx.state }));
      if (isTranscribe) {
        if (transcribeKickoffRef.current != null) {
          clearTimeout(transcribeKickoffRef.current);
        }
        transcribeKickoffRef.current = setTimeout(() => {
          transcribeKickoffRef.current = null;
          void runTranscribeCycle();
        }, FIRST_TRANSCRIBE_DELAY_MS);
        transcribeTimerRef.current = window.setInterval(() => {
          void runTranscribeCycle();
        }, TRANSCRIBE_INTERVAL_MS);
      }
    } catch (e) {
      recordingRef.current = false;
      if (transcribeTimerRef.current != null) {
        clearInterval(transcribeTimerRef.current);
        transcribeTimerRef.current = null;
      }
      if (transcribeKickoffRef.current != null) {
        clearTimeout(transcribeKickoffRef.current);
        transcribeKickoffRef.current = null;
      }
      transcribeRequestInFlightRef.current = false;
      setTranscribeRequestInFlight(false);
      const message = e instanceof Error ? e.message : String(e);
      setError(message);
      setLiveDebug((p) => ({ ...p, micPermission: "denied" }));
      if (liveConsoleDebug) {
        console.warn("[live] startRecording failed", message);
      }
    }
  }, [
    sendWav,
    clearPostStopTimers,
    liveConsoleDebug,
    liveMicDeviceId,
    liveInputBoost,
    liveChunkSeconds,
    liveExperienceMode,
    runTranscribeCycle,
  ]);

  const stopRecording = useCallback(async () => {
    recordingRef.current = false;
    liveEpochRef.current += 1;
    transcribeEpochRef.current += 1;
    if (transcribeTimerRef.current != null) {
      clearInterval(transcribeTimerRef.current);
      transcribeTimerRef.current = null;
    }
    if (transcribeKickoffRef.current != null) {
      clearTimeout(transcribeKickoffRef.current);
      transcribeKickoffRef.current = null;
    }
    transcribeRequestInFlightRef.current = false;
    setTranscribeRequestInFlight(false);
    setTranscribeBufferDisplaySec(0);
    const session = sessionRef.current;
    sessionRef.current = null;
    if (session) {
      await session.stop();
    }
    setRecording(false);
    setStatus("Stopped.");
    setTranscribePhase((p) => (p === "idle" ? "idle" : "ready"));
    const ctx = liveMicCtxRef.current;
    setLiveDebug((p) => ({ ...p, audioContextState: ctx && ctx.state !== "closed" ? ctx.state : "—" }));
  }, []);

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
      const res = await fetch(`${API_BASE}/analyze?debug=${analyzeQueryDebug ? "true" : "false"}`, {
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
  }, [analyzeFile, analyzeQueryDebug]);

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

  const liveInstantChordTonesLine = useMemo(() => {
    if (liveExperienceMode !== "instant") return null;
    if (chord == null) return null;
    const notes = liveTriadNoteNamesFromLabel(chord);
    if (!notes?.length) return null;
    return notes.join(" · ");
  }, [liveExperienceMode, chord]);

  const liveAudioProcessingSummary = useMemo(() => {
    if (!recording) {
      return "Idle (not recording)";
    }
    if (liveDebug.audioProcessCallbacks < 1) {
      return "No callbacks yet — if this stays at 0, audio is not reaching ScriptProcessor (suspended AudioContext, bad graph, or browser restriction)";
    }
    if (liveDebug.msSinceLastAudioProcess > 3000) {
      return `Stale: last onaudioprocess ${liveDebug.msSinceLastAudioProcess}ms ago (background tab / throttling?)`;
    }
    return `Firing — callback #${liveDebug.audioProcessCallbacks}, last tick ${liveDebug.msSinceLastAudioProcess}ms ago`;
  }, [recording, liveDebug.audioProcessCallbacks, liveDebug.msSinceLastAudioProcess]);

  const transcribeRecentTimeline = useMemo(() => {
    if (!transcribeTimeline.length) {
      return [];
    }
    const lastT = transcribeTimeline[transcribeTimeline.length - 1].t1;
    const cutoff = lastT - TRANSCRIBE_TIMELINE_KEEP_SEC;
    return transcribeTimeline.filter((s) => s.t1 > cutoff);
  }, [transcribeTimeline]);

  const liveDerivedProgression = useMemo(
    () =>
      deriveLiveStableProgression(transcribeTimeline, {
        analysisCount: transcribeDebug.analysisCount,
        bufferSec: transcribeBufferDisplaySec,
      }),
    [transcribeTimeline, transcribeDebug.analysisCount, transcribeBufferDisplaySec],
  );

  const liveDisplayProgressionChips = useMemo(() => {
    const fromTimeline = liveDerivedProgression.chipLabels;
    if (fromTimeline.length) {
      return fromTimeline;
    }
    if (liveServerCoreLabels.length) {
      return liveServerCoreLabels.slice(0, 8);
    }
    return deriveFallbackProgressionFromWindowChords(liveLastWindowChords, 8);
  }, [liveDerivedProgression.chipLabels, liveServerCoreLabels, liveLastWindowChords]);

  const liveDisplayProgressionUi = useMemo(() => {
    const chips = liveDisplayProgressionChips;
    if (!chips.length) {
      const r = liveProgressionMeta?.empty_reason;
      let primary = "Still listening for the progression…";
      if (r === "all_low_confidence") {
        primary = "Still listening";
      }
      return {
        chips,
        qualityPrimary: primary,
        qualitySecondary:
          r === "all_low_confidence"
            ? "Harmony reads uncertain — try a clearer moment or louder chords."
            : r && r !== "waiting_for_more_audio"
              ? `Reason: ${r.replace(/_/g, " ")}`
              : undefined,
      };
    }
    if (liveDerivedProgression.chipLabels.length) {
      return {
        chips,
        qualityPrimary: liveDerivedProgression.qualityLabel,
        qualitySecondary: liveDerivedProgression.usedLenientFallback
          ? "Coarse read from merged timeline"
          : undefined,
      };
    }
    if (liveServerCoreLabels.length) {
      const q = liveProgressionMeta?.quality;
      if (q === "likely") {
        return {
          chips,
          qualityPrimary: "Likely progression",
          qualitySecondary: "From recent full-window analysis",
        };
      }
      if (q === "stabilizing") {
        return {
          chips,
          qualityPrimary: "Pattern stabilizing",
          qualitySecondary: "From recent full-window analysis",
        };
      }
      return {
        chips,
        qualityPrimary: "Rough progression so far",
        qualitySecondary: "From recent full-window analysis",
      };
    }
    return {
      chips,
      qualityPrimary: "Rough progression so far",
      qualitySecondary: "From the latest short window only",
    };
  }, [
    liveDisplayProgressionChips,
    liveDerivedProgression.chipLabels.length,
    liveDerivedProgression.qualityLabel,
    liveDerivedProgression.usedLenientFallback,
    liveServerCoreLabels.length,
    liveProgressionMeta?.empty_reason,
    liveProgressionMeta?.quality,
  ]);

  const liveChordLabelForDisplay = useCallback(
    (raw: string | null | undefined) => {
      const d = transcribeChordDisplay(raw);
      if (d === "—" || displayTransposeSemitones === 0) return d;
      return transposeChordLabel(d, displayTransposeSemitones);
    },
    [displayTransposeSemitones],
  );

  const transcribeCurrentChordForUi = useMemo(() => {
    if (transcribeCurrentChord === "—" || displayTransposeSemitones === 0) return transcribeCurrentChord;
    return transposeChordLabel(transcribeCurrentChord, displayTransposeSemitones);
  }, [transcribeCurrentChord, displayTransposeSemitones]);

  const transcribeCurrentNotesForUi = useMemo(
    () =>
      displayTransposeSemitones === 0 || !transcribeCurrentNotes.length
        ? transcribeCurrentNotes
        : transposeNotes(transcribeCurrentNotes, displayTransposeSemitones),
    [transcribeCurrentNotes, displayTransposeSemitones],
  );

  const transcribeLiveStatus = useMemo((): { headline: string; sub?: string } => {
    if (liveExperienceMode !== "transcribe") {
      return { headline: "" };
    }
    const hasCore = liveDisplayProgressionChips.length > 0;
    const analysisCount = transcribeDebug.analysisCount;

    if (!recording) {
      if (transcribePhase === "idle") {
        return {
          headline: "Ready when you are.",
          sub: "Press Start listening and give us a few seconds of clear harmony from the room.",
        };
      }
      return { headline: "Stopped.", sub: "Your last readout is below." };
    }

    if (transcribeRequestInFlight) {
      return {
        headline: "Updating…",
        sub: "Refreshing the last few seconds of audio.",
      };
    }

    if (transcribeBufferDisplaySec < FIRST_TRANSCRIBE_AFTER_SEC) {
      return {
        headline: "Buffering…",
        sub: "First chords usually appear after a few seconds of steady listening.",
      };
    }

    if (analysisCount === 0) {
      return {
        headline: "Listening…",
        sub: "Hang tight — first chords are on the way.",
      };
    }

    if (hasCore) {
      return {
        headline: "Main progression shaping up",
        sub: "We will keep refreshing while the music plays. Live transcription is rough — upload the file for best accuracy.",
      };
    }

    return {
      headline: "Finding the pattern…",
      sub: "Keep playing; we need a bit more harmony to suggest a loop.",
    };
  }, [
    liveExperienceMode,
    recording,
    transcribePhase,
    transcribeRequestInFlight,
    transcribeBufferDisplaySec,
    transcribeDebug.analysisCount,
    liveDisplayProgressionChips.length,
  ]);

  const downloadTranscribeSnapshot = useCallback(() => {
    const inputModeLabel =
      LIVE_INPUT_MODE_OPTIONS.find((o) => o.value === liveInputMode)?.label ?? liveInputMode;
    const keyStabilityWord =
      transcribeKey?.label &&
      transcribeKey.label !== "—" &&
      typeof transcribeKey.confidence === "number" &&
      transcribeKey.confidence > 0
        ? confidenceLevel(transcribeKey.confidence)
        : null;
    const snap = buildLiveTranscribeSnapshot({
      sessionId: transcribeSessionId,
      likelyKey: transcribeKey,
      keyStabilityWord,
      mainProgressionLabels: liveDisplayProgressionChips,
      progressionIsLikelyLoop:
        liveDerivedProgression.chipLabels.length > 0 ? liveDerivedProgression.isLikelyLoop : false,
      progressionQualityLabel: liveDisplayProgressionUi.qualityPrimary,
      formatChordLabel: (l) => liveChordLabelForDisplay(l),
      recentSegments: transcribeRecentTimeline.map((s) => ({ t0: s.t0, t1: s.t1, label: s.label })),
      summary: transcribeSummary?.trim() || null,
      currentChord: transcribeCurrentChordForUi,
      currentChordNotes: transcribeCurrentNotesForUi,
      inputMode: liveInputMode,
      inputModeLabel,
      inputBoost: liveInputBoost,
      displayTransposeSemitones,
    });
    downloadLiveTranscribeSnapshotJson(snap);
  }, [
    liveInputMode,
    liveInputBoost,
    transcribeSessionId,
    transcribeKey,
    liveDisplayProgressionChips,
    liveDerivedProgression.chipLabels.length,
    liveDerivedProgression.isLikelyLoop,
    liveDisplayProgressionUi.qualityPrimary,
    transcribeRecentTimeline,
    transcribeSummary,
    transcribeCurrentChordForUi,
    transcribeCurrentNotesForUi,
    liveChordLabelForDisplay,
    displayTransposeSemitones,
  ]);

  const copyTranscribeProgression = useCallback(async () => {
    const labels = liveDisplayProgressionChips;
    if (!labels.length) {
      return;
    }
    const text = labels.map((l) => liveChordLabelForDisplay(l)).join(" → ");
    const ok = await copyTextToClipboard(text);
    if (!ok) {
      setError("Could not copy progression to the clipboard.");
      return;
    }
    setError(null);
    if (transcribeCopyDoneTimerRef.current != null) {
      clearTimeout(transcribeCopyDoneTimerRef.current);
    }
    setTranscribeCopyFlash(true);
    transcribeCopyDoneTimerRef.current = setTimeout(() => {
      setTranscribeCopyFlash(false);
      transcribeCopyDoneTimerRef.current = null;
    }, 2000);
  }, [liveDisplayProgressionChips, liveChordLabelForDisplay]);

  const transcribeInputQuiet = useMemo(() => {
    if (!recording || liveExperienceMode !== "transcribe") {
      return false;
    }
    return liveDebug.lastRawSampleRms < 0.008 && liveDebug.lastBoostedSampleRms < 0.025;
  }, [recording, liveExperienceMode, liveDebug.lastRawSampleRms, liveDebug.lastBoostedSampleRms]);

  const transcribeQuietSuggestUpload = useMemo(() => {
    if (!transcribeInputQuiet) {
      return false;
    }
    const maxBoost = LIVE_INPUT_BOOST_OPTIONS[LIVE_INPUT_BOOST_OPTIONS.length - 1] ?? 8;
    return liveInputBoost >= maxBoost;
  }, [transcribeInputQuiet, liveInputBoost]);

  const liveStartLabel = liveExperienceMode === "transcribe" ? "Start listening" : "Start recording";

  const liveStopLabel = "Stop";

  return (
    <main className={`demo${appMode === "file" ? " demo--file" : ""}`}>
      <header className="hero">
        <h1>Chord lab</h1>
        <p className="hero-sub">Learn from a recording, or listen in with the mic</p>
        <div className="mode-toggle mode-toggle--primary" role="tablist" aria-label="App mode">
          <button
            type="button"
            role="tab"
            aria-selected={appMode === "live"}
            className={appMode === "live" ? "active" : ""}
            onClick={() => setAppMode("live")}
          >
            Listen live
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={appMode === "file"}
            className={appMode === "file" ? "active" : ""}
            onClick={() => setAppMode("file")}
          >
            Learn from audio
          </button>
        </div>
        <p className="mode-intro" role="note">
          {appMode === "file"
            ? "Best for a full song: upload a take to get the clearest chart, main progression, practice parts, and loops."
            : "Quick checks and rough sketches from the mic. For the most accurate chart of a recording, use Learn from audio."}
        </p>
      </header>

      {appMode === "live" ? (
        <>
          <div className="mode-toggle live-experience-toggle" role="tablist" aria-label="Live listening mode">
            <button
              type="button"
              role="tab"
              aria-selected={liveExperienceMode === "instant"}
              className={liveExperienceMode === "instant" ? "active" : ""}
              disabled={recording}
              onClick={() => setLiveExperienceMode("instant")}
            >
              Instant chord check
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={liveExperienceMode === "transcribe"}
              className={liveExperienceMode === "transcribe" ? "active" : ""}
              disabled={recording}
              onClick={() => setLiveExperienceMode("transcribe")}
            >
              Live song transcription
            </button>
          </div>

          <p className="live-experience-copy" role="note">
            {liveExperienceMode === "instant"
              ? "Fastest read — aim the mic at your own instrument to check what you are playing."
              : "Rough live progression from the room — great for rehearsal, speakers, or a band. Not as exact as uploading the track."}
          </p>

          <div className="controls">
            <label className="live-mic-device-label">
              <span className="visually-hidden">Microphone device</span>
              <select
                className="live-mic-device-select"
                value={liveMicDeviceId}
                onChange={(e) => setLiveMicDeviceId(e.target.value)}
                disabled={recording}
                aria-label="Microphone device"
              >
                <option value="">Default microphone</option>
                {liveAudioInputs.map((d) => (
                  <option key={d.deviceId} value={d.deviceId}>
                    {d.label}
                  </option>
                ))}
              </select>
            </label>
            <button type="button" onClick={() => void startRecording()} disabled={recording}>
              {liveStartLabel}
            </button>
            <button type="button" onClick={() => void stopRecording()} disabled={!recording}>
              {liveStopLabel}
            </button>
          </div>

          <div className="live-input-mode">
            <label className="live-input-mode-label" htmlFor="live-input-mode-select">
              Sound source
            </label>
            <select
              id="live-input-mode-select"
              className="live-input-mode-select"
              value={liveInputMode}
              onChange={(e) => setLiveInputMode(e.target.value as LiveInputMode)}
              disabled={recording}
              aria-label="Live microphone input sensitivity"
            >
              {LIVE_INPUT_MODE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <p className="live-input-mode-hint">
              {LIVE_INPUT_MODE_OPTIONS.find((o) => o.value === liveInputMode)?.hint}
              {liveExperienceMode === "transcribe" ? (
                <span className="live-debug-muted">
                  {" "}
                  Transcription works best on the <strong>Speaker / room</strong> input — you can still switch if needed.
                </span>
              ) : null}
            </p>
          </div>

          <div className="live-input-mode">
            <label className="live-input-mode-label" htmlFor="live-input-boost-select">
              Input boost
            </label>
            <select
              id="live-input-boost-select"
              className="live-input-mode-select"
              value={liveInputBoost}
              onChange={(e) => {
                liveBoostUserTouchedRef.current = true;
                setLiveInputBoost(Number(e.target.value));
              }}
              disabled={recording}
              aria-label="Microphone input boost for analysis"
            >
              {LIVE_INPUT_BOOST_OPTIONS.map((n) => (
                <option key={n} value={n}>
                  {n}×
                </option>
              ))}
            </select>
            <p className="live-input-mode-hint">
              Use higher boost for quiet phone/speaker audio. Only the audio sent for chord detection is boosted — not
              your speakers.
            </p>
          </div>

          {liveExperienceMode === "instant" && liveInputMode === "song" ? (
            <div className="live-song-playback-notice" role="note">
              <p className="live-song-playback-notice-text">
                Song playback mode is experimental. For best accuracy, switch to <strong>Learn from audio</strong> and upload
                the track.
              </p>
              <button type="button" className="live-analyze-instead-btn" onClick={() => setAppMode("file")}>
                Analyze a file instead
              </button>
            </div>
          ) : null}

          {liveExperienceMode === "transcribe" ? (
            <div className="live-transcribe-honesty" role="note">
              <p className="live-transcribe-honesty-text">
                Live transcription works best with clear harmony in the room — handy for rehearsal or jotting a rough chart.{" "}
                <strong>For a precise chart, use Learn from audio.</strong>
              </p>
              <button type="button" className="live-analyze-instead-btn" onClick={() => setAppMode("file")}>
                Open Learn from audio
              </button>
            </div>
          ) : null}

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

          {transcribeInputQuiet ? (
            <div className="live-transcribe-quiet-warning" role="status">
              <p className="live-transcribe-quiet-warning-text">
                Input is quiet — move the source closer, raise volume, or increase input boost.
              </p>
              {transcribeQuietSuggestUpload ? (
                <p className="live-transcribe-quiet-warning-more">
                  Input is still low at max boost — try uploading the track under <strong>Learn from audio</strong> for the
                  clearest readout.
                </p>
              ) : null}
            </div>
          ) : null}

          {liveExperienceMode === "instant" ? (
            <>
              <section className="chord-stage" aria-live="polite" aria-atomic="true">
                <p className="chord-stage-label">Current chord</p>
                <p className={chordValueClass}>{chordDisplay.text}</p>
                {confidence !== null && chord !== null ? (
                  <p className="chord-stage-confidence chord-stage-confidence--soft">
                    Read strength: {readStrengthLabel(confidence)}
                  </p>
                ) : null}
                {liveInstantChordTonesLine ? (
                  <p className="chord-stage-confidence chord-stage-confidence--soft">
                    Chord tones: {liveInstantChordTonesLine}
                  </p>
                ) : null}
              </section>

              <section className="details details--live" aria-label="Key">
                <div className="detail-grid">
                  <div className="detail-block">
                    <span className="detail-label">Likely key</span>
                    <span className="detail-value">{key}</span>
                  </div>
                  {keyConfidence !== null && key !== "—" ? (
                    <div className="detail-block">
                      <span className="detail-label">Key read</span>
                      <span className="detail-value">{readStrengthLabel(keyConfidence)}</span>
                    </div>
                  ) : null}
                </div>
              </section>

              <section className="history-section" aria-label="Chord history">
                <h2 className="section-title">Recent chords</h2>
                {chordHistory.length === 0 ? (
                  <p className="history-empty">Recent chords show up here as they change.</p>
                ) : (
                  <ol className="history-list" aria-label="Recent chords, newest first">
                    {chordHistory.map((c, i) => (
                      <li key={`${c}-${i}`}>{c}</li>
                    ))}
                  </ol>
                )}
              </section>
            </>
          ) : (
            <>
              <section className="live-transcribe-panel" aria-live="polite">
                <h2 className="live-transcribe-title">Live sketch</h2>
                <p className="live-transcribe-dek">
                  Rough progression from the room — try these chords first. For a precise chart,{" "}
                  <button type="button" className="text-link-btn" onClick={() => setAppMode("file")}>
                    learn from an audio file
                  </button>
                  .
                </p>
                <p className="live-transcribe-status" role="status">
                  {transcribeLiveStatus.headline}
                </p>
                {transcribeLiveStatus.sub ? (
                  <p className="live-transcribe-status-sub">{transcribeLiveStatus.sub}</p>
                ) : null}

                <div className="detail-grid detail-grid--transcribe">
                  <div className="detail-block">
                    <span className="detail-label">Likely song key</span>
                    <span className="detail-value">{transcribeKey?.label ?? "—"}</span>
                  </div>
                  {transcribeKey?.label != null &&
                  transcribeKey.label !== "—" &&
                  typeof transcribeKey.confidence === "number" &&
                  transcribeKey.confidence > 0 ? (
                    <div className="detail-block">
                      <span className="detail-label">Key stability</span>
                      <span className="detail-value">{readStrengthLabel(transcribeKey.confidence)}</span>
                    </div>
                  ) : null}
                </div>

                <TransposeDisplayControl
                  id="live-transpose"
                  value={displayTransposeSemitones}
                  onChange={setDisplayTransposeSemitones}
                />

                <section className="chord-stage chord-stage--transcribe" aria-label="Current harmony">
                  <p className="chord-stage-label">Latest chord</p>
                  <p
                    className={
                      transcribeCurrentChord !== "—" ? "chord-stage-value" : "chord-stage-value chord-stage-placeholder"
                    }
                  >
                    {transcribeCurrentChordForUi}
                  </p>
                  {transcribeCurrentNotesForUi.length ? (
                    <p className="chord-stage-confidence chord-stage-confidence--soft">
                    Suggested notes: {transcribeCurrentNotesForUi.join(" · ")}
                  </p>
                  ) : (
                    <p className="chord-stage-confidence chord-stage-confidence--soft">
                      Slight lag behind what you hear — a harmonic hint, not a finished chart.
                    </p>
                  )}
                </section>

                <div className="live-transcribe-practice">
                  <div className="live-progression-header-row">
                    <h3 className="live-transcribe-subhead live-progression-heading">Main progression</h3>
                    {liveDerivedProgression.chipLabels.length > 0 && liveDerivedProgression.isLikelyLoop ? (
                      <span className="live-progression-badge" title="Repeated pattern detected in recent harmony">
                        Likely loop
                      </span>
                    ) : null}
                  </div>
                  <div className="live-transcribe-actions" role="group" aria-label="Transcription snapshot actions">
                    <button type="button" className="live-transcribe-action-btn" onClick={() => downloadTranscribeSnapshot()}>
                      Download snapshot
                    </button>
                    <button
                      type="button"
                      className="live-transcribe-action-btn"
                      onClick={() => void copyTranscribeProgression()}
                      disabled={liveDisplayProgressionChips.length === 0}
                      title={
                        liveDisplayProgressionChips.length === 0
                          ? "Listen a bit longer to build a progression first"
                          : undefined
                      }
                    >
                      Copy progression
                    </button>
                    {transcribeCopyFlash ? (
                      <span className="live-transcribe-copied" role="status" aria-live="polite">
                        Copied
                      </span>
                    ) : null}
                    <button type="button" className="live-transcribe-action-btn" onClick={() => clearTranscribeSession()}>
                      Clear session
                    </button>
                  </div>
                  {liveDisplayProgressionChips.length === 0 ? (
                    <p className="live-transcribe-actions-hint muted-hint">
                      Chords will appear here after a few seconds — then you can copy or save a snapshot.
                    </p>
                  ) : null}
                  <p className="live-progression-quality" role="status">
                    {liveDisplayProgressionUi.qualityPrimary}
                  </p>
                  {liveDisplayProgressionUi.qualitySecondary ? (
                    <p className="live-progression-pattern-hint">{liveDisplayProgressionUi.qualitySecondary}</p>
                  ) : null}
                  {(liveDerivedProgression.showPatternHint && liveDerivedProgression.chipLabels.length > 0) ||
                  (liveDisplayProgressionChips.length > 0 &&
                    liveDisplayProgressionChips.length < 2 &&
                    liveDerivedProgression.chipLabels.length === 0) ? (
                    <p className="live-progression-pattern-hint">Still listening for the pattern…</p>
                  ) : null}
                  {liveDisplayProgressionChips.length > 0 ? (
                    <div className="live-progression-chips" aria-label="Main progression chords to try first">
                      {liveDisplayProgressionChips.map((lab, i) => (
                        <Fragment key={`${lab}-${i}`}>
                          {i > 0 ? (
                            <span className="live-progression-arrow" aria-hidden>
                              →
                            </span>
                          ) : null}
                          <span className="live-progression-chip">{liveChordLabelForDisplay(lab)}</span>
                        </Fragment>
                      ))}
                    </div>
                  ) : (
                    <p className="live-transcribe-progression-line live-transcribe-progression-line--empty">
                      Still listening for the progression…
                    </p>
                  )}
                  {liveDisplayProgressionChips.length > 0 && transcribeKey?.label && transcribeKey.label !== "—" ? (
                    <p className="live-transcribe-loop-hint">
                      Try these in <strong>{transcribeKey.label}</strong> if that feels right — the fine timeline below is
                      reference only.
                    </p>
                  ) : null}
                  {transcribeSummary ? <p className="live-transcribe-summary">{transcribeSummary}</p> : null}
                </div>

                <details className="live-transcribe-timeline-details">
                  <summary className="live-transcribe-timeline-summary">Fine-grained timeline (optional)</summary>
                  <div className="live-transcribe-timeline live-transcribe-timeline--secondary">
                    <p className="live-transcribe-timeline-lead">
                      Shorter slices from roughly the last {TRANSCRIBE_TIMELINE_KEEP_SEC}s — for cross-checking, not for
                      practicing the main row.
                    </p>
                    {transcribeRecentTimeline.length === 0 ? (
                      <p className="history-empty">Segments appear after the listener has analyzed a few windows.</p>
                    ) : (
                      <ul className="live-transcribe-timeline-list" aria-label="Recent chords in time order">
                        {transcribeRecentTimeline.map((s, i) => (
                          <li key={`${s.label}-${s.t0}-${i}`}>
                            <span className="live-transcribe-timeline-span">
                              {s.t0.toFixed(1)}s–{s.t1.toFixed(1)}s
                            </span>{" "}
                            <strong>{liveChordLabelForDisplay(s.label)}</strong>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </details>
              </section>
            </>
          )}

          <details className="live-debug-panel">
            <summary>
              Diagnostics{" "}
              <span className="live-debug-summary-hint">for troubleshooting</span>
            </summary>
            <p className="live-debug-lead">
              Open while recording if the readout freezes. Verbose browser logs: add <code>?liveDebug=1</code> to the page
              URL or set <code>NEXT_PUBLIC_LIVE_MIC_DEBUG=1</code>.
            </p>
            <dl className="live-debug-dl">
              <dt>POST endpoint</dt>
              <dd>
                <code>
                  {liveExperienceMode === "transcribe"
                    ? `${API_BASE}/live-transcribe (rolling ~${TRANSCRIBE_WINDOW_SEC}s)`
                    : `${API_BASE}/stream?mode=${liveInputMode}`}
                </code>
              </dd>
              <dt>Live input mode (UI selector)</dt>
              <dd>
                <code>{liveInputMode}</code>
              </dd>
              <dt>Preset (last response)</dt>
              <dd>
                <code>{liveDebug.lastStreamInputMode}</code> — {liveDebug.lastStreamPresetName}
              </dd>
              <dt>WAV chunk length (this mode)</dt>
              <dd>
                {liveExperienceMode === "transcribe" ? (
                  <>
                    Rolling buffer up to {TRANSCRIBE_RING_MAX_SEC}s · analyze ~{TRANSCRIBE_WINDOW_SEC}s · every{" "}
                    {TRANSCRIBE_INTERVAL_SEC}s
                    <span className="live-debug-muted"> (no /stream chunks in this mode)</span>
                  </>
                ) : (
                  <>
                    {liveChunkSeconds} s <span className="live-debug-muted">(song uses shorter chunks for lower latency)</span>
                  </>
                )}
              </dd>
              <dt>Latency hint (wall clock)</dt>
              <dd>
                since last chunk sent: {formatMsOptional(liveDebug.msSinceLastChunkSent)} · since last /stream response:{" "}
                {formatMsOptional(liveDebug.msSinceLastStreamResponse)}{" "}
                <span className="live-debug-muted">(updated ~250ms while recording)</span>
              </dd>
              <dt>Preset tuning (last response)</dt>
              <dd>
                weak_confirm={liveDebug.lastPresetWeakConfirmChunks ?? "—"} · silence_clear=
                {liveDebug.lastPresetSilenceStreakClear ?? "—"} · invalid_clear=
                {liveDebug.lastPresetInvalidStreakClear ?? "—"}
              </dd>
              <dt>Key: instant raw / conf (chunk)</dt>
              <dd>
                <code>{liveDebug.lastStreamInstantKeyRaw}</code> ·{" "}
                {formatOptionalFloat(liveDebug.lastStreamInstantKeyConf, 3)}
              </dd>
              <dt>Key: smoothed internal (session state)</dt>
              <dd>
                <code>{liveDebug.lastStreamSmoothedKeyRaw}</code>
              </dd>
              <dt>Key display source</dt>
              <dd>
                <code>{liveDebug.lastStreamKeyDisplaySource}</code>
              </dd>
              <dt>Held last valid chord</dt>
              <dd>
                {liveDebug.lastStreamHeldLastValid != null ? String(liveDebug.lastStreamHeldLastValid) : "—"}
              </dd>
              <dt>Chord commit (last)</dt>
              <dd>
                <code>{liveDebug.lastStreamChordCommitKind}</code>
                <span className="live-debug-muted"> · displayed </span>
                <code>{liveDebug.lastStreamDisplayedChord}</code>
              </dd>
              <dt>Backend raw / final chord (last)</dt>
              <dd>
                <code>{liveDebug.lastBackendRaw}</code> → <code>{liveDebug.lastBackendFinalChord}</code>
              </dd>
              <dt>Mic permission (this session)</dt>
              <dd>{liveDebug.micPermission}</dd>
              <dt>Browser mic permission</dt>
              <dd>
                {liveDebug.browserMicPermission}{" "}
                <span className="live-debug-muted">(Permissions API; may stay &quot;prompt&quot; until Start)</span>
              </dd>
              <dt>AudioContext state</dt>
              <dd>{liveDebug.audioContextState}</dd>
              <dt>Audio processing (ScriptProcessor)</dt>
              <dd>{liveAudioProcessingSummary}</dd>
              <dt>InputBuffer channels (telemetry)</dt>
              <dd>{liveDebug.lastInputBufferChannels || "—"}</dd>
              <dt>Audio tracks / label</dt>
              <dd>
                {liveDebug.trackCount} · {liveDebug.trackLabel}
              </dd>
              <dt>Track enabled / muted / readyState</dt>
              <dd>
                {String(liveDebug.trackEnabled)} / {String(liveDebug.trackMuted)} / {liveDebug.trackReadyState}
              </dd>
              <dt>getSettings() — deviceId</dt>
              <dd>
                <code>{liveDebug.trackDeviceId}</code>
              </dd>
              <dt>getSettings() — sampleRate / channelCount</dt>
              <dd>
                {liveDebug.trackSampleRate} / {liveDebug.trackChannelCount}
              </dd>
              <dt>getSettings() — echo / noise / AGC</dt>
              <dd>
                {liveDebug.trackEchoCancellation} / {liveDebug.trackNoiseSuppression} /{" "}
                {liveDebug.trackAutoGainControl}
              </dd>
              <dt>Input boost (WAV / analysis path)</dt>
              <dd>
                {liveInputBoost}× <span className="live-debug-muted">(not applied to speaker output)</span>
              </dd>
              <dt>Raw mono peak / RMS (pre-boost)</dt>
              <dd>
                {formatLivePeakRms(liveDebug.lastRawSamplePeak, liveDebug.lastRawSampleRms)}{" "}
                <span className="live-debug-muted">(telemetry ~220ms)</span>
              </dd>
              <dt>Boosted mono peak / RMS (pre-WAV, clamped)</dt>
              <dd>{formatLivePeakRms(liveDebug.lastBoostedSamplePeak, liveDebug.lastBoostedSampleRms)}</dd>
              <dt>Clipping (last telemetry buffer)</dt>
              <dd>
                {liveDebug.lastBoostClipFraction > 0
                  ? `yes — ${(liveDebug.lastBoostClipFraction * 100).toFixed(2)}% of samples hit ±1 clamp`
                  : "no"}
              </dd>
              <dt>Chunks created (WAV emitted)</dt>
              <dd>{liveDebug.chunksCreated}</dd>
              <dt>Chunks posted (HTTP 200 + JSON parsed)</dt>
              <dd>{liveDebug.chunksPosted}</dd>
              <dt>Responses applied (to UI)</dt>
              <dd>{liveDebug.responsesApplied}</dd>
              <dt>Last WAV blob size</dt>
              <dd>{liveDebug.lastChunkSize > 0 ? `${liveDebug.lastChunkSize} B (${formatBytes(liveDebug.lastChunkSize)})` : "—"}</dd>
              <dt>Last HTTP status</dt>
              <dd>{liveDebug.lastHttpStatus != null ? liveDebug.lastHttpStatus : "—"}</dd>
              <dt>Last upload / response line</dt>
              <dd>{liveDebug.lastUploadStatus}</dd>
              <dt>Last backend chord / raw</dt>
              <dd>
                {liveDebug.lastBackendChord} / {liveDebug.lastBackendRaw}
              </dd>
              <dt>Backend rejection / gating</dt>
              <dd>
                <code>{liveDebug.lastStreamRejectionReason}</code>
                {liveDebug.lastBackendAccepted != null ? (
                  <>
                    {" "}
                    · accepted={String(liveDebug.lastBackendAccepted)}
                    {liveDebug.lastBackendClearDisplay != null
                      ? ` · clear_display=${String(liveDebug.lastBackendClearDisplay)}`
                      : ""}
                  </>
                ) : null}
              </dd>
              <dt>Backend waveform RMS / peak (chunk)</dt>
              <dd>
                {formatOptionalFloat(liveDebug.lastBackendWaveformRms, 6)} /{" "}
                {formatOptionalFloat(liveDebug.lastBackendWaveformPeak, 6)}
              </dd>
              <dt>Template scores: best / second / margin</dt>
              <dd>
                {formatOptionalFloat(liveDebug.lastBackendBestScore, 3)} / {formatOptionalFloat(liveDebug.lastBackendSecondScore, 3)}{" "}
                / {formatOptionalFloat(liveDebug.lastBackendTemplateMargin, 3)}
              </dd>
              <dt>Chroma: entropy / stability / strong bins</dt>
              <dd>
                {formatOptionalFloat(liveDebug.lastBackendChromaEntropy, 3)} /{" "}
                {formatOptionalFloat(liveDebug.lastBackendChromaStability, 3)} /{" "}
                {liveDebug.lastBackendStrongChromaBins != null ? liveDebug.lastBackendStrongChromaBins : "—"}
              </dd>
              <dt>Final chord (after gating, matches response)</dt>
              <dd>
                <code>{liveDebug.lastBackendFinalChord}</code> · silence_flag{" "}
                {liveDebug.lastBackendSilenceFlag != null ? String(liveDebug.lastBackendSilenceFlag) : "—"}
              </dd>
              <dt>Ignored / disposition</dt>
              <dd>{liveDebug.ignoredResponseReason}</dd>
              <dt>Last fetch / HTTP body error</dt>
              <dd>{liveDebug.lastFetchError}</dd>
              <dt>Last UI error (alert line)</dt>
              <dd>{liveDebug.lastUiError}</dd>
            </dl>
          </details>

          {liveExperienceMode === "transcribe" ? (
            <details className="live-debug-panel live-debug-panel--transcribe">
              <summary>
                Transcription timing <span className="live-debug-summary-hint">technical</span>
              </summary>
              <div className="live-debug-toolbar-row muted-hint">
                <label>
                  <input
                    type="checkbox"
                    checked={transcribeServerDebug}
                    onChange={(e) => setTranscribeServerDebug(e.target.checked)}
                  />{" "}
                  Request <code>debug=true</code> on <code>/live-transcribe</code> (server diagnostics)
                </label>
              </div>
              <dl className="live-debug-dl">
                <dt>Last core empty reason (server)</dt>
                <dd>
                  <code>{transcribeDebug.lastLtCoreEmptyReason}</code>
                </dd>
                <dt>Runs-for-core strategy (server)</dt>
                <dd>
                  <code>{transcribeDebug.lastLtRunsForCoreStrategy}</code>
                </dd>
                <dt>Progression source / quality (API)</dt>
                <dd>
                  <code>{transcribeDebug.lastProgressionSource}</code> · <code>{transcribeDebug.lastProgressionQuality}</code>
                </dd>
                <dt>Segment counts (server debug)</dt>
                <dd>{transcribeDebug.lastLtServerSegmentSummary}</dd>
                <dt>Client merged timeline seg count (echo)</dt>
                <dd>{transcribeDebug.lastLtClientTimelineSegEcho}</dd>
                <dt>First analysis threshold (min buffer, s)</dt>
                <dd>{FIRST_TRANSCRIBE_AFTER_SEC}</dd>
                <dt>Analysis window (max slice, s)</dt>
                <dd>{TRANSCRIBE_WINDOW_SEC}</dd>
                <dt>Analysis interval (s)</dt>
                <dd>{TRANSCRIBE_INTERVAL_SEC}</dd>
                <dt>Min audio slice before POST (s)</dt>
                <dd>{MIN_TRANSCRIBE_AUDIO_SEC}</dd>
                <dt>Current rolling buffer (s, polled)</dt>
                <dd>{transcribeBufferDisplaySec.toFixed(2)}</dd>
                <dt>Last request round-trip (ms)</dt>
                <dd>{transcribeDebug.lastRequestDurationMs != null ? transcribeDebug.lastRequestDurationMs : "—"}</dd>
                <dt>Last response <code>status</code> field</dt>
                <dd>
                  <code>{transcribeDebug.lastAnalysisStatus}</code>
                </dd>
                <dt>Request in flight</dt>
                <dd>{String(transcribeRequestInFlight)}</dd>
                <dt>Session id</dt>
                <dd>
                  <code>{transcribeSessionId || "—"}</code>
                </dd>
                <dt>Ring buffered (last cycle, s)</dt>
                <dd>{transcribeDebug.ringBufferedSec.toFixed(2)}</dd>
                <dt>Last WAV window sent (s)</dt>
                <dd>{transcribeDebug.lastWindowSec.toFixed(2)}</dd>
                <dt>ScriptProcessor callbacks (see main Live debug)</dt>
                <dd>{liveDebug.audioProcessCallbacks}</dd>
                <dt>Last raw peak / RMS (telemetry)</dt>
                <dd>{formatLivePeakRms(transcribeDebug.lastRawPeak, transcribeDebug.lastRawRms)}</dd>
                <dt>Last HTTP outcome</dt>
                <dd>{transcribeDebug.lastRequestStatus}</dd>
                <dt>Analyses completed (applied)</dt>
                <dd>{transcribeDebug.analysisCount}</dd>
                <dt>Last key (raw from server)</dt>
                <dd>{transcribeDebug.lastKeyLabel}</dd>
                <dt>Last core progression line</dt>
                <dd>{transcribeDebug.lastProgression}</dd>
                <dt>Last error</dt>
                <dd>{transcribeDebug.lastError}</dd>
              </dl>
            </details>
          ) : null}
        </>
      ) : (
        <section className="analyze-panel" aria-label="File analysis">
          <header className="analyze-panel-header">
            <h2>Learn from audio</h2>
            <p className="analyze-lead">
              Upload a recording for the best chord chart. Follow the main progression, then loop sections and slow the tempo
              until it feels easy.
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
              {analyzeLoading ? "Working…" : "Analyze"}
            </button>
            <label className="muted-hint analyze-debug-api-label">
              <input
                type="checkbox"
                checked={analyzeQueryDebug}
                onChange={(e) => setAnalyzeQueryDebug(e.target.checked)}
              />{" "}
              Include <code>debug</code> in API response
            </label>
          </div>

          {analyzeLoading ? (
            <div className="analyze-processing" role="status" aria-live="polite" aria-busy="true">
              <div className="analyze-spinner" aria-hidden />
              <div className="analyze-processing-body">
                <strong className="analyze-processing-title">Analyzing your track</strong>
                <p className="analyze-processing-detail">
                  Longer files take longer. The list below is a rough guide — the server may finish steps in a different order.
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
                <h3 className="analyze-song-summary-title">This track</h3>
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
                <TransposeDisplayControl
                  id="analyze-transpose"
                  value={displayTransposeSemitones}
                  onChange={setDisplayTransposeSemitones}
                />
                {analyzeResult.debug && typeof analyzeResult.debug === "object" ? (
                  <details className="analyze-api-debug">
                    <summary className="muted-hint">Server analysis debug (requested)</summary>
                    <pre className="analyze-api-debug-pre">{JSON.stringify(analyzeResult.debug, null, 2)}</pre>
                  </details>
                ) : null}
              </div>

              <div className="analyze-chord-rail-block analyze-chord-rail-block--core">
                <h2 className="analyze-learning-heading">Main progression</h2>
                <p className="analyze-learning-lead">
                  The song in broad strokes — tap a chord to jump where it first appears in the track.
                </p>
                <div className="analyze-core-row" aria-label="Core chord progression">
                  {                    coreProgressionDisplay.length === 0 ? (
                    <p className="analyze-core-empty">No chord summary available for this track.</p>
                  ) : (
                    coreProgressionDisplay.map((entry, i) => {
                      const orig = coreProgression[i];
                      const isActive =
                        orig.label === currentChordLabelForHighlight && orig.label !== "N";
                      return (
                        <div className="analyze-core-slot" key={`${orig.label}-core-${i}`}>
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
                              const t = firstChordTimeForLabel(analyzeResult.chords, orig.label);
                              if (t === null) return;
                              el.currentTime = t;
                              setAnalyzePlaybackTime(t);
                            }}
                            title={`Jump to first ${orig.label}`}
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
                    Some symbols are educated guesses — trust your ears if something feels off.
                  </p>
                ) : null}
              </div>

              {analyzeAudioUrl ? (
                <div className="analyze-practice-controls-card">
                  <h3 className="analyze-subhead">Practice setup</h3>
                  <p className="analyze-practice-controls-lead">
                    Playback, speed, and loop — set this first, then use <strong>Right now</strong> to play along with the chart.
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
                <span className="analyze-practice-view-label">Chart style</span>
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
                <h3 className="analyze-subhead">Right now</h3>
                <div className="analyze-practice-grid analyze-practice-grid--two">
                  <div className="analyze-practice-cell">
                    <span className="analyze-practice-eyebrow">Current chord</span>
                    <p className="analyze-practice-chord" aria-live="polite">
                      {currentAnalyzeChordForUi?.label ??
                        transposeChordLabel(
                          chordLabelAtTime(analyzePlaybackTime, analyzeResult.chords, analyzePlaybackDuration),
                          displayTransposeSemitones,
                        )}
                    </p>
                    {analyzePracticeView === "piano" ? (
                      (() => {
                        const h = getSimplePianoHands(currentAnalyzeChordForUi);
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
                    ) : formatPlayHint(currentAnalyzeChordForUi) ? (
                      <p className="analyze-practice-playhint">{formatPlayHint(currentAnalyzeChordForUi)}</p>
                    ) : chordNotesLine(currentAnalyzeChordForUi) !== "—" ? (
                      <p className="analyze-practice-tones">{chordNotesLine(currentAnalyzeChordForUi)}</p>
                    ) : null}
                    {analyzePracticeView === "chords" && currentAnalyzeChord?.low_confidence ? (
                      <p className="analyze-practice-ear">Check this one by ear</p>
                    ) : null}
                  </div>
                  <div className="analyze-practice-cell">
                    <span className="analyze-practice-eyebrow">Next chord</span>
                    <p className="analyze-practice-chord analyze-practice-chord--secondary" aria-live="polite">
                      {nextChordCountdown
                        ? transposeChordLabel(nextChordCountdown.label, displayTransposeSemitones)
                        : nextRunDisplayForUi.label}
                    </p>
                    {analyzePracticeView === "piano" && nextRunDisplayForUi.label !== "End of chart" ? (
                      (() => {
                        const h = getSimplePianoHands(displayedNextChordSegForUi);
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
                        {displayedNextChordSegForUi ? (
                          formatPlayHint(displayedNextChordSegForUi) ? (
                            <p className="analyze-practice-playhint">{formatPlayHint(displayedNextChordSegForUi)}</p>
                          ) : chordNotesLine(displayedNextChordSegForUi) !== "—" ? (
                            <p className="analyze-practice-tones">{chordNotesLine(displayedNextChordSegForUi)}</p>
                          ) : null
                        ) : !nextChordCountdown &&
                          nextRunDisplayForUi.notesLine &&
                          nextRunDisplayForUi.notesLine !== "—" ? (
                          <p className="analyze-practice-tones">{nextRunDisplayForUi.notesLine}</p>
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
                <h3 className="analyze-subhead">Jump to a part</h3>
                <p className="analyze-section-flow-hint analyze-section-flow-hint--short">
                  Same sections as in <strong>Practice setup</strong> — tap to seek.
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
                <summary>Song map &amp; extra detail (optional)</summary>
                <div className="analyze-advanced-inner">
                  <section className="analyze-advanced-chunk" aria-label="Song map">
                    <h4 className="analyze-advanced-h">Song map</h4>
                    <div className="analyze-timeline-block analyze-timeline-block--secondary">
                      <p className="analyze-legend analyze-legend--muted">
                        Overview only — click the bar to seek. Daily practice: use <strong>Main progression</strong> and{" "}
                        <strong>Right now</strong> above.
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
                            title={`${transposeChordLabel(c.label, displayTransposeSemitones)} ${formatTimeSec(c.start)}–${formatTimeSec(c.end)}`}
                          >
                            {transposeChordLabel(c.label, displayTransposeSemitones)}
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

                  <details className="analyze-nested-drawer">
                    <summary>Chord-by-chord timing</summary>
                    <div className="analyze-nested-drawer-body">
                      <p className="analyze-more-p analyze-more-p--nested">
                        Exact timestamps for each symbol — for detail work. Most players stay with the main progression.
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
                              title={`${transposeChordLabel(run.label, displayTransposeSemitones)} at ${formatTimeSec(run.start)} (${holdSec.toFixed(0)}s)`}
                            >
                              {isActive ? <span className="analyze-prog-badge">Now</span> : null}
                              {isNext ? <span className="analyze-prog-badge analyze-prog-badge--next">Next</span> : null}
                              <span className="analyze-prog-symbol">
                                {transposeChordLabel(run.label, displayTransposeSemitones)}
                                {run.repeatCount > 1 ? (
                                  <span className="analyze-prog-repeat"> ×{run.repeatCount}</span>
                                ) : null}
                              </span>
                              <span className="analyze-prog-time">
                                {formatTimeSec(run.start)} – {formatTimeSec(run.end)}
                              </span>
                              <span className="analyze-prog-notes">
                                {transposeChordToneLine(run.notesLine, displayTransposeSemitones)}
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  </details>

                  <details className="analyze-nested-drawer">
                    <summary>Raw sections &amp; beat estimate</summary>
                    <div className="analyze-nested-drawer-body">
                      <section className="analyze-advanced-chunk analyze-advanced-chunk--nested-block" aria-label="Raw sections">
                        <h4 className="analyze-advanced-h">Sections from the analyzer</h4>
                        <p className="analyze-more-p">
                          A/B-style markers from the server. Your practice <strong>Parts</strong> above are merged for easier
                          looping.
                        </p>
                        <ul className="analyze-raw-sections-list">
                          {(analyzeResult.sections ?? []).map((s, i) => (
                            <li key={`raw-sec-${s.index ?? i}-${s.start}`}>
                              {formatSectionDropdownLabel(s, analyzePlaybackDuration, i)}
                            </li>
                          ))}
                        </ul>
                      </section>
                      <section className="analyze-advanced-chunk analyze-advanced-chunk--nested-block" aria-label="Meter estimate">
                        <h4 className="analyze-advanced-h">Beat &amp; meter (estimate)</h4>
                        <p className="analyze-more-p">
                          Approximate place in the bar: {approxMeterReadout}. Assuming{" "}
                          {analyzeRhythmEffective.assumed_beats_per_bar} beats per bar (hint only).
                        </p>
                      </section>
                    </div>
                  </details>
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
            <p className="analyze-empty-hint">Choose a file and tap Analyze to see chords, sections, and practice tools.</p>
          ) : null}
        </section>
      )}

      <p className="meta-footer">
        API base: <code>{API_BASE}</code> — optional env <code>NEXT_PUBLIC_API_URL</code>. Optional verbose mic logs:{" "}
        <code>?liveDebug=1</code>.
      </p>
    </main>
  );
}
