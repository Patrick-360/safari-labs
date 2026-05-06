/**
 * Client-only practice snapshots for Live Song Transcription (JSON download).
 */

export const LIVE_TRANSCRIBE_SNAPSHOT_NOTE =
  "Rough live transcription from microphone — not a verified chart. Prefer Analyze File for accuracy.";

export type LiveTranscribeSnapshotV1 = {
  schema_version: 1;
  /** ISO-8601 when the snapshot was taken */
  created_at: string;
  disclaimer: string;
  source: {
    kind: "live_song_transcription";
    session_id: string;
    rough_transcription: true;
  };
  likely_key: {
    label: string;
    confidence: number | null;
    /** e.g. Low / Medium / High from UI tier; null if unknown */
    stability_word: string | null;
  };
  main_progression: {
    labels: string[];
    /** e.g. C → G → Am → F */
    progression_text: string;
    is_likely_loop: boolean;
    quality_label: string;
  };
  practice_guidance: {
    summary: string | null;
    current_chord: string;
    current_chord_notes: string[];
  };
  recent_chord_segments: Array<{
    start_sec: number;
    end_sec: number;
    label: string;
  }>;
  capture_settings: {
    input_mode: string;
    input_mode_label: string;
    input_boost: number;
    /** Semitone offset applied to chord/note labels in this snapshot (display-only). */
    display_transpose_semitones: number;
  };
};

export type BuildLiveTranscribeSnapshotInput = {
  sessionId: string;
  likelyKey: { label: string; confidence: number } | null;
  keyStabilityWord: string | null;
  mainProgressionLabels: string[];
  progressionIsLikelyLoop: boolean;
  progressionQualityLabel: string;
  formatChordLabel: (label: string) => string;
  recentSegments: { t0: number; t1: number; label: string }[];
  summary: string | null;
  currentChord: string;
  currentChordNotes: string[];
  inputMode: string;
  inputModeLabel: string;
  inputBoost: number;
  /** Semitones added to displayed labels in this snapshot; 0 = concert (as detected). */
  displayTransposeSemitones?: number;
};

export function buildLiveTranscribeSnapshot(input: BuildLiveTranscribeSnapshotInput): LiveTranscribeSnapshotV1 {
  const displayTranspose = input.displayTransposeSemitones ?? 0;
  const labels = input.mainProgressionLabels.map((l) => input.formatChordLabel(l));
  const progressionText = labels.join(" → ");
  const created = new Date();
  const keyLabel = input.likelyKey?.label?.trim() && input.likelyKey.label !== "—" ? input.likelyKey.label : "—";
  return {
    schema_version: 1,
    created_at: created.toISOString(),
    disclaimer: LIVE_TRANSCRIBE_SNAPSHOT_NOTE,
    source: {
      kind: "live_song_transcription",
      session_id: input.sessionId || "—",
      rough_transcription: true,
    },
    likely_key: {
      label: keyLabel,
      confidence:
        input.likelyKey && typeof input.likelyKey.confidence === "number" && keyLabel !== "—"
          ? input.likelyKey.confidence
          : null,
      stability_word: keyLabel !== "—" ? input.keyStabilityWord : null,
    },
    main_progression: {
      labels,
      progression_text: progressionText || "(empty — keep listening)",
      is_likely_loop: input.progressionIsLikelyLoop,
      quality_label: input.progressionQualityLabel,
    },
    practice_guidance: {
      summary: input.summary?.trim() || null,
      current_chord: input.currentChord,
      current_chord_notes: [...input.currentChordNotes],
    },
    recent_chord_segments: input.recentSegments.map((s) => ({
      start_sec: round4(s.t0),
      end_sec: round4(s.t1),
      label: input.formatChordLabel(s.label),
    })),
    capture_settings: {
      input_mode: input.inputMode,
      input_mode_label: input.inputModeLabel,
      input_boost: input.inputBoost,
      display_transpose_semitones: displayTranspose,
    },
  };
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

/** Local calendar date for filename: live-transcription-snapshot-YYYY-MM-DD.json */
export function formatSnapshotDownloadDate(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

export function downloadLiveTranscribeSnapshotJson(snapshot: LiveTranscribeSnapshotV1): void {
  const json = `${JSON.stringify(snapshot, null, 2)}\n`;
  const blob = new Blob([json], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const d = new Date(snapshot.created_at);
  const safeDate = Number.isNaN(d.getTime()) ? formatSnapshotDownloadDate(new Date()) : formatSnapshotDownloadDate(d);
  a.href = url;
  a.download = `live-transcription-snapshot-${safeDate}.json`;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export async function copyTextToClipboard(text: string): Promise<boolean> {
  try {
    if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall through */
  }
  try {
    if (typeof document === "undefined") {
      return false;
    }
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}
