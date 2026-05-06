/**
 * Display-only chord / note transposition helpers for practice UI.
 * Unknown or non-standard labels are returned unchanged (never throws).
 */

const SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];

const LETTER_PC: Record<string, number> = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

function normalizeAcc(raw: string): "" | "#" | "b" {
  if (!raw) return "";
  if (raw === "♯" || raw === "#") return "#";
  if (raw === "♭" || raw === "b") return "b";
  return "";
}

export type TransposableChordSegment = {
  label: string;
  notes?: string[];
  practice_hint?: string;
};

function parsePitchClass(note: string): { pc: number; preferFlat: boolean } | null {
  const t = note.trim();
  const m = t.match(/^([A-Ga-g])([#b♯♭]?)$/u);
  if (!m) return null;
  const L = m[1].toUpperCase();
  const acc = normalizeAcc(m[2] ?? "");
  const base = LETTER_PC[L];
  if (base === undefined) return null;
  let pc = base;
  if (acc === "#") pc = (pc + 1) % 12;
  if (acc === "b") pc = (pc + 11) % 12;
  return { pc, preferFlat: acc === "b" };
}

function shiftPc(pc: number, semitones: number): number {
  return ((((pc + semitones) % 12) + 12) % 12);
}

/**
 * Transpose a simple pitch-class name (e.g. C, F#, Bb). Optional octave digits are preserved.
 */
export function transposeNoteName(note: string, semitones: number): string {
  const raw = note.trim();
  if (!raw || semitones === 0) return note;

  const m = raw.match(/^([A-Ga-g])([#b♯♭]?)(\d*)$/u);
  if (!m) return note;

  const parsed = parsePitchClass(`${m[1]}${m[2] ?? ""}`);
  if (!parsed) return note;

  const pc = shiftPc(parsed.pc, semitones);
  const name = (parsed.preferFlat ? FLAT_NAMES : SHARP_NAMES)[pc];
  return name + (m[3] ?? "");
}

/**
 * Transpose a chord symbol: root + suffix, optional slash bass (single-letter root + accidental).
 */
export function transposeChordLabel(chord: string, semitones: number): string {
  const t = chord.trim();
  if (!t || semitones === 0) return chord;
  if (t === "N" || t === "—" || t === "-") return t;
  if (!/^[A-Ga-g]/u.test(t)) return chord;

  const slashIdx = t.indexOf("/");
  const head = slashIdx >= 0 ? t.slice(0, slashIdx) : t;
  const bassRaw = slashIdx >= 0 ? t.slice(slashIdx + 1).trim() : "";

  if (bassRaw.includes("/")) return chord;

  const hm = head.match(/^([A-Ga-g])([#b♯♭]?)(.*)$/u);
  if (!hm) return chord;

  const parsed = parsePitchClass(`${hm[1]}${hm[2] ?? ""}`);
  if (!parsed) return chord;

  const newPc = shiftPc(parsed.pc, semitones);
  const newRoot = (parsed.preferFlat ? FLAT_NAMES : SHARP_NAMES)[newPc];
  let out = `${newRoot}${hm[3]}`;

  if (bassRaw) {
    const bm = bassRaw.match(/^([A-Ga-g])([#b♯♭]?)$/u);
    if (!bm) return chord;
    const bparsed = parsePitchClass(`${bm[1]}${bm[2] ?? ""}`);
    if (!bparsed) return chord;
    const bPc = shiftPc(bparsed.pc, semitones);
    const bName = (bparsed.preferFlat ? FLAT_NAMES : SHARP_NAMES)[bPc];
    out = `${out}/${bName}`;
  }

  return out;
}

export function transposeNotes(notes: string[], semitones: number): string[] {
  if (!semitones || !notes.length) return [...notes];
  return notes.map((n) => transposeNoteName(n, semitones));
}

/**
 * Line built from chord tones (e.g. "C · E · G" or "C - E - G"); otherwise whole line treated as one chord label.
 */
export function transposeChordToneLine(line: string, semitones: number): string {
  const t = line.trim();
  if (!t || semitones === 0) return line;
  if (t.includes(" · ")) {
    return t
      .split(" · ")
      .map((p) => transposeNoteName(p, semitones))
      .join(" · ");
  }
  if (t.includes(" - ")) {
    return t
      .split(" - ")
      .map((p) => transposeNoteName(p, semitones))
      .join(" - ");
  }
  return transposeChordLabel(t, semitones);
}

/**
 * Shallow segment for display: transposed label and notes; drops practice_hint when notes exist (regenerate from tones).
 */
export function transposeChordSegment<T extends TransposableChordSegment>(seg: T | null | undefined, semitones: number): T | null {
  if (!seg) return null;
  if (!semitones) return seg;

  return {
    ...seg,
    label: transposeChordLabel(seg.label, semitones),
    notes: seg.notes?.length ? transposeNotes(seg.notes, semitones) : seg.notes,
    practice_hint: seg.notes?.length ? undefined : seg.practice_hint,
  };
}

/**
 * Spell simple triad tones from a live/stream-style symbol (C, Am, Bb, F#m). Slash chords use the left side only.
 * Returns null for empty, N, placeholders, or unparseable labels.
 */
export function liveTriadNoteNamesFromLabel(label: string | null | undefined): string[] | null {
  const t = (label ?? "").trim();
  if (!t || t === "N" || t === "n" || t === "—" || t === "-" || t === "Listening..." || t === "--") {
    return null;
  }
  const head = t.split("/")[0]?.trim() ?? t;
  let qual: "maj" | "min" | "dim";
  let rootRaw: string;
  if (head.length >= 4 && head.endsWith("dim")) {
    rootRaw = head.slice(0, -3);
    qual = "dim";
  } else if (head.length > 1 && head.endsWith("m") && !head.endsWith("maj")) {
    rootRaw = head.slice(0, -1);
    qual = "min";
  } else {
    rootRaw = head;
    qual = "maj";
  }
  const parsed = parsePitchClass(rootRaw);
  if (!parsed) return null;
  const iv = qual === "maj" ? [0, 4, 7] : qual === "min" ? [0, 3, 7] : [0, 3, 6];
  const names = parsed.preferFlat ? FLAT_NAMES : SHARP_NAMES;
  return iv.map((off) => names[(parsed.pc + off) % 12]);
}
