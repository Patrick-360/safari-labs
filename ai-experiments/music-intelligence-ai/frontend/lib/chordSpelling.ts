/**
 * Client-side chord spelling — kept in sync with backend/app/audio/chord_spellings.py (spell_chord_tones).
 *
 * Examples (no test runner in this package; mirror backend tests):
 *   spellChordTones("C") → ["C","E","G"]
 *   spellChordTones("Am") → ["A","C","E"]
 *   spellChordTones("Cm7") → ["C","Eb","G","Bb"]
 *   spellChordTones("C/E") → ["C","E","G"]  (slash: spell left side only)
 *   spellChordTones("N") → []
 */

const PC_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const PC_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];

const ROOT_ALIASES: Record<string, number> = {
  C: 0,
  "C#": 1,
  Db: 1,
  D: 2,
  "D#": 3,
  Eb: 3,
  E: 4,
  F: 5,
  "F#": 6,
  Gb: 6,
  G: 7,
  "G#": 8,
  Ab: 8,
  A: 9,
  "A#": 10,
  Bb: 10,
  B: 11,
};

function pcDisplay(pc: number, preferFlat: boolean): string {
  const table = preferFlat ? PC_FLAT : PC_SHARP;
  return table[((pc % 12) + 12) % 12];
}

function rootPrefersFlat(body: string): boolean {
  return body.toLowerCase().includes("b");
}

function normalizeColonChord(token: string): string | null {
  const idx = token.indexOf(":");
  if (idx < 0) {
    return token.trim();
  }
  const root = token.slice(0, idx).trim();
  const qualRawSt = token.slice(idx + 1).trim();
  const qual = qualRawSt.toLowerCase();
  if (!(root in ROOT_ALIASES)) return null;
  if (qual === "maj" || qual === "major" || qualRawSt === "M") {
    return root;
  }
  if (qual === "min" || qual === "minor" || qual === "m") {
    return `${root}m`;
  }
  return null;
}

/**
 * Bass note after slash (single-letter style), for LH hints. Unrecognized → null.
 */
export function slashBassNote(label: string): string | null {
  const t = label.trim();
  const idx = t.indexOf("/");
  if (idx < 0) return null;
  const bass = t.slice(idx + 1).trim();
  if (bass.includes("/")) return null;
  const m = bass.match(/^([A-Ga-g])([#b♯♭]?)$/u);
  if (!m) return null;
  return m[1].toUpperCase() + (m[2] ?? "").replace("♯", "#").replace("♭", "b");
}

function rootHasSharp(body: string): boolean {
  return body.includes("#");
}

function preferFlatSpelling(body: string, intervals: readonly number[]): boolean {
  if (rootPrefersFlat(body)) return true;
  if (rootHasSharp(body)) return false;
  return intervals.includes(3) || intervals.includes(10);
}

export function spellChordTones(symbol: string): string[] {
  if (!symbol || !String(symbol).trim()) {
    return [];
  }
  const raw = String(symbol).trim();
  if (raw.toUpperCase() === "N") {
    return [];
  }

  const chordSide = raw.split("/")[0].trim();
  const core = normalizeColonChord(chordSide);
  if (core === null) {
    return [];
  }

  const cl = core.toLowerCase();
  let body: string;
  let intervals: readonly number[];

  if (cl.endsWith("maj7")) {
    body = core.slice(0, -4);
    intervals = [0, 4, 7, 11];
  } else if (cl.endsWith("m7b5")) {
    body = core.slice(0, -5);
    intervals = [0, 3, 6, 10];
  } else if (cl.endsWith("m7")) {
    body = core.slice(0, -2);
    intervals = [0, 3, 7, 10];
  } else if (cl.endsWith("sus4")) {
    body = core.slice(0, -4);
    intervals = [0, 5, 7];
  } else if (cl.endsWith("sus2")) {
    body = core.slice(0, -4);
    intervals = [0, 2, 7];
  } else if (cl.endsWith("dim")) {
    body = core.slice(0, -3);
    intervals = [0, 3, 6];
  } else if (cl.endsWith("aug")) {
    body = core.slice(0, -3);
    intervals = [0, 4, 8];
  } else if (core.length > 1 && core.endsWith("+") && !core.toLowerCase().endsWith("aug")) {
    body = core.slice(0, -1);
    intervals = [0, 4, 8];
  } else if (cl.length >= 3 && cl.endsWith("maj") && !cl.endsWith("maj7")) {
    body = core.slice(0, -3);
    intervals = [0, 4, 7];
  } else if (core.length >= 2 && core.endsWith("7") && !cl.endsWith("m7") && !cl.endsWith("maj7")) {
    body = core.slice(0, -1);
    intervals = [0, 4, 7, 10];
  } else if (core.length > 1 && core.endsWith("m") && !cl.endsWith("maj")) {
    body = core.slice(0, -1);
    intervals = [0, 3, 7];
  } else {
    body = core;
    intervals = [0, 4, 7];
  }

  if (!(body in ROOT_ALIASES)) {
    return [];
  }
  const rootPc = ROOT_ALIASES[body];
  const preferFlat = preferFlatSpelling(body, intervals);
  return intervals.map((iv) => pcDisplay(rootPc + iv, preferFlat));
}

/** Short UI line under the chord symbol ( Analyze File ). */
export function chordNotesDisplayForLabel(label: string): string {
  const t = label.trim();
  if (!t || t.toUpperCase() === "N") {
    return "—";
  }
  const notes = spellChordTones(t);
  if (notes.length) {
    return notes.join(" · ");
  }
  return "Check this one by ear";
}

export function playableChordNotesAndHint(symbol: string): { notes: string[]; hint: string } {
  const notes = spellChordTones(symbol);
  if (!notes.length) {
    if (!symbol || !String(symbol).trim() || String(symbol).trim().toUpperCase() === "N") {
      return { notes: [], hint: "—" };
    }
    return { notes: [], hint: "Check this one by ear — unfamiliar or abbreviated symbol." };
  }
  const line = notes.join(" · ");
  return {
    notes,
    hint: `${line} (simplified close spelling, any octave; check by ear if unsure)`,
  };
}
