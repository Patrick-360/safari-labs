"use client";

import { useCallback, useRef, useState } from "react";

import { startMicWavChunks } from "@/lib/micWavChunks";

const CHUNK_SECONDS = 0.5;
const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

const CHORD_HISTORY_MAX = 12;

type StreamResponse = {
  chord: string;
  confidence: number;
  key: string;
  key_confidence: number;
  timestamp: number;
};

/** Backend margin scores are in [0, 1]. */
function confidenceLevel(value: number): "Low" | "Medium" | "High" {
  if (value >= 0.5) return "High";
  if (value >= 0.2) return "Medium";
  return "Low";
}

export default function Home() {
  const [recording, setRecording] = useState(false);
  const [chord, setChord] = useState("—");
  const [confidence, setConfidence] = useState<number | null>(null);
  const [key, setKey] = useState("—");
  const [keyConfidence, setKeyConfidence] = useState<number | null>(null);
  const [chordHistory, setChordHistory] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const sessionRef = useRef<{ stop: () => Promise<void> } | null>(null);
  const lastTsRef = useRef(0);

  const applyResponse = useCallback((data: StreamResponse) => {
    if (data.timestamp <= lastTsRef.current) {
      return;
    }
    lastTsRef.current = data.timestamp;
    setChord(data.chord);
    setConfidence(data.confidence);
    setKey(data.key);
    setKeyConfidence(data.key_confidence);
    setChordHistory((prev) => {
      if (prev[0] === data.chord) {
        return prev;
      }
      return [data.chord, ...prev].slice(0, CHORD_HISTORY_MAX);
    });
  }, []);

  const sendWav = useCallback(
    async (blob: Blob) => {
      const form = new FormData();
      form.append("file", blob, "chunk.wav");

      const res = await fetch(`${API_BASE}/stream`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`${res.status} ${res.statusText}: ${text}`);
      }

      const data = (await res.json()) as StreamResponse;
      applyResponse(data);
    },
    [applyResponse],
  );

  const startRecording = useCallback(async () => {
    setError(null);
    setStatus(null);
    lastTsRef.current = 0;
    setChordHistory([]);

    try {
      const session = await startMicWavChunks({
        chunkSeconds: CHUNK_SECONDS,
        tailMinSeconds: 0.2,
        onChunk: ({ blob }) =>
          sendWav(blob).catch((e) => {
            const message = e instanceof Error ? e.message : String(e);
            setError(message);
          }),
        onError: (err) => setError(err.message),
      });
      sessionRef.current = session;
      setRecording(true);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setError(message);
    }
  }, [sendWav]);

  const stopRecording = useCallback(async () => {
    const session = sessionRef.current;
    sessionRef.current = null;
    if (session) {
      await session.stop();
    }
    setRecording(false);
    setStatus("Stopped.");
  }, []);

  return (
    <main className="demo">
      <header className="hero">
        <h1>Live chord recognition</h1>
        <p className="hero-sub">Real-time harmony from your microphone</p>
      </header>

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
        <p className="chord-stage-value">{chord}</p>
        <p className="chord-stage-confidence">
          Chord confidence:{" "}
          {confidence === null ? "—" : confidenceLevel(confidence)}
        </p>
      </section>

      <section className="details" aria-label="Key and confidence">
        <div className="detail-grid">
          <div className="detail-block">
            <span className="detail-label">Key</span>
            <span className="detail-value">{key}</span>
          </div>
          <div className="detail-block">
            <span className="detail-label">Key confidence</span>
            <span className="detail-value">
              {keyConfidence === null ? "—" : confidenceLevel(keyConfidence)}
            </span>
          </div>
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

      <p className="meta-footer">
        API: <code>{API_BASE}</code> — set <code>NEXT_PUBLIC_API_URL</code> to override
      </p>
    </main>
  );
}
