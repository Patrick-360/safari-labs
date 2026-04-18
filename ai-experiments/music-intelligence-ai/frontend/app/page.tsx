"use client";

import { useCallback, useRef, useState } from "react";

import { startMicWavChunks } from "@/lib/micWavChunks";

const CHUNK_SECONDS = 2;
const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

type StreamResponse = {
  chord: string;
  confidence: number;
  key: string;
  key_confidence: number;
  timestamp: number;
};

export default function Home() {
  const [recording, setRecording] = useState(false);
  const [chord, setChord] = useState("—");
  const [confidence, setConfidence] = useState<number | null>(null);
  const [key, setKey] = useState("—");
  const [keyConfidence, setKeyConfidence] = useState<number | null>(null);
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
    <main>
      <h1>Live chord</h1>
      <p>
        <button type="button" onClick={() => void startRecording()} disabled={recording}>
          Start Recording
        </button>
        <button type="button" onClick={() => void stopRecording()} disabled={!recording}>
          Stop Recording
        </button>
      </p>
      {status ? <p>{status}</p> : null}
      {error ? <p className="error">{error}</p> : null}
      <dl>
        <dt>Current chord</dt>
        <dd>{chord}</dd>
        <dt>Confidence</dt>
        <dd>{confidence === null ? "—" : confidence.toFixed(3)}</dd>
        <dt>Key</dt>
        <dd>{key}</dd>
        <dt>Key confidence</dt>
        <dd>{keyConfidence === null ? "—" : keyConfidence.toFixed(3)}</dd>
      </dl>
      <p style={{ fontSize: "0.85rem", color: "#555" }}>
        API: <code>{API_BASE}</code> (set <code>NEXT_PUBLIC_API_URL</code> to override)
      </p>
    </main>
  );
}
