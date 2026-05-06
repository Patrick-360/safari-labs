/**
 * Microphone → fixed-duration mono PCM WAV blobs (ScriptProcessor path).
 * Replace this module with an AudioWorklet-based implementation later; keep the same exports.
 */

export type MicWavChunk = {
  blob: Blob;
  sampleRate: number;
};

/** Snapshotted MediaStreamTrack fields for Live debug (serializable). */
export type MicTrackDebugSnapshot = {
  trackCount: number;
  label: string;
  enabled: boolean;
  muted: boolean;
  readyState: string;
  settingsDeviceId: string | undefined;
  settingsSampleRate: number | undefined;
  settingsChannelCount: number | undefined;
  settingsEchoCancellation: boolean | undefined;
  settingsNoiseSuppression: boolean | undefined;
  settingsAutoGainControl: boolean | undefined;
};

function buildTrackSnapshot(stream: MediaStream): MicTrackDebugSnapshot {
  const tracks = stream.getAudioTracks();
  const t = tracks[0];
  const s = t?.getSettings?.() ?? {};
  return {
    trackCount: tracks.length,
    label: t?.label ?? "—",
    enabled: t?.enabled ?? false,
    muted: t?.muted ?? false,
    readyState: t?.readyState != null ? String(t.readyState) : "ended",
    settingsDeviceId: s.deviceId,
    settingsSampleRate: s.sampleRate,
    settingsChannelCount: s.channelCount,
    settingsEchoCancellation: s.echoCancellation,
    settingsNoiseSuppression: s.noiseSuppression,
    settingsAutoGainControl: s.autoGainControl,
  };
}

export type StartMicWavChunksOptions = {
  chunkSeconds?: number;
  /** Emit trailing audio on stop only if at least this many seconds remain. */
  tailMinSeconds?: number;
  /** When set, requests this capture device (after permission). */
  deviceId?: string;
  /**
   * Multiply mono capture by this factor before enqueueing WAV samples.
   * Output to speakers stays silent (`gain` node at 0); only analysis path is boosted.
   * Typical: 1–8. Values outside [1,32] are clamped for safety.
   */
  inputBoost?: number;
  /**
   * If false, do not build fixed-duration WAV chunks for `onChunk` (rolling-buffer modes only).
   * Still runs `onMonoFrames` each callback when set.
   */
  streamChunks?: boolean;
  /**
   * Each ScriptProcessor buffer: boosted mono + sampleRate (for rolling live transcription ring).
   */
  onMonoFrames?: (mono: Float32Array, sampleRate: number) => void;
  onChunk?: (chunk: MicWavChunk) => void | Promise<void>;
  onError?: (err: Error) => void;
  /** Optional trace for live-mode diagnostics (keep quiet in production). */
  onDebug?: (message: string, detail?: Record<string, unknown>) => void;
  /** Called when the audio track state/settings should be re-read (mute, device, etc.). */
  onTrackSnapshot?: (snapshot: MicTrackDebugSnapshot) => void;
  /**
   * Low-rate input levels + context state (throttle in the consumer if you drive React state).
   * `inputPeak` / `inputRms` are pre-boost downmixed mono; boosted* are after gain + clamp to [-1,1].
   */
  onTelemetry?: (info: {
    audioContextState: AudioContextState;
    callbackIndex: number;
    inputPeak: number;
    inputRms: number;
    inputFrames: number;
    inputChannels: number;
    inputBoost: number;
    boostedPeak: number;
    boostedRms: number;
    clippedSamplesInBuffer: number;
    clippedFractionInBuffer: number;
  }) => void;
  /**
   * Fires on every ScriptProcessor callback (cheap — use to prove `onaudioprocess` is running).
   * Prefer batching to React state in the consumer.
   */
  onProcessTick?: (info: { callbackIndex: number }) => void;
  /**
   * Build the capture graph on this context; it will NOT be closed in `stop()`.
   * Create with `new AudioContext()` and call `resume()` in the same user gesture as Start (before awaits).
   */
  audioContext?: AudioContext;
};

function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

function floatTo16BitPCM(float32: Float32Array): Int16Array {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff);
  }
  return out;
}

function encodeWavPcm16Mono(samples: Int16Array, sampleRate: number): Blob {
  const numChannels = 1;
  const bitsPerSample = 16;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    view.setInt16(offset, samples[i], true);
  }

  return new Blob([buffer], { type: "audio/wav" });
}

/** Mono float samples → PCM16 WAV blob (for rolling-window POST). */
export function encodeFloat32MonoToWav(samples: Float32Array, sampleRate: number): Blob {
  return encodeWavPcm16Mono(floatTo16BitPCM(samples), sampleRate);
}

/**
 * Downmix all `inputBuffer` channels to mono for WAV + level metering.
 * Some devices expose a stereo stream with signal only on channel 1 — reading only ch0 yields silence.
 */
function monoFromInputBuffer(inputBuffer: AudioBuffer): { mono: Float32Array; peak: number; rms: number } {
  const nCh = inputBuffer.numberOfChannels;
  const len = inputBuffer.length;
  const mono = new Float32Array(len);
  let peak = 0;
  let sumSq = 0;
  const denom = Math.max(1, nCh);
  for (let i = 0; i < len; i++) {
    let acc = 0;
    for (let c = 0; c < nCh; c++) {
      acc += inputBuffer.getChannelData(c)[i];
    }
    const v = acc / denom;
    mono[i] = v;
    const a = Math.abs(v);
    if (a > peak) {
      peak = a;
    }
    sumSq += v * v;
  }
  const rms = len > 0 ? Math.sqrt(sumSq / len) : 0;
  return { mono, peak, rms };
}

/** Apply linear gain for analysis/WAV path; clamp to [-1,1]; count samples that required clamping. */
function boostMonoForAnalysis(
  mono: Float32Array,
  linearGain: number,
): { boosted: Float32Array; boostedPeak: number; boostedRms: number; clippedSamples: number } {
  const n = mono.length;
  const boosted = new Float32Array(n);
  let boostedPeak = 0;
  let sumSq = 0;
  let clippedSamples = 0;
  for (let i = 0; i < n; i++) {
    const pre = mono[i] * linearGain;
    if (pre > 1 || pre < -1) {
      clippedSamples += 1;
    }
    const v = Math.max(-1, Math.min(1, pre));
    boosted[i] = v;
    const a = Math.abs(v);
    if (a > boostedPeak) {
      boostedPeak = a;
    }
    sumSq += v * v;
  }
  const boostedRms = n > 0 ? Math.sqrt(sumSq / n) : 0;
  return { boosted, boostedPeak, boostedRms, clippedSamples };
}

function takeSamples(queue: Float32Array[], count: number): Float32Array {
  const out = new Float32Array(count);
  let written = 0;
  while (written < count) {
    const head = queue[0];
    if (!head) {
      break;
    }
    const need = count - written;
    if (head.length <= need) {
      out.set(head, written);
      written += head.length;
      queue.shift();
    } else {
      out.set(head.subarray(0, need), written);
      queue[0] = head.subarray(need);
      written += need;
    }
  }
  return out;
}

function queuedSampleCount(queue: Float32Array[]): number {
  let n = 0;
  for (const chunk of queue) {
    n += chunk.length;
  }
  return n;
}

function resolveAudioContext(): typeof AudioContext {
  return (
    window.AudioContext ||
    (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
  );
}

function processorInputChannelCount(stream: MediaStream): number {
  const t = stream.getAudioTracks()[0];
  const n = t?.getSettings?.().channelCount;
  if (typeof n === "number" && n >= 1) {
    return Math.min(8, Math.max(2, n));
  }
  /** Stereo-safe default: ch0-only ScriptProcessors often see silence when hardware uses ch1. */
  return 2;
}

/**
 * Opens the mic and calls `onChunk` with WAV blobs (~`chunkSeconds` each).
 * Tear down with `stop()` on the returned handle.
 */
export async function startMicWavChunks(
  options: StartMicWavChunksOptions,
): Promise<{ stop: () => Promise<void> }> {
  const chunkSeconds = options.chunkSeconds ?? 2;
  const tailMinSeconds = options.tailMinSeconds ?? 0.2;
  const {
    onChunk,
    onError,
    onDebug,
    onTelemetry,
    onProcessTick,
    onTrackSnapshot,
    onMonoFrames,
    audioContext: providedCtx,
    deviceId,
    inputBoost: requestedBoost,
    streamChunks: streamChunksOpt = true,
  } = options;
  const streamChunks = streamChunksOpt !== false;
  const inputBoost = Math.min(32, Math.max(1, requestedBoost ?? 1));
  const ownsAudioContext = providedCtx == null;

  /**
   * When no external context: create + resume BEFORE the first `await` on getUserMedia so the
   * click's user-activation chain is not broken (otherwise the context can stay suspended and
   * ScriptProcessor never runs).
   */
  const AudioContextClass = resolveAudioContext();
  const audioCtx = providedCtx ?? new AudioContextClass();

  if (audioCtx.state === "closed") {
    throw new Error("AudioContext is closed; cannot start microphone capture.");
  }

  if (ownsAudioContext) {
    onDebug?.("audio_context_created", { state: audioCtx.state, sampleRate: audioCtx.sampleRate });
    if (audioCtx.state === "suspended") {
      await audioCtx.resume();
      onDebug?.("audio_context_resume_after_suspended", { state: audioCtx.state });
    }
  } else {
    onDebug?.("audio_context_reused", { state: audioCtx.state, sampleRate: audioCtx.sampleRate });
  }

  const audioConstraints: MediaTrackConstraints = {
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
  };
  if (deviceId) {
    audioConstraints.deviceId = { ideal: deviceId };
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: audioConstraints,
  });

  const audioTracks = stream.getAudioTracks();
  for (const tr of audioTracks) {
    tr.enabled = true;
  }

  const pushTrackSnapshot = () => onTrackSnapshot?.(buildTrackSnapshot(stream));
  pushTrackSnapshot();
  for (const tr of audioTracks) {
    tr.addEventListener("mute", pushTrackSnapshot);
    tr.addEventListener("unmute", pushTrackSnapshot);
    tr.addEventListener("ended", pushTrackSnapshot);
  }

  onDebug?.("get_user_media_ok", { ...buildTrackSnapshot(stream) });

  if (audioCtx.state === "suspended") {
    await audioCtx.resume();
    onDebug?.("audio_context_resume_after_mic", { state: audioCtx.state });
  }

  const source = audioCtx.createMediaStreamSource(stream);
  const inCh = processorInputChannelCount(stream);
  const processor = audioCtx.createScriptProcessor(4096, inCh, 1);
  const gain = audioCtx.createGain();
  gain.gain.value = 0;

  /** Hold references until stop() so the graph is not GC'd mid-capture. */
  const keepAlive = { source, processor, gain, stream };

  const pending: Float32Array[] = [];
  const chunkSamples = Math.floor(audioCtx.sampleRate * chunkSeconds);

  const emitChunk = (samples: Float32Array, sampleRate: number) => {
    const pcm = floatTo16BitPCM(samples);
    const blob = encodeWavPcm16Mono(pcm, sampleRate);
    return onChunk ? Promise.resolve(onChunk({ blob, sampleRate })) : Promise.resolve();
  };

  let processCbCount = 0;
  processor.onaudioprocess = (event) => {
    try {
      processCbCount += 1;
      onProcessTick?.({ callbackIndex: processCbCount });

      const ib = event.inputBuffer;
      const { mono, peak, rms } = monoFromInputBuffer(ib);
      const { boosted, boostedPeak, boostedRms, clippedSamples } = boostMonoForAnalysis(mono, inputBoost);
      const clippedFractionInBuffer = mono.length > 0 ? clippedSamples / mono.length : 0;

      const out = event.outputBuffer.getChannelData(0);
      out.fill(0);

      if (processCbCount <= 3 || processCbCount % 24 === 0) {
        onTelemetry?.({
          audioContextState: audioCtx.state,
          callbackIndex: processCbCount,
          inputPeak: peak,
          inputRms: rms,
          inputFrames: ib.length,
          inputChannels: ib.numberOfChannels,
          inputBoost,
          boostedPeak,
          boostedRms,
          clippedSamplesInBuffer: clippedSamples,
          clippedFractionInBuffer,
        });
      }
      if (processCbCount === 1) {
        const perChPeak: number[] = [];
        for (let c = 0; c < ib.numberOfChannels; c++) {
          const ch = ib.getChannelData(c);
          let pc = 0;
          for (let i = 0; i < ch.length; i++) {
            const a = Math.abs(ch[i]);
            if (a > pc) {
              pc = a;
            }
          }
          perChPeak.push(pc);
        }
        onDebug?.("first_onaudioprocess", {
          state: audioCtx.state,
          frames: ib.length,
          inputChannels: ib.numberOfChannels,
          processorInputChannels: inCh,
          peakDownmix: peak,
          rmsDownmix: rms,
          inputBoost,
          boostedPeak,
          boostedRms,
          clippedFractionInBuffer,
          perChannelPeak: perChPeak,
        });
      }

      if (audioCtx.state === "suspended") {
        void audioCtx.resume();
      }

      const copy = new Float32Array(boosted.length);
      copy.set(boosted);
      onMonoFrames?.(new Float32Array(copy), audioCtx.sampleRate);

      if (streamChunks) {
        pending.push(copy);

        while (queuedSampleCount(pending) >= chunkSamples) {
          const merged = takeSamples(pending, chunkSamples);
          onDebug?.("chunk_ready", { samples: merged.length, sampleRate: audioCtx.sampleRate });
          void emitChunk(merged, audioCtx.sampleRate).catch((e) => {
            const err = e instanceof Error ? e : new Error(String(e));
            onError?.(err);
          });
        }
      }
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      onError?.(err);
    }
  };

  keepAlive.source.connect(keepAlive.processor);
  keepAlive.processor.connect(keepAlive.gain);
  keepAlive.gain.connect(audioCtx.destination);
  onDebug?.("graph_connected", {
    state: audioCtx.state,
    chunkSamples,
    chunkSeconds,
    scriptProcessorInputChannels: inCh,
    inputBoost,
  });

  const onVisibility = () => {
    if (!document.hidden && audioCtx.state === "suspended") {
      void audioCtx.resume();
      onDebug?.("audio_context_resume_visibility", { state: audioCtx.state });
    }
  };
  document.addEventListener("visibilitychange", onVisibility);

  return {
    stop: async () => {
      document.removeEventListener("visibilitychange", onVisibility);
      for (const tr of audioTracks) {
        tr.removeEventListener("mute", pushTrackSnapshot);
        tr.removeEventListener("unmute", pushTrackSnapshot);
        tr.removeEventListener("ended", pushTrackSnapshot);
      }

      keepAlive.processor.onaudioprocess = null;
      keepAlive.processor.disconnect();
      keepAlive.gain.disconnect();
      keepAlive.source.disconnect();
      keepAlive.stream.getTracks().forEach((t) => t.stop());

      const sampleRate = audioCtx.sampleRate;
      const remaining = queuedSampleCount(pending);
      const minFlush = Math.floor(sampleRate * tailMinSeconds);
      if (streamChunks && onChunk && remaining >= minFlush) {
        const merged = takeSamples(pending, remaining);
        await emitChunk(merged, sampleRate);
      } else {
        pending.length = 0;
      }

      if (ownsAudioContext) {
        await audioCtx.close();
      }
    },
  };
}
