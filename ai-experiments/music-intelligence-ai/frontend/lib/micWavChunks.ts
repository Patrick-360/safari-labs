/**
 * Microphone → fixed-duration mono PCM WAV blobs (ScriptProcessor path).
 * Replace this module with an AudioWorklet-based implementation later; keep the same exports.
 */

export type MicWavChunk = {
  blob: Blob;
  sampleRate: number;
};

export type StartMicWavChunksOptions = {
  chunkSeconds?: number;
  /** Emit trailing audio on stop only if at least this many seconds remain. */
  tailMinSeconds?: number;
  onChunk: (chunk: MicWavChunk) => void | Promise<void>;
  onError?: (err: Error) => void;
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

/**
 * Opens the mic and calls `onChunk` with WAV blobs (~`chunkSeconds` each).
 * Tear down with `stop()` on the returned handle.
 */
export async function startMicWavChunks(
  options: StartMicWavChunksOptions,
): Promise<{ stop: () => Promise<void> }> {
  const chunkSeconds = options.chunkSeconds ?? 2;
  const tailMinSeconds = options.tailMinSeconds ?? 0.2;
  const { onChunk, onError } = options;

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });

  const AudioContextClass = resolveAudioContext();
  const audioCtx = new AudioContextClass();
  const source = audioCtx.createMediaStreamSource(stream);
  const processor = audioCtx.createScriptProcessor(4096, 1, 1);
  const gain = audioCtx.createGain();
  gain.gain.value = 0;

  const pending: Float32Array[] = [];
  const chunkSamples = Math.floor(audioCtx.sampleRate * chunkSeconds);

  const emitChunk = (samples: Float32Array, sampleRate: number) => {
    const pcm = floatTo16BitPCM(samples);
    const blob = encodeWavPcm16Mono(pcm, sampleRate);
    return Promise.resolve(onChunk({ blob, sampleRate }));
  };

  processor.onaudioprocess = (event) => {
    try {
      const input = event.inputBuffer.getChannelData(0);
      const copy = new Float32Array(input.length);
      copy.set(input);
      pending.push(copy);

      while (queuedSampleCount(pending) >= chunkSamples) {
        const merged = takeSamples(pending, chunkSamples);
        void emitChunk(merged, audioCtx.sampleRate).catch((e) => {
          const err = e instanceof Error ? e : new Error(String(e));
          onError?.(err);
        });
      }
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      onError?.(err);
    }
  };

  source.connect(processor);
  processor.connect(gain);
  gain.connect(audioCtx.destination);

  return {
    stop: async () => {
      processor.onaudioprocess = null;
      processor.disconnect();
      gain.disconnect();
      stream.getTracks().forEach((t) => t.stop());

      const sampleRate = audioCtx.sampleRate;
      const remaining = queuedSampleCount(pending);
      const minFlush = Math.floor(sampleRate * tailMinSeconds);
      if (remaining >= minFlush) {
        const merged = takeSamples(pending, remaining);
        await emitChunk(merged, sampleRate);
      } else {
        pending.length = 0;
      }

      await audioCtx.close();
    },
  };
}
