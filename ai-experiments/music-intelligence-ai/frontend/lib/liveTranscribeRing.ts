/**
 * Rolling mono buffer for live transcription: keep last N seconds of boosted mic audio.
 */

export class LiveTranscribeRing {
  private chunks: Float32Array[] = [];
  private total = 0;
  private sampleRate = 48000;

  constructor(private readonly maxSeconds: number) {}

  push(frame: Float32Array, sr: number): void {
    if (sr > 0) {
      this.sampleRate = sr;
    }
    const f = frame.length ? frame.slice() : new Float32Array(0);
    if (!f.length) {
      return;
    }
    this.chunks.push(f);
    this.total += f.length;
    const cap = Math.floor(this.sampleRate * this.maxSeconds);
    while (this.total > cap && this.chunks.length) {
      const h = this.chunks[0];
      this.total -= h.length;
      this.chunks.shift();
    }
  }

  getSampleRate(): number {
    return this.sampleRate;
  }

  /** Total seconds currently buffered. */
  bufferedSeconds(): number {
    return this.sampleRate > 0 ? this.total / this.sampleRate : 0;
  }

  /** Concatenate last `seconds` of audio (or less if not enough yet). */
  sliceLastSeconds(seconds: number): Float32Array {
    const need = Math.floor(this.sampleRate * seconds);
    const len = Math.min(need, this.total);
    if (len <= 0) {
      return new Float32Array(0);
    }
    const out = new Float32Array(len);
    let pos = len;
    for (let i = this.chunks.length - 1; i >= 0 && pos > 0; i--) {
      const c = this.chunks[i];
      const take = Math.min(pos, c.length);
      const start = c.length - take;
      out.set(c.subarray(start, start + take), pos - take);
      pos -= take;
    }
    return out;
  }

  clear(): void {
    this.chunks = [];
    this.total = 0;
  }
}
