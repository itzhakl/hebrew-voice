'use strict';

/* Energy-VAD endpointing. Decides WHEN to commit a segment; the audio itself
 * streams to the provider continuously and is never buffered here. */

const DEFAULTS = {
  sampleRate: 16000,
  vadThreshold: 0.005,
  endpointMs: 600,
  // Chirp finalizes long Hebrew sentences on its own; a short cap only chops
  // them mid-thought, so this sits well above the extension's 4000.
  maxSegmentMs: 12000,
  minUtteranceMs: 300
};

class Endpointer {
  constructor(options = {}) {
    this.o = Object.assign({}, DEFAULTS, options);
    this.spoke = false;
    this.silenceSamples = 0;
    this.segSamples = 0;
  }

  get speaking() {
    return this.spoke;
  }

  reset() {
    this.spoke = false;
    this.silenceSamples = 0;
    this.segSamples = 0;
  }

  pushFrame(pcm) {
    let sum = 0;
    for (let i = 0; i < pcm.length; i++) {
      const s = pcm[i] / 32768;
      sum += s * s;
    }
    const rms = pcm.length ? Math.sqrt(sum / pcm.length) : 0;
    if (rms > this.o.vadThreshold) {
      this.spoke = true;
      this.silenceSamples = 0;
    } else if (this.spoke) {
      this.silenceSamples += pcm.length;
    }
    if (this.spoke) this.segSamples += pcm.length;
    if (this.spoke && this.silenceSamples >= this._samples(this.o.endpointMs)) {
      return this._finish('silence');
    }
    if (this.spoke && this.segSamples >= this._samples(this.o.maxSegmentMs)) {
      return this._finish('max-segment');
    }
    return {};
  }

  _finish(reason) {
    const utteranceSamples = this.segSamples - this.silenceSamples;
    this.reset();
    if (utteranceSamples < this._samples(this.o.minUtteranceMs)) return { discarded: true };
    return { commit: reason };
  }

  _samples(ms) {
    return (this.o.sampleRate * ms) / 1000;
  }
}

module.exports = { Endpointer, DEFAULTS };
