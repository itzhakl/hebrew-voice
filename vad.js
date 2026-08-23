'use strict';

/* Energy-VAD endpointing. Decides WHEN to commit a segment; the audio itself
 * streams to the provider continuously and is never buffered here. */

const DEFAULTS = {
  sampleRate: 16000,
  // The absolute floor, and the only threshold used until the room has been
  // measured. A microphone whose own noise sits above this never falls silent,
  // so nothing ever commits and dictation only lands when the mic is released.
  vadThreshold: 0.005,
  // Speech has to be this much louder than the room. Three is about 10 dB.
  noiseRatio: 3,
  // Claude opens the socket and starts streaming the moment the mic opens,
  // and nobody begins talking inside the first third of a second, so this
  // window is the room rather than the speaker.
  calibrationMs: 300,
  // The wire carries 20 ms frames. Anything much coarser is a caller feeding
  // us whole seconds at a time - a test, or a file - and one such frame is not
  // a measurement of a room, so calibration is skipped and the absolute
  // threshold stands.
  calibrationMinFrames: 8,
  endpointMs: 600,
  // Scribe finalizes long Hebrew sentences on its own; a short cap only chops
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
    // Zero means "not measured yet" - the absolute threshold is used until it is.
    this.noiseFloor = 0;
    this.calibration = [];
    this.calibrationSamples = 0;
    this.calibrated = false;
  }

  get speaking() {
    return this.spoke;
  }

  /* What the next frame has to beat to count as speech. */
  get threshold() {
    if (!this.calibrated) return this.o.vadThreshold;
    return Math.max(this.o.vadThreshold, this.noiseFloor * this.o.noiseRatio);
  }

  /* The room survives a commit: it is a property of the microphone and the
   * place, not of the sentence that just ended. */
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
    this._learn(rms, pcm.length);
    const threshold = this.threshold;
    if (rms > threshold) {
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

  _learn(rms, samples) {
    if (!this.calibrated) {
      this.calibration.push(rms);
      this.calibrationSamples += samples;
      if (this.calibrationSamples < this._samples(this.o.calibrationMs)) return;
      if (this.calibration.length >= this.o.calibrationMinFrames) {
        // The quietest frame in the window, not the average: a cough or a
        // chair inside the first 300 ms would otherwise be learned as the room.
        this.noiseFloor = Math.min(...this.calibration);
        this.calibrated = true;
        // Those frames were judged against the absolute threshold, which a
        // noisy room clears - so the room has already opened a segment that,
        // 600 ms later, commits itself as if someone had spoken. If nothing in
        // the window beats what we now know the threshold to be, none of it was
        // speech and the segment it opened never happened.
        if (this.spoke && Math.max(...this.calibration) <= this.threshold) this.reset();
      }
      this.calibration.length = 0;
      this.calibrationSamples = 0;
      return;
    }
    // A room that went quiet is believed at once - a fan switching off must not
    // cost several seconds of dead endpointing. One that got louder is believed
    // slowly, and only from audio already judged not to be speech, so a held
    // vowel cannot raise the threshold over the speaker saying it.
    if (rms < this.noiseFloor) this.noiseFloor = this.noiseFloor * 0.8 + rms * 0.2;
    else if (rms <= this.threshold) this.noiseFloor = this.noiseFloor * 0.995 + rms * 0.005;
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
