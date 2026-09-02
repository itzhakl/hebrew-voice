'use strict';

/* Gemini 3.5 Transcribe (non-streaming) - the accurate half of the pair.
 *
 * The Interactions API takes a whole utterance in one POST and answers with
 * the model's best reading of it: 2.6% WER against the live model's 4.0%,
 * because nothing has to be decided before the speech is over. There are no
 * interims here by construction, so this provider is only useful behind the
 * hybrid, where the live engine paints and this one commits.
 *
 * Wire reference: POST https://generativelanguage.googleapis.com/v1beta/interactions
 * body -> {model, input:[{type:'audio',data,mime_type:'audio/wav'}],
 *          generation_config:{transcription_config:{language_codes,custom_vocabulary,mode}}}
 * reply -> {output_text} | {steps:[{content:[{type:'text',text}]}]} | {error:{…}}
 */

const { parseGeminiCredential, mapGeminiError, languageCodes, vocabularyList, SAMPLE_RATE } = require('./gemini');

const DEFAULT_ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/interactions';
const DEFAULT_MODEL = 'gemini-3.5-transcribe';

/* Measured against the live API, not read off the docs: `audio/l16` with
 * `sample_rate`/`channels` beside it is rejected with a generic
 * "Request contains an invalid argument", while the same PCM behind a 44-byte
 * WAV header returns 200. The rate and channel count then travel in the
 * header, and sending them as fields as well is rejected in turn ("Rate and
 * channels are only supported for TYPE_L16 audio"). */
const MIME_TYPE = 'audio/wav';

const WAV_HEADER_BYTES = 44;

function wavHeader(byteLength) {
  const h = Buffer.alloc(WAV_HEADER_BYTES);
  h.write('RIFF', 0);
  h.writeUInt32LE(36 + byteLength, 4);
  h.write('WAVE', 8);
  h.write('fmt ', 12);
  h.writeUInt32LE(16, 16);
  h.writeUInt16LE(1, 20);
  h.writeUInt16LE(1, 22);
  h.writeUInt32LE(SAMPLE_RATE, 24);
  h.writeUInt32LE(SAMPLE_RATE * 2, 28);
  h.writeUInt16LE(2, 32);
  h.writeUInt16LE(16, 34);
  h.write('data', 36);
  h.writeUInt32LE(byteLength, 40);
  return h;
}

function toWav(pcm) {
  return Buffer.concat([wavHeader(pcm.length), pcm]);
}

/* An utterance of dictation is seconds long; anything past this is a runaway
 * segment and posting it would cost more than dropping it. */
const MAX_AUDIO_MS = 120000;
const BYTES_PER_MS = (SAMPLE_RATE * 2) / 1000;

/* The transcript sits at output_text on a completed interaction, but the same
 * text is also reachable through the step list, and which one arrives has
 * changed once already during the preview. Read both. */
function readTranscript(body) {
  if (!body || typeof body !== 'object') return '';
  if (typeof body.output_text === 'string') return body.output_text.trim();
  const steps = Array.isArray(body.steps) ? body.steps : [];
  const parts = [];
  for (const step of steps) {
    for (const item of (step && step.content) || []) {
      if (item && item.type === 'text' && typeof item.text === 'string') parts.push(item.text);
    }
  }
  return parts.join(' ').trim();
}

function buildBody(opts, audio) {
  const codes = languageCodes(opts);
  const vocabulary = vocabularyList(opts.keyterms);
  const transcription = { mode: opts.noVerbatim ? 'smart' : 'verbatim' };
  if (codes.length) transcription.language_codes = codes;
  // custom_vocabulary and diarization are mutually exclusive; dictation has
  // one speaker, so the vocabulary is the half worth keeping.
  if (vocabulary.length) transcription.custom_vocabulary = vocabulary;
  return {
    model: opts.model || DEFAULT_MODEL,
    input: [{ type: 'audio', data: toWav(audio).toString('base64'), mime_type: MIME_TYPE }],
    generation_config: { transcription_config: transcription }
  };
}

class GeminiTranscribeProvider {
  /* postFn is injectable so the request shape can be tested without the API. */
  constructor(opts, postFn) {
    this.id = 'gemini-transcribe';
    this.opts = opts || {};
    this.post =
      postFn ||
      (async (body) => {
        const res = await fetch(this.opts.endpoint || DEFAULT_ENDPOINT, {
          method: 'POST',
          headers: {
            'x-goog-api-key': parseGeminiCredential(this.opts.credential),
            'content-type': 'application/json'
          },
          body: JSON.stringify(body)
        });
        const parsed = await res.json().catch(() => ({}));
        if (!res.ok) {
          const err = parsed && parsed.error ? parsed : { error: { code: res.status, message: res.statusText } };
          throw mapGeminiError(err);
        }
        return parsed;
      });
  }

  async createSession(cb) {
    const log = typeof this.opts.log === 'function' ? this.opts.log : () => {};
    const maxBytes = Math.round(MAX_AUDIO_MS * BYTES_PER_MS);
    let audio = [];
    let bytes = 0;
    let dead = false;
    let text = '';
    let inFlight = null;

    const transcribe = async () => {
      const pcm = audio.length === 1 ? audio[0] : Buffer.concat(audio);
      audio = [];
      bytes = 0;
      if (!pcm.length) return;
      const started = Date.now();
      try {
        const body = await this.post(buildBody(this.opts, pcm));
        text = readTranscript(body);
        log(`transcribed ${Math.round(pcm.length / BYTES_PER_MS)}ms of audio in ${Date.now() - started}ms`);
      } catch (e) {
        dead = true;
        // The hybrid falls back to the live engine's text on this, so the
        // failure must be reported without also blanking the segment.
        cb.onError(e && e.message ? e : mapGeminiError(e));
      }
    };

    return {
      sendAudio: (pcm) => {
        if (dead || inFlight) return;
        if (bytes >= maxBytes) return;
        audio.push(pcm);
        bytes += pcm.length;
      },
      flush: () => {
        if (dead || inFlight) return;
        inFlight = transcribe();
      },
      endSegment: async () => {
        if (inFlight) await inFlight;
        const out = text;
        text = '';
        inFlight = null;
        return out;
      },
      close: async () => {
        dead = true;
        if (inFlight) await inFlight.catch(() => {});
      }
    };
  }
}

module.exports = {
  GeminiTranscribeProvider,
  toWav,
  buildBody,
  readTranscript,
  DEFAULT_MODEL,
  DEFAULT_ENDPOINT,
  MAX_AUDIO_MS
};
