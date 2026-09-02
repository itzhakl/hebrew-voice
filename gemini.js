'use strict';

/* Gemini 3.5 Transcribe Live - one WebSocket, JSON in, JSON out.
 *
 * The Live API's dedicated transcription path: audio in as base64 PCM,
 * `interimInputTranscription` while the speaker talks and `inputTranscription`
 * once the server endpoints the utterance. Same push shape as Scribe, so the
 * server and the endpointer around it stay untouched.
 *
 * Wire reference:
 * wss://generativelanguage.googleapis.com/ws/...BidiGenerateContent?key=…
 * client -> {setup:{…}} | {realtimeInput:{audio:{data,mimeType}}}
 *           | {realtimeInput:{audioStreamEnd:true}}
 * server -> {setupComplete:{}} | {serverContent:{interimInputTranscription|
 *           inputTranscription:{text}}} | {error|{code,message,status}}
 */

const DEFAULT_BASE_URL = 'wss://generativelanguage.googleapis.com';
const BIDI_PATH = '/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent';
const DEFAULT_MODEL = 'gemini-3.5-transcribe-live';
const SAMPLE_RATE = 16000;
const MIME_TYPE = `audio/pcm;rate=${SAMPLE_RATE}`;

/* Server-side limit on the biasing list. */
const MAX_VOCABULARY = 1000;

const DEFAULT_CHUNK_MS = 100;
const BYTES_PER_MS = (SAMPLE_RATE * 2) / 1000;

function sttError(message, hint) {
  const e = new Error(message);
  e.hint = hint;
  return e;
}

/* Google wants BCP-47 with a region: "he" alone is rejected by the Live API,
 * and "iw" is the legacy tag Cloud STT V2 uses - neither is what this one
 * takes. */
const REGIONS = {
  he: 'he-IL',
  iw: 'he-IL',
  en: 'en-US',
  ar: 'ar-EG',
  ru: 'ru-RU',
  fr: 'fr-FR',
  es: 'es-ES',
  de: 'de-DE'
};

function normalizeLanguage(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  if (raw.includes('-') || raw.includes('_')) return raw.replace('_', '-');
  const bare = raw.toLowerCase();
  return REGIONS[bare] || bare;
}

function toList(value) {
  if (value == null) return [];
  const raw = Array.isArray(value) ? value : String(value).split(',');
  return raw.map((v) => String(v).trim()).filter(Boolean);
}

/* Hebrew first, then whatever else the sentence code-switches into. An empty
 * list means auto-detect, which drifts on short Hebrew utterances. */
function languageCodes(opts = {}) {
  const primary = normalizeLanguage(opts.languageCode);
  const rest = toList(opts.secondaryLanguages).map(normalizeLanguage);
  return [...new Set([primary, ...rest].filter(Boolean))];
}

function vocabularyList(value) {
  return toList(value).slice(0, MAX_VOCABULARY);
}

function parseGeminiCredential(raw) {
  const trimmed = String(raw || '').trim();
  if (!trimmed) {
    throw sttError('No Gemini API key - run: hebrew-voice setup --provider gemini', 'set-api-key');
  }
  if (trimmed.startsWith('{')) {
    throw sttError(
      'That is service-account JSON, not a Gemini API key - the Live API takes an "AIza…" key',
      'set-api-key'
    );
  }
  if (/^sk_/.test(trimmed)) {
    throw sttError('That is an ElevenLabs API key, not a Gemini API key', 'set-api-key');
  }
  if (/\s/.test(trimmed)) {
    throw sttError('The Gemini API key contains whitespace - paste it on one line', 'set-api-key');
  }
  return trimmed;
}

function buildSetup(opts = {}) {
  const codes = languageCodes(opts);
  const vocabulary = vocabularyList(opts.keyterms);
  const inputAudioTranscription = {
    // SMART rewrites fillers and self-corrections; VERBATIM returns what was
    // actually said. Same meaning as the Scribe no_verbatim knob.
    mode: opts.noVerbatim ? 'SMART' : 'VERBATIM'
  };
  if (codes.length) inputAudioTranscription.languageCodes = codes;
  if (vocabulary.length) inputAudioTranscription.customVocabulary = vocabulary;
  return {
    setup: {
      model: `models/${opts.model || DEFAULT_MODEL}`,
      generationConfig: { responseModalities: ['TEXT'] },
      inputAudioTranscription
    }
  };
}

function buildRealtimeUrl(opts = {}) {
  const base = String(opts.baseUrl || DEFAULT_BASE_URL).replace(/\/+$/, '');
  const key = encodeURIComponent(parseGeminiCredential(opts.credential));
  return `${base}${BIDI_PATH}?key=${key}`;
}

/* Accepts a protocol error frame, a thrown socket error, or a close reason. */
function mapGeminiError(e) {
  const status = (e && (e.status || (e.error && e.error.status))) || '';
  const code = (e && (e.code || (e.error && e.error.code))) || 0;
  const text = String(
    (e && (e.message || (e.error && e.error.message))) || e || ''
  ).slice(0, 300);
  const m = `${status} ${text}`.toLowerCase();
  if (
    code === 401 ||
    code === 403 ||
    status === 'UNAUTHENTICATED' ||
    status === 'PERMISSION_DENIED' ||
    m.includes('api key') ||
    m.includes('unauthorized')
  ) {
    // Keep the raw server text: a disabled API and a wrong key read the same
    // from here, and only the server says which.
    return { message: `Gemini rejected the credential - ${text}`, hint: 'set-api-key' };
  }
  if (code === 429 || status === 'RESOURCE_EXHAUSTED' || m.includes('quota')) {
    return { message: `Gemini quota or rate limit reached - ${text}`, hint: 'quota' };
  }
  if (
    m.includes('enotfound') ||
    m.includes('econnrefused') ||
    m.includes('etimedout') ||
    m.includes('econnreset') ||
    m.includes('network') ||
    m.includes('getaddrinfo')
  ) {
    return { message: 'No network connection - voice dictation requires internet', hint: 'network' };
  }
  return { message: text || 'Gemini failed without a reason' };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

class GeminiProvider {
  /* socketFactory is injectable so the protocol can be tested without the API.
   * It resolves once the socket is open, to a `ws`-shaped object. */
  constructor(opts, socketFactory) {
    this.id = 'gemini';
    this.opts = opts || {};
    this.socketFactory =
      socketFactory ||
      (async () => {
        const WebSocket = require('ws');
        const ws = new WebSocket(buildRealtimeUrl(this.opts));
        // Frames that land between open and the session's own listener would
        // otherwise be dropped - setupComplete arrives that fast.
        const early = [];
        const buffer = (data) => early.push(data);
        ws.on('message', buffer);
        ws.drainEarly = (handler) => {
          ws.removeListener('message', buffer);
          for (const data of early.splice(0, early.length)) handler(data);
        };
        await new Promise((resolve, reject) => {
          const onOpen = () => {
            ws.removeListener('error', onError);
            resolve();
          };
          const onError = (e) => {
            ws.removeListener('open', onOpen);
            const mapped = mapGeminiError(e);
            reject(sttError(mapped.message, mapped.hint));
          };
          ws.once('open', onOpen);
          ws.once('error', onError);
        });
        return ws;
      });
  }

  async createSession(cb) {
    const log = typeof this.opts.log === 'function' ? this.opts.log : () => {};
    const settleTimeoutMs = this.opts.settleTimeoutMs == null ? 3000 : this.opts.settleTimeoutMs;
    const chunkBytes = Math.max(
      1,
      Math.round((this.opts.chunkMs == null ? DEFAULT_CHUNK_MS : this.opts.chunkMs) * BYTES_PER_MS)
    );
    const push = typeof cb.onFinal === 'function';

    let committed = '';
    let pending = '';
    let awaitingFinal = false;
    const display = () => `${committed} ${pending}`.trim();

    const socket = await this.socketFactory();
    let dead = false;
    let flushed = false;
    let ready = false;
    let endPending = false;
    let outbox = [];
    let outboxBytes = 0;

    const sendStreamEnd = () => {
      endPending = false;
      socket.send(JSON.stringify({ realtimeInput: { audioStreamEnd: true } }));
    };

    const shipChunk = () => {
      if (!ready || !outbox.length) return;
      const audio = outbox.length === 1 ? outbox[0] : Buffer.concat(outbox);
      outbox = [];
      outboxBytes = 0;
      socket.send(
        JSON.stringify({
          realtimeInput: { audio: { data: audio.toString('base64'), mimeType: MIME_TYPE } }
        })
      );
    };

    const onMessage = (raw) => {
      let msg;
      try {
        msg = JSON.parse(raw.toString());
      } catch (e) {
        return;
      }
      if (msg.setupComplete) {
        ready = true;
        log('setup complete');
        shipChunk();
        if (endPending) sendStreamEnd();
        return;
      }
      if (msg.error || msg.status === 'error') {
        dead = true;
        cb.onError(mapGeminiError(msg));
        return;
      }
      const content = msg.serverContent;
      if (!content) return;
      const interim = content.interimInputTranscription && content.interimInputTranscription.text;
      const final = content.inputTranscription && content.inputTranscription.text;
      if (interim != null) {
        pending = String(interim).trim();
        awaitingFinal = true;
        cb.onInterim(display());
      }
      if (final != null) {
        const text = String(final).trim();
        pending = '';
        awaitingFinal = false;
        if (!text) return;
        if (push) cb.onFinal(text);
        else committed = `${committed} ${text}`.trim();
        cb.onInterim(display());
      }
    };

    socket.on('message', onMessage);
    if (typeof socket.drainEarly === 'function') socket.drainEarly(onMessage);
    socket.on('error', (e) => {
      dead = true;
      cb.onError(mapGeminiError(e));
    });
    socket.on('close', () => {
      dead = true;
      if (cb.onClosed) cb.onClosed();
    });

    socket.send(JSON.stringify(buildSetup(this.opts)));

    return {
      sendAudio: (pcm) => {
        if (dead || flushed) return;
        outbox.push(pcm);
        outboxBytes += pcm.length;
        if (outboxBytes >= chunkBytes) shipChunk();
      },
      // Mic stopped. audioStreamEnd is the server's immediate-finalize prompt,
      // so the tail lands now instead of after its own silence timer.
      flush: () => {
        if (dead || flushed) return;
        flushed = true;
        awaitingFinal = true;
        shipChunk();
        // Held audio and the end marker have to stay in order, so a flush
        // before setupComplete waits for it rather than racing past.
        if (ready) sendStreamEnd();
        else endPending = true;
      },
      endSegment: async () => {
        const started = Date.now();
        while (Date.now() - started < settleTimeoutMs) {
          if (dead) break;
          if (!awaitingFinal && !pending) break;
          await sleep(25);
        }
        const text = push ? '' : display();
        committed = '';
        pending = '';
        awaitingFinal = false;
        return text;
      },
      close: async () => {
        dead = true;
        try {
          socket.close();
        } catch (e) {
          /* already gone */
        }
      }
    };
  }
}

module.exports = {
  GeminiProvider,
  parseGeminiCredential,
  mapGeminiError,
  buildRealtimeUrl,
  buildSetup,
  normalizeLanguage,
  languageCodes,
  vocabularyList,
  MAX_VOCABULARY,
  DEFAULT_MODEL,
  DEFAULT_BASE_URL,
  SAMPLE_RATE
};
