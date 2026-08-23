'use strict';

/* ElevenLabs Scribe v2 Realtime - one WebSocket, JSON in, JSON out.
 *
 * The server runs its own voice-activity endpointer, so every finished
 * utterance arrives as `committed_transcript` while the user is still talking.
 * That is the same push shape Chirp had, minus gRPC: `ws` is already a
 * dependency of the local server, so nothing heavier is loaded for dictation.
 *
 * Wire reference: wss://api.elevenlabs.io/v1/speech-to-text/realtime
 *   client -> {message_type:'input_audio_chunk', audio_base_64, commit, sample_rate}
 *   server -> session_started | partial_transcript | final_transcript |
 *             committed_transcript | <one of the error types below> */

const DEFAULT_BASE_URL = 'wss://api.elevenlabs.io';
const REALTIME_PATH = '/v1/speech-to-text/realtime';
const DEFAULT_MODEL = 'scribe_v2_realtime';
const SAMPLE_RATE = 16000;

/* One JSON+base64 frame per 20 ms of audio is 50 messages a second for no
 * gain. Batching to ~100 ms costs latency the model cannot use anyway. */
const DEFAULT_CHUNK_MS = 100;
const BYTES_PER_MS = (SAMPLE_RATE * 2) / 1000;

/* Every message_type the server uses to report a failure. All of them close
 * the socket except commit_throttled, which only rejects that one commit. */
const ERROR_TYPES = new Set([
  'error',
  'auth_error',
  'quota_exceeded',
  'commit_throttled',
  'transcriber_error',
  'unaccepted_terms',
  'unaccepted_terms_error',
  'rate_limited',
  'input_error',
  'invalid_request',
  'queue_overflow',
  'resource_exhausted',
  'session_time_limit_exceeded',
  'chunk_size_exceeded',
  'insufficient_audio_activity'
]);
const RECOVERABLE_TYPES = new Set(['commit_throttled', 'insufficient_audio_activity']);

function sttError(message, hint) {
  const e = new Error(message);
  if (hint) e.hint = hint;
  return e;
}

/* ElevenLabs wants a bare ISO-639-1/639-3 code. "iw-IL" is what the Google
 * config carried, and "iw" is the retired code for Hebrew - both have to keep
 * working or every existing voice.json breaks on upgrade. */
const LEGACY_CODES = { iw: 'he', in: 'id', ji: 'yi' };

function normalizeLanguage(code) {
  const bare = String(code || '')
    .trim()
    .toLowerCase()
    .split(/[-_]/)[0];
  if (!bare) return '';
  return LEGACY_CODES[bare] || bare;
}

/* An API key is opaque, so the only checks worth making are the two mistakes a
 * migration actually produces: an empty box, and the Google credential that
 * used to live in it. */
function parseElevenLabsCredential(raw) {
  const trimmed = String(raw || '').trim();
  if (!trimmed) {
    throw sttError('No ElevenLabs API key - run: rtl-caret voice setup', 'set-api-key');
  }
  if (trimmed.startsWith('{')) {
    throw sttError(
      'That is a Google Cloud service-account JSON, not an ElevenLabs API key - run: rtl-caret voice setup',
      'set-api-key'
    );
  }
  if (/^AIza/.test(trimmed)) {
    throw sttError(
      'That is a Google Cloud API key, not an ElevenLabs API key - run: rtl-caret voice setup',
      'set-api-key'
    );
  }
  if (/\s/.test(trimmed)) {
    throw sttError('The ElevenLabs API key contains whitespace - paste it on one line', 'set-api-key');
  }
  return trimmed;
}

function buildRealtimeUrl(opts = {}) {
  const params = new URLSearchParams({
    model_id: opts.model || DEFAULT_MODEL,
    audio_format: `pcm_${SAMPLE_RATE}`,
    // Server-side endpointing: without it committed_transcript never fires for
    // microphone input and the whole utterance rides on our own flush.
    commit_strategy: opts.commitStrategy || 'vad'
  });
  const language = normalizeLanguage(opts.languageCode);
  if (language) params.set('language_code', language);
  if (opts.vadSilenceThresholdSecs != null) {
    params.set('vad_silence_threshold_secs', String(opts.vadSilenceThresholdSecs));
  }
  const base = String(opts.baseUrl || DEFAULT_BASE_URL).replace(/\/+$/, '');
  return `${base}${REALTIME_PATH}?${params.toString()}`;
}

/* Accepts either a protocol error frame or a thrown socket error. */
function mapElevenLabsError(e) {
  const type = (e && e.message_type) || '';
  const text = String((e && (e.error || e.message)) || e || '').slice(0, 300);
  const m = text.toLowerCase();
  if (type === 'auth_error' || type === 'unaccepted_terms' || type === 'unaccepted_terms_error' ||
      m.includes('401') || m.includes('403') || m.includes('unauthorized') || m.includes('invalid api key')) {
    // Keep the raw server text: which permission, on which resource, is the
    // difference between a five-minute fix and a guessing game.
    return { message: `ElevenLabs rejected the credential - ${text}`, hint: 'set-api-key' };
  }
  if (type === 'quota_exceeded' || type === 'rate_limited' || type === 'resource_exhausted' ||
      type === 'queue_overflow' || m.includes('quota') || m.includes('429')) {
    return { message: `ElevenLabs quota or rate limit reached - ${text}`, hint: 'quota' };
  }
  if (m.includes('enotfound') || m.includes('econnrefused') || m.includes('etimedout') ||
      m.includes('econnreset') || m.includes('network') || m.includes('getaddrinfo')) {
    return { message: 'No network connection - voice dictation requires internet', hint: 'network' };
  }
  return { message: text || 'ElevenLabs failed without a reason' };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

class ElevenLabsProvider {
  /* socketFactory is injectable so the protocol can be tested without the API.
   * It resolves once the socket is open, to a `ws`-shaped object. */
  constructor(opts, socketFactory) {
    this.id = 'elevenlabs';
    this.opts = opts || {};
    this.socketFactory =
      socketFactory ||
      (async () => {
        const apiKey = parseElevenLabsCredential(this.opts.credential);
        const WebSocket = require('ws');
        const ws = new WebSocket(buildRealtimeUrl(this.opts), { headers: { 'xi-api-key': apiKey } });
        await new Promise((resolve, reject) => {
          const onOpen = () => {
            ws.removeListener('error', onError);
            resolve();
          };
          const onError = (e) => {
            ws.removeListener('open', onOpen);
            const mapped = mapElevenLabsError(e);
            reject(sttError(mapped.message, mapped.hint));
          };
          ws.once('open', onOpen);
          ws.once('error', onError);
        });
        return ws;
      });
  }

  async createSession(cb) {
    const settleTimeoutMs = this.opts.settleTimeoutMs == null ? 3000 : this.opts.settleTimeoutMs;
    const chunkBytes = Math.max(1, Math.round((this.opts.chunkMs == null ? DEFAULT_CHUNK_MS : this.opts.chunkMs) * BYTES_PER_MS));

    // Push mode (onFinal wired): each server-endpointed segment is handed over
    // the moment it lands. Pull mode accumulates for endSegment instead.
    const push = typeof cb.onFinal === 'function';
    let committed = '';
    let pending = '';
    let awaitingCommit = false;
    const display = () => `${committed} ${pending}`.trim();

    const socket = await this.socketFactory();
    let dead = false;
    let flushed = false;
    let outbox = [];
    let outboxBytes = 0;

    const shipChunk = (commit) => {
      const audio = outbox.length === 1 ? outbox[0] : Buffer.concat(outbox);
      outbox = [];
      outboxBytes = 0;
      if (!audio.length && !commit) return;
      socket.send(
        JSON.stringify({
          message_type: 'input_audio_chunk',
          audio_base_64: audio.toString('base64'),
          commit,
          sample_rate: SAMPLE_RATE
        })
      );
    };

    socket.on('message', (raw) => {
      let msg;
      try {
        msg = JSON.parse(raw.toString());
      } catch (e) {
        return;
      }
      const type = msg && msg.message_type;
      const text = String((msg && msg.text) || '').trim();
      if (type === 'partial_transcript' || type === 'final_transcript') {
        // final_transcript is a settled hypothesis, not a commit. Emitting it
        // as one too would send the same words twice - committed_transcript is
        // the only frame that ends a segment.
        if (!text) return;
        pending = text;
        cb.onInterim(display());
        return;
      }
      if (type === 'committed_transcript') {
        pending = '';
        awaitingCommit = false;
        if (!text) return;
        if (push) cb.onFinal(text);
        else committed = `${committed} ${text}`.trim();
        return;
      }
      if (ERROR_TYPES.has(type)) {
        if (RECOVERABLE_TYPES.has(type)) {
          // A throttled commit is answered by the server's own VAD a moment
          // later, and painting TranscriptError over live text loses it.
          awaitingCommit = false;
          return;
        }
        dead = true;
        cb.onError(mapElevenLabsError(msg));
      }
    });
    socket.on('error', (e) => {
      dead = true;
      cb.onError(mapElevenLabsError(e));
    });
    socket.on('close', () => {
      dead = true;
      if (cb.onClosed) cb.onClosed();
    });

    return {
      sendAudio: (pcm) => {
        if (dead || flushed) return;
        outbox.push(pcm);
        outboxBytes += pcm.length;
        if (outboxBytes >= chunkBytes) shipChunk(false);
      },
      // Mic stopped. Ship the tail with commit set so the segment is finalized
      // now instead of after the server's silence threshold.
      flush: () => {
        if (dead || flushed) return;
        flushed = true;
        awaitingCommit = true;
        shipChunk(true);
      },
      endSegment: async () => {
        const started = Date.now();
        while (Date.now() - started < settleTimeoutMs) {
          if (dead) break;
          if (!awaitingCommit && !pending) break;
          await sleep(25);
        }
        const text = push ? '' : display();
        committed = '';
        pending = '';
        awaitingCommit = false;
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
  ElevenLabsProvider,
  parseElevenLabsCredential,
  mapElevenLabsError,
  buildRealtimeUrl,
  normalizeLanguage,
  DEFAULT_MODEL,
  DEFAULT_BASE_URL,
  SAMPLE_RATE
};
