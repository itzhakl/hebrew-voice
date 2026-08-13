'use strict';

/* Chirp - Google Cloud Speech-to-Text V2 StreamingRecognize.
 *
 * A dedicated streaming ASR: interim hypotheses arrive continuously and the
 * server finalizes each utterance itself, so latency is bounded by the model
 * rather than by our own endpointer.
 *
 * @google-cloud/speech is required lazily: the gax/grpc stack is heavy and
 * nothing outside dictation should pay for loading it. */

const PROJECT_ID_RE = /^[a-z][a-z0-9-]{4,28}[a-z0-9]$/;

function speechEndpoint(location) {
  return location === 'global' ? 'speech.googleapis.com' : `${location}-speech.googleapis.com`;
}

function sttError(message, hint) {
  const e = new Error(message);
  if (hint) e.hint = hint;
  return e;
}

/* Catches the real-world failure of an API key ("AIzaSy...") pasted into the
 * project-id box, which otherwise only surfaces as PERMISSION_DENIED on a
 * nonsense recognizer path at mic time. */
function requireValidProjectId(project) {
  if (!PROJECT_ID_RE.test(project)) {
    throw sttError(
      `"${project}" is not a valid Google Cloud project ID ` +
        '(if you pasted an API key here, clear "projectId" in voice.json)',
      'set-api-key'
    );
  }
  return project;
}

/* Accepts either a bare API key or pasted service-account JSON and shapes it
 * into SpeechClient constructor options. */
function parseGoogleCredential(raw, projectId, location) {
  const trimmed = String(raw || '').trim();
  const base = { apiEndpoint: speechEndpoint(location) };
  if (!trimmed) {
    throw sttError('No Google Cloud credential - run: rtl-caret voice setup', 'set-api-key');
  }
  if (trimmed.startsWith('{')) {
    let json;
    try {
      json = JSON.parse(trimmed);
    } catch (e) {
      throw sttError('The service-account JSON is not valid', 'set-api-key');
    }
    if (!json.client_email || !json.private_key) {
      throw sttError('Service-account JSON is missing client_email/private_key', 'set-api-key');
    }
    // The JSON's own project_id wins: a stale or typo'd configured projectId
    // must not redirect a valid service account at a foreign project.
    const project = json.project_id || projectId;
    if (!project) throw sttError('Missing project_id - set "projectId" in voice.json', 'set-api-key');
    requireValidProjectId(project);
    return { clientOptions: Object.assign({}, base, { credentials: json, projectId: project }), projectId: project };
  }
  if (!projectId) {
    throw sttError('An API key needs a project ID - set "projectId" in voice.json', 'set-api-key');
  }
  requireValidProjectId(projectId);
  return { clientOptions: Object.assign({}, base, { apiKey: trimmed, projectId }), projectId };
}

/* gRPC status codes -> user-facing error. */
function mapChirpError(e) {
  const message = (e && e.message) || String(e);
  const code = e && e.code;
  const m = message.toLowerCase();
  if (code === 7 || code === 16 || m.includes('api key') || m.includes('unauthenticated') ||
      m.includes('permission') || m.includes('401') || m.includes('403')) {
    // Keep the raw server text - "which permission, on which resource" is the
    // difference between a five-minute fix and a guessing game.
    return { message: `Google Cloud rejected the credential - ${message.slice(0, 300)}`, hint: 'set-api-key' };
  }
  if (code === 8 || m.includes('quota') || m.includes('resource_exhausted') || m.includes('429')) {
    return { message: 'Speech-to-Text quota exceeded - try again later', hint: 'quota' };
  }
  if (code === 14 || m.includes('enotfound') || m.includes('econnrefused') ||
      m.includes('etimedout') || m.includes('network') || m.includes('unavailable')) {
    return { message: 'No network connection - voice dictation requires internet', hint: 'network' };
  }
  return { message };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

class ChirpProvider {
  /* streamFactory is injectable so the protocol can be tested without Google. */
  constructor(opts, streamFactory) {
    this.id = 'chirp';
    this.opts = opts;
    this.clientPromise = undefined;
    this.streamFactory =
      streamFactory ||
      (async () => {
        // One SpeechClient for the provider's lifetime: the OAuth token
        // exchange happens once instead of on every mic press.
        if (!this.clientPromise) {
          this.clientPromise = (async () => {
            const { clientOptions } = parseGoogleCredential(opts.credential, opts.projectId, opts.location);
            const speech = require('@google-cloud/speech');
            return new speech.v2.SpeechClient(clientOptions);
          })();
        }
        const client = await this.clientPromise;
        // _streamingRecognize is the generated bidi method; the public
        // streamingRecognize helper targets the v1 request shape.
        const stream = client._streamingRecognize();
        return { stream, close: async () => {} };
      });
  }

  async createSession(cb) {
    const settleMs = this.opts.settleMs == null ? 250 : this.opts.settleMs;
    const settleTimeoutMs = this.opts.settleTimeoutMs == null ? 1500 : this.opts.settleTimeoutMs;
    const { projectId } = parseGoogleCredential(this.opts.credential, this.opts.projectId, this.opts.location);
    const recognizer = `projects/${projectId}/locations/${this.opts.location}/recognizers/_`;

    // Push mode (onFinal wired): every server-finalized utterance is handed to
    // the client the moment it lands - the caller's VAD never gates the commit.
    // Pull mode: finals accumulate for endSegment.
    const push = typeof cb.onFinal === 'function';
    let finals = '';
    let pending = '';
    let lastEventAt = 0;
    const currentText = () => `${finals} ${pending}`.trim();

    const { stream, close } = await this.streamFactory();
    // Once the gRPC stream errors it is destroyed; further writes throw
    // synchronously and would spam one error per queued audio frame.
    let dead = false;
    let flushed = false;

    stream.on('data', (resp) => {
      let sawFragment = false;
      const interimParts = [];
      for (const result of (resp && resp.results) || []) {
        const transcript = (result.alternatives && result.alternatives[0] && result.alternatives[0].transcript) || '';
        if (!transcript) continue;
        sawFragment = true;
        if (result.isFinal) {
          if (push) cb.onFinal(transcript.trim());
          else finals = `${finals} ${transcript}`.trim();
          pending = '';
        } else {
          interimParts.push(transcript);
        }
      }
      if (interimParts.length > 0) pending = interimParts.join(' ').trim();
      if (sawFragment) {
        lastEventAt = Date.now();
        cb.onInterim(currentText());
      }
    });
    stream.on('error', (e) => {
      dead = true;
      cb.onError(mapChirpError(e));
    });
    stream.on('end', () => {
      if (cb.onClosed) cb.onClosed();
    });

    stream.write({
      recognizer,
      streamingConfig: {
        config: {
          explicitDecodingConfig: {
            encoding: 'LINEAR16',
            sampleRateHertz: 16000,
            audioChannelCount: 1
          },
          languageCodes: [this.opts.languageCode],
          model: this.opts.model
        },
        streamingFeatures: { interimResults: true }
      }
    });

    return {
      sendAudio: (pcm) => {
        if (!dead && !flushed) stream.write({ audio: pcm });
      },
      // Half-close the gRPC stream: the server finalizes whatever it holds and
      // answers in ~100-300 ms, instead of its natural ~1 s silence endpointer.
      flush: () => {
        if (dead || flushed) return;
        flushed = true;
        stream.end();
      },
      endSegment: async () => {
        const started = Date.now();
        while (Date.now() - started < settleTimeoutMs) {
          // Push mode: finals already went out via onFinal; only wait for a
          // dangling interim to finalize (it arrives through onFinal too).
          if (push ? !pending : finals && !pending) break;
          if (!push && currentText() && Date.now() - lastEventAt >= settleMs) break;
          await sleep(25);
        }
        const text = push ? '' : currentText();
        finals = '';
        pending = '';
        return text;
      },
      close: async () => {
        dead = true;
        try {
          stream.end();
        } finally {
          await close();
        }
      }
    };
  }
}

module.exports = { ChirpProvider, parseGoogleCredential, mapChirpError, speechEndpoint };
