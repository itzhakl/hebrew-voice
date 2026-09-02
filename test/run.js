#!/usr/bin/env node
'use strict';

/* Dictation has no pty recordings to check against: the protocol is Claude's,
 * not ours. So the socket round-trip below drives the real server with a fake
 * provider, which is the closest thing to a fixture we can hold - the frames
 * are exactly what the CLI sends and expects. */

const fs = require('fs');
const os = require('os');
const path = require('path');

const { Endpointer } = require('../src/vad');
const {
  ElevenLabsProvider,
  parseElevenLabsCredential,
  mapElevenLabsError,
  buildRealtimeUrl,
  normalizeLanguage,
  MAX_KEYTERMS,
  MAX_KEYTERM_LENGTH
} = require('../src/elevenlabs');
const { WhisperProvider, mapWhisperError, encodeFrame, resolvePython, venvPython } = require('../src/whisper');
const {
  GeminiProvider,
  buildSetup,
  buildRealtimeUrl: buildGeminiUrl,
  mapGeminiError,
  languageCodes,
  DEFAULT_MODEL: DEFAULT_GEMINI_MODEL
} = require('../src/gemini');
const { HybridProvider } = require('../src/hybrid');
const { GeminiTranscribeProvider, readTranscript } = require('../src/transcribe');
const config = require('../src/config');
const server = require('../src/server');
const cli = require('../src/cli');

let failures = 0;
let checks = 0;

function ok(cond, msg) {
  checks++;
  if (!cond) {
    failures++;
    console.log(`  FAIL ${msg}`);
  }
}

function eq(actual, expected, msg) {
  ok(actual === expected, `${msg}: got ${JSON.stringify(actual)}, want ${JSON.stringify(expected)}`);
}

function throwsWith(fn, fragment, msg) {
  try {
    fn();
    ok(false, `${msg}: expected a throw`);
  } catch (e) {
    ok(String(e.message).includes(fragment), `${msg}: got "${e.message}"`);
  }
}

const SAMPLE_RATE = 16000;
function frame(ms, amplitude) {
  const n = (SAMPLE_RATE * ms) / 1000;
  const pcm = new Int16Array(n);
  for (let i = 0; i < n; i++) pcm[i] = i % 2 ? amplitude : -amplitude;
  return pcm;
}
const LOUD = 8000;
const SILENT = 0;

// ---------- VAD ----------

function testVad() {
  const ep = new Endpointer();
  eq(ep.pushFrame(frame(400, LOUD)).commit, undefined, 'speech alone does not commit');
  ok(ep.speaking, 'speaking after loud frame');
  eq(ep.pushFrame(frame(500, SILENT)).commit, undefined, 'silence under endpointMs holds');
  eq(ep.pushFrame(frame(200, SILENT)).commit, 'silence', 'silence past endpointMs commits');
  ok(!ep.speaking, 'reset after commit');

  const short = new Endpointer();
  short.pushFrame(frame(100, LOUD));
  const r = short.pushFrame(frame(700, SILENT));
  eq(r.commit, undefined, 'utterance under minUtteranceMs is not committed');
  eq(r.discarded, true, 'utterance under minUtteranceMs is discarded');

  // The extension capped segments at 4 s, which chops long Hebrew sentences.
  const capped = new Endpointer({ maxSegmentMs: 12000 });
  for (let i = 0; i < 39; i++) {
    eq(capped.pushFrame(frame(300, LOUD)).commit, undefined, `continuous speech at ${i * 300}ms holds`);
  }
  eq(capped.pushFrame(frame(300, LOUD)).commit, 'max-segment', 'commits at maxSegmentMs');

  const quiet = new Endpointer();
  eq(quiet.pushFrame(frame(2000, SILENT)).commit, undefined, 'silence before any speech never commits');
  ok(!quiet.speaking, 'silence alone does not open a segment');
}

/* The bug this exists to prevent: a microphone whose own noise sits above the
 * absolute threshold never falls silent, so nothing ever commits and dictation
 * only lands when the key is released. Real frames are 20 ms - the room is
 * measured from them, not from the whole-second frames the tests above use. */
function testNoisyRoom() {
  const ROOM = 900; // ~0.027 rms, five times the absolute threshold
  const SPEECH = 9000;
  // Returns the frame that ended a segment, not the last frame pushed - the
  // commit lands mid-run and the frames after it are a new segment.
  const feed = (ep, ms, amplitude) => {
    let out = {};
    for (let i = 0; i < ms / 20; i++) {
      const r = ep.pushFrame(frame(20, amplitude));
      if ((r.commit || r.discarded) && !out.commit && !out.discarded) out = r;
    }
    return out;
  };

  const fixed = new Endpointer({ calibrationMinFrames: Infinity });
  feed(fixed, 300, ROOM);
  feed(fixed, 1000, SPEECH);
  eq(feed(fixed, 2000, ROOM).commit, undefined, 'without calibration a noisy room never falls silent');

  const ep = new Endpointer();
  feed(ep, 300, ROOM);
  ok(ep.calibrated, 'the room is measured from the leading pause');
  ok(ep.threshold > ROOM / 32768, `speech has to beat the room: threshold ${ep.threshold}`);
  ok(!ep.speaking, 'the segment the room opened before it was measured is taken back');
  eq(feed(ep, 3000, ROOM).commit, undefined, 'three seconds of room alone commits nothing');
  feed(ep, 1000, SPEECH);
  ok(ep.speaking, 'speech over a noisy room is still speech');
  eq(feed(ep, 400, ROOM).commit, undefined, 'a pause under endpointMs holds');
  eq(feed(ep, 400, ROOM).commit, 'silence', 'the pause after speech commits in a noisy room');

  // A room that goes quiet must not leave the threshold stranded high.
  const hushed = new Endpointer();
  feed(hushed, 300, ROOM);
  const loudThreshold = hushed.threshold;
  feed(hushed, 1000, 30);
  ok(hushed.threshold < loudThreshold, 'a room that went quiet lowers the threshold');

  // A caller handing over whole seconds is not measuring a room, and must be
  // left on the absolute threshold rather than told that speech is the floor.
  const coarse = new Endpointer();
  coarse.pushFrame(frame(400, SPEECH));
  ok(!coarse.calibrated, 'one coarse frame does not calibrate a room');
  eq(coarse.threshold, 0.005, 'an uncalibrated endpointer keeps the absolute threshold');
}

// ---------- credentials ----------

function testCredentials() {
  eq(parseElevenLabsCredential(' sk_abc123 '), 'sk_abc123', 'the key is trimmed');
  throwsWith(() => parseElevenLabsCredential(''), 'No ElevenLabs API key', 'empty credential');
  // The two things a migration off Google actually leaves in the box.
  throwsWith(() => parseElevenLabsCredential('AIzaSyExample'), 'Google Cloud API key',
    'a leftover Google API key is named, not just rejected');
  throwsWith(() => parseElevenLabsCredential('{"project_id":"x"}'), 'service-account JSON',
    'a leftover service account is named, not just rejected');
  throwsWith(() => parseElevenLabsCredential('sk_a sk_b'), 'whitespace', 'a mangled paste is caught');

  eq(normalizeLanguage('iw-IL'), 'he', 'the tag the Google config carried still means Hebrew');
  eq(normalizeLanguage('he'), 'he', 'a bare code passes through');
  eq(normalizeLanguage('ar-SA'), 'ar', 'the region is dropped');
  eq(normalizeLanguage(''), '', 'no language is not a language');

  const url = new URL(buildRealtimeUrl({ languageCode: 'iw-IL' }));
  eq(url.host, 'api.elevenlabs.io', 'the default host');
  eq(url.pathname, '/v1/speech-to-text/realtime', 'the realtime path');
  eq(url.searchParams.get('model_id'), 'scribe_v2_realtime', 'the default model');
  eq(url.searchParams.get('audio_format'), 'pcm_16000', 'declares linear16 at 16 kHz');
  eq(url.searchParams.get('language_code'), 'he', 'the language is normalized into the URL');
  // Without server-side endpointing, committed_transcript never fires for a
  // microphone and the whole utterance would ride on our own flush.
  eq(url.searchParams.get('commit_strategy'), 'vad', 'server-side endpointing is on');

  eq(url.searchParams.get('secondary_languages'), null, 'no second language unless asked for');
  eq(url.searchParams.get('no_verbatim'), null, 'verbatim is the default');

  eq(mapElevenLabsError({ message_type: 'auth_error', error: 'bad key' }).hint, 'set-api-key', 'auth hint');
  eq(mapElevenLabsError({ message_type: 'quota_exceeded', error: 'no credits' }).hint, 'quota', 'quota hint');
  eq(mapElevenLabsError(new Error('getaddrinfo ENOTFOUND api.elevenlabs.io')).hint, 'network', 'network hint');
  eq(mapElevenLabsError({ message_type: 'transcriber_error', error: 'something else' }).hint, undefined,
    'unknown errors carry no hint');
  ok(mapElevenLabsError({ message_type: 'auth_error', error: 'bad key' }).message.includes('bad key'),
    'the raw server text survives');
}

/* Both list parameters have to be repeated, never comma-joined. The live API
 * rejects a comma-joined secondary_languages outright, and accepts comma-joined
 * keyterms as ONE long term - biasing the model at a string nobody will say. */
function testHebrewTuning() {
  const tuned = new URL(
    buildRealtimeUrl({
      languageCode: 'he',
      secondaryLanguages: ['en', 'ar'],
      keyterms: ['claude', 'קומיט'],
      noVerbatim: true,
      filterBackgroundAudio: true,
      vadSilenceThresholdSecs: 0.8
    })
  );
  eq(tuned.searchParams.getAll('secondary_languages').join('|'), 'en|ar', 'each language is its own parameter');
  eq(tuned.searchParams.getAll('keyterms').join('|'), 'claude|קומיט', 'each keyterm is its own parameter');
  eq(tuned.searchParams.get('no_verbatim'), 'true', 'no_verbatim is passed on');
  eq(tuned.searchParams.get('filter_background_audio'), 'true', 'background filtering is passed on');
  eq(tuned.searchParams.get('vad_silence_threshold_secs'), '0.8', 'the silence threshold is passed on');

  // Naming the primary language again narrows nothing and reads as a mistake.
  const dupe = new URL(buildRealtimeUrl({ languageCode: 'iw-IL', secondaryLanguages: ['he', 'en'] }));
  eq(dupe.searchParams.getAll('secondary_languages').join('|'), 'en', 'the primary language is not repeated as a secondary');

  const coerced = new URL(buildRealtimeUrl({ languageCode: 'he', secondaryLanguages: 'en, ar' }));
  eq(coerced.searchParams.getAll('secondary_languages').join('|'), 'en|ar', 'a comma string is split, not sent whole');

  // Over the server's limits earns an invalid_request at mic time, where the
  // failure is invisible behind Claude's UI.
  const long = 'x'.repeat(MAX_KEYTERM_LENGTH + 1);
  const capped = new URL(buildRealtimeUrl({ keyterms: ['ok', long] }));
  eq(capped.searchParams.getAll('keyterms').join('|'), 'ok', 'an over-long keyterm is dropped, not truncated');
  const many = new URL(buildRealtimeUrl({ keyterms: Array.from({ length: MAX_KEYTERMS + 10 }, (_, i) => `t${i}`) }));
  eq(many.searchParams.getAll('keyterms').length, MAX_KEYTERMS, 'the keyterm list is capped');
}

// ---------- config ----------

function testConfig() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rtl-voice-'));
  const file = path.join(dir, 'voice.json');

  const empty = config.load({}, {}, file);
  eq(empty.port, 8765, 'default port');
  eq(empty.language, 'he', 'Hebrew is the default language');
  eq(empty.provider, 'elevenlabs', 'elevenlabs is the default provider');
  eq(empty.model, 'scribe_v2_realtime', 'the realtime Scribe model is the default');
  // Terminal Hebrew is code-switched: paths and commands arrive in English.
  eq(empty.secondaryLanguages.join('|'), 'en', 'English rides along with Hebrew by default');
  eq(empty.noVerbatim, false, 'dictation returns what was said by default');
  eq(empty.vadSilenceThresholdSecs, null, 'the server keeps its own silence threshold by default');

  config.save({ credential: 'sk_saved', port: 9000 }, file);
  eq(fs.statSync(file).mode & 0o777, 0o600, 'config file is 0600');
  const saved = config.load({}, {}, file);
  eq(saved.port, 9000, 'file overrides the default');
  eq(saved.credential, 'sk_saved', 'credential read back from the file');
  eq(saved.language, 'he', 'unsaved keys keep their defaults');

  const env = config.load({}, { RTL_VOICE_PORT: '9100', ELEVENLABS_API_KEY: 'sk_env' }, file);
  eq(env.port, 9100, 'environment overrides the file');
  eq(env.credential, 'sk_env', 'inline env credential wins over the file');
  eq(config.load({}, { XI_API_KEY: 'sk_xi' }, file).credential, 'sk_xi', "ElevenLabs' own variable is honoured");

  const flags = config.load({ port: 9200, language: 'ar-SA' }, { RTL_VOICE_PORT: '9100' }, file);
  eq(flags.port, 9200, 'explicit flags override the environment');
  eq(flags.language, 'ar-SA', 'language flag applies');

  eq(config.load({ provider: 'nonsense' }, {}, file).provider, 'elevenlabs', 'unknown provider falls back');
  eq(config.load({}, {}, file).vadNoiseRatio, 3, 'speech is asked to beat the room threefold by default');
  eq(config.load({ vadNoiseRatio: 0.5 }, {}, file).vadNoiseRatio, 1.2, 'a ratio that cannot separate speech from the room is clamped');

  // A voice.json left over from the Google backend must not send "long" as a
  // Scribe model_id. "hybrid" names a pair again, but a Chirp-era one whose
  // halves it never described, so the halves come from the defaults.
  config.save({ provider: 'hybrid', model: 'long', location: 'eu' }, file);
  const migrated = config.load({}, {}, file);
  eq(migrated.hybrid.fast, 'gemini', 'the undescribed fast half falls back');
  eq(migrated.model, 'gemini-3.5-transcribe-live', 'the retired Chirp model is replaced');
  config.save({ provider: 'chirp' }, file);
  eq(config.load({}, {}, file).provider, 'elevenlabs', 'a retired provider is replaced');

  // A hand-edited voice.json holds whatever the user typed.
  config.save({ secondaryLanguages: 'en, ar', keyterms: 'קלוד', noVerbatim: 'true', vadSilenceThresholdSecs: 9 }, file);
  const typed = config.load({}, {}, file);
  eq(typed.secondaryLanguages.join('|'), 'en|ar', 'a bare string becomes a list');
  eq(typed.keyterms.join('|'), 'קלוד', 'a single keyterm becomes a list');
  eq(typed.noVerbatim, true, 'a string boolean is read as one');
  // The server rejects anything outside 0.3-3.0 rather than clamping it.
  eq(typed.vadSilenceThresholdSecs, 3, 'an out-of-range silence threshold is clamped');

  fs.writeFileSync(file, '{not json');
  throwsWith(() => config.load({}, {}, file), 'not valid JSON', 'corrupt config is reported, not swallowed');
  fs.rmSync(dir, { recursive: true, force: true });
}

// ---------- CLI parsing ----------

function testCliParse() {
  const run = cli.parse(['--port', '9000', '--', 'claude', '--continue']);
  eq(run.overrides.port, 9000, 'port flag parsed');
  eq(run.command.join(' '), 'claude --continue', 'everything after -- is the command');

  const model = cli.parse(['serve', '--model', 'scribe_v2_realtime']);
  eq(model.overrides.model, 'scribe_v2_realtime', 'model flag parsed');

  const tuned = cli.parse(['serve', '--secondary', 'en', '--keyterm', 'claude', '--keyterm', 'קומיט']);
  eq(tuned.overrides.secondaryLanguages, 'en', 'secondary flag parsed');
  eq(tuned.overrides.keyterms.join('|'), 'claude|קומיט', 'keyterm repeats accumulate');

  const serve = cli.parse(['serve', '--verbose']);
  eq(serve.sub, 'serve', 'subcommand parsed');
  eq(serve.verbose, true, 'verbose parsed');
  eq(serve.command.length, 0, 'no command without --');

  // A wrapped command carrying its own flags must not be re-parsed as ours.
  const nested = cli.parse(['--', 'claude', '--port', '1234']);
  eq(nested.overrides.port, undefined, "the wrapped command's flags stay its own");
  eq(nested.command.length, 3, 'wrapped command keeps its arguments');
}

// ---------- provider selection ----------

function testProviderSelection() {
  const base = config.load({ credential: 'sk_test' }, {}, '/nonexistent');
  eq(cli.buildProvider(base).id, 'elevenlabs', 'the elevenlabs provider is built');
  throwsWith(() => cli.buildProvider(Object.assign({}, base, { credential: '' })),
    'No ElevenLabs API key', 'a missing credential fails before the mic opens');
  throwsWith(() => cli.buildProvider(Object.assign({}, base, { credential: 'AIzaSyOld' })),
    'not an ElevenLabs API key', 'a leftover Google key fails before the mic opens');
}

// ---------- fake engines ----------

/* Stands in for the realtime WebSocket, so the provider and the socket can be
 * driven without ElevenLabs. */
function fakeSocket() {
  const listeners = {};
  return {
    sent: [],
    closed: false,
    on(event, fn) {
      (listeners[event] = listeners[event] || []).push(fn);
      return this;
    },
    send(text) {
      this.sent.push(JSON.parse(text));
    },
    close() {
      this.closed = true;
      this.emit('close');
    },
    emit(event, arg) {
      for (const fn of listeners[event] || []) fn(arg);
    },
    reply(obj) {
      this.emit('message', Buffer.from(JSON.stringify(obj)));
    }
  };
}

/* chunkMs 20 makes one CLI frame a full chunk, so batching does not have to be
 * spelled out in every assertion below. */
function elevenWith(socket, opts) {
  return new ElevenLabsProvider(
    Object.assign({ credential: 'sk_test', languageCode: 'iw-IL', chunkMs: 20 }, opts),
    async () => socket
  );
}

async function testElevenLabsSession() {
  const socket = fakeSocket();
  const interims = [];
  const finals = [];
  const session = await elevenWith(socket).createSession({
    onInterim: (t) => interims.push(t),
    onFinal: (t) => finals.push(t),
    onError: (e) => finals.push(`ERR ${e.message}`)
  });

  socket.reply({ message_type: 'session_started', session_id: 'abc' });
  eq(interims.length, 0, 'the session handshake is not a transcript');
  socket.reply({ message_type: 'partial_transcript', text: 'שלום' });
  eq(interims[0], 'שלום', 'a partial surfaces as an interim');
  // final_transcript is a settled hypothesis, not a commit. Treating it as one
  // as well as committed_transcript would send the same words twice.
  socket.reply({ message_type: 'final_transcript', text: 'שלום עולם' });
  eq(interims[1], 'שלום עולם', 'a settled final is still only a hypothesis');
  eq(finals.length, 0, 'final_transcript does not commit');
  socket.reply({ message_type: 'committed_transcript', text: 'שלום עולם' });
  eq(finals[0], 'שלום עולם', 'committed_transcript is the commit');

  session.sendAudio(Buffer.alloc(640));
  eq(socket.sent.length, 1, 'a full chunk is shipped');
  eq(socket.sent[0].message_type, 'input_audio_chunk', 'audio goes out as input_audio_chunk');
  eq(socket.sent[0].sample_rate, 16000, 'declares the CLI sample rate');
  eq(socket.sent[0].commit, false, 'audio alone does not commit');
  eq(Buffer.from(socket.sent[0].audio_base_64, 'base64').length, 640, 'the audio survives base64');

  session.sendAudio(Buffer.alloc(320));
  eq(socket.sent.length, 1, 'a part chunk is held back');
  session.flush();
  eq(socket.sent.length, 2, 'flush ships the tail');
  eq(socket.sent[1].commit, true, 'flush commits the segment instead of waiting out the silence');
  eq(Buffer.from(socket.sent[1].audio_base_64, 'base64').length, 320, 'the held audio rides the commit');
  session.sendAudio(Buffer.alloc(640));
  eq(socket.sent.length, 2, 'no audio is written after flush');

  socket.reply({ message_type: 'committed_transcript', text: 'הזנב' });
  eq(finals[1], 'הזנב', 'the flushed tail commits');
  eq(await session.endSegment(), '', 'push mode commits through onFinal, not endSegment');

  // A commit we asked for and never got must not hang the close path.
  const slow = fakeSocket();
  const slowSession = await elevenWith(slow, { settleTimeoutMs: 100 }).createSession({
    onInterim: () => {},
    onFinal: () => {},
    onError: () => {}
  });
  slowSession.sendAudio(Buffer.alloc(640));
  slowSession.flush();
  const waited = Date.now();
  eq(await slowSession.endSegment(), '', 'an unanswered commit falls through');
  ok(Date.now() - waited >= 90, 'endSegment waits for the commit it asked for');

  // An engine that dies must not throw once per remaining audio frame.
  const dying = fakeSocket();
  const errors = [];
  const dyingSession = await elevenWith(dying).createSession({
    onInterim: () => {},
    onFinal: () => {},
    onError: (e) => errors.push(e)
  });
  dying.reply({ message_type: 'quota_exceeded', error: 'out of credits' });
  eq(errors.length, 1, 'the error is reported once');
  eq(errors[0].hint, 'quota', 'the error is mapped');
  dyingSession.sendAudio(Buffer.alloc(640));
  dyingSession.sendAudio(Buffer.alloc(640));
  eq(dying.sent.length, 0, 'a dead socket takes no further audio');

  // A throttled commit is answered by the server's own VAD a moment later.
  // Painting TranscriptError over live text would lose the transcript.
  const throttled = fakeSocket();
  const throttledErrors = [];
  const throttledSession = await elevenWith(throttled).createSession({
    onInterim: () => {},
    onFinal: () => {},
    onError: (e) => throttledErrors.push(e)
  });
  throttled.reply({ message_type: 'commit_throttled', error: 'too soon' });
  eq(throttledErrors.length, 0, 'a throttled commit is not painted over the transcript');
  throttledSession.sendAudio(Buffer.alloc(640));
  eq(throttled.sent.length, 1, 'and the session stays alive');
}

// ---------- the socket, end to end ----------

function scriptedProvider(script) {
  return {
    id: 'scripted',
    async createSession(cb) {
      let audioFrames = 0;
      return {
        sendAudio: () => {
          audioFrames++;
          if (audioFrames === 1) cb.onInterim(script.interim);
        },
        flush: () => {},
        endSegment: async () => script.final,
        close: async () => {},
        frames: () => audioFrames
      };
    }
  };
}

async function testSocketRoundTrip() {
  const WebSocket = require('ws');
  const instance = await server.VoiceStreamServer.start({
    port: 0,
    provider: scriptedProvider({ interim: 'שלום‎', final: 'שלום עולם' }),
    makeEndpointer: () => new Endpointer()
  });
  const ws = new WebSocket(`ws://127.0.0.1:${instance.port}${server.VOICE_STREAM_PATH}`);
  const received = [];
  ws.on('message', (d) => received.push(JSON.parse(d.toString())));
  await new Promise((resolve, reject) => {
    ws.on('open', resolve);
    ws.on('error', reject);
  });

  // 20 ms of linear16 at 16 kHz, the frame size the CLI sends.
  ws.send(Buffer.alloc(640), { binary: true });
  await new Promise((r) => setTimeout(r, 50));
  eq(received.length, 1, 'the interim is sent back');
  eq(received[0].type, 'TranscriptInterim', 'a hypothesis is sent as TranscriptInterim');
  eq(received[0].data, 'שלום', 'bidi controls are stripped from the transcript');

  const closed = new Promise((r) => ws.on('close', r));
  ws.send(JSON.stringify({ type: 'KeepAlive' }));
  const closeStreamAt = Date.now();
  ws.send(JSON.stringify({ type: 'CloseStream' }));
  await closed;

  // The client arms a 1500 ms no-data timer when it sends CloseStream, and any
  // frame at all cancels it. Going quiet here costs the accurate transcript.
  const answered = received.findIndex((m, i) => i > 0 && m.type === 'TranscriptInterim');
  ok(answered !== -1, 'CloseStream is answered with a keepalive interim');

  const text = received.filter((m) => m.type === 'TranscriptText').map((m) => m.data);
  eq(text[text.length - 1], 'שלום עולם', 'the final transcript is committed on CloseStream');
  eq(received[received.length - 1].type, 'TranscriptEndpoint', 'the endpoint frame closes the utterance');
  ok(Date.now() - closeStreamAt < server.SAFETY_MS, 'the socket closes inside the client safety timeout');

  await instance.close();
}

/* Only TranscriptEndpoint commits: the client replaces its pending buffer on
 * every Text/Interim frame and promotes it on Endpoint. A commit is therefore
 * always a pair, and the text must be whole rather than a delta. */
async function testCommitIsAlwaysAPair() {
  const WebSocket = require('ws');
  const instance = await server.VoiceStreamServer.start({
    port: 0,
    provider: {
      id: 'two-utterances',
      async createSession(cb) {
        return {
          sendAudio: () => {
            cb.onFinal('משפט ראשון');
            cb.onFinal('משפט שני');
          },
          flush: () => {},
          endSegment: async () => '',
          close: async () => {}
        };
      }
    },
    makeEndpointer: () => new Endpointer()
  });
  const ws = new WebSocket(`ws://127.0.0.1:${instance.port}${server.VOICE_STREAM_PATH}`);
  const received = [];
  ws.on('message', (d) => received.push(JSON.parse(d.toString())));
  await new Promise((r) => ws.on('open', r));
  ws.send(Buffer.alloc(640), { binary: true });
  await new Promise((r) => setTimeout(r, 30));

  eq(received.length, 4, 'two utterances, two frames each');
  eq(received[0].type, 'TranscriptText', 'a commit leads with the text');
  eq(received[1].type, 'TranscriptEndpoint', 'and is closed by an endpoint');
  eq(received[2].data, 'משפט שני', 'the second utterance is sent whole, not as a delta');
  eq(received[3].type, 'TranscriptEndpoint', 'the second commit is endpointed too');
  ws.close();
  await instance.close();
}

/* The CLI starts sending audio as soon as the socket is open, but opening a
 * provider session costs a TLS handshake and an authenticated upgrade on the
 * first mic press. Frames that arrive in that window must be replayed, not
 * dropped - dropping them eats the first words of the utterance. */
async function testSlowSessionKeepsTheFirstWords() {
  const WebSocket = require('ws');
  const frames = [];
  let release;
  const opening = new Promise((r) => (release = r));
  const instance = await server.VoiceStreamServer.start({
    port: 0,
    provider: {
      id: 'slow',
      async createSession() {
        await opening;
        return {
          sendAudio: (buf) => frames.push(buf.length),
          flush: () => {},
          endSegment: async () => 'המילים הראשונות',
          close: async () => {}
        };
      }
    },
    makeEndpointer: () => new Endpointer()
  });

  const ws = new WebSocket(`ws://127.0.0.1:${instance.port}${server.VOICE_STREAM_PATH}`);
  const received = [];
  ws.on('message', (d) => received.push(JSON.parse(d.toString())));
  await new Promise((r) => ws.on('open', r));

  for (let i = 0; i < 5; i++) ws.send(Buffer.alloc(640), { binary: true });
  await new Promise((r) => setTimeout(r, 50));
  eq(frames.length, 0, 'nothing reaches a session that has not opened');

  release();
  await new Promise((r) => setTimeout(r, 50));
  eq(frames.length, 5, 'every frame from the opening window is replayed');

  const closed = new Promise((r) => ws.on('close', r));
  ws.send(JSON.stringify({ type: 'CloseStream' }));
  await closed;
  eq(received.filter((m) => m.type === 'TranscriptText')[0].data, 'המילים הראשונות', 'the utterance survives a slow open');
  await instance.close();
}

/* An engine that shows interim text and then never finalises it would leave
 * the user watching words disappear: Claude paints interims grey and drops
 * them when the mic stops. */
async function testInterimFallbackOnClose() {
  const WebSocket = require('ws');
  const instance = await server.VoiceStreamServer.start({
    port: 0,
    provider: {
      id: 'never-finalises',
      async createSession(cb) {
        return {
          sendAudio: () => cb.onInterim('מה שראיתי על המסך'),
          flush: () => {},
          endSegment: async () => '',
          close: async () => {}
        };
      }
    },
    makeEndpointer: () => new Endpointer()
  });

  const ws = new WebSocket(`ws://127.0.0.1:${instance.port}${server.VOICE_STREAM_PATH}`);
  const received = [];
  ws.on('message', (d) => received.push(JSON.parse(d.toString())));
  await new Promise((r) => ws.on('open', r));
  ws.send(Buffer.alloc(640), { binary: true });
  await new Promise((r) => setTimeout(r, 30));

  const closed = new Promise((r) => ws.on('close', r));
  ws.send(JSON.stringify({ type: 'CloseStream' }));
  await closed;

  const text = received.filter((m) => m.type === 'TranscriptText').map((m) => m.data);
  eq(text[text.length - 1], 'מה שראיתי על המסך', 'the unfinalised interim is committed on close');
  eq(received[received.length - 1].type, 'TranscriptEndpoint', 'and it is endpointed like any commit');
  await instance.close();
}

/* The fallback must not double-commit text the engine did finalise. */
async function testNoDoubleCommit() {
  const WebSocket = require('ws');
  const instance = await server.VoiceStreamServer.start({
    port: 0,
    provider: {
      id: 'finalises',
      async createSession(cb) {
        return {
          sendAudio: () => {
            cb.onInterim('שלום');
            cb.onFinal('שלום עולם');
          },
          flush: () => {},
          endSegment: async () => '',
          close: async () => {}
        };
      }
    },
    makeEndpointer: () => new Endpointer()
  });

  const ws = new WebSocket(`ws://127.0.0.1:${instance.port}${server.VOICE_STREAM_PATH}`);
  const received = [];
  ws.on('message', (d) => received.push(JSON.parse(d.toString())));
  await new Promise((r) => ws.on('open', r));
  ws.send(Buffer.alloc(640), { binary: true });
  await new Promise((r) => setTimeout(r, 30));

  const closed = new Promise((r) => ws.on('close', r));
  ws.send(JSON.stringify({ type: 'CloseStream' }));
  await closed;

  const committed = received.filter((m) => m.type === 'TranscriptEndpoint');
  eq(committed.length, 1, 'a finalised utterance is committed exactly once');
  await instance.close();
}

async function testSocketReportsProviderFailure() {
  const WebSocket = require('ws');
  const instance = await server.VoiceStreamServer.start({
    port: 0,
    provider: {
      id: 'broken',
      createSession: async () => {
        throw new Error('ElevenLabs rejected the credential');
      }
    },
    makeEndpointer: () => new Endpointer()
  });
  const ws = new WebSocket(`ws://127.0.0.1:${instance.port}${server.VOICE_STREAM_PATH}`);
  const messages = [];
  ws.on('message', (d) => messages.push(JSON.parse(d.toString())));
  await new Promise((r) => ws.on('close', r));
  eq(messages.length, 1, 'one error frame');
  eq(messages[0].type, 'TranscriptError', 'failures reach the client as TranscriptError');
  ok(messages[0].description.includes('rejected the credential'), 'the reason survives');
  await instance.close();
}

async function testHealthAndAdoption() {
  const provider = scriptedProvider({ interim: '', final: '' });
  const opts = { port: 8999, provider, makeEndpointer: () => new Endpointer() };
  const first = await server.startWithPortFallback(opts);
  eq(first.adopted, false, 'the first server binds');
  eq(await server.isOurServer(first.port), true, 'healthz identifies our server');

  const second = await server.startWithPortFallback(opts);
  eq(second.adopted, true, 'a second start adopts the first');
  eq(second.port, first.port, 'adoption reuses the port');
  eq(second.server, null, 'nothing new was bound');

  eq(await server.isOurServer(first.port + 1), false, 'an empty port is not ours');
  eq((await server.probe(first.port)).kind, 'ours', 'our own server probes as ours');
  eq((await server.probe(first.port + 1)).kind, 'none', 'an empty port probes as none');
  await first.server.close();

  /* The VS Code extension runs the same protocol on the same default port.
   * Calling that "no server" sends the user hunting for a dead socket that is
   * alive and serving the CLI perfectly well. */
  const http = require('http');
  const foreign = http.createServer((req, res) => {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ app: 'claude-code-voice', protocol: 'voice_stream', provider: 'hybrid', pid: 1234 }));
  });
  await new Promise((r) => foreign.listen(0, '127.0.0.1', r));
  const foreignPort = foreign.address().port;
  const probed = await server.probe(foreignPort);
  eq(probed.kind, 'foreign', "another app's voice_stream server is reported, not hidden");
  eq(probed.health.app, 'claude-code-voice', 'its identity is carried back');
  eq(await server.isOurServer(foreignPort), false, 'but it is never adopted as ours');
  await new Promise((r) => foreign.close(r));
}

/* The model is unloaded only after a quiet stretch, and any of the three things
 * that mean "the user is back" must call the unload off. Real timings would put
 * a ten-minute sleep in the suite, so idleUnloadMs is set to a few ticks - the
 * clock the code reads is the one the config hands it. */
/* keyterms is one list in voice.json and two wires out of it: Scribe takes it
 * as repeated query parameters, Whisper as a decoder prompt. The list reaching
 * only the cloud engine is the bug this guards - it looks configured and does
 * nothing. */
function testWhisperHotwords() {
  const cfg = config.load({ provider: 'whisper', keyterms: ['git', 'npm', 'hebrew-tty'] }, {}, path.join(os.tmpdir(), 'no-such-voice.json'));
  const provider = cli.buildProvider(cfg, () => {});
  const sent = provider.sidecarOptions();
  eq(sent.hotwords.join(','), 'git,npm,hebrew-tty', 'keyterms reach the sidecar as hotwords');

  const none = cli.buildProvider(config.load({ provider: 'whisper' }, {}, path.join(os.tmpdir(), 'no-such-voice.json')), () => {});
  eq(none.sidecarOptions().hotwords.length, 0, 'no keyterms means no prompt, not an empty one');

  // The style prompt is the stronger of the two and does a different job:
  // hotwords names terms, this names how they are written. Measured on one
  // recording, it fixed "dev" and "deployment" - neither of them in any list.
  ok(sent.initialPrompt.length > 0, 'a style prompt ships by default');
  ok(/[a-z]/.test(sent.initialPrompt) && /[\u0590-\u05ff]/.test(sent.initialPrompt),
    'the style prompt is code-switched, which is the whole point of it');
  const blank = cli.buildProvider(
    config.load(
      { provider: 'whisper', whisper: { initialPrompt: '  ' } },
      {},
      path.join(os.tmpdir(), 'no-such-voice.json')
    ),
    () => {}
  );
  eq(blank.sidecarOptions().initialPrompt, '', 'a blank style prompt is an absence, not an empty prompt');
  eq(blank.sidecarOptions().partialModel, '', 'no second model unless one is named');
  eq(new WhisperProvider({ partialModel: 'a/b' }, () => fakeSidecar()).sidecarOptions().partialModel, 'a/b',
    'the hypothesis model reaches the sidecar');

  console.log('whisper hotwords: keyterms and the style prompt reach the local decoder');
  console.log(`  ${sent.hotwords.length} terms forwarded`);
}

async function testWhisperIdleUnload() {
  const tick = () => new Promise((r) => setTimeout(r, 25));

  // 1. quiet past the deadline unloads.
  let proc = fakeSidecar();
  let w = new WhisperProvider({ idleUnloadMs: 40 }, () => proc);
  const p1 = w.createSession({ onInterim() {}, onError() {} });
  proc.say(READY);
  const s1 = await p1;
  ok(w.proc !== null, 'the sidecar is up while dictating');
  await s1.close();
  ok(w.idleTimer !== null, 'closing the microphone arms the unload');
  await tick(); await tick();
  eq(w.proc, null, 'a quiet stretch unloads the model');

  // 2. a new session inside the window keeps it loaded.
  proc = fakeSidecar();
  w = new WhisperProvider({ idleUnloadMs: 10000 }, () => proc);
  const p2 = w.createSession({ onInterim() {}, onError() {} });
  proc.say(READY);
  const s2 = await p2;
  await s2.close();
  ok(w.idleTimer !== null, 'the unload is armed again');
  const p3 = w.createSession({ onInterim() {}, onError() {} });
  const s3 = await p3;
  eq(w.idleTimer, null, 'pressing the microphone again cancels the unload');
  ok(w.proc === proc, 'and the same loaded sidecar is reused - no reload');
  await s3.close();
  w._cancelIdle();

  // 3. the opt-out is honoured, and is not raised by the one-minute floor.
  proc = fakeSidecar();
  w = new WhisperProvider({ idleUnloadMs: 0 }, () => proc);
  const p4 = w.createSession({ onInterim() {}, onError() {} });
  proc.say(READY);
  await (await p4).close();
  eq(w.idleTimer, null, 'idleUnloadMs 0 never arms an unload');
  ok(w.proc === proc, 'and the model stays resident, as it always did');
  await w.shutdown();

  eq(config.normalizeWhisper({ idleUnloadMs: 5000 }).idleUnloadMs, 60000,
    'a sub-minute idle window is raised to the floor');
  eq(config.normalizeWhisper({ idleUnloadMs: 0 }).idleUnloadMs, 0,
    'but 0 survives the floor as the explicit opt-out');
  eq(config.normalizeWhisper({}).idleUnloadMs, 300000, 'five minutes is the default');

  // 4. a preload with no session behind it still gives the model back. This is
  // the long-running server: `_ensure()` cancels the unload on the way in, and
  // nothing closes to arm it again.
  proc = fakeSidecar();
  w = new WhisperProvider({ idleUnloadMs: 40 }, () => proc);
  const p5 = w.preload();
  proc.say(READY);
  await p5;
  ok(w.idleTimer !== null, 'preloading arms the unload with no session to close');
  await tick(); await tick();
  eq(w.proc, null, 'and a preloaded model is given back after a quiet stretch');
}

// ---------- run ----------


// ---------- whisper (local engine) ----------

/* A child-process-shaped stand-in. The sidecar's own behaviour is Python's
 * business; what has to hold here is the framing and the commit bookkeeping,
 * neither of which needs a GPU to be wrong. */
function fakeSidecar() {
  const { EventEmitter } = require('events');
  const proc = new EventEmitter();
  proc.written = [];
  proc.killed = false;
  proc.stdin = {
    destroyed: false,
    write(buf) {
      proc.written.push(Buffer.from(buf));
      return true;
    },
    end() {
      this.destroyed = true;
    }
  };
  proc.stdout = new EventEmitter();
  proc.stderr = new EventEmitter();
  proc.kill = () => {
    proc.killed = true;
  };
  proc.say = (obj) => proc.stdout.emit('data', Buffer.from(`${JSON.stringify(obj)}\n`));
  return proc;
}

function decodeFrames(chunks) {
  const buf = Buffer.concat(chunks);
  const out = [];
  let i = 0;
  while (i + 5 <= buf.length) {
    const type = buf.readUInt8(i);
    const len = buf.readUInt32BE(i + 1);
    out.push({ type, payload: buf.subarray(i + 5, i + 5 + len) });
    i += 5 + len;
  }
  return out;
}

const READY = { type: 'ready', model: 'm', device: 'cuda', computeType: 'int8_float16', loadMs: 10 };

function testWhisperConfig() {
  const w = config.load({ provider: 'whisper' }, {}, '/nonexistent');
  eq(w.provider, 'whisper', 'whisper is a selectable provider');
  eq(w.whisper.model, 'ivrit-ai/whisper-large-v3-turbo-ct2', 'the Hebrew model is the local default');
  eq(w.whisper.device, 'auto', 'the device is resolved in the sidecar, not here');
  // The Scribe model check must not reach into the local engine: "model" names
  // a Scribe id and has nothing to say about which Whisper is loaded.
  const named = config.load({ provider: 'whisper', model: 'long' }, {}, '/nonexistent');
  eq(named.whisper.model, 'ivrit-ai/whisper-large-v3-turbo-ct2', 'the local model is untouched by the Scribe check');

  const typed = config.normalizeWhisper({ device: 'gpu', partialMs: 10, finalBeamSize: 99, offline: 'true' });
  eq(typed.device, 'auto', 'a device name CTranslate2 does not know falls back to auto');
  eq(typed.partialMs, 250, 'a hypothesis interval below the decode cost is clamped');
  eq(typed.finalBeamSize, 10, 'the beam size is clamped');
  eq(typed.offline, true, 'a string boolean is read as one');
  eq(config.normalizeWhisper({}).vadFilter, true, 'Silero is on by default');
  eq(config.normalizeWhisper({ vadFilter: 'false' }).vadFilter, false, 'a string boolean turns Silero off');
  eq(config.normalizeWhisper('nonsense').partialBeamSize, 1, 'a non-object whisper block falls back wholesale');
  eq(config.normalizeWhisper({}).partialModel, '', 'one model does both jobs unless a second is named');
  eq(config.normalizeWhisper({ partialModel: '  a/b  ' }).partialModel, 'a/b', 'the hypothesis model is trimmed');

  eq(cli.buildProvider(w).id, 'whisper', 'the whisper provider is built');
  // No credential is involved, so a missing one must not fail the way it does
  // for Scribe - the whole point of the local engine.
  eq(cli.buildProvider(Object.assign({}, w, { credential: '' })).id, 'whisper', 'whisper needs no credential');

  ok(/whisper-venv/.test(venvPython()), 'the venv python lives under the rtl-caret data dir');
  // A typo in whisper.python must fail naming the path that was typed, not
  // quietly run a different interpreter.
  eq(resolvePython('/definitely/not/here'), '/definitely/not/here', 'an explicitly named python is used as named');
  ok(/faster-whisper/.test(mapWhisperError({ message: 'ModuleNotFoundError: No module named faster_whisper' }, 'py').message),
    'a missing module is answered with the install command');
  eq(mapWhisperError({ message: 'CUDA failed with error out of memory' }, 'py').hint, 'vram',
    'an exhausted card is named as one');
}

async function testWhisperSession() {
  const proc = fakeSidecar();
  let spawns = 0;
  const provider = new WhisperProvider({ languageCode: 'he', settleTimeoutMs: 200 }, () => {
    spawns++;
    return proc;
  });
  const interims = [];
  const errors = [];
  const opening = provider.createSession({ onInterim: (t) => interims.push(t), onError: (e) => errors.push(e) });
  // The model takes seconds to load; nothing may be sent at it until it says so.
  eq(decodeFrames(proc.written).length, 0, 'no frames go out before the model is ready');
  proc.say(READY);
  const session = await opening;
  eq(decodeFrames(proc.written).map((f) => JSON.parse(f.payload).cmd).join('|'), 'reset',
    'a new session starts from an empty buffer');

  proc.written.length = 0;
  const pcm = Buffer.from([1, 0, 2, 0, 3, 0]);
  session.sendAudio(pcm);
  const audio = decodeFrames(proc.written);
  eq(audio.length, 1, 'one audio frame went out');
  eq(audio[0].type, 0, 'audio rides the binary type, never base64');
  eq(audio[0].payload.toString('hex'), pcm.toString('hex'), 'the samples arrive unchanged');

  proc.say({ type: 'partial', text: 'שלום', ms: 470, audioMs: 700 });
  eq(interims.join('|'), 'שלום', 'a hypothesis is painted as an interim');

  proc.written.length = 0;
  const committing = session.endSegment();
  const commit = decodeFrames(proc.written).map((f) => JSON.parse(f.payload));
  eq(commit[0].cmd, 'commit', 'endSegment asks the sidecar to transcribe');
  // A late final from the previous utterance must not answer this commit.
  proc.say({ type: 'final', id: 'stale', text: 'ישן', ms: 5, audioMs: 5 });
  proc.say({ type: 'final', id: commit[0].id, text: 'שלום עולם', ms: 470, audioMs: 3000 });
  eq(await committing, 'שלום עולם', 'the matching final is what endSegment returns');

  // The engine owes a final and never sends it: the commit has to resolve, or
  // the client sits until its own 5000 ms safety timer fires.
  const stranded = await session.endSegment();
  eq(stranded, '', 'a commit the sidecar never answers gives up on its own');

  // A second session reuses the loaded model - the whole reason the process
  // outlives the microphone.
  await session.close();
  const second = await provider.createSession({ onInterim: () => {}, onError: () => {} });
  eq(spawns, 1, 'the model is loaded once, not once per microphone press');

  proc.written.length = 0;
  const orphan = second.endSegment();
  proc.emit('exit', 1, null);
  eq(await orphan, '', 'a commit outstanding when the sidecar dies resolves rather than hangs');
  ok(errors.length === 0 || /stopped/.test(errors[errors.length - 1].message), 'the death is reported to the live session');
}

async function testWhisperSidecarFailureIsExplained() {
  const proc = fakeSidecar();
  const provider = new WhisperProvider({ languageCode: 'he' }, () => proc);
  const opening = provider.createSession({ onInterim: () => {}, onError: () => {} });
  proc.stderr.emit('data', Buffer.from('Traceback (most recent call last):\nModuleNotFoundError: No module named \'faster_whisper\'\n'));
  proc.say({ type: 'error', message: "ModuleNotFoundError: No module named 'faster_whisper'", fatal: true });
  let message = '';
  try {
    await opening;
  } catch (e) {
    message = e.message;
  }
  ok(/faster-whisper/.test(message), `the install command is in the failure: ${message}`);
  ok(/Traceback|ModuleNotFound/.test(message), 'the python traceback is kept, not swallowed');
}

// ---------- gemini ----------

function testGeminiSetup() {
  const setup = buildSetup({
    languageCode: 'he',
    secondaryLanguages: ['en'],
    keyterms: ['git', 'npm'],
    noVerbatim: false
  }).setup;
  eq(setup.model, `models/${DEFAULT_GEMINI_MODEL}`, 'the model rides as a model path');
  eq(setup.generationConfig.responseModalities[0], 'TEXT', 'transcription asks for TEXT, not AUDIO');
  // "he" and "iw" are both rejected by the Live API - it wants a region.
  eq(setup.inputAudioTranscription.languageCodes.join(','), 'he-IL,en-US', 'languages carry a region');
  eq(setup.inputAudioTranscription.mode, 'VERBATIM', 'dictation returns what was said by default');
  eq(buildSetup({ noVerbatim: true }).setup.inputAudioTranscription.mode, 'SMART', 'no-verbatim maps to SMART');
  eq(setup.inputAudioTranscription.customVocabulary.join(','), 'git,npm', 'keyterms bias as custom vocabulary');
  eq(languageCodes({ languageCode: 'he', secondaryLanguages: ['he', 'en'] }).length, 2, 'a repeated language is sent once');

  ok(/key=AIzaTest/.test(buildGeminiUrl({ credential: 'AIzaTest' })), 'the key goes on the query string');
  throwsWith(() => buildGeminiUrl({ credential: 'sk_eleven' }), 'not a Gemini API key', 'a Scribe key is caught before the mic opens');
  throwsWith(() => buildGeminiUrl({ credential: '{"type":"service_account"}' }), 'not a Gemini API key', 'service-account JSON is caught');
  eq(mapGeminiError({ error: { code: 429, status: 'RESOURCE_EXHAUSTED', message: 'quota' } }).hint, 'quota', 'quota errors are labelled');
  eq(mapGeminiError({ error: { code: 403, status: 'PERMISSION_DENIED', message: 'no' } }).hint, 'set-api-key', 'a rejected key is labelled');

  const cfg = config.load({ provider: 'gemini', geminiCredential: 'AIzaTest', model: 'scribe_v2_realtime' }, {}, '/nonexistent');
  eq(cfg.model, DEFAULT_GEMINI_MODEL, 'a leftover Scribe model is replaced when the provider switches');
  eq(cli.buildProvider(cfg).id, 'gemini', 'the gemini provider is built');
  eq(config.load({ provider: 'gemini' }, { GEMINI_API_KEY: 'AIzaEnv' }, '/nonexistent').geminiCredential, 'AIzaEnv', 'the environment key is honoured');
}

async function testGeminiSession() {
  const socket = fakeSocket();
  const interims = [];
  const finals = [];
  const session = await new GeminiProvider(
    { credential: 'AIzaTest', languageCode: 'he', chunkMs: 20 },
    async () => socket
  ).createSession({
    onInterim: (t) => interims.push(t),
    onFinal: (t) => finals.push(t),
    onError: (e) => finals.push(`ERR ${e.message}`)
  });

  eq(socket.sent[0].setup.model, `models/${DEFAULT_GEMINI_MODEL}`, 'setup is the first frame');
  // Audio before setupComplete would be dropped by the server, so it is held.
  session.sendAudio(Buffer.alloc(640));
  eq(socket.sent.length, 1, 'audio waits for the handshake');
  socket.reply({ setupComplete: {} });
  eq(socket.sent.length, 2, 'the held audio ships once setup is acknowledged');
  eq(socket.sent[1].realtimeInput.audio.mimeType, 'audio/pcm;rate=16000', 'the declared rate matches the CLI');
  eq(Buffer.from(socket.sent[1].realtimeInput.audio.data, 'base64').length, 640, 'the audio survives base64');

  socket.reply({ serverContent: { interimInputTranscription: { text: 'שלום' } } });
  eq(interims[0], 'שלום', 'an interim hypothesis surfaces as interim');
  eq(finals.length, 0, 'an interim does not commit');
  socket.reply({ serverContent: { inputTranscription: { text: 'שלום עולם' } } });
  eq(finals[0], 'שלום עולם', 'inputTranscription is the commit');

  session.sendAudio(Buffer.alloc(320));
  eq(socket.sent.length, 2, 'a part chunk is held back');
  session.flush();
  eq(Buffer.from(socket.sent[2].realtimeInput.audio.data, 'base64').length, 320, 'flush ships the tail');
  eq(socket.sent[3].realtimeInput.audioStreamEnd, true, 'flush asks for immediate finalization');
  session.sendAudio(Buffer.alloc(640));
  eq(socket.sent.length, 4, 'no audio is written after flush');

  socket.reply({ serverContent: { inputTranscription: { text: 'הזנב' } } });
  eq(finals[1], 'הזנב', 'the flushed tail commits');
  eq(await session.endSegment(), '', 'push mode commits through onFinal, not endSegment');

  // A final we asked for and never got must not hang the close path.
  const slow = fakeSocket();
  const slowSession = await new GeminiProvider(
    { credential: 'AIzaTest', settleTimeoutMs: 100 },
    async () => slow
  ).createSession({ onInterim: () => {}, onFinal: () => {}, onError: () => {} });
  slow.reply({ setupComplete: {} });
  slowSession.sendAudio(Buffer.alloc(640));
  slowSession.flush();
  const waited = Date.now();
  eq(await slowSession.endSegment(), '', 'an unanswered final falls through');
  ok(Date.now() - waited >= 90, 'endSegment waits for the final it asked for');

  const errors = [];
  const broken = fakeSocket();
  await new GeminiProvider({ credential: 'AIzaTest' }, async () => broken).createSession({
    onInterim: () => {},
    onError: (e) => errors.push(e)
  });
  broken.reply({ error: { code: 401, status: 'UNAUTHENTICATED', message: 'API key not valid' } });
  eq(errors[0].hint, 'set-api-key', 'a protocol error frame reaches the user');
}

// ---------- hybrid: fast paints, accurate commits ----------

function fakeStreamingProvider(id) {
  const sessions = [];
  return {
    id,
    sessions,
    async createSession(cb) {
      const session = {
        cb,
        audio: 0,
        flushed: false,
        sendAudio: (pcm) => {
          session.audio += pcm.length;
        },
        flush: () => {
          session.flushed = true;
        },
        endSegment: async () => session.pull || '',
        close: async () => {
          session.closed = true;
        }
      };
      sessions.push(session);
      return session;
    }
  };
}

async function testHybridPairing() {
  const fast = fakeStreamingProvider('fast');
  const accurate = fakeStreamingProvider('accurate');
  const interims = [];
  const session = await new HybridProvider(fast, accurate, { finalWaitMs: 200 }).createSession({
    onInterim: (t) => interims.push(t),
    onError: () => {}
  });
  const [fastSide, accurateSide] = [fast.sessions[0], accurate.sessions[0]];

  session.sendAudio(Buffer.alloc(640));
  eq(fastSide.audio, 640, 'the fast engine hears the microphone');
  eq(accurateSide.audio, 640, 'the accurate engine hears the same microphone');

  fastSide.cb.onInterim('שלום');
  eq(interims[0], 'שלום', 'the fast engine paints');
  // A fast final is still a hypothesis here: the accurate engine owns commits.
  fastSide.cb.onFinal('שלום עולם');
  eq(interims[1], 'שלום עולם', 'a fast final is demoted to interim');
  eq(await session.endSegment(), '', 'a mid-recording tick commits nothing');

  session.flush();
  ok(fastSide.flushed && accurateSide.flushed, 'both engines are flushed on stop');
  accurateSide.pull = 'שלום עולם, מה קורה';
  eq(await session.endSegment(), 'שלום עולם, מה קורה', 'the accurate engine wins the commit');

  // The whole point of the fallback: a dictation is never lost to the slower
  // engine going quiet or failing.
  const f2 = fakeStreamingProvider('fast');
  const a2 = fakeStreamingProvider('accurate');
  const second = await new HybridProvider(f2, a2, { finalWaitMs: 150 }).createSession({
    onInterim: () => {},
    onError: () => {}
  });
  f2.sessions[0].cb.onInterim('טקסט מהיר');
  a2.sessions[0].endSegment = () => new Promise(() => {});
  second.flush();
  const waited = Date.now();
  eq(await second.endSegment(), 'טקסט מהיר', 'a silent accurate engine falls back to the fast text');
  ok(Date.now() - waited >= 140, 'the fallback waits out its deadline first');

  const f3 = fakeStreamingProvider('fast');
  const a3 = fakeStreamingProvider('accurate');
  const third = await new HybridProvider(f3, a3, { finalWaitMs: 150 }).createSession({
    onInterim: () => {},
    onError: () => {}
  });
  f3.sessions[0].cb.onFinal('טקסט מהיר');
  a3.sessions[0].cb.onError({ message: 'quota' });
  third.flush();
  eq(await third.endSegment(), 'טקסט מהיר', 'a failed accurate engine does not blank the segment');

  // An engine that fails every time must stop being waited for, or every
  // commit pays its deadline for nothing.
  const f4 = fakeStreamingProvider('fast');
  const a4 = fakeStreamingProvider('accurate');
  const pair = new HybridProvider(f4, a4, { finalWaitMs: 50 });
  for (let i = 0; i < 3; i++) {
    const s = await pair.createSession({ onInterim: () => {}, onError: () => {} });
    f4.sessions[i].cb.onFinal('טקסט מהיר');
    a4.sessions[i].cb.onError({ message: 'quota' });
    s.flush();
    eq(await s.endSegment(), 'טקסט מהיר', `failure ${i + 1} still commits the fast text`);
  }
  ok(pair.accurateDropped, 'three failures in a row drop the accurate engine');
  const after = await pair.createSession({ onInterim: () => {}, onError: () => {} });
  eq(a4.sessions.length, 3, 'a dropped engine is not opened again');
  f4.sessions[3].cb.onFinal('אחרי הנפילה');
  after.flush();
  const solo = Date.now();
  eq(await after.endSegment(), 'אחרי הנפילה', 'the fast engine commits alone');
  ok(Date.now() - solo < 40, 'and commits without waiting out a deadline');
}

async function testTranscribeBatch() {
  const posted = [];
  const provider = new GeminiTranscribeProvider(
    { credential: 'AIzaTest', languageCode: 'he', secondaryLanguages: ['en'], keyterms: ['npm'] },
    async (body) => {
      posted.push(body);
      return { output_text: 'שלום עולם' };
    }
  );
  const session = await provider.createSession({ onInterim: () => {}, onError: () => {} });
  session.sendAudio(Buffer.alloc(640));
  session.sendAudio(Buffer.alloc(320));
  session.flush();
  eq(await session.endSegment(), 'שלום עולם', 'the batch model answers once, at the end');
  eq(posted.length, 1, 'the whole utterance goes up in one request');
  const audio = posted[0].input[0];
  // Measured: l16 with sample_rate/channels is rejected; the same PCM behind a
  // WAV header returns 200.
  eq(audio.mime_type, 'audio/wav', 'the PCM travels behind a WAV header');
  eq(audio.sample_rate, undefined, 'rate and channels are rejected for non-l16 audio');
  const wav = Buffer.from(audio.data, 'base64');
  eq(wav.length, 44 + 960, 'both frames ride the request, behind a 44-byte header');
  eq(wav.toString('ascii', 0, 4), 'RIFF', 'it is a RIFF file');
  eq(wav.readUInt32LE(24), 16000, 'the header carries the sample rate');
  eq(wav.readUInt32LE(40), 960, 'the data chunk names the payload length');
  const tc = posted[0].generation_config.transcription_config;
  eq(tc.language_codes.join(','), 'he-IL,en-US', 'languages carry a region');
  eq(tc.mode, 'verbatim', 'dictation returns what was said by default');
  eq(tc.custom_vocabulary.join(','), 'npm', 'keyterms bias the batch model too');

  // The transcript has moved between output_text and the step list once
  // already during the preview; both are read.
  eq(readTranscript({ steps: [{ content: [{ type: 'text', text: 'מהצעדים' }] }] }), 'מהצעדים', 'the step list is read as a fallback');

  const errors = [];
  const failing = new GeminiTranscribeProvider({ credential: 'AIzaTest' }, async () => {
    throw { message: 'quota', hint: 'quota' };
  });
  const bad = await failing.createSession({ onInterim: () => {}, onError: (e) => errors.push(e) });
  bad.sendAudio(Buffer.alloc(640));
  bad.flush();
  eq(await bad.endSegment(), '', 'a failed request commits nothing of its own');
  eq(errors[0].hint, 'quota', 'the failure is still reported');
}

function testHybridConfig() {
  const cfg = config.load({ provider: 'hybrid', credential: 'sk_test', geminiCredential: 'AIzaTest' }, {}, '/nonexistent');
  // Both halves default to one vendor: a pair that quietly reaches for a
  // second vendor's key is not what "switch to Gemini" asked for.
  eq(cfg.hybrid.fast, 'gemini', 'the fast half defaults to the live Gemini model');
  eq(cfg.hybrid.accurate, 'gemini-transcribe', 'the accurate half defaults to the batch model');
  eq(cfg.model, 'gemini-3.5-transcribe-live', 'model names the streaming half');
  eq(cli.buildProvider(cfg).id, 'hybrid', 'the hybrid provider is built');

  const geminiFast = config.load(
    { provider: 'hybrid', hybrid: { fast: 'gemini' }, geminiCredential: 'AIzaTest', model: 'scribe_v2_realtime' },
    {},
    '/nonexistent'
  );
  eq(geminiFast.model, 'gemini-3.5-transcribe-live', 'a leftover Scribe id is replaced when the fast half is gemini');
  // An accurate half that cannot stream would never commit anything.
  eq(config.load({ provider: 'hybrid', hybrid: { accurate: 'whisper' } }, {}, '/nonexistent').hybrid.accurate,
    'gemini-transcribe', 'a bogus accurate half falls back');
}

async function main() {
  testVad();
  testNoisyRoom();
  testCredentials();
  testHebrewTuning();
  testConfig();
  testCliParse();
  testProviderSelection();
  testWhisperConfig();
  testGeminiSetup();
  testHybridConfig();
  await testHybridPairing();
  await testTranscribeBatch();
  await testGeminiSession();
  await testElevenLabsSession();
  await testSocketRoundTrip();
  await testCommitIsAlwaysAPair();
  await testSlowSessionKeepsTheFirstWords();
  await testInterimFallbackOnClose();
  await testNoDoubleCommit();
  await testSocketReportsProviderFailure();
  await testHealthAndAdoption();
  await testWhisperSession();
  await testWhisperSidecarFailureIsExplained();
  await testWhisperIdleUnload();
  testWhisperHotwords();

  console.log(`voice: ${checks - failures}/${checks} checks passed`);
  if (failures) {
    console.log(`voice: ${failures} FAILED`);
    process.exit(1);
  }
}

main().catch((e) => {
  console.log(`voice: crashed - ${e && e.stack ? e.stack : e}`);
  process.exit(1);
});
