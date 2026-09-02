'use strict';

const { spawn } = require('child_process');
const config = require('./config');
const vad = require('./vad');
const { Endpointer } = vad;
const {
  ElevenLabsProvider,
  parseElevenLabsCredential,
  normalizeLanguage,
  keytermList,
  MAX_KEYTERMS,
  MAX_KEYTERM_LENGTH
} = require('./elevenlabs');
const { WhisperProvider, venvPython, resolvePython } = require('./whisper');
const { GeminiProvider, parseGeminiCredential } = require('./gemini');
const server = require('./server');

const ENV_VAR = 'VOICE_STREAM_BASE_URL';

const USAGE = `hebrew-voice - Hebrew dictation for Claude Code's terminal /voice

  hebrew-voice -- <command...>   run a command with dictation redirected here
  hebrew-voice serve             run the server in the foreground
  hebrew-voice status            report whether a server is reachable
  hebrew-voice env               print the export line for an existing server
  hebrew-voice setup             store the API key for the chosen provider
  hebrew-voice test [seconds]    record from the microphone and transcribe
  hebrew-voice levels [seconds]  measure this microphone: room, speech, bar

  --port <n>       port to bind or probe (default 8765)
  --provider <p>   gemini (Gemini 3.5 Transcribe Live), elevenlabs (Scribe),
                   or whisper (local faster-whisper)
  --lang <code>    ISO-639-1 language, "he" for Hebrew
  --model <id>     Scribe model id (default scribe_v2_realtime)
  --secondary <c>  other languages in the same sentence (default "en")
  --keyterm <t>    bias a word the model mishears; repeatable, max 50
  --verbose        log every protocol step

Configuration lives in ${config.configPath()}.
`;

function baseUrl(port) {
  return `ws://127.0.0.1:${port}`;
}

function parse(argv) {
  const opts = { sub: null, command: [], verbose: false, overrides: {} };
  const sep = argv.indexOf('--');
  const head = sep === -1 ? argv : argv.slice(0, sep);
  if (sep !== -1) opts.command = argv.slice(sep + 1);
  for (let i = 0; i < head.length; i++) {
    const a = head[i];
    if (a === '--port') opts.overrides.port = Number(head[++i]);
    else if (a === '--provider') opts.overrides.provider = head[++i];
    else if (a === '--lang') opts.overrides.language = head[++i];
    else if (a === '--model') opts.overrides.model = head[++i];
    else if (a === '--secondary') opts.overrides.secondaryLanguages = head[++i];
    else if (a === '--keyterm') {
      opts.overrides.keyterms = (opts.overrides.keyterms || []).concat(head[++i]);
    }
    else if (a === '--verbose') opts.verbose = true;
    else if (!opts.sub) opts.sub = a;
    else opts.args = (opts.args || []).concat(a);
  }
  return opts;
}

function buildProvider(cfg, log) {
  if (cfg.provider === 'whisper') {
    return new WhisperProvider(
      Object.assign({}, cfg.whisper, {
        languageCode: normalizeLanguage(cfg.language),
        settleTimeoutMs: cfg.settleTimeoutMs,
        keyterms: keytermList(cfg.keyterms),
        log
      })
    );
  }
  if (cfg.provider === 'gemini') {
    parseGeminiCredential(cfg.geminiCredential);
    return new GeminiProvider({
      credential: cfg.geminiCredential,
      baseUrl: cfg.baseUrl,
      model: cfg.model,
      languageCode: cfg.language,
      secondaryLanguages: cfg.secondaryLanguages,
      keyterms: cfg.keyterms,
      noVerbatim: cfg.noVerbatim,
      settleTimeoutMs: cfg.settleTimeoutMs,
      log
    });
  }
  // Fail fast on a missing or wrong-vendor key instead of erroring on the
  // first mic press, when the failure is invisible behind Claude's UI.
  parseElevenLabsCredential(cfg.credential);
  return new ElevenLabsProvider({
    credential: cfg.credential,
    baseUrl: cfg.baseUrl,
    model: cfg.model,
    commitStrategy: cfg.commitStrategy,
    languageCode: cfg.language,
    secondaryLanguages: cfg.secondaryLanguages,
    keyterms: cfg.keyterms,
    noVerbatim: cfg.noVerbatim,
    filterBackgroundAudio: cfg.filterBackgroundAudio,
    vadSilenceThresholdSecs: cfg.vadSilenceThresholdSecs,
    settleTimeoutMs: cfg.settleTimeoutMs,
    log
  });
}

function engineLabel(cfg) {
  if (cfg.provider === 'whisper') {
    return `${cfg.whisper.model} on ${cfg.whisper.device}, ${normalizeLanguage(cfg.language)}`;
  }
  return `${cfg.model}, ${normalizeLanguage(cfg.language)}`;
}

function serverOptions(cfg, verbose) {
  return {
    port: cfg.port,
    provider: buildProvider(cfg, verbose ? (m) => console.error(`[${cfg.provider}] ${m}`) : undefined),
    makeEndpointer: () =>
      new Endpointer({
        vadThreshold: cfg.vadThreshold,
        noiseRatio: cfg.vadNoiseRatio,
        endpointMs: cfg.endpointMs,
        maxSegmentMs: cfg.maxSegmentMs
      }),
    log: verbose ? (m) => console.error(`[voice] ${m}`) : undefined
  };
}

/** First port in the scan range answering our own /healthz. */
async function findRunning(port, attempts = 10) {
  for (let i = 0; i < attempts; i++) {
    if (await server.isOurServer(port + i)) return port + i;
  }
  return null;
}

/** Everything answering the voice_stream health probe in the scan range. */
async function scan(port, attempts = 10) {
  const found = [];
  for (let i = 0; i < attempts; i++) {
    const result = await server.probe(port + i);
    if (result.kind !== 'none') found.push(result);
  }
  return found;
}

async function cmdServe(cfg, verbose) {
  const { server: instance, port, adopted } = await server.startWithPortFallback(serverOptions(cfg, verbose));
  if (adopted) {
    console.error(`another hebrew-voice server already owns ${baseUrl(port)}`);
    return 0;
  }
  console.error(`hebrew-voice on ${baseUrl(port)}  (${engineLabel(cfg)})`);
  console.error(`export ${ENV_VAR}=${baseUrl(port)}`);
  // The local model takes seconds to load. A service that waits for the first
  // microphone press to find that out spends them in front of the user.
  const provider = instance && instance.provider;
  if (provider && typeof provider.preload === 'function') {
    provider.preload().then(
      (info) => console.error(`engine ready: ${info.model} on ${info.device}/${info.computeType} (${info.loadMs}ms)`),
      (e) => console.error(`engine failed to load: ${e.message}`)
    );
  }
  await new Promise((resolve) => {
    const stop = () => {
      instance.close().then(resolve, resolve);
    };
    process.on('SIGINT', stop);
    process.on('SIGTERM', stop);
  });
  return 0;
}

async function cmdRun(cfg, verbose, command) {
  if (!cfg.enabled) {
    console.error('voice is disabled in voice.json - running the command unchanged');
    return exec(command, {});
  }
  const { server: instance, port, adopted } = await server.startWithPortFallback(serverOptions(cfg, verbose));
  if (verbose) console.error(`[voice] ${adopted ? 'adopted' : 'listening on'} ${baseUrl(port)}`);
  const code = await exec(command, { [ENV_VAR]: baseUrl(port) });
  if (instance) await instance.close();
  return code;
}

function exec(command, extraEnv) {
  return new Promise((resolve) => {
    const child = spawn(command[0], command.slice(1), {
      stdio: 'inherit',
      env: Object.assign({}, process.env, extraEnv)
    });
    // The wrapped program owns the terminal: forward the signals the user aims
    // at it and let its own exit end us, rather than dying first and orphaning it.
    const forward = (sig) => child.kill(sig);
    process.on('SIGINT', forward);
    process.on('SIGTERM', forward);
    child.on('error', (e) => {
      console.error(`cannot run ${command[0]}: ${e.message}`);
      resolve(127);
    });
    child.on('exit', (code, signal) => resolve(signal ? 128 : code == null ? 0 : code));
  });
}

/* Where dictation would actually go: our own server, a foreign voice_stream
 * server, or nothing. Shared by both engines - the redirect is the same. */
function statusServers(cfg, found) {
  const ours = found.find((f) => f.kind === 'ours');
  if (process.env[ENV_VAR]) console.log(`${ENV_VAR.padEnd(10)} ${process.env[ENV_VAR]} (already in this shell)`);

  if (!found.length) {
    console.log(`server     none on ${cfg.port}..${cfg.port + 9}`);
    console.log('           dictation runs for as long as "hebrew-voice -- <command>" does');
    return 1;
  }
  for (const f of found) {
    // A foreign voice_stream server is almost always the VS Code extension,
    // which serves the same CLI just as well - saying "no server" here sends
    // you hunting for a dead socket that is alive and working.
    const who = f.kind === 'ours' ? 'ours' : `another app (${f.health.app})`;
    console.log(`server     ${baseUrl(f.port)}  ${who}, provider ${f.health.provider}, pid ${f.health.pid}`);
  }
  if (ours) console.log(`export     ${ENV_VAR}=${baseUrl(ours.port)}`);
  return ours ? 0 : 1;
}

async function cmdStatus(cfg) {
  const found = await scan(cfg.port);
  const whisper = cfg.provider === 'whisper';

  console.log(`config     ${config.configPath()}`);
  if (whisper) {
    const python = resolvePython(cfg.whisper.python);
    console.log('credential not needed - whisper runs on this machine');
    console.log(`python     ${python}${python === venvPython() ? '' : '  (not the rtl-caret venv)'}`);
  } else {
    const key = cfg.provider === 'gemini' ? cfg.geminiCredential : cfg.credential;
    const setupCmd = cfg.provider === 'gemini' ? 'hebrew-voice setup --provider gemini' : 'hebrew-voice setup';
    console.log(`credential ${key ? 'set' : `MISSING - run: ${setupCmd}`}`);
  }
  console.log(`provider   ${cfg.provider} (${engineLabel(cfg)})`);

  if (whisper) {
    // Whisper is given one language and decodes code-switched English inside
    // it on its own; there is no secondary-language list to report.
    console.log(`languages  ${normalizeLanguage(cfg.language)}`);
    console.log(`decoding   partials every ${cfg.whisper.partialMs}ms at beam ${cfg.whisper.partialBeamSize}, commit at beam ${cfg.whisper.finalBeamSize}`);
    const hot = keytermList(cfg.keyterms);
    console.log(`hotwords   ${hot.length ? `${hot.length}: ${hot.join(', ')}` : 'none - set "keyterms" in voice.json'}`);
    const prompt = cfg.whisper.initialPrompt;
    console.log(`style      ${prompt ? `"${prompt}"` : 'none - English will come back transliterated'}`);
    // With no server-side endpointer, these two numbers ARE the endpointing -
    // "it cuts me off" and "it never commits" both land here.
    console.log(`endpoint   local VAD: ${cfg.endpointMs}ms of silence commits, ${cfg.maxSegmentMs}ms caps a segment`);
    console.log(`speech     room measured over the first 300ms, ${cfg.vadNoiseRatio}x above it counts as speech (floor ${cfg.vadThreshold}), Silero ${cfg.whisper.vadFilter ? 'on' : 'OFF'}`);
    console.log('           run "hebrew-voice levels" if pauses never commit');
    return statusServers(cfg, found);
  }

  const secondary = cfg.secondaryLanguages.map(normalizeLanguage).filter((c) => c && c !== normalizeLanguage(cfg.language));
  console.log(`languages  ${normalizeLanguage(cfg.language)}${secondary.length ? ` + ${secondary.join(' ')}` : ' only'}`);
  // The knobs that decide whether speech is heard at all, and how long a pause
  // may be before the sentence is cut - the two things a user reports as
  // "it does not pick me up" and "it cuts me off".
  console.log(`audio      silence ${cfg.vadSilenceThresholdSecs == null ? 'server default (1.5s)' : `${cfg.vadSilenceThresholdSecs}s`}, background filter ${cfg.filterBackgroundAudio ? 'on' : 'off'}, no-verbatim ${cfg.noVerbatim ? 'on' : 'off'}`);
  // Silently dropped keyterms would otherwise look like the model ignoring them.
  const terms = keytermList(cfg.keyterms);
  if (cfg.keyterms.length) {
    const dropped = cfg.keyterms.length - terms.length;
    console.log(`keyterms   ${terms.join(', ')}${dropped ? `  (${dropped} dropped: over ${MAX_KEYTERM_LENGTH} chars or past ${MAX_KEYTERMS})` : ''}`);
  }
  return statusServers(cfg, found);
}

async function cmdEnv(cfg) {
  const port = await findRunning(cfg.port);
  if (port === null) {
    console.error('no hebrew-voice server running - start one with: hebrew-voice serve');
    return 1;
  }
  console.log(`export ${ENV_VAR}=${baseUrl(port)}`);
  return 0;
}

function readStdin() {
  return new Promise((resolve) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (c) => (data += c));
    process.stdin.on('end', () => resolve(data.trim()));
  });
}

async function cmdSetup(overrides) {
  // The key is stored per vendor, so which one is being asked for has to be
  // settled before the prompt: --provider, else whatever is configured.
  const provider = overrides.provider || config.load({}).provider;
  const gemini = provider === 'gemini';
  console.error(`Paste your ${gemini ? 'Gemini' : 'ElevenLabs'} API key, then Ctrl-D:`);
  const credential = await readStdin();
  if (!credential) {
    console.error('nothing read - aborted');
    return 1;
  }
  const patch = gemini ? { geminiCredential: credential, provider } : { credential };
  if (overrides.language) patch.language = overrides.language;
  if (overrides.model) patch.model = overrides.model;
  if (overrides.secondaryLanguages) patch.secondaryLanguages = overrides.secondaryLanguages;
  if (overrides.keyterms) patch.keyterms = overrides.keyterms;
  if (overrides.port) patch.port = overrides.port;
  const cfg = config.load(patch);
  // Reject the credential now, at the one moment the user is looking at it.
  if (gemini) parseGeminiCredential(cfg.geminiCredential);
  else parseElevenLabsCredential(cfg.credential);
  const file = config.save(patch);
  console.error(`saved to ${file} (mode 0600)`);
  return 0;
}

const RECORDERS = [
  ['arecord', ['-q', '-f', 'S16_LE', '-r', '16000', '-c', '1', '-t', 'raw', '-d', null]],
  ['sox', ['-q', '-d', '-t', 'raw', '-b', '16', '-e', 'signed-integer', '-r', '16000', '-c', '1', '-', 'trim', '0', null]]
];

/* First recorder that actually produces audio, with the frame it produced -
 * dropping that frame would eat the start of the recording. */
async function startRecorder(secs) {
  for (const [bin, template] of RECORDERS) {
    const args = template.map((a) => (a === null ? String(secs) : a));
    const child = spawn(bin, args, { stdio: ['ignore', 'pipe', 'ignore'] });
    const first = await new Promise((resolve) => {
      child.on('error', () => resolve(null));
      child.stdout.once('data', (chunk) => resolve(chunk));
      // No audio at all within a second means this recorder is not working.
      setTimeout(() => resolve(child.exitCode === null ? Buffer.alloc(0) : null), 1000);
    });
    if (first) return { child, first };
    child.kill();
  }
  return null;
}

/* Records from the microphone and drives the real socket, so a green result
 * proves the whole chain - protocol, provider, credential - not just the API. */
async function cmdTest(cfg, verbose, seconds) {
  const secs = Number(seconds) > 0 ? Number(seconds) : 5;
  const { server: instance, port } = await server.startWithPortFallback(serverOptions(cfg, verbose));
  const WebSocket = require('ws');
  const ws = new WebSocket(`${baseUrl(port)}${server.VOICE_STREAM_PATH}`);
  const transcripts = [];
  let recorder = null;

  const finished = new Promise((resolve) => {
    ws.on('message', (data) => {
      let msg;
      try {
        msg = JSON.parse(data.toString());
      } catch (e) {
        return;
      }
      if (msg.type === 'TranscriptText') transcripts.push(msg.data);
      else if (msg.type === 'TranscriptError') console.error(`error: ${msg.description}`);
    });
    ws.on('close', resolve);
    ws.on('error', (e) => {
      console.error(`socket: ${e.message}`);
      resolve();
    });
  });

  await new Promise((resolve, reject) => {
    ws.on('open', resolve);
    ws.on('error', reject);
  });

  const started = await startRecorder(secs);
  if (started) {
    recorder = started.child;
    if (started.first.length) ws.send(started.first, { binary: true });
  }
  if (!recorder) {
    console.error('no working recorder found (tried arecord, sox) - install one to use "voice test"');
    ws.close();
    if (instance) await instance.close();
    return 1;
  }

  console.error(`recording ${secs}s - speak now`);
  recorder.stdout.on('data', (chunk) => ws.send(chunk, { binary: true }));
  await new Promise((resolve) => recorder.on('exit', resolve));
  ws.send(JSON.stringify({ type: 'CloseStream' }));
  await finished;
  if (instance) await instance.close();

  const text = transcripts.join(' ').trim();
  if (!text) {
    console.error('no transcript came back');
    return 1;
  }
  console.log(text);
  return 0;
}

/* Records without transcribing and reports the levels the endpointer works
 * from. This is the answer to "dictation only commits when I let go": either
 * the room clears the bar, or speech does not, and no amount of staring at
 * transcripts will say which. Nothing is written to disk and nothing is sent
 * anywhere - the audio is turned into one RMS number per 20 ms and dropped. */
async function cmdLevels(cfg, seconds) {
  const secs = Number(seconds) > 0 ? Number(seconds) : 8;
  const started = await startRecorder(secs);
  if (!started) {
    console.error('no working recorder found (tried arecord, sox)');
    return 1;
  }
  const recorder = started.child;
  console.error(`recording ${secs}s - stay QUIET for the first 3, then talk normally`);

  const FRAME_BYTES = 640; // 20 ms of 16 kHz mono linear16
  const levels = [];
  let carry = started.first;
  recorder.stdout.on('data', (chunk) => {
    let buf = carry.length ? Buffer.concat([carry, chunk]) : chunk;
    let off = 0;
    for (; off + FRAME_BYTES <= buf.length; off += FRAME_BYTES) {
      let sum = 0;
      for (let i = 0; i < FRAME_BYTES; i += 2) {
        const s = buf.readInt16LE(off + i) / 32768;
        sum += s * s;
      }
      levels.push(Math.sqrt(sum / (FRAME_BYTES / 2)));
    }
    carry = buf.subarray(off);
  });
  await new Promise((resolve) => recorder.on('exit', resolve));

  if (levels.length < 50) {
    console.error(`only ${levels.length} frames arrived - the microphone produced almost nothing`);
    return 1;
  }
  const pct = (arr, p) => {
    const sorted = arr.slice().sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))];
  };
  const f = (n) => n.toFixed(4);
  const quiet = levels.slice(0, Math.min(levels.length, 150));   // the first 3 s
  const room = pct(quiet, 0.5);
  const loud = pct(levels, 0.9);
  const bar = Math.max(cfg.vadThreshold, room * cfg.vadNoiseRatio);

  // Where the level sits second by second. A microphone that opens loud and
  // settles has a room that cannot be measured from the first frames, and that
  // is invisible in any single number.
  const BUCKET = 25; // 500 ms
  const timeline = [];
  for (let i = 0; i + BUCKET <= levels.length; i += BUCKET) {
    timeline.push(pct(levels.slice(i, i + BUCKET), 0.5));
  }
  const top = Math.max(...timeline) || 1;
  console.log(`frames     ${levels.length} (${(levels.length * 20) / 1000}s)`);
  console.log('timeline   median per 500ms, loudest bar is the loudest half-second');
  timeline.forEach((v, i) => {
    const bar = '#'.repeat(Math.max(1, Math.round((v / top) * 40)));
    console.log(`  ${String((i * BUCKET * 20) / 1000).padStart(5)}s ${f(v)} ${bar}`);
  });
  console.log(`room       ${f(room)}   median of the quiet opening`);
  console.log(`speech     ${f(pct(levels, 0.5))} median, ${f(loud)} at the 90th, ${f(Math.max(...levels))} peak`);
  console.log(`headroom   speech sits ${(loud / (room || 1e-9)).toFixed(1)}x over the room`);
  console.log(`bar        ${f(bar)}   what the endpointer would ask speech to beat`);
  // Whether this microphone can be endpointed at all, and at what ratio.
  if (loud <= room * 1.3) {
    console.log('verdict    speech is not louder than the room - check the input gain or use a closer mic');
  } else if (loud > bar) {
    console.log(`verdict    OK - speech clears the bar, pauses will commit`);
  } else {
    const ratio = Math.max(1.2, Math.floor((loud / room) * 10) / 10 - 0.3);
    console.log(`verdict    the bar is too high for this microphone - set "vadNoiseRatio": ${ratio} in voice.json`);
  }
  return 0;
}

async function run(argv) {
  const opts = parse(argv);
  if (opts.sub === 'help' || opts.sub === '--help' || opts.sub === '-h') {
    process.stdout.write(USAGE);
    return 0;
  }
  if (opts.sub === 'setup') return cmdSetup(opts.overrides);

  let cfg;
  try {
    cfg = config.load(opts.overrides);
  } catch (e) {
    console.error(e.message);
    return 1;
  }

  try {
    if (opts.command.length) return await cmdRun(cfg, opts.verbose, opts.command);
    if (opts.sub === 'serve') return await cmdServe(cfg, opts.verbose);
    if (opts.sub === 'status' || !opts.sub) return await cmdStatus(cfg);
    if (opts.sub === 'env') return await cmdEnv(cfg);
    if (opts.sub === 'test') return await cmdTest(cfg, opts.verbose, (opts.args || [])[0]);
    if (opts.sub === 'levels') return await cmdLevels(cfg, (opts.args || [])[0]);
  } catch (e) {
    console.error(e.message || String(e));
    if (e.hint === 'set-api-key' && !e.message.includes('voice setup')) {
      console.error('run: hebrew-voice setup');
    }
    return 1;
  }

  console.error(`unknown voice command: ${opts.sub}`);
  process.stdout.write(USAGE);
  return 1;
}

module.exports = { run, parse, buildProvider, engineLabel, USAGE, ENV_VAR };
