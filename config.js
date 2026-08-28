'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');

const { DEFAULT_MODEL, DEFAULT_BASE_URL, toList } = require('./elevenlabs');
const { DEFAULT_MODEL: DEFAULT_WHISPER_MODEL } = require('./whisper');

const PROVIDERS = new Set(['elevenlabs', 'whisper']);

/* The local engine's knobs live in their own object: none of them mean
 * anything to Scribe, and a flat namespace would make "model" ambiguous. */
const WHISPER_DEFAULTS = {
  model: DEFAULT_WHISPER_MODEL,
  // Resolved in the sidecar, which is the only place that can ask CTranslate2
  // whether a usable card is actually there.
  device: 'auto',
  computeType: 'auto',
  // Empty means the venv under ~/.local/share/rtl-caret.
  python: '',
  cacheDir: '',
  offline: false,
  // A hypothesis costs one pass, ~470 ms, and the card is idle between them.
  // The cadence the user reads is this plus that pass.
  partialMs: 400,
  // The hypothesis model. Empty means the accurate model does both jobs. The
  // encoder cost is flat and dominates a hypothesis, so the only way under it
  // is a smaller model - the commit corrects whatever it got wrong.
  partialModel: '',
  // A hypothesis that will be replaced in under a second does not earn a beam
  // search; the commit does.
  partialBeamSize: 1,
  finalBeamSize: 5,
  // Silero in front of the decoder. Fed room noise, Whisper does not return
  // nothing - it invents a sentence. Off only if it ever eats quiet speech.
  vadFilter: true,
  // Whisper's hotwords list biases the decoder at named terms; this biases it
  // at a WAY OF WRITING. ivrit-ai learned Hebrew transcripts where English is
  // transliterated, so "git commit" comes back "גיט קומית" however many times
  // the words appear in hotwords. One example sentence in the target style
  // fixes terms that are not in any list - measured: "dev" and "deployment"
  // came back in Latin without ever being named.
  initialPrompt: 'תעשה git commit ואז push ל-branch, ואז תריץ את ה-deployment עם docker ו-npm.',
  startupTimeoutMs: 120000,
  // The model is ~1.3 GB resident and a desktop that dictates twice a day
  // pays for it around the clock - swapped out, so the next press waits on
  // the disk anyway. Unloading after a quiet stretch trades that for a
  // reload the user asked for. 0 keeps the old always-resident behaviour.
  idleUnloadMs: 600000
};

const DEFAULTS = {
  enabled: true,
  provider: 'elevenlabs',
  whisper: WHISPER_DEFAULTS,
  language: 'he',
  model: DEFAULT_MODEL,
  baseUrl: DEFAULT_BASE_URL,
  commitStrategy: 'vad',
  // Dictating into a terminal is code-switched by nature: paths, commands and
  // library names arrive in English mid-Hebrew-sentence. Naming English as a
  // secondary language is what stops them being transliterated into Hebrew
  // letters - the job the two-engine hybrid provider used to do.
  secondaryLanguages: ['en'],
  // Words the model would otherwise mishear. Max 50, each at most 20 chars.
  keyterms: [],
  // Strips fillers and false starts. Off by default: it also edits speech that
  // was not a filler, and dictation should return what was said.
  noVerbatim: false,
  filterBackgroundAudio: false,
  // null leaves the server on its own 1.5 s. Lower commits sooner mid-sentence.
  vadSilenceThresholdSecs: null,
  // How long endSegment waits for the commit the engine still owes us. The
  // client's hard budget after CloseStream is 5000 ms, so this must leave room
  // for the socket to close inside it.
  settleTimeoutMs: 3000,
  port: 8765,
  vadThreshold: 0.005,
  // How much louder than the measured room speech has to be. Three is about
  // 10 dB and suits a headset; a laptop microphone with its gain wound up
  // hears itself almost as loudly as it hears you, and needs less.
  // `hebrew-voice levels` measures both and names the number.
  vadNoiseRatio: 3,
  endpointMs: 600,
  maxSegmentMs: 12000,
  credential: ''
};

function configDir() {
  const base = process.env.XDG_CONFIG_HOME || path.join(os.homedir(), '.config');
  return path.join(base, 'rtl-caret');
}

function configPath() {
  return process.env.RTL_VOICE_CONFIG || path.join(configDir(), 'voice.json');
}

function readFileConfig(file = configPath()) {
  if (!fs.existsSync(file)) return {};
  try {
    const parsed = JSON.parse(fs.readFileSync(file, 'utf8'));
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch (e) {
    throw new Error(`${file} is not valid JSON: ${e.message}`);
  }
}

/* The key may live in the config file or in the environment. XI_API_KEY is
 * what ElevenLabs' own tooling exports, so both spellings are honoured. */
function resolveCredential(fileCfg, env) {
  const inline = (env.ELEVENLABS_API_KEY || env.XI_API_KEY || '').trim();
  if (inline) return inline;
  return String(fileCfg.credential || '').trim();
}

function num(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function bool(value) {
  return value === true || value === 'true' || value === 1 || value === '1';
}

/* File first, then environment overrides, then explicit CLI flags. */
function load(overrides = {}, env = process.env, file = configPath()) {
  const fileCfg = readFileConfig(file);
  const cfg = Object.assign({}, DEFAULTS, fileCfg, {
    credential: resolveCredential(fileCfg, env)
  });
  if (env.RTL_VOICE_PORT) cfg.port = num(env.RTL_VOICE_PORT, cfg.port);
  if (env.RTL_VOICE_LANGUAGE) cfg.language = env.RTL_VOICE_LANGUAGE;
  if (env.RTL_VOICE_PROVIDER) cfg.provider = env.RTL_VOICE_PROVIDER;
  if (env.RTL_VOICE_MODEL) cfg.model = env.RTL_VOICE_MODEL;
  for (const [key, value] of Object.entries(overrides)) {
    if (value !== undefined) cfg[key] = value;
  }
  cfg.port = num(cfg.port, DEFAULTS.port);
  if (!PROVIDERS.has(cfg.provider)) cfg.provider = DEFAULTS.provider;
  // A voice.json written for the Google backend names a Chirp model ("long",
  // "chirp_3"). Sending that as model_id only earns an invalid_request at mic
  // time, so anything not a Scribe model falls back to the default. Only the
  // remote engine is checked: `model` names a Scribe id, and the local engine
  // names its own under `whisper.model`.
  if (cfg.provider === 'elevenlabs' && !/^scribe/.test(String(cfg.model || ''))) {
    cfg.model = DEFAULTS.model;
  }
  cfg.whisper = normalizeWhisper(cfg.whisper);
  if (cfg.commitStrategy !== 'manual') cfg.commitStrategy = 'vad';
  // A hand-edited voice.json may hold a bare string where a list belongs.
  cfg.secondaryLanguages = toList(cfg.secondaryLanguages);
  cfg.keyterms = toList(cfg.keyterms);
  cfg.noVerbatim = bool(cfg.noVerbatim);
  cfg.vadNoiseRatio = Math.min(10, Math.max(1.2, num(cfg.vadNoiseRatio, DEFAULTS.vadNoiseRatio)));
  cfg.settleTimeoutMs = Math.min(4500, Math.max(500, num(cfg.settleTimeoutMs, DEFAULTS.settleTimeoutMs)));
  cfg.filterBackgroundAudio = bool(cfg.filterBackgroundAudio);
  // The server rejects anything outside 0.3-3.0 rather than clamping it.
  cfg.vadSilenceThresholdSecs =
    cfg.vadSilenceThresholdSecs == null
      ? null
      : Math.min(3, Math.max(0.3, num(cfg.vadSilenceThresholdSecs, 1.5)));
  return cfg;
}

function oneOf(value, allowed, fallback) {
  const v = String(value || '').trim();
  return allowed.includes(v) ? v : fallback;
}

/* A file config replaces the whole `whisper` object rather than merging into
 * it, so every key has to be filled back in here. */
function normalizeWhisper(raw) {
  const w = Object.assign({}, WHISPER_DEFAULTS, raw && typeof raw === 'object' ? raw : {});
  w.model = String(w.model || '').trim() || WHISPER_DEFAULTS.model;
  w.device = oneOf(w.device, ['auto', 'cuda', 'cpu'], 'auto');
  w.computeType = String(w.computeType || '').trim() || 'auto';
  w.partialModel = String(w.partialModel || '').trim();
  w.initialPrompt = w.initialPrompt == null ? '' : String(w.initialPrompt).trim();
  w.python = String(w.python || '').trim();
  w.cacheDir = String(w.cacheDir || '').trim();
  w.offline = bool(w.offline);
  // Below ~250 ms the card spends its whole time on hypotheses that are
  // replaced before anyone reads them, and the commit queues behind them.
  w.partialMs = Math.max(250, num(w.partialMs, WHISPER_DEFAULTS.partialMs));
  w.partialBeamSize = Math.min(5, Math.max(1, Math.round(num(w.partialBeamSize, 1))));
  w.finalBeamSize = Math.min(10, Math.max(1, Math.round(num(w.finalBeamSize, 5))));
  w.vadFilter = w.vadFilter !== false && !['false', '0', 'off'].includes(String(w.vadFilter).toLowerCase());
  w.startupTimeoutMs = Math.max(5000, num(w.startupTimeoutMs, WHISPER_DEFAULTS.startupTimeoutMs));
  // Anything under a minute would unload between two sentences of the same
  // thought; 0 is the explicit opt-out and must survive the floor.
  const idle = num(w.idleUnloadMs, WHISPER_DEFAULTS.idleUnloadMs);
  w.idleUnloadMs = idle <= 0 ? 0 : Math.max(60000, idle);
  return w;
}

/* 0600 from creation, never after: the credential must not exist on disk
 * world-readable even briefly. */
function save(patch, file = configPath()) {
  const dir = path.dirname(file);
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  const current = readFileConfig(file);
  const next = Object.assign({}, current, patch);
  const tmp = `${file}.tmp`;
  fs.writeFileSync(tmp, `${JSON.stringify(next, null, 2)}\n`, { encoding: 'utf8', mode: 0o600 });
  fs.renameSync(tmp, file);
  return file;
}

module.exports = {
  DEFAULTS,
  WHISPER_DEFAULTS,
  PROVIDERS,
  load,
  save,
  configPath,
  configDir,
  readFileConfig,
  resolveCredential,
  normalizeWhisper
};
