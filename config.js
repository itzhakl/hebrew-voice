'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');

const { DEFAULT_MODEL, DEFAULT_BASE_URL, toList } = require('./elevenlabs');

const DEFAULTS = {
  enabled: true,
  provider: 'elevenlabs',
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
  if (cfg.provider !== 'elevenlabs') cfg.provider = DEFAULTS.provider;
  // A voice.json written for the Google backend names a Chirp model ("long",
  // "chirp_3"). Sending that as model_id only earns an invalid_request at mic
  // time, so anything not a Scribe model falls back to the default.
  if (!/^scribe/.test(String(cfg.model || ''))) cfg.model = DEFAULTS.model;
  if (cfg.commitStrategy !== 'manual') cfg.commitStrategy = 'vad';
  // A hand-edited voice.json may hold a bare string where a list belongs.
  cfg.secondaryLanguages = toList(cfg.secondaryLanguages);
  cfg.keyterms = toList(cfg.keyterms);
  cfg.noVerbatim = bool(cfg.noVerbatim);
  cfg.settleTimeoutMs = Math.min(4500, Math.max(500, num(cfg.settleTimeoutMs, DEFAULTS.settleTimeoutMs)));
  cfg.filterBackgroundAudio = bool(cfg.filterBackgroundAudio);
  // The server rejects anything outside 0.3-3.0 rather than clamping it.
  cfg.vadSilenceThresholdSecs =
    cfg.vadSilenceThresholdSecs == null
      ? null
      : Math.min(3, Math.max(0.3, num(cfg.vadSilenceThresholdSecs, 1.5)));
  return cfg;
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

module.exports = { DEFAULTS, load, save, configPath, configDir, readFileConfig, resolveCredential };
