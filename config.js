'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');

/* Measured on live Hebrew (2026-07): chirp_3 streams no interim text until the
 * stream closes; "long" streams interims ~0.2-0.4 s behind speech and
 * finalizes +0.77 s after flush, at equal or better quality. "long" is the
 * latency winner by ~3x - but that is language-dependent. chirp_3 measured
 * faster from the US multi-region than from eu. */
const DEFAULTS = {
  enabled: true,
  provider: 'hybrid',
  language: 'iw-IL',
  projectId: '',
  location: 'eu',
  model: 'long',
  hybridFinalModel: 'chirp_3',
  hybridFinalLocation: 'us',
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

/* The credential may live in the config file, inline in the environment, or in
 * a file the environment points at - the shape voice-shim already used. */
function resolveCredential(fileCfg, env) {
  const inline = (env.GOOGLE_STT_CREDENTIAL || '').trim();
  if (inline) return inline;
  const keyFile = (env.GOOGLE_APPLICATION_CREDENTIALS || '').trim();
  if (keyFile && fs.existsSync(keyFile)) return fs.readFileSync(keyFile, 'utf8').trim();
  return String(fileCfg.credential || '').trim();
}

function num(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
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
  if (env.GOOGLE_STT_PROJECT_ID) cfg.projectId = env.GOOGLE_STT_PROJECT_ID;
  for (const [key, value] of Object.entries(overrides)) {
    if (value !== undefined) cfg[key] = value;
  }
  cfg.port = num(cfg.port, DEFAULTS.port);
  if (cfg.provider !== 'chirp' && cfg.provider !== 'hybrid') cfg.provider = DEFAULTS.provider;
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
