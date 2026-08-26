'use strict';

/* Local dictation: faster-whisper in a Python sidecar behind a pipe.
 *
 * Nothing leaves the machine and there is no credential, which is the whole
 * reason to prefer it over Scribe. The costs are the mirror image: the model
 * takes seconds to load and a card to run on, so the process is started once
 * and outlives every microphone press.
 *
 * Whisper has no server-side endpointer, so unlike Scribe this provider is
 * PULL, not push: the local energy VAD in vad.js decides when an utterance
 * ended, endSegment() asks the sidecar to transcribe what it has, and the text
 * comes back as the return value. onFinal is never called.
 *
 * Wire format, chosen so audio never has to be base64'd:
 *   to sidecar    [1 byte type][4 byte big-endian length][payload]
 *   from sidecar  one JSON object per line
 */

const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const SAMPLE_RATE = 16000;
const TYPE_AUDIO = 0;
const TYPE_COMMAND = 1;
const HEADER_BYTES = 5;

const DEFAULT_MODEL = 'ivrit-ai/whisper-large-v3-turbo-ct2';
const SIDECAR_PATH = path.join(__dirname, 'whisper_sidecar.py');

/* Loading large-v3-turbo off a cold page cache is tens of seconds, and the
 * first press after a reboot is exactly when it happens. Failing at 30 s would
 * report "whisper is broken" for something that was merely still reading. */
const DEFAULT_STARTUP_TIMEOUT_MS = 120000;

const SETUP_HINT = 'install-whisper';

function sttError(message, hint) {
  const e = new Error(message);
  if (hint) e.hint = hint;
  return e;
}

function dataDir() {
  const base = process.env.XDG_DATA_HOME || path.join(os.homedir(), '.local', 'share');
  return path.join(base, 'rtl-caret');
}

function venvPython() {
  return path.join(dataDir(), 'whisper-venv', 'bin', 'python');
}

/* The venv is preferred over whatever python is on PATH: faster-whisper pulls
 * in CUDA libraries that have no business in a system interpreter. A python
 * the user named explicitly is returned even when it is not there - silently
 * substituting another one turns a typo into "whisper picked the wrong GPU". */
function resolvePython(explicit) {
  const named = String(explicit || process.env.RTL_VOICE_PYTHON || '').trim();
  if (named) return named;
  return fs.existsSync(venvPython()) ? venvPython() : 'python3';
}

function setupInstructions(python) {
  return [
    `whisper needs faster-whisper in ${python}`,
    `  python3 -m venv ${path.join(dataDir(), 'whisper-venv')}`,
    `  ${venvPython()} -m pip install faster-whisper nvidia-cudnn-cu12`
  ].join('\n');
}

function mapWhisperError(raw, python) {
  const text = String((raw && (raw.message || raw.error)) || raw || '').slice(0, 400);
  const m = text.toLowerCase();
  if (m.includes('enoent') || m.includes('no such file or directory')) {
    return { message: `cannot start ${python}\n${setupInstructions(python)}`, hint: SETUP_HINT };
  }
  if (m.includes('modulenotfounderror') || m.includes('no module named')) {
    return { message: `${text}\n${setupInstructions(python)}`, hint: SETUP_HINT };
  }
  if (m.includes('out of memory') || m.includes('cuda_error_out_of_memory')) {
    return {
      message: `the GPU ran out of memory loading the model - ${text}\nset whisper.device to "cpu", or a smaller whisper.model`,
      hint: 'vram'
    };
  }
  if (m.includes('couldn\'t find') || m.includes('repository not found') || m.includes('offline')) {
    return { message: `the model could not be fetched - ${text}`, hint: SETUP_HINT };
  }
  return { message: text || 'the whisper sidecar failed without a reason' };
}

function encodeFrame(type, payload) {
  const head = Buffer.alloc(HEADER_BYTES);
  head.writeUInt8(type, 0);
  head.writeUInt32BE(payload.length, 1);
  return Buffer.concat([head, payload]);
}

class WhisperProvider {
  /* processFactory is injectable so the protocol can be tested without a GPU,
   * a model, or a Python interpreter. It returns a child-process-shaped object. */
  constructor(opts = {}, processFactory) {
    this.id = 'whisper';
    this.opts = opts;
    this.log = typeof opts.log === 'function' ? opts.log : () => {};
    this.python = resolvePython(opts.python);
    this.processFactory = processFactory || (() => this._spawn());
    this.proc = null;
    this.starting = null;
    this.info = null;
    // One microphone, one audio buffer in the sidecar: only the newest session
    // may be handed hypotheses, or a stale one paints over live text.
    this.session = null;
    this.pending = new Map();
    this.commitSeq = 0;
    this.stderrTail = [];
    this.idleTimer = null;
  }

  sidecarOptions() {
    const o = this.opts;
    return {
      model: o.model || DEFAULT_MODEL,
      device: o.device || 'auto',
      computeType: o.computeType || 'auto',
      language: o.languageCode || 'he',
      cacheDir: o.cacheDir || '',
      offline: !!o.offline,
      partialMs: o.partialMs == null ? 400 : o.partialMs,
      partialBeamSize: o.partialBeamSize == null ? 1 : o.partialBeamSize,
      finalBeamSize: o.finalBeamSize == null ? 5 : o.finalBeamSize,
      vadFilter: o.vadFilter !== false
    };
  }

  _spawn() {
    return spawn(this.python, [SIDECAR_PATH], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: Object.assign({}, process.env, {
        WHISPER_OPTS: JSON.stringify(this.sidecarOptions()),
        PYTHONUNBUFFERED: '1'
      })
    });
  }

  /* Resolves once the model is loaded and warmed. Every caller shares the one
   * load; a failed load clears itself so the next microphone press retries. */
  _ensure() {
    this._cancelIdle();
    if (this.starting) return this.starting;
    this.starting = new Promise((resolve, reject) => {
      let proc;
      try {
        proc = this.processFactory();
      } catch (e) {
        const mapped = mapWhisperError(e, this.python);
        return reject(sttError(mapped.message, mapped.hint));
      }
      this.proc = proc;
      this.stderrTail = [];

      const fail = (e) => {
        const mapped = mapWhisperError(e, this.python);
        const tail = this.stderrTail.join('').trim().split('\n').slice(-4).join('\n');
        reject(sttError(tail ? `${mapped.message}\n${tail}` : mapped.message, mapped.hint));
      };

      const timer = setTimeout(() => {
        fail({ message: `the whisper sidecar did not become ready within ${this.startupTimeoutMs()}ms` });
        this._teardown();
      }, this.startupTimeoutMs());
      if (timer.unref) timer.unref();

      let buffered = '';
      if (proc.stdout) {
        proc.stdout.on('data', (chunk) => {
          buffered += chunk.toString('utf8');
          let nl;
          while ((nl = buffered.indexOf('\n')) !== -1) {
            const line = buffered.slice(0, nl);
            buffered = buffered.slice(nl + 1);
            if (line.trim()) this._onLine(line, { resolve, fail, timer });
          }
        });
      }
      // Python writes its tracebacks here, and that text is the only useful
      // thing to show when a load fails - "spawn failed" explains nothing.
      if (proc.stderr) {
        proc.stderr.on('data', (chunk) => {
          const text = chunk.toString('utf8');
          this.stderrTail.push(text);
          if (this.stderrTail.length > 40) this.stderrTail.shift();
          this.log(`sidecar stderr: ${text.trim()}`);
        });
      }
      proc.on('error', (e) => {
        clearTimeout(timer);
        fail(e);
        this._teardown();
      });
      proc.on('exit', (code, signal) => {
        clearTimeout(timer);
        this.log(`sidecar exited (code ${code}, signal ${signal})`);
        fail({ message: `the whisper sidecar exited (code ${code}${signal ? `, ${signal}` : ''})` });
        this._teardown();
      });
    }).catch((e) => {
      this.starting = null;
      throw e;
    });
    return this.starting;
  }

  startupTimeoutMs() {
    const n = Number(this.opts.startupTimeoutMs);
    return Number.isFinite(n) && n > 0 ? n : DEFAULT_STARTUP_TIMEOUT_MS;
  }

  _onLine(line, startup) {
    let msg;
    try {
      msg = JSON.parse(line);
    } catch (e) {
      this.log(`sidecar said something that is not JSON: ${line.slice(0, 200)}`);
      return;
    }
    if (msg.type === 'ready') {
      clearTimeout(startup.timer);
      this.info = msg;
      this.log(`sidecar ready: ${msg.model} on ${msg.device}/${msg.computeType} in ${msg.loadMs}ms`);
      startup.resolve(msg);
      return;
    }
    if (msg.type === 'warning') {
      this.log(`sidecar warning: ${msg.message}`);
      return;
    }
    if (msg.type === 'partial') {
      this.log(`partial: ${String(msg.text || '').length} chars, ${msg.ms}ms on ${msg.audioMs}ms of audio`);
      if (this.session && !this.session.closed && msg.text) this.session.cb.onInterim(msg.text);
      return;
    }
    if (msg.type === 'final') {
      const waiter = this.pending.get(String(msg.id));
      this.log(`final for ${msg.id}: ${String(msg.text || '').length} chars, ${msg.ms}ms on ${msg.audioMs}ms of audio`);
      if (!waiter) return;
      this.pending.delete(String(msg.id));
      clearTimeout(waiter.timer);
      waiter.resolve(String(msg.text || '').trim());
      return;
    }
    if (msg.type === 'error') {
      if (msg.fatal) {
        clearTimeout(startup.timer);
        startup.fail(msg);
        this._teardown();
        return;
      }
      this.log(`sidecar error: ${msg.message}`);
      if (this.session && !this.session.closed) this.session.cb.onError(mapWhisperError(msg, this.python));
    }
  }

  /* The process is gone: every commit still waiting for it must resolve rather
   * than hang until the client's 5000 ms safety timer fires. */
  _teardown() {
    this._cancelIdle();
    this.proc = null;
    this.starting = null;
    this.info = null;
    for (const [, waiter] of this.pending) {
      clearTimeout(waiter.timer);
      waiter.resolve('');
    }
    this.pending.clear();
    if (this.session && !this.session.closed) {
      this.session.closed = true;
      this.session.cb.onError({ message: 'the whisper sidecar stopped' });
    }
  }

  _write(frame) {
    if (!this.proc || !this.proc.stdin || this.proc.stdin.destroyed) return false;
    try {
      this.proc.stdin.write(frame);
      return true;
    } catch (e) {
      this.log(`write to sidecar failed: ${e.message}`);
      return false;
    }
  }

  _command(obj) {
    return this._write(encodeFrame(TYPE_COMMAND, Buffer.from(JSON.stringify(obj), 'utf8')));
  }

  async createSession(cb) {
    await this._ensure();
    this._cancelIdle();
    const settleTimeoutMs = this.opts.settleTimeoutMs == null ? 3000 : this.opts.settleTimeoutMs;
    const session = { cb, closed: false };
    this.session = session;
    // Whatever the last press left behind is not part of this utterance.
    this._command({ cmd: 'reset' });

    return {
      sendAudio: (pcm) => {
        if (session.closed) return;
        this._write(encodeFrame(TYPE_AUDIO, pcm));
      },
      // Nothing is buffered on this side - the sidecar already holds every
      // frame - and stdin preserves order, so the commit that follows sees all
      // of it. Kept for the provider contract.
      flush: () => {},
      endSegment: async () => {
        if (session.closed || !this.proc) return '';
        const id = String(++this.commitSeq);
        const text = new Promise((resolve) => {
          // Deliberately not unref'd: a commit in flight is work the process
          // owes the user, and exiting out from under it loses the utterance.
          const timer = setTimeout(() => {
            this.pending.delete(id);
            this.log(`commit ${id} timed out after ${settleTimeoutMs}ms`);
            resolve('');
          }, settleTimeoutMs);
          this.pending.set(id, { resolve, timer });
        });
        if (!this._command({ cmd: 'commit', id })) {
          const waiter = this.pending.get(id);
          if (waiter) {
            this.pending.delete(id);
            clearTimeout(waiter.timer);
            waiter.resolve('');
          }
        }
        return text;
      },
      // The microphone stopped, not the engine: the model stays loaded so the
      // next press costs nothing, and only the audio buffer is dropped. It is
      // only after whisper.idleUnloadMs of quiet that the model is given up.
      close: async () => {
        session.closed = true;
        if (this.session === session) this.session = null;
        this._command({ cmd: 'reset' });
        this._armIdle();
      }
    };
  }

  /* Loads the model before anyone speaks. A long-running server should pay the
   * load at startup, not on the first microphone press. */
  preload() {
    return this._ensure();
  }

  idleUnloadMs() {
    const n = Number(this.opts.idleUnloadMs);
    return Number.isFinite(n) && n > 0 ? n : 0;
  }

  _cancelIdle() {
    if (!this.idleTimer) return;
    clearTimeout(this.idleTimer);
    this.idleTimer = null;
  }

  /* The microphone stopped and the model is idle. Hold it for one quiet stretch
   * in case the user is only drawing breath, then give the memory back.
   * Unref'd: a pending unload is not work the process owes anyone, and it must
   * never be the reason a one-shot CLI run refuses to exit. */
  _armIdle() {
    this._cancelIdle();
    const ms = this.idleUnloadMs();
    if (!ms || !this.proc) return;
    this.idleTimer = setTimeout(() => {
      this.idleTimer = null;
      // A session opened while the timer was in flight owns the model again.
      if (this.session && !this.session.closed) return;
      this.log(`unloading the model after ${ms}ms idle`);
      this.shutdown();
    }, ms);
    if (this.idleTimer.unref) this.idleTimer.unref();
  }

  /* Ends the sidecar process. Only the server's own shutdown wants this. */
  async shutdown() {
    if (!this.proc) return;
    const proc = this.proc;
    this._command({ cmd: 'stop' });
    try {
      if (proc.stdin) proc.stdin.end();
    } catch (e) {
      /* already gone */
    }
    this._teardown();
    if (typeof proc.kill === 'function') {
      setTimeout(() => proc.kill('SIGTERM'), 500).unref();
    }
  }
}

module.exports = {
  WhisperProvider,
  mapWhisperError,
  resolvePython,
  venvPython,
  dataDir,
  encodeFrame,
  setupInstructions,
  DEFAULT_MODEL,
  DEFAULT_STARTUP_TIMEOUT_MS,
  SIDECAR_PATH,
  SAMPLE_RATE,
  TYPE_AUDIO,
  TYPE_COMMAND
};
