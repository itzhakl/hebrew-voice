'use strict';

/* The local end of Claude Code's dictation socket.
 *
 * The CLI builds its socket URL as
 *   ${VOICE_STREAM_BASE_URL}/api/ws/speech_to_text/voice_stream?...
 * so pointing that variable here is the whole redirect - the CLI's own
 * microphone keeps recording and we answer instead of Anthropic. It sends
 * linear16 16 kHz mono frames as binary messages and JSON control frames
 * (CloseStream, KeepAlive); it expects TranscriptText / TranscriptEndpoint /
 * TranscriptError back. */

const http = require('http');

const HEALTH_APP = 'rtl-caret-voice';
const VOICE_STREAM_PATH = '/api/ws/speech_to_text/voice_stream';

// Unicode bidi control marks. STT engines wrap Latin runs inside RTL
// transcripts with them; they carry no content, and in a terminal they only
// confuse the caret mapping this project exists to fix.
// LRM RLM | ALM | LRE-RLO (embeddings/overrides + PDF) | LRI-PDI (isolates)
const BIDI_CONTROLS_RE = /[‎‏؜‪-‮⁦-⁩]/g;

function stripBidiControls(text) {
  return text.replace(BIDI_CONTROLS_RE, '');
}

/* Read out of the CLI's own finalize(): after it sends CloseStream it arms two
 * timers - a 1500 ms "no data" timer and a 5000 ms safety timer. ANY
 * TranscriptText/TranscriptInterim frame that arrives after CloseStream clears
 * the first one; a TranscriptEndpoint resolves finalize immediately. Whichever
 * fires, the client promotes the last interim it holds and stops listening.
 *
 * So the budget is 1500 ms if we go quiet, and 5000 ms if we keep the socket
 * talking - which is why CloseStream is answered instantly with a keepalive
 * interim below, before the engine's real final is waited for. */
const NO_DATA_MS = 1500;
const SAFETY_MS = 5000;
const CLOSE_STREAM_DEADLINE_MS = 4500;

function handleConnection(ws, opts) {
  const log = opts.log || (() => {});
  const endpointer = opts.makeEndpointer();
  const WebSocket = require('ws');

  const send = (obj) => {
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  };
  const reportError = (error) => {
    send({ type: 'TranscriptError', description: error.message });
    if (opts.onProviderError) opts.onProviderError(error);
  };

  log('connection open (mic start) - creating provider session');
  const openedAt = Date.now();
  let session;
  // Opening a session costs an OAuth exchange and a gRPC channel on the first
  // mic press. The CLI starts sending audio the moment the socket is up, so
  // frames that arrive inside that window are held here and replayed - dropping
  // them silently eats the first words of the utterance.
  const preroll = [];
  let prerollDropped = 0;

  // The last interim we showed the user and have not superseded with a final.
  // Claude paints interims grey and discards them when the mic stops, so an
  // utterance whose final never lands would vanish in front of the user.
  let unfinalised = '';

  // Time-to-first-ink is the whole felt latency of dictation: an engine that
  // streams partials paints grey text while you talk, and one that only speaks
  // at commit leaves the screen empty until its endpointer fires. The two feel
  // nothing alike at the same accuracy, so both are timed here.
  let firstInterimAt = 0;
  let interims = 0;

  const emit = (text, why) => {
    const clean = stripBidiControls(text);
    if (!clean) return;
    log(`${why}: ${clean.length} chars, ${Date.now() - openedAt}ms after mic start, ${interims} interims so far`);
    send({ type: 'TranscriptText', data: clean });
    send({ type: 'TranscriptEndpoint' });
    unfinalised = '';
  };

  const ready = opts.provider
    .createSession({
      // TranscriptInterim and TranscriptText are handled identically by the
      // client - both only replace its pending buffer - but naming the grey
      // hypothesis correctly is free and survives the two diverging.
      onInterim: (text) => {
        interims++;
        if (!firstInterimAt) {
          firstInterimAt = Date.now();
          log(`first interim ${firstInterimAt - openedAt}ms after mic start`);
        }
        unfinalised = text;
        send({ type: 'TranscriptInterim', data: stripBidiControls(text) });
      },
      onError: reportError,
      // Providers with server-side endpointing push each finalized utterance
      // here the moment it lands - commit latency is the ASR's own endpointer.
      onFinal: (text) => {
        if (text) emit(text, 'final (server-endpointed)');
      }
    })
    .then((s) => {
      session = s;
      log(`session ready ${Date.now() - openedAt}ms after mic start, ${preroll.length} frames buffered`);
      for (const buf of preroll) s.sendAudio(buf);
      preroll.length = 0;
      if (prerollDropped) log(`WARNING: ${prerollDropped} frames dropped before the session opened`);
    })
    .catch((e) => {
      reportError({ message: e && e.message ? e.message : String(e) });
      ws.close();
    });

  // Serialize commits so segments cannot interleave out of order.
  let commitChain = Promise.resolve();
  const commit = (reason) => {
    commitChain = commitChain
      .then(async () => {
        if (!session) return;
        const text = await session.endSegment();
        // The measured room and the level speech has to beat: when dictation
        // "never commits until I let go", these two numbers are the answer.
        log(`commit (${reason}): ${text.length} chars, room ${endpointer.noiseFloor.toFixed(4)}, speech above ${endpointer.threshold.toFixed(4)}`);
        // Committed DURING recording - Claude ignores transcripts after stop.
        if (text) emit(text, `commit (${reason})`);
        // The engine owed us a final and the deadline is the socket closing:
        // send back what the user already read rather than nothing at all.
        else if (reason === 'close-stream' && unfinalised) emit(unfinalised, 'commit (interim fallback)');
      })
      .catch((e) => log(`commit failed: ${e}`));
  };

  ws.on('message', (data, isBinary) => {
    if (isBinary) {
      const buf = Buffer.isBuffer(data) ? data : Buffer.concat(data);
      try {
        if (session) session.sendAudio(buf);
        // ~10 s of 20 ms frames. A session that has not opened by then is not
        // going to, and the memory is not worth holding.
        else if (preroll.length < 500) preroll.push(buf);
        else prerollDropped++;
        const pcm = new Int16Array(buf.byteLength >> 1);
        for (let i = 0; i < pcm.length; i++) pcm[i] = buf.readInt16LE(i * 2);
        const result = endpointer.pushFrame(pcm);
        if (result.commit) commit(result.commit);
      } catch (e) {
        // The session died mid-recording (quota/network): report once and close
        // so we don't throw for every remaining audio frame.
        reportError({ message: e && e.message ? e.message : String(e) });
        ws.close();
      }
      return;
    }
    let ctrl;
    try {
      ctrl = JSON.parse(data.toString());
    } catch (e) {
      return;
    }
    if (ctrl.type === 'CloseStream') {
      const closeStreamAt = Date.now();
      log('CloseStream received');
      // Answer before doing any work: this frame alone cancels the client's
      // 1500 ms no-data timer and buys the full 5000 ms, which is the
      // difference between committing the accurate engine's transcript and
      // losing it to the fast engine's guess. Empty data is deliberate when
      // there is nothing yet - the client clears the timer before it looks at
      // the payload, and an empty payload cannot clobber what it holds.
      send({ type: 'TranscriptInterim', data: stripBidiControls(unfinalised) });
      // Mic stopped - no more audio. Let the provider force-finalize buffered
      // speech instead of waiting out server-side endpointing.
      void ready.then(() => {
        if (session && session.flush) session.flush();
        endpointer.reset();
        commit('close-stream');
        // Claude resolves its stop only once we close the socket (or after its
        // own 3 s grace). Race the commit chain against a hard deadline.
        const committed = commitChain.then(() => 'committed');
        const deadline = new Promise((resolve) => setTimeout(() => resolve('deadline'), CLOSE_STREAM_DEADLINE_MS));
        return Promise.race([committed, deadline]).then((via) => {
          log(`closing socket ${Date.now() - closeStreamAt}ms after CloseStream (${via})`);
          if (ws.readyState === WebSocket.OPEN) ws.close();
        });
      });
    }
    // KeepAlive: intentionally ignored.
  });

  ws.on('close', () => {
    log('connection closed');
    if (session) void Promise.resolve(session.close()).catch((e) => log(`session close failed: ${e}`));
  });
}

class VoiceStreamServer {
  constructor(httpServer, wss, port, provider) {
    this.httpServer = httpServer;
    this.wss = wss;
    this.port = port;
    this.provider = provider;
  }

  /* Binds 127.0.0.1 only. Rejects with EADDRINUSE when the port is taken. */
  static start(opts) {
    const { WebSocketServer } = require('ws');
    return new Promise((resolve, reject) => {
      const httpServer = http.createServer((req, res) => {
        if (req.method === 'GET' && req.url === '/healthz') {
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(JSON.stringify({
            app: HEALTH_APP,
            protocol: 'voice_stream',
            provider: opts.provider.id,
            pid: process.pid
          }));
        } else {
          res.writeHead(404);
          res.end();
        }
      });
      const wss = new WebSocketServer({ server: httpServer });
      wss.on('connection', (ws) => handleConnection(ws, opts));
      // `ws` forwards httpServer 'error' events onto the WebSocketServer for
      // the server's lifetime, and Node throws an unhandled-error exception if
      // nothing listens there - which would crash the process on EADDRINUSE
      // before our own listener below could reject().
      wss.on('error', (e) => (opts.log || (() => {}))(`wss error: ${e}`));
      httpServer.once('error', reject);
      httpServer.listen(opts.port, '127.0.0.1', () => {
        httpServer.removeListener('error', reject);
        resolve(new VoiceStreamServer(httpServer, wss, httpServer.address().port, opts.provider));
      });
    });
  }

  close() {
    for (const client of this.wss.clients) client.terminate();
    return new Promise((resolve) => {
      this.wss.close(() => this.httpServer.close(() => resolve()));
    }).then(() => {
      // A local engine holds a child process and a card's worth of memory:
      // closing the socket is not enough to give either of them back.
      if (this.provider && typeof this.provider.shutdown === 'function') {
        return this.provider.shutdown();
      }
      return undefined;
    });
  }
}

/* What is answering /healthz on this port: our own server, someone else's
 * voice_stream server (the VS Code extension runs one), or nothing. Reporting
 * "no server" for a port a foreign voice_stream owns sends you hunting for a
 * dead socket that is in fact alive and serving the CLI. */
function probe(port, timeoutMs = 800) {
  return new Promise((resolve) => {
    const req = http.get({ host: '127.0.0.1', port, path: '/healthz', timeout: timeoutMs }, (res) => {
      let body = '';
      res.on('data', (c) => (body += c));
      res.on('end', () => {
        try {
          const health = JSON.parse(body);
          if (health.app === HEALTH_APP) resolve({ kind: 'ours', port, health });
          else if (health.protocol === 'voice_stream') resolve({ kind: 'foreign', port, health });
          else resolve({ kind: 'none', port });
        } catch (e) {
          resolve({ kind: 'none', port });
        }
      });
    });
    req.on('timeout', () => {
      req.destroy();
      resolve({ kind: 'none', port });
    });
    req.on('error', () => resolve({ kind: 'none', port }));
  });
}

async function isOurServer(port, timeoutMs = 800) {
  return (await probe(port, timeoutMs)).kind === 'ours';
}

/* Bind opts.port, or adopt a healthy instance of our own server already on it
 * (another shell), or fall through to the next free port. */
async function startWithPortFallback(opts, attempts = 10) {
  for (let i = 0; i < attempts; i++) {
    const port = opts.port + i;
    try {
      const server = await VoiceStreamServer.start(Object.assign({}, opts, { port }));
      return { server, port, adopted: false };
    } catch (e) {
      if (e.code !== 'EADDRINUSE') throw e;
      if (await isOurServer(port)) return { server: null, port, adopted: true };
    }
  }
  throw new Error(`no free port in ${opts.port}..${opts.port + attempts - 1}`);
}

module.exports = {
  VoiceStreamServer,
  startWithPortFallback,
  isOurServer,
  probe,
  stripBidiControls,
  HEALTH_APP,
  VOICE_STREAM_PATH,
  CLOSE_STREAM_DEADLINE_MS,
  NO_DATA_MS,
  SAFETY_MS
};
