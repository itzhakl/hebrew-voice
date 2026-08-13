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

/* Claude's client waits at most 3000 ms for the socket to close after
 * CloseStream before its own fallback timer fires and the mic UI hangs
 * visibly. Provider endSegment timeouts are tuned to fit under that, but a VAD
 * commit already queued ahead of the close-stream commit stacks its wait on
 * top. This deadline is the hard backstop. */
const CLOSE_STREAM_DEADLINE_MS = 2500;

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
  let session;
  const ready = opts.provider
    .createSession({
      onInterim: (text) => send({ type: 'TranscriptText', data: stripBidiControls(text) }),
      onError: reportError,
      // Providers with server-side endpointing push each finalized utterance
      // here the moment it lands - commit latency is the ASR's own endpointer.
      onFinal: (text) => {
        if (!text) return;
        log(`final (server-endpointed): ${text.length} chars`);
        send({ type: 'TranscriptText', data: stripBidiControls(text) });
        send({ type: 'TranscriptEndpoint' });
      }
    })
    .then((s) => {
      session = s;
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
        log(`commit (${reason}): ${text.length} chars`);
        if (text) {
          // Committed DURING recording - Claude ignores transcripts after stop.
          send({ type: 'TranscriptText', data: stripBidiControls(text) });
          send({ type: 'TranscriptEndpoint' });
        }
      })
      .catch((e) => log(`commit failed: ${e}`));
  };

  ws.on('message', (data, isBinary) => {
    if (isBinary) {
      const buf = Buffer.isBuffer(data) ? data : Buffer.concat(data);
      try {
        if (session) session.sendAudio(buf);
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
  constructor(httpServer, wss, port) {
    this.httpServer = httpServer;
    this.wss = wss;
    this.port = port;
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
        resolve(new VoiceStreamServer(httpServer, wss, httpServer.address().port));
      });
    });
  }

  close() {
    for (const client of this.wss.clients) client.terminate();
    return new Promise((resolve) => {
      this.wss.close(() => this.httpServer.close(() => resolve()));
    });
  }
}

function isOurServer(port, timeoutMs = 800) {
  return new Promise((resolve) => {
    const req = http.get({ host: '127.0.0.1', port, path: '/healthz', timeout: timeoutMs }, (res) => {
      let body = '';
      res.on('data', (c) => (body += c));
      res.on('end', () => {
        try {
          resolve(JSON.parse(body).app === HEALTH_APP);
        } catch (e) {
          resolve(false);
        }
      });
    });
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
    req.on('error', () => resolve(false));
  });
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
  stripBidiControls,
  HEALTH_APP,
  VOICE_STREAM_PATH,
  CLOSE_STREAM_DEADLINE_MS
};
