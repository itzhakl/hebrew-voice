'use strict';

/* Hybrid = two engines on the same microphone feed.
 *
 *   fast     (long @ eu)    - streams live interims; paints the grey text.
 *   accurate (chirp_3 @ us) - multilingual (Hebrew/English code-switching);
 *                             its flush-final is what actually gets committed.
 *
 * Nothing commits mid-recording: the fast engine's finals are demoted to
 * interim display. On stop, both streams flush; endSegment waits for the
 * accurate engine and commits its text - falling back to the fast engine's
 * transcript if the accurate one errors, is silent, or misses the deadline. */

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

class HybridProvider {
  constructor(fast, accurate, opts = {}) {
    this.id = 'hybrid';
    this.fast = fast;
    this.accurate = accurate;
    this.opts = opts;
  }

  async createSession(cb) {
    // The server answers CloseStream with a keepalive interim, which lifts the
    // client's window from 1500 ms to 5000 ms. chirp_3 lands its flush-final
    // 1.4-2.0 s after the stop, so waiting this long is what makes the accurate
    // transcript - not the fast engine's guess - the one that gets committed.
    const finalWaitMs = this.opts.finalWaitMs == null ? 4000 : this.opts.finalWaitMs;
    let fastCommitted = '';
    let fastInterim = '';
    const fastText = () => `${fastCommitted} ${fastInterim}`.trim();
    let accurateFinals = [];
    let accurateDone = false;
    let accurateDiedEarly = false;
    let flushed = false;

    const [fastSession, accurateSession] = await Promise.all([
      this.fast.createSession({
        onInterim: (t) => {
          fastInterim = t;
          cb.onInterim(fastText());
        },
        // Demote fast finals to display: the accurate engine owns the commit.
        onFinal: (t) => {
          fastCommitted = `${fastCommitted} ${t}`.trim();
          fastInterim = '';
          cb.onInterim(fastText());
        },
        onError: cb.onError
      }),
      this.accurate.createSession({
        onInterim: () => {},
        onFinal: (t) => accurateFinals.push(t),
        // An engine that ends or errors while the user is still talking has
        // only heard part of the utterance. Its transcript is more accurate
        // per word and yet missing words, so the fast engine - which heard all
        // of it - has to win, or the tail of the sentence disappears.
        onClosed: () => {
          accurateDone = true;
          if (!flushed) accurateDiedEarly = true;
        },
        // The accurate engine failing must not kill dictation - the fast
        // transcript is the fallback, so swallow and stop waiting.
        onError: () => {
          accurateDone = true;
          if (!flushed) accurateDiedEarly = true;
        }
      })
    ]);

    return {
      sendAudio: (pcm) => {
        fastSession.sendAudio(pcm);
        accurateSession.sendAudio(pcm);
      },
      flush: () => {
        flushed = true;
        if (fastSession.flush) fastSession.flush();
        if (accurateSession.flush) accurateSession.flush();
      },
      endSegment: async () => {
        // Mid-recording VAD tick: everything is display-only until stop.
        if (!flushed) return '';
        const started = Date.now();
        while (!accurateDone && Date.now() - started < finalWaitMs) await sleep(25);
        const accurate = accurateDiedEarly ? '' : accurateFinals.join(' ').trim();
        const text = accurate || fastText();
        fastCommitted = '';
        fastInterim = '';
        accurateFinals = [];
        accurateDiedEarly = false;
        flushed = false;
        return text;
      },
      close: async () => {
        await Promise.all([fastSession.close(), accurateSession.close()]);
      }
    };
  }
}

module.exports = { HybridProvider };
