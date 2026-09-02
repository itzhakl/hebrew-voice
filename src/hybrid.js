'use strict';

/* Two engines on one microphone feed.
 *
 * fast     - streams interims and paints the grey text while you talk.
 * accurate - answers once, at the end, and its answer is what gets committed.
 *
 * Nothing commits mid-recording: the fast engine's finals are demoted to
 * interim display. On stop, both flush, and endSegment waits for the accurate
 * engine - falling back to the fast engine's accumulated text if the accurate
 * one errors, goes silent, or misses the deadline. The user never loses a
 * dictation to the slower engine's failure.
 */

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

class HybridProvider {
  constructor(fast, accurate, opts = {}) {
    this.id = 'hybrid';
    this.fast = fast;
    this.accurate = accurate;
    this.opts = opts;
  }

  async createSession(cb) {
    const log = typeof this.opts.log === 'function' ? this.opts.log : () => {};
    /* Must stay inside the client's grace window after CloseStream, which the
     * server budgets with settleTimeoutMs. */
    const finalWaitMs = this.opts.finalWaitMs == null ? 2300 : this.opts.finalWaitMs;

    let fastCommitted = '';
    let fastInterim = '';
    const fastText = () => `${fastCommitted} ${fastInterim}`.trim();

    let accurateDone = false;
    let accurateFailed = false;
    let flushed = false;

    const [fastSession, accurateSession] = await Promise.all([
      this.fast.createSession({
        onInterim: (t) => {
          fastInterim = t;
          cb.onInterim(fastText());
        },
        // Demoted on purpose: the accurate engine owns the commit.
        onFinal: (t) => {
          fastCommitted = `${fastCommitted} ${t}`.trim();
          fastInterim = '';
          cb.onInterim(fastText());
        },
        onError: cb.onError
      }),
      this.accurate.createSession({
        onInterim: () => {},
        // A failed accurate engine must not blank the segment - it only means
        // the fast engine's text is the best we hold.
        onError: (e) => {
          accurateFailed = true;
          accurateDone = true;
          log(`accurate engine failed, falling back: ${e && e.message}`);
        },
        onClosed: () => {
          accurateDone = true;
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
        // A mid-recording VAD tick: everything is display-only until stop.
        if (!flushed) return '';
        const started = Date.now();
        const settled = accurateSession.endSegment().then((t) => {
          accurateDone = true;
          return t;
        });
        const raced = await Promise.race([
          settled,
          (async () => {
            while (!accurateDone && Date.now() - started < finalWaitMs) await sleep(25);
            return null;
          })()
        ]);
        const accurate = accurateFailed ? '' : String(raced || '').trim();
        const text = accurate || fastText();
        log(
          `commit from ${accurate ? 'accurate' : 'fast'} engine after ${Date.now() - started}ms`
        );
        fastCommitted = '';
        fastInterim = '';
        flushed = false;
        accurateDone = false;
        accurateFailed = false;
        return text;
      },
      close: async () => {
        await Promise.all([fastSession.close(), accurateSession.close()]);
      }
    };
  }
}

module.exports = { HybridProvider };
