# hebrew-voice

Hebrew dictation for Claude Code's `/voice`: one local WebSocket server that
Claude's microphone talks to. Extracted from `hebrew-tty`, which keeps the
terminal RTL proxy and shares no code with this.

## Layout

| Path | Holds |
| --- | --- |
| `bin/hebrew-voice` | entry point: `serve`, `status`, `setup`, `test`, `levels` |
| `src/cli.js` | flags, config resolution, provider construction |
| `src/server.js` | the `voice_stream` socket Claude Code talks to |
| `src/vad.js` | room calibration and the client-side endpointer |
| `src/gemini.js` | Gemini 3.5 Transcribe Live over the Live API |
| `src/elevenlabs.js` | ElevenLabs Scribe v2 Realtime |
| `src/whisper.js`, `src/whisper_sidecar.py` | local faster-whisper behind a pipe |
| `test/run.js` | assertion runner over the whole service |

## Commands

```sh
npm test
node bin/hebrew-voice serve     # the service; needs no install and no root
node bin/hebrew-voice status
node bin/hebrew-voice levels    # measure the microphone before blaming the engine
```

## Invariants

- The redirect is one environment variable: Claude Code's CLI builds its
  dictation socket from `VOICE_STREAM_BASE_URL`. Nothing is patched, and the
  CLI's own microphone keeps recording.
- The wire protocol belongs to Claude, not to us. Binary frames are linear16
  16 kHz mono; replies are `TranscriptInterim` / `TranscriptText` /
  `TranscriptEndpoint` / `TranscriptError`.
- **Only `TranscriptEndpoint` commits.** The client replaces a single pending
  buffer on every `TranscriptInterim`/`TranscriptText` (they are handled
  identically) and promotes that buffer on `Endpoint`. So a commit is always a
  pair, and the text sent must be the whole utterance, never a delta.
- After `CloseStream` the client arms a 1500 ms no-data timer and a 5000 ms
  safety timer. **Any frame clears the no-data timer**, which is why
  `CloseStream` is answered instantly with a keepalive interim — going quiet
  costs the accurate engine's transcript. Every settle and wait must fit inside
  5000 ms, not 1500.
- These numbers were read out of the CLI binary's `finalize()`. Re-check them
  when dictation starts truncating after a Claude Code upgrade.
- The engine is ElevenLabs Scribe v2 Realtime over one WebSocket:
  `input_audio_chunk` frames carry base64 PCM up, and `commit_strategy=vad`
  makes the server endpoint each utterance itself. Without VAD,
  `committed_transcript` never fires for a microphone.
- **Only `committed_transcript` ends a segment.** `partial_transcript` and
  `final_transcript` are both hypotheses - the latter is settled, not
  committed. Emitting either as a commit sends the same words twice.
- Audio is batched to ~100 ms before it goes out. One JSON+base64 frame per
  20 ms CLI frame is 50 messages a second, for latency the model cannot use.
- ElevenLabs wants a bare ISO-639-1 code, so Hebrew is `he`. A `voice.json`
  left over from the Google backend carries `iw-IL` and a Chirp model name;
  both are translated on load rather than sent as-is.
- Scribe's list parameters are **repeated** query parameters, never
  comma-joined. A comma-joined `secondary_languages` is rejected outright; a
  comma-joined `keyterms` is accepted as ONE long term, which biases the model
  at a string nobody will ever say. The live API echoes its resolved config in
  `session_started` - read it there, not in the docs, when a knob looks ignored.
- Terminal Hebrew is code-switched: paths, commands and library names arrive in
  English mid-sentence. `secondary_languages=en` is what stops them coming back
  transliterated into Hebrew letters, and it is the default.
- `provider: "hybrid"` is two engines on one microphone feed: the fast one
  streams interims and paints, the accurate one answers once at the end and
  owns the commit. **Nothing commits mid-recording** - the fast engine's finals
  are demoted to interim display, or the same words land twice.
- The accurate half is `gemini-3.5-transcribe` over the Interactions API: one
  POST with the whole utterance as inline `audio/l16`, which is headerless, so
  `sample_rate` and `channels` have to be declared beside it. It is worth its
  round trip only because it reads ~2.6% WER against the live model's ~4.0%.
- **The fallback is the point.** If the accurate half errors, goes quiet, or
  misses `finalWaitMs`, the fast engine's accumulated text is committed. A
  dictation is never lost to the slower engine's failure, and `finalWaitMs`
  must stay inside `settleTimeoutMs` so the socket still closes in the client's
  window.
- `custom_vocabulary` cannot combine with diarization or word timestamps - the
  API rejects the request. Dictation has one speaker, so the vocabulary wins.
- The local engine is `provider: "whisper"` - faster-whisper in a Python
  sidecar (`whisper_sidecar.py`) behind a pipe, spoken to with
  `[1 byte type][4 byte big-endian length][payload]` so audio is never
  base64'd. It outlives every microphone press: loading the model costs about
  five seconds, which is not a price to pay per press.
  The venv is `~/.local/share/rtl-caret/whisper-venv` (the path predates the rename and is
  left alone: moving it costs a five second model reload for nothing), and CUDA comes from
  pip - the sidecar dlopens `site-packages/nvidia/*/lib` itself rather than
  making the caller set `LD_LIBRARY_PATH`.
- **The model is resident, not immortal.** It is ~1.3 GB, and on a desktop that
  dictates a few times a day the kernel swaps it out long before the next press
  - so "stays loaded" was already costing a disk read without saving one.
  `whisper.idleUnloadMs` (default five minutes, `0` to opt out) ends the sidecar
  after a quiet stretch; `_ensure()` reloads it on the next press. The window is
  floored at a minute so it can never unload between two sentences of one
  thought, and the timer is `unref`'d - a pending unload must never be the
  reason a one-shot CLI run refuses to exit.
- **Whisper is PULL, Scribe is PUSH.** Whisper has no endpointer, so the local
  energy VAD decides when an utterance ended and `endSegment()` returns the
  text; `onFinal` is never called. Wiring it as a push provider would commit
  nothing until the microphone stopped.
- **Fed silence, Whisper invents a sentence** - in Hebrew, reliably
  "תודה רבה". That is not an edge case: the tail of every microphone press is
  silence. Silero (`vad_filter`) in front of the decoder is the only thing that
  tells the two apart - measured against noise at four levels, `no_speech_prob`
  came back 0.0 and `avg_logprob` looked like ordinary speech. It also pays for
  itself: a buffer with no voice in it skips the encoder, 30 ms instead of 600.
  An energy floor cannot do this job - the room clears it.
- Silero and the endpointer answer different questions. The endpointer asks
  "did the level drop", Silero asks "was that a voice". Neither substitutes for
  the other, and the endpointer must stay the only holder of absolute levels -
  two places holding the same threshold is how they drift.
- Whisper's default temperature is a fallback ladder: a decode whose logprob or
  compression ratio looks wrong is retried at 0.2, 0.4 ... 1.0, each retry a
  full pass. Half an utterance trips it constantly, which turned a 470 ms
  hypothesis into 1900 ms. Hypotheses run at a fixed `temperature=0`; only the
  commit gets the ladder.
- One card, one inference at a time. A hypothesis and a commit that overlap
  queue on the GPU and can exhaust it, so a lock serializes them and a
  hypothesis that has not started yet yields to a waiting commit.
- The encoder cost is flat: the mel is padded to thirty seconds, so twelve
  seconds of speech costs the same ~470 ms as one. Segment length is not a
  latency knob; `partialMs` and `endpointMs` are.
- Nothing is on screen until the first hypothesis lands, so what the user calls
  slowness is that one number: the audio the hypothesis waits to accumulate,
  plus one decode. The commit's own decode is not felt - the last hypothesis is
  already painted by the time it runs, and it is what carries the words spoken
  after that hypothesis started. Committing the last hypothesis instead would
  drop the end of every sentence.
- `endpointMs` is not free to shorten. At 350 ms an ordinary gap between two
  words ended the sentence: the buffer was emptied, the hypothesis started over,
  and the first text arrived 500 ms LATER than at 450. A commit of zero
  characters in the log is that, and it costs more than the shorter endpoint
  saves.
- A hypothesis is allowed on less audio than a commit - `MIN_PARTIAL_BYTES`,
  0.2 s against the commit's 0.5. Half a second of floor is half a second of
  talking to a screen that has not moved, and a bad hypothesis is replaced.
- **The second model is a knob with nothing to put in it.** `whisper.partialModel`
  runs hypotheses on a small model while the commit stays accurate - measured,
  `faster-whisper-small` hypothesises in 137 ms against the turbo's 550. It is
  off because no small Hebrew model exists: every ivrit-ai release is large, and
  a multilingual small invents Hebrew. `medium` does not fit beside the turbo on
  a 4 GB card. Quantising a second copy is not the way round it either - int8
  buys 13% (467 ms against 535), not a second channel.
- `int8_float16` is what `computeType: auto` already resolves to on a card, and
  a config that names `float16` is giving up a gigabyte for nothing. Measured on
  the same recording, the two return the same transcript word for word - the
  hard case included, a Hebrew sentence carrying `git`, `npm` and `deployment` -
  while int8 holds 1241 MiB against 2233 and decodes no slower.
- **A correction pass over the committed text does not fit on this card.** The
  idea is sound - punctuation, and the English terms that come back
  transliterated - but the room left beside whisper is ~2.7 GB, and what fits
  there cannot do the job. Measured: Gemma 3 4B took 3.3 s, left every
  transliterated term in Hebrew and deleted words that were there; Gemma 3 1B
  took 1.2 s and answered the sentence instead of correcting it ("I'm sorry, I
  can't do that"). That answer would have landed in the user's input box. The
  model that could do this is one the card cannot hold at the same time as
  whisper, and swapping them costs whisper's four second load per sentence.
- **A fixed energy threshold does not survive a real microphone.** At 0.005 a
  quiet room already reads as speech, so silence never arrives, nothing ever
  commits, and dictation only lands when the key is released - which looks like
  slowness, not like a broken endpointer. `vad.js` measures the room over the
  first 300 ms of each press (Claude streams from the moment the mic opens, and
  nobody starts talking that fast) and puts the bar at three times it. The
  segment those first frames opened against the absolute threshold is taken
  back once the room is known, or it would commit itself 600 ms later.
- `noiseRatio` is not one number for every microphone. A headset hears speech
  ten times louder than its room; a laptop microphone with its gain wound up
  hears itself at 0.15 RMS with nobody in the room, and asking speech to beat
  three times that asks for more than it produces. `hebrew-voice levels`
  records without transcribing and reports both ends, which is the only way to
  tell "the room clears the bar" from "speech does not".
- Calibration needs the real 20 ms wire frames. A caller handing over whole
  seconds at a time is not measuring a room, so it is left on the absolute
  threshold rather than told that speech is the floor.
- The hypothesis loop skips a buffer whose tail has gone quiet: the speaker has
  stopped, the text would repeat the last one, and starting it now only makes
  the commit queue behind half a second of decoding.

