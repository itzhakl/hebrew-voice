# hebrew-voice

Hebrew dictation for Claude Code's `/voice`. A local WebSocket server that
Claude's microphone talks to, and nothing else.

Extracted from [hebrew-tty](https://github.com/itzhakl/hebrew-tty), which keeps
the terminal RTL proxy. The two share no code.

## Install

```sh
npm install
hebrew-voice setup --provider gemini   # stores the AIza… key, mode 0600
hebrew-voice serve
hebrew-voice -- claude
```

## Providers

| provider | engine | notes |
| --- | --- | --- |
| `hybrid` | two engines on one feed | fast one paints, accurate one commits |
| `gemini` | Gemini 3.5 Transcribe Live | streaming, ~4.0% WER |
| `elevenlabs` | Scribe v2 Realtime | streaming, lowest time to first ink |
| `whisper` | local faster-whisper sidecar | no network, ~1.3 GB resident |

`hybrid` is the pair: the fast engine streams interims so grey text appears
while you talk, and `gemini-3.5-transcribe` reads the whole utterance back at
the end (~2.6% WER against the live model's ~4.0%) and owns the commit. If the
accurate half fails or misses its deadline, the fast engine's text is committed
instead — a dictation is never lost to the slower engine.

```json
{
  "provider": "hybrid",
  "hybrid": { "fast": "gemini", "accurate": "gemini-transcribe" }
}
```

Configuration lives in `~/.config/rtl-caret/voice.json`. `hebrew-voice status`
reports the resolved engine, credential, languages, and endpointer thresholds.

## Verification

```sh
npm test
hebrew-voice levels        # measure the microphone: room, speech, threshold
hebrew-voice test 5        # record and transcribe, end to end
```

MIT.
