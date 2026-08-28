"""Local Hebrew dictation engine: faster-whisper behind a pipe.

Whisper is not a streaming model - it transcribes a window, and its encoder
costs the same whether that window holds one second of speech or ten, because
the mel is padded to thirty either way. Measured on an RTX 3050 Ti with
ivrit-ai/whisper-large-v3-turbo-ct2 that cost is ~480 ms flat. So the way to
stream it is to re-transcribe the utterance so far on a timer and send the
result as a hypothesis, which is what the caller paints grey.

Protocol, chosen so audio never has to be base64'd:
  stdin   [1 byte type][4 byte big-endian length][payload]
            type 0 - payload is int16 little-endian PCM at 16 kHz
            type 1 - payload is a UTF-8 JSON command
  stdout  one JSON object per line

The model is loaded once and lives for the life of the process: loading it
costs several seconds, which is not a price to pay per microphone press.
"""

import ctypes
import glob
import json
import os
import struct
import sys
import threading
import time

SAMPLE_RATE = 16000
HEADER = struct.Struct('>BI')
TYPE_AUDIO = 0
TYPE_COMMAND = 1


def emit(**payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + '\n')
    sys.stdout.flush()


def read_exactly(stream, count):
    chunks = []
    while count:
        chunk = stream.read(count)
        if not chunk:
            return None
        chunks.append(chunk)
        count -= len(chunk)
    return b''.join(chunks)


def preload_cuda_libraries():
    """CTranslate2 dlopens libcudnn/libcublas by soname, and pip puts them under
    site-packages/nvidia/*/lib where the loader does not look. Opening them
    RTLD_GLOBAL here is what LD_LIBRARY_PATH would otherwise have to do, and it
    keeps the caller from having to set an environment variable to get a GPU."""
    roots = [os.path.join(p, 'nvidia') for p in sys.path if p.endswith('site-packages')]
    for root in roots:
        for so in sorted(glob.glob(os.path.join(root, '*', 'lib', 'lib*.so*'))):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def resolve_device(requested):
    if requested and requested != 'auto':
        return requested
    try:
        import ctranslate2

        return 'cuda' if ctranslate2.get_cuda_device_count() > 0 else 'cpu'
    except Exception:
        return 'cpu'


def resolve_compute_type(requested, device):
    if requested and requested != 'auto':
        return requested
    # int8_float16 halves the weights without the accuracy loss of plain int8,
    # and on a 4 GB laptop card that is the difference between fitting and not.
    return 'int8_float16' if device == 'cuda' else 'int8'


class Engine:
    def __init__(self, opts):
        import numpy as np
        from faster_whisper import WhisperModel

        self.np = np
        self.opts = opts
        self.device = resolve_device(opts.get('device'))
        self.compute_type = resolve_compute_type(opts.get('computeType'), self.device)
        try:
            self.model = self._load(self.device, self.compute_type)
        except Exception as e:
            # A driver mismatch or an out-of-memory card must not leave the user
            # with no dictation at all; the CPU is slower, not broken.
            if self.device != 'cuda':
                raise
            emit(type='warning', message=f'cuda unavailable ({type(e).__name__}: {e}) - falling back to cpu')
            self.device = 'cpu'
            self.compute_type = resolve_compute_type(opts.get('computeType'), 'cpu')
            self.model = self._load(self.device, self.compute_type)
        self.lock = threading.Lock()
        # Set the moment a commit arrives, before it queues for the card, so a
        # hypothesis that has not started yet gives up its turn instead of
        # making the user wait half a second for text they will never read.
        self.commit_wanted = False
        # One model, one card. Two transcriptions at once would queue on the GPU
        # anyway and can exhaust its memory, so only one runs at a time and a
        # hypothesis that cannot get in simply skips its turn.
        self.infer_lock = threading.Lock()
        self.audio = bytearray()
        # Bumped on every reset so a transcription that was already in flight
        # when the utterance ended cannot publish itself over the next one.
        self.generation = 0

    def _load(self, device, compute_type):
        from faster_whisper import WhisperModel

        return WhisperModel(
            self.opts['model'],
            device=device,
            compute_type=compute_type,
            download_root=self.opts.get('cacheDir') or None,
            local_files_only=bool(self.opts.get('offline')),
        )

    def warmup(self):
        """The first transcription after a load pays for CUDA kernel setup and
        the tokenizer. Paying it here means the first thing the user says is not
        the slowest thing they say."""
        silence = self.np.zeros(SAMPLE_RATE, dtype=self.np.float32)
        segments, _ = self.model.transcribe(silence, language=self.opts['language'], beam_size=1)
        for _ in segments:
            pass
        # Silero loads its own ONNX session on first use - another 200 ms that
        # would otherwise land on the first thing the user says.
        segments, _ = self.model.transcribe(
            silence, language=self.opts['language'], beam_size=1, vad_filter=True
        )
        for _ in segments:
            pass

    def push(self, pcm):
        with self.lock:
            self.audio.extend(pcm)

    def snapshot(self):
        with self.lock:
            return bytes(self.audio), self.generation

    def reset(self):
        with self.lock:
            self.audio.clear()
            self.generation += 1

    def tail_is_quiet(self, pcm, tail_ms=400, ratio=0.25):
        """Whether the speaker has already stopped. Compares the tail against
        the buffer as a whole rather than against a threshold, so it holds for
        any microphone and any room - the caller's endpointer owns the absolute
        numbers, and two places holding the same number is how they drift."""
        audio = self.np.frombuffer(pcm, dtype=self.np.int16).astype(self.np.float32)
        tail_n = int(SAMPLE_RATE * tail_ms / 1000)
        if len(audio) < tail_n * 2:
            return False
        whole = float(self.np.sqrt((audio * audio).mean()))
        tail = audio[-tail_n:]
        return whole > 0 and float(self.np.sqrt((tail * tail).mean())) < whole * ratio

    def transcribe(self, pcm, beam_size, partial=False):
        if len(pcm) < SAMPLE_RATE:  # under 0.5 s of audio, nothing to say yet
            return ''
        audio = self.np.frombuffer(pcm, dtype=self.np.int16).astype(self.np.float32) / 32768.0
        options = {}
        # Terminal Hebrew is code-switched: "git", "npm", a library name arrive
        # mid-sentence and come back transliterated into Hebrew letters. The
        # list biases the decoder at them. Passed only when non-empty - an empty
        # string here is a prompt, not an absence, and it costs a decode.
        if self.opts.get('hotwords'):
            options['hotwords'] = ' '.join(self.opts['hotwords'])
        # The stronger of the two, and they do not overlap: hotwords names
        # terms, this names a way of writing them.
        if self.opts.get('initialPrompt'):
            options['initial_prompt'] = self.opts['initialPrompt']
        if partial:
            # Whisper's default temperature list is a fallback ladder: a decode
            # whose logprob or compression ratio looks wrong is retried at 0.2,
            # 0.4, ... 1.0. Half an utterance trips that constantly, and each
            # retry is another full pass - measured 470 ms turning into 1900.
            # A hypothesis about to be replaced is not worth five passes.
            options['temperature'] = 0.0
        segments, _ = self.model.transcribe(
            audio,
            language=self.opts['language'],
            beam_size=beam_size,
            **options,
            # Whisper will happily continue a sentence it invented; feeding it
            # its own previous output is how a stuck loop starts.
            condition_on_previous_text=False,
            without_timestamps=True,
            # Not a second endpointer - a different question. The caller's
            # endpointer asks "did the level drop", which any quiet room
            # answers wrong; Silero asks "was that a voice". Whisper does not
            # return nothing for nothing: fed room noise it invents a sentence,
            # and the one it invents in Hebrew is reliably "תודה רבה". Measured
            # against noise at four levels, no_speech_prob came back 0.0 and
            # avg_logprob looked like ordinary speech - this pass is the only
            # thing that tells the two apart. It also PAYS for itself: a buffer
            # with no voice in it skips the encoder, 30 ms instead of 600.
            vad_filter=self.opts['vadFilter'],
            # Silero's own default of 0.5 clips speech that trails off, and the
            # cost of keeping a little noise is one wasted pass, not a wrong
            # transcript.
            vad_parameters={'threshold': 0.3, 'min_speech_duration_ms': 100},
        )
        return ' '.join(segment.text for segment in segments).strip()


def partial_loop(engine, stop, interval):
    last_len = 0
    while not stop.is_set():
        stop.wait(interval)
        if stop.is_set():
            return
        pcm, generation = engine.snapshot()
        if len(pcm) == last_len:
            continue
        last_len = len(pcm)
        # A commit is running: it owns the card, and this hypothesis is about to
        # be superseded by it anyway.
        # The speaker has stopped: a commit is about to be asked for, this
        # hypothesis would say what the last one already said, and starting it
        # now only makes the commit queue behind half a second of decoding.
        if engine.tail_is_quiet(pcm):
            continue
        if engine.commit_wanted or not engine.infer_lock.acquire(blocking=False):
            continue
        if engine.commit_wanted:
            engine.infer_lock.release()
            continue
        began = time.perf_counter()
        try:
            text = engine.transcribe(pcm, engine.opts['partialBeamSize'], partial=True)
        except Exception as e:  # a failed hypothesis is not worth killing the process over
            emit(type='error', message=f'partial failed: {e}', fatal=False)
            continue
        finally:
            engine.infer_lock.release()
        # The utterance was committed while this ran - publishing now would
        # paint the previous sentence over the new one.
        if text and generation == engine.generation:
            emit(
                type='partial',
                text=text,
                ms=round((time.perf_counter() - began) * 1000),
                audioMs=round(len(pcm) / 2 / SAMPLE_RATE * 1000),
            )


def main():
    opts = json.loads(os.environ.get('WHISPER_OPTS', '{}'))
    opts.setdefault('model', 'ivrit-ai/whisper-large-v3-turbo-ct2')
    opts.setdefault('device', 'auto')
    opts.setdefault('computeType', 'auto')
    opts.setdefault('language', 'he')
    opts.setdefault('partialMs', 400)
    opts.setdefault('partialBeamSize', 1)
    opts.setdefault('finalBeamSize', 5)
    opts.setdefault('vadFilter', True)
    opts.setdefault('hotwords', [])
    opts.setdefault('initialPrompt', '')

    preload_cuda_libraries()
    started = time.perf_counter()
    try:
        engine = Engine(opts)
        engine.warmup()
    except Exception as e:
        emit(type='error', message=f'{type(e).__name__}: {e}', fatal=True)
        return 1
    emit(
        type='ready',
        model=opts['model'],
        device=engine.device,
        computeType=engine.compute_type,
        language=opts['language'],
        loadMs=round((time.perf_counter() - started) * 1000),
    )

    stop = threading.Event()
    worker = threading.Thread(
        target=partial_loop, args=(engine, stop, opts['partialMs'] / 1000.0), daemon=True
    )
    worker.start()

    stream = sys.stdin.buffer
    while True:
        header = read_exactly(stream, HEADER.size)
        if header is None:
            break
        kind, length = HEADER.unpack(header)
        payload = read_exactly(stream, length) if length else b''
        if payload is None:
            break
        if kind == TYPE_AUDIO:
            engine.push(payload)
            continue
        try:
            command = json.loads(payload.decode('utf-8'))
        except ValueError:
            continue
        name = command.get('cmd')
        if name == 'commit':
            engine.commit_wanted = True
            pcm, _ = engine.snapshot()
            engine.reset()
            began = time.perf_counter()
            with engine.infer_lock:
                try:
                    text = engine.transcribe(pcm, opts['finalBeamSize'])
                except Exception as e:
                    emit(type='error', message=f'commit failed: {e}', fatal=False)
                    text = ''
                finally:
                    engine.commit_wanted = False
            emit(
                type='final',
                id=command.get('id'),
                text=text,
                ms=round((time.perf_counter() - began) * 1000),
                audioMs=round(len(pcm) / 2 / SAMPLE_RATE * 1000),
            )
        elif name == 'reset':
            engine.reset()
        elif name == 'stop':
            break

    stop.set()
    # The worker may be mid-decode: dropping the model out from under it, or
    # letting CUDA tear its context down while a kernel is still queued, throws
    # a std::runtime_error from C++ and the process aborts instead of exiting.
    worker.join(timeout=10)
    with engine.infer_lock:
        engine.model = None
    return 0


if __name__ == '__main__':
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # CTranslate2 frees its CUDA allocations from a static destructor that runs
    # after the interpreter is already torn down, which aborts. Nothing else is
    # owed at this point, so the process leaves before that can happen.
    os._exit(code)
