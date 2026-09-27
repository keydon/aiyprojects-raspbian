#
# Ported from pico.py: wake-word detection using
# openWakeWord (https://github.com/dscripka/openWakeWord)
# instead of Picovoice Porcupine.
#
# Install with:
#     pip install openwakeword
#     # Download the pre-trained built-in models on first run:
#     python -c "import openwakeword.utils; openwakeword.utils.download_models()"
#
# Audio capture shells out to `arecord` (already present on Raspbian) rather
# than sounddevice, because PortAudio does not expose ALSA plug-family PCMs.
#
# We capture at the Voice HAT's *native* 48 kHz / S32_LE / 2ch and do the 3:1
# decimation and the mic gain here in Python, deliberately bypassing the stock
# `plug:micboost` chain. Two reasons:
#
#   1. `micboost` (scripts/asound.conf) applies a 30x route gain *below* the
#      rate converter, and ALSA's route plugin saturates. Ordinary speech
#      already peaks within 0.2 dB of full scale, so anything louder hard-clips
#      at 48 kHz and sprays harmonics up to 24 kHz.
#   2. ALSA's `plug` only uses speexrate when libasound2-plugins is installed,
#      otherwise falling back to linear interpolation with no anti-aliasing
#      filter. 3:1 decimation then folds all of that 8-24 kHz energy straight
#      back into the 0-8 kHz speech band.
#
# Together those wreck a small wake-word model (which has a frozen embedding
# and no language model to recover with) while barely troubling the cloud ASR
# the Assistant uses. Doing the gain *after* an anti-aliased decimation means
# an over-hot level shows up as a logged clip count instead of silent aliasing.
#

import argparse
import collections
import json
import logging
import math
import os
import re
import threading
import time
import wave

import subprocess

import numpy as np
from openwakeword.model import Model

import requests

import aiy.assistant.grpc
import aiy.audio
import aiy.voicehat

_log_format = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=_log_format,
    handlers=[
        logging.FileHandler('/var/log/voice.log'),
        logging.StreamHandler(),
    ],
)

HA_WEBHOOK_TIMER = "http://localhost:8123/api/webhook/timer7306dfcab90143a3aaf082353dab7a2f1f3ff36dd6df4486bb42f45ea9b8fc05"
HA_WEBHOOK_HEADERS = {
    "Content-Type": "application/json"
}

SAMPLE_RATE = 16000
# openWakeWord requires audio in multiples of 80 ms. Larger chunks do NOT save
# meaningful CPU (predict() still evaluates every 80 ms position internally),
# but they do weaken `patience`, which counts one entry per predict() call.
FRAME_MS = 80
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 1280

# Voice HAT capture hardware is fixed at these parameters.
NATIVE_RATE = 48000
NATIVE_FORMAT = 'S32_LE'
NATIVE_CHANNELS = 2
DECIM = NATIVE_RATE // SAMPLE_RATE  # 3

FIR_TAPS = 127
FIR_CUTOFF_HZ = 7000.0

# S32_LE holds the mics' 24-bit data left-justified, so the low 8 bits are
# zero and full scale is +/-2^31. Dividing by 2^16 maps it to the int16 domain
# openWakeWord expects.
S32_TO_S16 = 65536.0

try:
    from scipy.signal import lfilter
    _HAVE_LFILTER = True
except ImportError:  # pragma: no cover - depends on the Pi's env
    _HAVE_LFILTER = False

logger = logging.getLogger(__name__)


def hotword_from_model_name(name):
    # Normalize model keys ("hey_kodi", "hey_jarvis_v0.1", ".../hey_kodi.tflite")
    # to hotword names that match the routing logic below (e.g. "hey-kodi").
    base = os.path.splitext(os.path.basename(name))[0]
    base = re.sub(r'_v\d+(\.\d+)*$', '', base)
    return base.replace('_', '-')


def on_hey_kodi(phrase):
    logger.info('calling kodi broker with [de] %s' % (phrase))
    params = {'lang': 'de', 'phrase': phrase}
    resp = requests.post('http://localhost:8099/broker', params=params, json={'token': '6a24afcc-9e33-43d0-96dc-cc9a457e306b'})
    if resp.status_code == 500:
        logger.info('calling kodi broker with [en] %s' % (phrase))
        params = {'lang': 'en', 'phrase': phrase}
        resp = requests.post('http://localhost:8099/broker', params=params, json={'token': '6a24afcc-9e33-43d0-96dc-cc9a457e306b'})
    logger.info('resp: %d %s' % (resp.status_code, resp.text))


def on_hey_google(hotword):
    logger.info('getting assistant')
    assistant = aiy.assistant.grpc.get_assistant()
    logger.info('got assistant')
    status_ui.status('listening')
    logger.info('listening for user command')
    text, audio = assistant.recognize()
    if text:
        if hotword == 'hey-kodi':
            on_hey_kodi(text)
            status_ui.status('ready')
            return
        match = re.search(r"\btimer\b.*\b(\d+)\b", text, re.IGNORECASE)
        if match:
            duration = "00:%02d:00" % (int(match.group(1)))
            request_body = {"duration": duration}
            logger.info('time requested: %s' % (duration))
            try:
                resp = requests.post(HA_WEBHOOK_TIMER, headers=HA_WEBHOOK_HEADERS, data=json.dumps(request_body))
                resp.raise_for_status()
            except requests.exceptions.RequestException as e:
                logger.error('HA timer webhook failed: %s', e)
            status_ui.status('ready')
            return
        logger.info('user said: %s', text)
    if audio:
        aiy.audio.play_audio(audio)

    status_ui.status('ready')


def design_lowpass(cutoff_hz, fs, numtaps, gain):
    """Windowed-sinc low-pass, with the capture gain folded into the taps.

    Folding `gain / S32_TO_S16` in here means the decimation and the gain are a
    single multiply-accumulate, and `--gain 30` reproduces the loudness of the
    old `micboost` path exactly — minus the clipping and the aliasing, which
    makes the before/after comparison single-variable.

    Hand-rolled rather than scipy.signal.firwin so an old Python env on the Pi
    needs no new dependency.
    """
    n = np.arange(numtaps, dtype=np.float64)
    h = np.sinc(2.0 * cutoff_hz / fs * (n - (numtaps - 1) / 2.0))
    h *= np.hamming(numtaps)
    h /= h.sum()            # unity DC gain
    h *= gain / S32_TO_S16  # replaces micboost, and rescales S32 -> int16
    return h.astype(np.float32)


class Decimator:
    """Stateful anti-aliased 3:1 decimator, polyphase.

    The state matters: filtering each chunk independently would inject an edge
    transient at every chunk boundary (which is also why scipy's resample_poly
    is unusable here — it zero-pads at array edges).

    Only the samples we keep are computed. Writing this as
    `np.convolve(buf, taps, 'valid')[::DECIM]` spends 3x the multiply-
    accumulates on outputs that are thrown away; splitting the filter into
    DECIM phase sub-filters and convolving the correspondingly decimated input
    phases gives the same result. Measured 1.5x faster than convolve (the
    theoretical 3x does not materialise because np.convolve is already well
    optimised and the per-call overhead is fixed). Worth it because
    openWakeWord alone runs at RTF ~0.82 on the Pi 3, so the capture path has
    very little budget.
    """

    def __init__(self, taps):
        self._numtaps = len(taps)
        self._taps = taps
        # y[n] = sum_j hr[j]*buf[DECIM*n + j] with hr = reversed taps. Split
        # j = DECIM*q + p to get one sub-filter per phase; each is applied to
        # buf[p::DECIM] as a correlation, hence the second reversal.
        hr = taps[::-1]
        self._phases = [np.ascontiguousarray(hr[p::DECIM][::-1], dtype=np.float32)
                        for p in range(DECIM)]
        self._tail = np.zeros(self._numtaps - 1, dtype=np.float32)

    def process(self, x):
        buf = np.concatenate((self._tail, x))
        # Callers guarantee len(x) is a multiple of DECIM, so the decimation
        # phase stays aligned across chunks without an explicit phase counter.
        n_out = len(x) // DECIM
        y = None
        for p, phase_taps in enumerate(self._phases):
            # Phases differ in length by at most one sample; taking the first
            # n_out of each is the correct common alignment.
            yp = np.convolve(buf[p::DECIM], phase_taps, mode='valid')
            y = yp[:n_out] if y is None else y + yp[:n_out]
        assert len(y) == n_out, 'filter length bookkeeping is wrong'
        self._tail = buf[len(buf) - self._numtaps + 1:]
        return np.ascontiguousarray(y, dtype=np.float32)


class DcBlocker:
    """One-pole DC/rumble remover, applied after decimation.

    MEMS I2S mics often carry a DC offset. Off by default; enable only if the
    per-channel `sox stat` mean amplitude says it is needed.
    """

    def __init__(self, cutoff_hz, fs):
        self._a = math.exp(-2.0 * math.pi * cutoff_hz / fs)
        self._b = [1.0, -1.0]
        self._zi = np.zeros(1, dtype=np.float64)
        self._x1 = 0.0
        self._y1 = 0.0

    def process(self, x):
        if _HAVE_LFILTER:
            y, self._zi = lfilter(self._b, [1.0, -self._a], x, zi=self._zi)
            return y.astype(np.float32, copy=False)
        y = np.empty_like(x)
        x1, y1, a = self._x1, self._y1, self._a
        for i in range(len(x)):
            xi = x[i]
            y1 = xi - x1 + a * y1
            x1 = xi
            y[i] = y1
        self._x1, self._y1 = x1, y1
        return y


class AudioSource:
    """Native-rate capture, decimated and gain-scaled to 16 kHz mono int16.

    Owns the arecord process, a drain thread, the filter state and the level
    statistics. `read_chunk()` is the only thing the detection loop needs.
    """

    def __init__(self, device, chunk_samples, gain=30.0, mic_channels='mix',
                 highpass_hz=0.0, stats_interval=0.0, max_buffer_s=4.0):
        self._device = device
        self._chunk_samples = chunk_samples
        self._native_samples = chunk_samples * DECIM
        self._native_bytes = self._native_samples * NATIVE_CHANNELS * 4
        self._mic_channels = mic_channels
        self._decim = Decimator(
            design_lowpass(FIR_CUTOFF_HZ, NATIVE_RATE, FIR_TAPS, gain))
        self._dc = DcBlocker(highpass_hz, SAMPLE_RATE) if highpass_hz > 0 else None

        self._stats_interval = stats_interval
        self._reset_stats()
        self._stats_due = time.monotonic() + stats_interval

        cmd = ['arecord', '-q', '-t', 'raw',
               '-f', NATIVE_FORMAT,
               '-r', str(NATIVE_RATE),
               '-c', str(NATIVE_CHANNELS)]
        if device:
            cmd += ['-D', device]
        # No --buffer-size: the ring belongs to the shared dsnoop instance and
        # cannot be resized from here. The drain thread below supplies the
        # elasticity instead, and reports overflow instead of corrupting the
        # stream the way an ALSA overrun does.
        logger.info('capture: %s', ' '.join(cmd))
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)

        self._buf = collections.deque()
        self._nbytes = 0
        self._cap = int(max_buffer_s * NATIVE_RATE) * NATIVE_CHANNELS * 4
        self._dropped = 0
        self._dropped_logged = 0
        self._drop_log_due = 0.0
        self._eof = False
        self._cv = threading.Condition()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _drain(self):
        try:
            while True:
                data = self._proc.stdout.read(65536)
                if not data:
                    break
                with self._cv:
                    self._buf.append(data)
                    self._nbytes += len(data)
                    while self._nbytes > self._cap:
                        stale = self._buf.popleft()
                        self._nbytes -= len(stale)
                        self._dropped += len(stale)
                    self._cv.notify()
        finally:
            with self._cv:
                self._eof = True
                self._cv.notify_all()

    def _read_exact(self, n):
        with self._cv:
            while self._nbytes < n and not self._eof:
                self._cv.wait(timeout=5.0)
            if self._nbytes < n:
                raise RuntimeError(
                    'arecord exited unexpectedly on device %r' % (self._device,))
            out = bytearray()
            while len(out) < n:
                piece = self._buf.popleft()
                need = n - len(out)
                if len(piece) > need:
                    out += piece[:need]
                    self._buf.appendleft(piece[need:])
                    self._nbytes -= need
                else:
                    out += piece
                    self._nbytes -= len(piece)
            return bytes(out)

    def _reset_stats(self):
        self._peak = 0.0
        self._sumsq = 0.0
        self._nsamp = 0
        self._clipped = 0

    def _accumulate(self, y):
        energy = float(np.dot(y, y))
        # Per-chunk values, exposed so the score log can correlate detections
        # with input level without recomputing them.
        self.last_peak = float(np.abs(y).max())
        self.last_rms = math.sqrt(energy / len(y)) if len(y) else 0.0
        self._peak = max(self._peak, self.last_peak)
        self._sumsq += energy
        self._nsamp += len(y)
        self._clipped += int(np.count_nonzero(np.abs(y) > 32767.0))
        if self._stats_interval <= 0:
            return
        now = time.monotonic()
        if now < self._stats_due:
            return
        rms = math.sqrt(self._sumsq / self._nsamp) if self._nsamp else 0.0
        logger.info('level: peak=%d rms=%d clipped=%.2f%% dropped=%dB',
                    int(self._peak), int(rms),
                    100.0 * self._clipped / max(self._nsamp, 1),
                    self._dropped)
        self._reset_stats()
        self._stats_due = now + self._stats_interval

    def _maybe_log_drops(self):
        """Surface backlog overflow while it is happening, not at shutdown.

        An ALSA overrun is silent; so was this, which defeats the point of
        replacing one with the other.
        """
        if self._dropped == self._dropped_logged:
            return
        now = time.monotonic()
        if now < self._drop_log_due:
            return
        self._drop_log_due = now + 10.0
        lost = self._dropped - self._dropped_logged
        self._dropped_logged = self._dropped
        logger.warning(
            'capture backlog overflow: dropped %.0f ms of audio in the last '
            '10s (inference is not keeping up)',
            1000.0 * lost / (NATIVE_RATE * NATIVE_CHANNELS * 4.0))

    def read_chunk(self):
        self._maybe_log_drops()
        raw = self._read_exact(self._native_bytes)
        x = np.frombuffer(raw, dtype='<i4').reshape(-1, NATIVE_CHANNELS)
        x = x.astype(np.float32)
        if self._mic_channels == 'left':
            m = x[:, 0]
        elif self._mic_channels == 'right':
            m = x[:, 1]
        else:
            m = (x[:, 0] + x[:, 1]) * 0.5
        # Gain lives in the FIR taps, so nothing can clip before this point:
        # raw 24-bit data carries ~55 dB of headroom.
        y = self._decim.process(np.ascontiguousarray(m))
        if self._dc is not None:
            y = self._dc.process(y)
        self._accumulate(y)
        np.clip(y, -32768.0, 32767.0, out=y)
        return y.astype(np.int16)

    def close(self):
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                logger.warning('arecord did not exit after kill')
        self._thread.join(timeout=2)
        if self._dropped:
            logger.warning(
                'dropped %d bytes of capture: inference could not keep up',
                self._dropped)


class DetectConfig:
    def __init__(self, thresholds, default_threshold, dead_time, chunk_samples,
                 patience=None, debounce_time=0.0, near_threshold=0.0):
        self.thresholds = thresholds
        self.default_threshold = default_threshold
        self.dead_time = dead_time
        self.chunk_samples = chunk_samples
        self.patience = patience or {}
        self.debounce_time = debounce_time
        self.near_threshold = near_threshold

    def threshold_for(self, name):
        return self.thresholds.get(name, self.default_threshold)


def decide(scores, cfg):
    """Pick the winning model among those over threshold, or None.

    Ranking by score/threshold rather than raw score keeps the choice
    meaningful once models carry different thresholds. The previous code
    returned on the first dict key over threshold, so when two models crossed
    on the same frame the winner depended on dict iteration order.
    """
    best = None
    best_ratio = 0.0
    for name, score in scores.items():
        threshold = cfg.threshold_for(name)
        if score < threshold:
            continue
        ratio = score / threshold if threshold > 0 else float('inf')
        if ratio > best_ratio:
            best = (name, float(score))
            best_ratio = ratio
    return best


def raw_scores(oww, scores):
    """Ungated scores straight from openWakeWord's prediction buffer.

    When `patience` or `debounce_time` is in play, predict() returns 0.0 for a
    model whose gate was not satisfied. Logging the ungated value alongside it
    is what distinguishes "the model never saw the wake word" from "the model
    fired and a gate suppressed it" — the two halves of this whole problem.
    """
    buf = getattr(oww, 'prediction_buffer', {})
    out = {}
    for name in scores:
        seq = buf.get(name)
        out[name] = float(seq[-1]) if seq else float(scores[name])
    return out


def wait_for_wake(oww, source, cfg, instr=None):
    # Feed the model during the dead window so its buffers stay filled with
    # real audio, but suppress detections — otherwise the speaker echo of the
    # assistant's response (mic + speaker share the Voice HAT) or the tail of
    # the wake utterance itself re-triggers immediately.
    dead_frames = int(cfg.dead_time * SAMPLE_RATE / cfg.chunk_samples)
    frames_seen = 0
    near_peak, near_model, near_quiet = 0.0, '', 0

    while True:
        chunk = source.read_chunk()
        if instr:
            instr.push_audio(chunk)

        t0 = time.monotonic()
        # patience and debounce_time are mutually exclusive upstream (an elif),
        # and both require `threshold` to be passed as a dict.
        if cfg.patience:
            scores = oww.predict(chunk, threshold=cfg.thresholds,
                                 patience=cfg.patience)
        elif cfg.debounce_time:
            scores = oww.predict(chunk, threshold=cfg.thresholds,
                                 debounce_time=cfg.debounce_time)
        else:
            scores = oww.predict(chunk)
        predict_ms = (time.monotonic() - t0) * 1000.0
        frames_seen += 1

        raw = raw_scores(oww, scores)
        if instr:
            instr.log_frame(scores, raw, None,
                            source.last_rms, source.last_peak, predict_ms)

        if logger.isEnabledFor(logging.DEBUG):
            marker = ' (dead)' if frames_seen <= dead_frames else ''
            logger.debug(
                '%s%s',
                ' '.join('%s=%.2f' % (k, v) for k, v in scores.items()),
                marker)
        if frames_seen <= dead_frames:
            continue

        hit = decide(scores, cfg)
        if hit:
            return hit

        # Track near-miss episodes on the *ungated* scores, and report one
        # event per episode rather than one per frame.
        if instr and cfg.near_threshold > 0:
            best = max(raw.items(), key=lambda kv: kv[1]) if raw else ('', 0.0)
            if best[1] >= cfg.near_threshold:
                near_quiet = 0
                if best[1] > near_peak:
                    near_peak, near_model = best[1], best[0]
            elif near_peak > 0:
                near_quiet += 1
                if near_quiet >= 3:
                    gate = 'patience' if cfg.patience else 'below_threshold'
                    instr.on_near_miss(near_model, near_peak, gate=gate)
                    near_peak, near_model, near_quiet = 0.0, '', 0


def build_model(models, framework, vad_threshold, ncpu):
    """Construct the openWakeWord Model with a usable thread count.

    openWakeWord hardcodes `num_threads=1` for the per-wakeword classifiers,
    but AudioFeatures takes an `ncpu` that reaches the melspectrogram and
    embedding interpreters — and those dominate, ~72 ms of the ~83 ms budget
    per 80 ms of audio on a Pi 3. Measured there: ncpu=1 gives RTF 1.04 (i.e.
    it cannot keep up, hence constant ALSA overruns), ncpu=2 gives 0.82. More
    threads are worse (0.87 at 3, 0.84 at 4) — these models are small enough
    that threading overhead outweighs the parallelism.

    `Model` does not declare `ncpu` itself and only forwards it through
    **kwargs on newer versions, so fall back to swapping the preprocessor.
    """
    try:
        return Model(wakeword_models=models,
                     inference_framework=framework,
                     vad_threshold=vad_threshold,
                     ncpu=ncpu)
    except TypeError:
        logger.info('Model() does not accept ncpu; swapping preprocessor instead')
        from openwakeword.utils import AudioFeatures
        oww = Model(wakeword_models=models,
                    inference_framework=framework,
                    vad_threshold=vad_threshold)
        oww.preprocessor = AudioFeatures(inference_framework=framework, ncpu=ncpu)
        return oww


def record_to_wav(source, path, seconds, chunk_samples):
    """Capture through the exact production pipeline, for A/B comparison."""
    n_chunks = int(math.ceil(seconds * SAMPLE_RATE / float(chunk_samples)))
    w = wave.open(path, 'wb')
    try:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        for _ in range(n_chunks):
            w.writeframes(source.read_chunk().tobytes())
    finally:
        w.close()
    logger.info('wrote %.1fs to %s', n_chunks * chunk_samples / float(SAMPLE_RATE), path)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--models',
        nargs='+',
        help='Wake-word models to load. Each entry is either a path to a '
             '.tflite/.onnx file or the name of a built-in openWakeWord model '
             '(e.g. hey_jarvis, alexa, hey_mycroft). A model whose name '
             'normalizes to "hey-kodi" routes the transcript to Kodi. '
             'Not required with --record.')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Detection score threshold in [0,1]; higher is stricter. Default: 0.5')
    parser.add_argument(
        '--framework',
        choices=['tflite', 'onnx'],
        default='tflite',
        help='Inference framework for openWakeWord. Default: tflite')
    parser.add_argument(
        '--vad_threshold',
        type=float,
        default=0.0,
        help='Optional VAD gate to suppress non-speech false triggers. '
             '0 disables (default). Typical enabled value: 0.5')
    parser.add_argument(
        '--ncpu',
        type=int,
        default=2,
        help='Threads for the melspectrogram/embedding interpreters, which '
             'dominate inference cost. Measured on the Pi 3: 1 -> RTF 1.04 '
             '(cannot keep up), 2 -> 0.82, 3 -> 0.87, 4 -> 0.84. Default: 2')
    parser.add_argument(
        '--capture_device',
        default='dsnoop',
        help='ALSA capture PCM passed to arecord -D. Default "dsnoop" opens '
             'the Voice HAT at its native 48 kHz/S32_LE/2ch with no plug '
             'conversion, sharing the card with the assistant recorder. Use '
             '"default" to go back through plug:micboost for comparison. '
             'Note "plughw:0,0" will fail with EBUSY while the assistant '
             'recorder holds the card.')
    parser.add_argument(
        '--gain',
        type=float,
        default=30.0,
        help='Capture gain applied after decimation, replacing the 30x '
             'micboost route gain. Default 30.0 matches the old loudness. '
             'Aim for speech RMS 1500-3000 and peaks below ~24000.')
    parser.add_argument(
        '--mic_channels',
        choices=['mix', 'left', 'right'],
        default='mix',
        help='Which of the two Voice HAT mics to use. "mix" averages both '
             '(default); "left"/"right" isolate one, which diagnoses a dead '
             'or blocked mic without touching ALSA.')
    parser.add_argument(
        '--highpass_hz',
        type=float,
        default=0.0,
        help='DC/rumble high-pass cutoff applied after decimation. '
             '0 disables (default). Try 60 if the mics show a DC offset.')
    parser.add_argument(
        '--stats_interval',
        type=float,
        default=0.0,
        help='Log rolling peak/RMS/clip%% every N seconds. 0 disables '
             '(default). Use 5 while calibrating --gain.')
    parser.add_argument(
        '--dead_time',
        type=float,
        default=1.5,
        help='Seconds after starting a new capture during which detections '
             'are suppressed. Suppresses speaker-echo re-triggers on shared '
             'mic/speaker hardware like the Voice HAT. Default: 1.5')
    parser.add_argument(
        '--chunk_ms',
        type=int,
        default=160,
        help='Audio chunk size fed to the model, in ms. Must be a positive '
             'multiple of 80. Larger chunks barely save CPU but weaken '
             'patience-style gating and add latency. Default: 160')
    parser.add_argument(
        '--record',
        metavar='PATH',
        help='Capture through the exact production pipeline to a 16 kHz mono '
             'WAV and exit. This is the A/B harness against a plain '
             '"arecord -D default -f S16_LE -r 16000 -c 1".')
    parser.add_argument(
        '--record_seconds',
        type=float,
        default=10.0,
        help='Duration for --record. Default: 10')
    parser.add_argument(
        '--model_threshold',
        action='append',
        default=[],
        metavar='NAME=VALUE',
        help='Per-model threshold override, repeatable (e.g. '
             '--model_threshold alexa=0.6). Overrides --threshold for that '
             'model. alexa and hey_kodi generally want different operating '
             'points.')
    parser.add_argument(
        '--patience',
        type=int,
        default=0,
        help='Require N consecutive frames over threshold before firing. '
             '0 disables (default). This is the cheapest false-positive '
             'lever, but it counts one entry per predict() call, so it is '
             'only meaningful at --chunk_ms 80. Mutually exclusive with '
             '--debounce_time.')
    parser.add_argument(
        '--debounce_time',
        type=float,
        default=0.0,
        help='Seconds of cooldown enforced by openWakeWord between '
             'detections. 0 disables (default). Mutually exclusive with '
             '--patience.')

    group = parser.add_argument_group('instrumentation (off by default)')
    group.add_argument(
        '--capture',
        action='store_true',
        help='Record score logs and audio clips around detections and '
             'near-misses, so false activations can be listened to instead '
             'of guessed at.')
    group.add_argument(
        '--capture_dir',
        default=os.path.expanduser('~/wake_data'),
        help='Where instrumentation writes. Default: ~/wake_data')
    group.add_argument(
        '--near_threshold',
        type=float,
        default=None,
        help='Also record clips for scores reaching this but not firing, '
             'which diagnoses the "hard to trigger" case. '
             'Default: max(0.1, threshold - 0.25)')
    group.add_argument('--clip_pre_s', type=float, default=4.0,
                       help='Seconds of audio kept before an event. Default: 4')
    group.add_argument('--clip_post_s', type=float, default=1.5,
                       help='Seconds kept after an event. Default: 1.5')
    group.add_argument('--mark_pre_s', type=float, default=6.0,
                       help='Pre-roll for a button mark, which arrives after '
                            'you have finished speaking. Default: 6')
    group.add_argument('--log_scores', choices=['events', 'all'],
                       default='events',
                       help='"events" logs scores around events plus a 10s '
                            'summary row (~0.5 MB/day). "all" logs every '
                            'frame (~50 MB/day). Default: events')
    group.add_argument('--keep_days', type=int, default=14,
                       help='Prune clips and logs older than this. Default: 14')
    group.add_argument('--max_clips_per_hour', type=int, default=30,
                       help='Rate limit on clip writing. Default: 30')
    group.add_argument('--fa_window', type=float, default=20.0,
                       help='A button press within this many seconds of a '
                            'detection tags it as a false activation rather '
                            'than marking a miss. Default: 20')
    group.add_argument('--mark_button', action='store_true',
                       help='Use the VoiceHat button for ground truth: press '
                            'after a wrong trigger to tag it, or press after '
                            'a wake word that did not register to mark it.')
    group.add_argument('--profile', action='store_true',
                       help='Log predict() p50/p95 and real-time factor every '
                            '10s. RTF must stay below 1.0 or capture overruns.')

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Log per-frame prediction scores at DEBUG level.')
    return parser


def resolve_model_keys(oww, names):
    """Map user-supplied model names onto openWakeWord's score-dict keys.

    Fails loudly rather than silently ignoring an unmatched name — a
    --model_threshold that quietly does nothing would invalidate days of
    tuning without any visible symptom.
    """
    keys = list(getattr(oww, 'models', {}))
    resolved = {}
    for name in names:
        want = hotword_from_model_name(name)
        match = [k for k in keys if hotword_from_model_name(k) == want]
        if not match:
            raise ValueError(
                '%r matches no loaded model; loaded: %s' % (name, ', '.join(keys)))
        resolved[name] = match[0]
    return resolved


def main():
    global status_ui

    parser = build_parser()
    args = parser.parse_args()

    if args.chunk_ms <= 0 or args.chunk_ms % FRAME_MS != 0:
        parser.error('--chunk_ms must be a positive multiple of %d' % FRAME_MS)
    chunk_samples = SAMPLE_RATE * args.chunk_ms // 1000

    if args.patience and args.debounce_time:
        # openWakeWord picks one with an elif; silently honouring only one
        # would make the other look ineffective.
        parser.error('--patience and --debounce_time are mutually exclusive')
    if args.patience and args.chunk_ms != FRAME_MS:
        logger.warning(
            '--patience counts one entry per predict() call, so at '
            '--chunk_ms %d it requires %d ms of wall time where each call is '
            'already a max over %d frames. Use --chunk_ms %d.',
            args.chunk_ms, args.patience * args.chunk_ms,
            args.chunk_ms // FRAME_MS, FRAME_MS)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    def open_source():
        return AudioSource(
            args.capture_device, chunk_samples,
            gain=args.gain,
            mic_channels=args.mic_channels,
            highpass_hz=args.highpass_hz,
            stats_interval=args.stats_interval)

    if args.record:
        # No model and no VoiceHat UI needed for a capture-only run.
        with open_source() as source:
            record_to_wav(source, args.record, args.record_seconds, chunk_samples)
        return

    if not args.models:
        parser.error('--models is required unless --record is given')

    status_ui = aiy.voicehat.get_status_ui()
    status_ui.status('starting')

    oww = build_model(args.models, args.framework, args.vad_threshold, args.ncpu)

    thresholds = {name: args.threshold for name in getattr(oww, 'models', {})}
    for item in args.model_threshold:
        name, _, value = item.partition('=')
        if not value:
            parser.error('--model_threshold expects NAME=VALUE, got %r' % item)
        try:
            key = resolve_model_keys(oww, [name])[name]
        except ValueError as e:
            parser.error(str(e))
        thresholds[key] = float(value)

    patience = {k: args.patience for k in thresholds} if args.patience else {}
    near_threshold = args.near_threshold
    if near_threshold is None:
        near_threshold = max(0.1, args.threshold - 0.25)

    cfg = DetectConfig(thresholds, args.threshold, args.dead_time, chunk_samples,
                       patience=patience, debounce_time=args.debounce_time,
                       near_threshold=near_threshold if args.capture else 0.0)

    instr = None
    if args.capture:
        import wake_instrument
        instr = wake_instrument.Instrumentation(
            args.capture_dir, args.threshold, near_threshold,
            clip_pre_s=args.clip_pre_s, clip_post_s=args.clip_post_s,
            mark_pre_s=args.mark_pre_s, log_scores=args.log_scores,
            keep_days=args.keep_days,
            max_clips_per_hour=args.max_clips_per_hour,
            fa_window=args.fa_window, mark_button=args.mark_button,
            profile=args.profile, chunk_samples=chunk_samples)
        logger.info('instrumentation writing to %s', args.capture_dir)

    logger.info('loaded wake-word models: %s', args.models)
    logger.info('thresholds: %s%s', thresholds,
                '  patience=%d' % args.patience if args.patience else '')
    logger.info('listening for wake word (Ctrl+C to exit)')

    try:
        with aiy.audio.get_recorder():
            status_ui.status('ready')
            while True:
                # A fresh capture each cycle drops audio buffered during the
                # preceding assistant round-trip.
                with open_source() as source:
                    name, score = wait_for_wake(oww, source, cfg, instr)
                hotword = hotword_from_model_name(name)
                logger.info('detected %s (score %.3f)', hotword, score)
                if instr:
                    instr.on_fire(name, score)
                    instr.on_deaf(True)
                on_hey_google(hotword)
                if instr:
                    instr.on_deaf(False)
                oww.reset()
    except KeyboardInterrupt:
        logger.info('stopping')
    finally:
        if instr:
            instr.close()


if __name__ == '__main__':
    main()
