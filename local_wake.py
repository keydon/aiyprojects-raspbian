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
# Audio capture is done by shelling out to `arecord` (already present on
# Raspbian) rather than sounddevice, because PortAudio does not expose ALSA
# plug-family PCMs and cannot resample the Voice HAT's fixed native rate.
#

import argparse
import json
import logging
import os
import re
from datetime import datetime

import subprocess

import numpy as np
from openwakeword.model import Model

import requests

import aiy.assistant.grpc
import aiy.audio
import aiy.voicehat

logging.basicConfig(
    filename='/var/log/voice.log',
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
)

HA_WEBHOOK_TIMER = "http://localhost:8123/api/webhook/timer7306dfcab90143a3aaf082353dab7a2f1f3ff36dd6df4486bb42f45ea9b8fc05"
HA_WEBHOOK_HEADERS = {
    "Content-Type": "application/json"
}

SAMPLE_RATE = 16000
# openWakeWord expects 80 ms int16 frames at 16 kHz.
CHUNK_SAMPLES = 1280

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
    print('Listening...')
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
                print(f"Error during request: {e}")
            status_ui.status('ready')
            return
        print('You said "', text, '"')
    if audio:
        aiy.audio.play_audio(audio)

    status_ui.status('ready')


def wait_for_wake(oww, threshold, debug, device, dead_time_s):
    # Capture via arecord so we use ALSA's plug layer (which handles the
    # Voice HAT's fixed native rate → 16 kHz resample) instead of PortAudio,
    # which cannot see plug-family PCMs. A fresh process each call also drops
    # audio buffered during the preceding assistant round-trip.
    cmd = ['arecord', '-q', '-f', 'S16_LE',
           '-r', str(SAMPLE_RATE), '-c', '1', '-t', 'raw',
           '--buffer-size=32000']  # ~1s of slack so brief inference stalls don't overrun
    if device:
        cmd += ['-D', device]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
    # Feed the model during the dead window so its buffers stay filled with
    # real audio, but suppress detections — otherwise the speaker echo of the
    # assistant's response (mic + speaker share the Voice HAT) or the tail of
    # the wake utterance itself re-triggers immediately.
    dead_frames = int(dead_time_s * SAMPLE_RATE / CHUNK_SAMPLES)
    try:
        chunk_bytes = CHUNK_SAMPLES * 2  # int16 = 2 bytes/sample
        frames_seen = 0
        while True:
            raw = b''
            while len(raw) < chunk_bytes:
                more = proc.stdout.read(chunk_bytes - len(raw))
                if not more:
                    raise RuntimeError('arecord exited unexpectedly on device %r' % device)
                raw += more
            chunk = np.frombuffer(raw, dtype=np.int16)
            scores = oww.predict(chunk)
            frames_seen += 1
            if debug:
                marker = ' (dead)' if frames_seen <= dead_frames else ''
                print(' '.join('%s=%.2f' % (k, v) for k, v in scores.items()) + marker)
            if frames_seen <= dead_frames:
                continue
            for name, score in scores.items():
                if score >= threshold:
                    return name, score
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    global status_ui
    status_ui = aiy.voicehat.get_status_ui()
    status_ui.status('starting')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='Wake-word models to load. Each entry is either a path to a '
             '.tflite/.onnx file or the name of a built-in openWakeWord model '
             '(e.g. hey_jarvis, alexa, hey_mycroft). A model whose name '
             'normalizes to "hey-kodi" routes the transcript to Kodi.')
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
        '--capture_device',
        default='',
        help='ALSA capture PCM name passed to arecord -D. Empty (default) '
             'uses arecord\'s system default — matches the path used by a '
             'bare "arecord test.wav". Try "plughw:0,0" or "micboost" if '
             'the default routes to the wrong source.')
    parser.add_argument(
        '--dead_time',
        type=float,
        default=1.5,
        help='Seconds after starting a new capture during which detections '
             'are suppressed. Suppresses speaker-echo re-triggers on shared '
             'mic/speaker hardware like the Voice HAT. Default: 1.5')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Print per-frame prediction scores to stdout.')
    args = parser.parse_args()

    oww = Model(
        wakeword_models=args.models,
        inference_framework=args.framework,
        vad_threshold=args.vad_threshold)

    print('Loaded wake-word models: %s' % args.models)
    print('Listening ... (press Ctrl+C to exit)')

    try:
        with aiy.audio.get_recorder():
            status_ui.status('ready')
            while True:
                name, score = wait_for_wake(
                    oww, args.threshold, args.debug,
                    args.capture_device, args.dead_time)
                hotword = hotword_from_model_name(name)
                logger.info('[%s] Detected %s (score %.3f)' % (
                    str(datetime.now()), hotword, score))
                on_hey_google(hotword)
                oww.reset()
    except KeyboardInterrupt:
        print('Stopping ...')


if __name__ == '__main__':
    main()
