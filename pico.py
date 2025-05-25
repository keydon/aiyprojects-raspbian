#
# Copyright 2018-2023 Picovoice Inc.
#
# You may not use this file except in compliance with the license. A copy of the license is located in the "LICENSE"
# file accompanying this source.
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#

import argparse
import os
import struct
import wave
import re
from datetime import datetime

import pvporcupine
from pvrecorder import PvRecorder

import json
import requests
import logging

import aiy.assistant.grpc
import aiy.audio
import aiy.voicehat

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
)

HA_WEBHOOK_TIMER = "http://localhost:8123/api/webhook/timer7306dfcab90143a3aaf082353dab7a2f1f3ff36dd6df4486bb42f45ea9b8fc05"
HA_WEBHOOK_HEADERS = {
    "Content-Type": "application/json"
}


logger = logging.getLogger(__name__)

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
    #button = aiy.voicehat.get_button()
    #with aiy.audio.get_recorder():
        #while True:
            #status_ui.status('ready')
            #print('Press the button and speak')
            #button.wait_for_press()
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
                    request_body = {
                      "duration": duration
                    }
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



def main():
    global status_ui
    status_ui = aiy.voicehat.get_status_ui()
    status_ui.status('starting')
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--access_key',
        help='AccessKey obtained from Picovoice Console (https://console.picovoice.ai/)')

    parser.add_argument(
        '--keywords',
        nargs='+',
        help='List of default keywords for detection. Available keywords: %s' % ', '.join(
            '%s' % w for w in sorted(pvporcupine.KEYWORDS)),
        choices=sorted(pvporcupine.KEYWORDS),
        metavar='')

    parser.add_argument(
        '--keyword_paths',
        nargs='+',
        help="Absolute paths to keyword model files. If not set it will be populated from `--keywords` argument")

    parser.add_argument(
        '--library_path',
        help='Absolute path to dynamic library. Default: using the library provided by `pvporcupine`')

    parser.add_argument(
        '--model_path',
        help='Absolute path to the file containing model parameters. '
             'Default: using the library provided by `pvporcupine`')

    parser.add_argument(
        '--sensitivities',
        nargs='+',
        help="Sensitivities for detecting keywords. Each value should be a number within [0, 1]. A higher "
             "sensitivity results in fewer misses at the cost of increasing the false alarm rate. If not set 0.5 "
             "will be used.",
        type=float,
        default=None)

    parser.add_argument('--audio_device_index', help='Index of input audio device.', type=int, default=-1)

    parser.add_argument('--output_path', help='Absolute path to recorded audio for debugging.', default=None)

    parser.add_argument('--show_audio_devices', action='store_true')

    args = parser.parse_args()

    if args.show_audio_devices:
        for i, device in enumerate(PvRecorder.get_available_devices()):
            print('Device %d: %s' % (i, device))
        return

    if args.keyword_paths is None:
        if args.keywords is None:
            raise ValueError("Either `--keywords` or `--keyword_paths` must be set.")

        keyword_paths = [pvporcupine.KEYWORD_PATHS[x] for x in args.keywords]
    else:
        keyword_paths = args.keyword_paths

    if args.sensitivities is None:
        args.sensitivities = [0.5] * len(keyword_paths)

    if len(keyword_paths) != len(args.sensitivities):
        raise ValueError('Number of keywords does not match the number of sensitivities.')

    try:
        porcupine = pvporcupine.create(
            access_key=args.access_key,
            library_path=args.library_path,
            model_path=args.model_path,
            keyword_paths=keyword_paths,
            sensitivities=args.sensitivities)
    except pvporcupine.PorcupineInvalidArgumentError as e:
        print("One or more arguments provided to Porcupine is invalid: ", args)
        print(e)
        raise e
    except pvporcupine.PorcupineActivationError as e:
        print("AccessKey activation error")
        raise e
    except pvporcupine.PorcupineActivationLimitError as e:
        print("AccessKey '%s' has reached it's temporary device limit" % args.access_key)
        raise e
    except pvporcupine.PorcupineActivationRefusedError as e:
        print("AccessKey '%s' refused" % args.access_key)
        raise e
    except pvporcupine.PorcupineActivationThrottledError as e:
        print("AccessKey '%s' has been throttled" % args.access_key)
        raise e
    except pvporcupine.PorcupineError as e:
        print("Failed to initialize Porcupine")
        raise e

    keywords = list()
    for x in keyword_paths:
        keyword_phrase_part = os.path.basename(x).replace('.ppn', '').split('_')
        if len(keyword_phrase_part) > 6:
            keywords.append(' '.join(keyword_phrase_part[0:-6]))
        else:
            keywords.append(keyword_phrase_part[0])

    print('Porcupine version: %s' % porcupine.version)

    recorder = PvRecorder(
        frame_length=porcupine.frame_length,
        device_index=args.audio_device_index)
    recorder.start()

    wav_file = None
    if args.output_path is not None:
        wav_file = wave.open(args.output_path, "w")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)

    print('Listening ... (press Ctrl+C to exit)')

    try:
      with aiy.audio.get_recorder():
        status_ui.status('ready')
        while True:
            pcm = recorder.read()
            result = porcupine.process(pcm)

            if wav_file is not None:
                wav_file.writeframes(struct.pack("h" * len(pcm), *pcm))

            if result >= 0:
                logger.info('[%s] Detected %s' % (str(datetime.now()), keywords[result]))
                on_hey_google(keywords[result])
    except KeyboardInterrupt:
        print('Stopping ...')
    finally:
        recorder.delete()
        porcupine.delete()
        if wav_file is not None:
            wav_file.close()


if __name__ == '__main__':
    main()
