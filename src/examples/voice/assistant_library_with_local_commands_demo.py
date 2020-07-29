#!/usr/bin/env python3
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a recognizer using the Google Assistant Library.

The Google Assistant Library has direct access to the audio API, so this Python
code doesn't need to record audio. Hot word detection "OK, Google" is supported.

The Google Assistant Library can be installed with:
    env/bin/pip install google-assistant-library==0.0.2

It is available for Raspberry Pi 2/3 only; Pi Zero is not supported.
"""

LOG_FILENAME = '/var/log/voice.log'

import logging
import subprocess
import sys
import threading
import socket

#import asyncio
import urllib.error
import aiy.assistant.auth_helpers
import aiy.audio
import aiy.voicehat
from google.assistant.library import Assistant
from google.assistant.library.event import EventType
from kodipydent import Kodi

logging.basicConfig(
    filename='/var/log/voice.log',
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
)

kodi = None
assistant = None
alarm_is_buzzing = False
ttl = 3
isResponding = False

def set_kodi_volume(volume):
    try:
        global kodi
        if kodi is None:
            kodi = Kodi('openelec', username='keydon', password='desperad0')
        kodi.Application.SetVolume(volume)
    except urllib.error.URLError:
        logging.error('kodi not reachable')

def power_off_pi():
    aiy.audio.say('Good bye!')
    subprocess.call('sudo shutdown now', shell=True)


def reboot_pi():
    aiy.audio.say('See you in a bit!')
    subprocess.call('sudo reboot', shell=True)

def soft_reboot():
    aiy.audio.say('See you in a soft bit!')
    sys.exit(1)

def kill_kodi():
    aiy.audio.say('Restarting kodi')
    subprocess.call('ssh -l root openelec killall -9 kodi.bin', shell=True)

def wakeup_kodi():
    aiy.audio.say('waking up')
    openelec_mac = "54:04:a6:d1:0e:8e"
    target = openelec_mac.replace(":", "")
    magic = bytes.fromhex("F" * 12 + target * 16)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
      sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
      sock.connect(("255.255.255.255", 9))
      sock.send(magic)

    print("sent magic")

def say_ip():
    ip_address = subprocess.check_output("hostname -I | cut -d' ' -f1", shell=True)
    aiy.audio.say('My IP address is %s' % ip_address.decode('utf-8'))

def process_event(assistant, event):
    logging.info('processing event type: ' + str(event.type));
    status_ui = aiy.voicehat.get_status_ui()
    if event.type == EventType.ON_START_FINISHED:
        set_kodi_volume(60)
        status_ui.status('ready')
        aiy.voicehat.get_button().on_press(on_button_pressed)
        aiy.audio.say('I am here to serve you my master!')
        if sys.stdout.isatty():
            print('Say "OK, Google" then speak, or press Ctrl+C to quit...')
        set_kodi_volume(100)

    elif event.type == EventType.ON_CONVERSATION_TURN_STARTED:
        status_ui.status('listening')
        set_kodi_volume(60)

    elif event.type == EventType.ON_RECOGNIZING_SPEECH_FINISHED and event.args:
        logging.info('You said: %s', event.args['text'])
        global ttl
        ttl = 3
        text = event.args['text'].lower()
        if text == 'power off':
            assistant.stop_conversation()
            power_off_pi()
        elif text == 'reboot':
            assistant.stop_conversation()
            reboot_pi()
        elif text == 'ip address':
            assistant.stop_conversation()
            say_ip()
        elif text == 'soft reboot' or text == 'soft restart':
            assistant.stop_conversation()
            soft_reboot()
        elif text == 'wake up' or text == 'kodi wake up':
            assistant.stop_conversation()
            wakeup_kodi()
        elif (text == 'kill kodi' or text == 'restart kodi'
                or text == 'kodi restart'):
            assistant.stop_conversation()
            kill_kodi()
        else:
            pass
    elif event.type == EventType.ON_RENDER_RESPONSE and event.args:
        logging.info('She said: %s', event.args['text'])
    elif event.type == EventType.ON_END_OF_UTTERANCE:
        status_ui.status('thinking')

    elif event.type == EventType.ON_CONVERSATION_TURN_FINISHED:
        global ttl
        ttl = 3
        status_ui.status('ready')
        set_kodi_volume(100)

    elif event.type == EventType.ON_ASSISTANT_ERROR and event.args and event.args['is_fatal']:
        sys.exit(1)
    elif event.type == EventType.ON_ASSISTANT_ERROR:
        status_ui.status('ready')
        global ttl
        ttl -= -1
        logging.warn('TTL: %s', ttl)
        if ttl == 0:
        	sys.exit(1)
        set_kodi_volume(100)
    elif event.type == EventType.ON_RESPONDING_STARTED:
        set_kodi_volume(60)
    elif event.type == EventType.ON_RESPONDING_FINISHED:
        set_kodi_volume(100)
    elif event.type == EventType.ON_ALERT_STARTED:
        set_kodi_volume(60)
        global alarm_is_buzzing
        alarm_is_buzzing = True
    elif event.type == EventType.ON_ALERT_FINISHED:
        set_kodi_volume(100)
        global alarm_is_buzzing
        alarm_is_buzzing = False
    elif event.type == EventType.ON_CONVERSATION_TURN_TIMEOUT:
        set_kodi_volume(100)
    else:
        logging.info('Unkown event type: ' + str(event.type));


def run_task():
    global assistant
    credentials = aiy.assistant.auth_helpers.get_assistant_credentials()
    with Assistant(credentials, 'my-home-speech-script-AIY-Model') as assist:
        assistant = assist
        for event in assist.start():
            try:
                process_event(assist, event)
            except Exception as e:
                logging.exception("processing event failed")            

def on_button_pressed():
    logging.info("buttooon is pressed!")
    global alarm_is_buzzing
    if alarm_is_buzzing:
        logging.info("Stoppping the buzz")
        assistant.send_text_query("stop")
        alarm_is_buzzing = False
        return

    logging.info("regular button press, i am listening!")
    assistant.stop_conversation()
    assistant.start_conversation()
    logging.info("conversation started")

if __name__ == '__main__':
    try:
        task = threading.Thread(target=run_task)
        task.start()
    except Exception as e:
        logging.exception("main failed")
