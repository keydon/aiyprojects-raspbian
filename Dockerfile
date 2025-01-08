FROM balenalib/raspberry-pi3-debian:buster
MAINTAINER "Ugly-Wan" <oliver@mcblain.co.uk>

RUN apt-get purge -y python.*
RUN apt-get update
RUN apt-get install -y \
  curl \
  alsa-utils \
  git \
  python3 \
  python3-dev \
  python3-venv \
  build-essential \
  vim \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /home/pi/AIY-projects-python/requirements.txt
RUN cd /home/pi/AIY-projects-python \
  && python3 -m venv env \
  && env/bin/python -m pip install --upgrade pip numpy rpi.gpio google_auth_oauthlib google-assistant-library setuptools wheel kodipydent \
  && env/bin/pip install -r requirements.txt
RUN echo "/home/pi/AIY-projects-python/src" > /home/pi/AIY-projects-python/env/lib/python3.4/site-packages/aiy.pth
RUN echo "/home/pi/AIY-projects-python/src" > /home/pi/AIY-projects-python/env/lib/python3.7/site-packages/aiy.pth

COPY . /home/pi/AIY-projects-python
RUN cp /home/pi/AIY-projects-python/assistant.json /home/pi/assistant.json
RUN whoami
USER pi
RUN realpath ~
CMD /bin/bash ; sleep infinity
