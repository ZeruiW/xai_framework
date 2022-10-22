# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /xai_service

COPY requirements.txt requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt

COPY . .

CMD [ "flask", "--app", "pytorch_cam", "run", "--host=0.0.0.0", "-p", "5003"]