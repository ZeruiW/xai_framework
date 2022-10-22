# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /model

COPY ../requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD [ "flask", "--app", "resnet50", "run", "--host=0.0.0.0", "-p", "5001"]