FROM python:3.8-alpine
WORKDIR /xai_framework

RUN python -m pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
RUN pip install -r model_service/resnet50/requirements.txt
RUN pip install -r xai_service/pytorch_cam/requirements.txt
CMD ["flask", "--app", "db_service", "--debug", "run", "-p", "5001"]
CMD ["flask", "--app", "model_service/resnet50", "--debug", "run", "-p", "5002"]
CMD ["flask", "--app", "xai_service/pytorch_cam", "--debug", "run", "-p", "5000"]
