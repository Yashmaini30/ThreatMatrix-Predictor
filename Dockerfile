FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN apt update -y && apt install git awscli -y
RUN apt-get update && pip install -r requirements.txt
EXPOSE 8000
CMD [ "python3", "app.py" ]