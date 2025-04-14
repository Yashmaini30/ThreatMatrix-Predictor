FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app
#COPY .env .env for #####local deployment

RUN apt update -y && apt install git awscli -y
RUN apt-get update && pip install -r requirements.txt
EXPOSE 8000
CMD [ "python3", "app.py" ]