# syntax=docker/dockerfile:1
FROM python:3.9-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN apt-get update
RUN apt-get install -y gcc default-mysql-server default-libmysqlclient-dev
RUN pip3 install -r requirements.txt
COPY . .
CMD [ "python3", "-m" , "project.server", "--host", "0.0.0.0", "--port", "8080"]
