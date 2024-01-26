FROM python:3.8

ENV PYTHONUNBUFFERED True

ENV APP_HOME /webapp
WORKDIR $APP_HOME
COPY . ./

RUN pip install -r requirements.txt

EXPOSE 8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:webapp
