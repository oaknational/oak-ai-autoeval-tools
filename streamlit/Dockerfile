FROM python:3

ENV PYTHONUNBUFFERED True

EXPOSE 8080


ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN pip install -r requirements.txt


CMD streamlit run --serverport 8080 --server.enableCORS false Hello.py
