FROM python:3

ENV PYTHONUNBUFFERED True

EXPOSE 8080

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN pip install -r requirements.txt

CMD streamlit run --server.port $PORT --server.enableCORS false streamlit/Hello.py
