# 1
FROM python:3.11-slim

#  2
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /usr/src/app

#  3
WORKDIR $APP_HOME

#  4 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#  5
COPY . $APP_HOME

#  6
EXPOSE 8000

#  7 

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
