FROM python:3.7-slim

# libgomp1: OpenMP runtime required by xgboost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-py37.txt .
RUN pip install --no-cache-dir -r requirements-py37.txt

COPY river_forecast/ river_forecast/
COPY models/ models/
COPY data/ data/
COPY bin/create_river_forecast.py bin/

# forecast.png is written to the working directory; mount /output to collect it
RUN mkdir /output
WORKDIR /output
ENV PYTHONPATH=/app

CMD ["python", "/app/bin/create_river_forecast.py"]
