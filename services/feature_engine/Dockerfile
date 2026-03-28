FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    bytewax==0.21 \
    kafka-python \
    redis \
    psycopg2-binary \
    python-dotenv

COPY services/feature_engine/ ./services/feature_engine/

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "services.feature_engine.dataflow"]