FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Follow the original README setup: CPU core environment (GUI + transcription + conversion + rendering).
RUN python scripts/setup_core.py

# Same application entry point as the original launcher; only bind to Render's public port.
CMD ["sh", "-c", "MUSICMASTER_HOST=0.0.0.0 MUSICMASTER_PORT=${PORT:-10000} .venv/bin/python -m musicmaster.web.server"]
