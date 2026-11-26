FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Audio decoding and sound-file runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[gui,render,transcribe]"

# Render supplies PORT. Keep the original local defaults unchanged in server.py.
CMD ["sh", "-c", "export MUSICMASTER_HOST=0.0.0.0; export MUSICMASTER_PORT=${PORT:-10000}; exec python -m musicmaster.web.server"]
