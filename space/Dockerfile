# FinSage API on Hugging Face Spaces (Docker SDK, CPU).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# HF cache lives here; make it writable for the Space's non-root user.
ENV HF_HOME=/app/.cache
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

ENV HF_REPO=AshutoshMore96/finsage-gguf
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
