# FinSage API — CPU GGUF inference container.
# Build: docker build -t finsage-api .
# Run:   docker run -p 8000:8000 -v $(pwd)/models:/app/models finsage-api
FROM python:3.11-slim

# Build tools needed to compile llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src ./src
COPY config ./config

# Models are mounted at runtime (kept out of the image — they're large).
ENV FINSAGE_GGUF=/app/models/finsage-merged-16bit-gguf/finsage.Q4_K_M.gguf
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').read() else 1)"

CMD ["uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
