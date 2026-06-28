# FinSage API on Hugging Face Spaces.
# Compile the LATEST llama-cpp-python from source (glibc) — the prebuilt wheels on the
# index are too old (bundle an outdated llama.cpp that crashes on this Qwen2.5 GGUF with
# `llama_decode returned -1`). Build flags below keep the compile portable and within RAM.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Portable build + limit parallelism so the compile doesn't exhaust the builder's memory.
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF"
ENV CMAKE_BUILD_PARALLEL_LEVEL=2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV HF_HOME=/app/.cache
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

ENV HF_REPO=AshutoshMore96/finsage-gguf
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
