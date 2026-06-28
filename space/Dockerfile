# FinSage API on Hugging Face Spaces.
# Alpine (musl) base — matches the prebuilt llama-cpp-python CPU wheel (musl-linked),
# so it installs fast AND loads at runtime (no source compile, no glibc/musl mismatch).
FROM python:3.11-alpine

# Runtime libs the llama.cpp shared object needs (C++ stdlib + OpenMP).
RUN apk add --no-cache libstdc++ libgomp

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Writable HF cache for the non-root Spaces user.
ENV HF_HOME=/app/.cache
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

ENV HF_REPO=AshutoshMore96/finsage-gguf
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
