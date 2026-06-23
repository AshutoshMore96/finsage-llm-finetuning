"""
FinSage production API — FastAPI + llama.cpp (GGUF, CPU inference).

Runs anywhere CPU is available: a $5 cloud VM, Render/Railway/Fly.io, HF Spaces,
or locally on the Mac. Uses the quantized GGUF so it needs no GPU.

Local:   uvicorn src.serve.api:app --host 0.0.0.0 --port 8000
Docker:  docker compose up   (see docker-compose.yml)

Endpoints:
  GET  /health            -> liveness/readiness
  POST /chat              -> {"message": "...", "history": [["u","a"], ...]}
  GET  /                  -> minimal info
Optional auth: set api.api_key in config -> clients send header  X-API-Key: <value>
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from src.config_utils import abspath, load_config
from src.data.prompts import SYSTEM_PROMPT

cfg = load_config()
_llm = None  # lazy global, loaded on startup


def _load_llm():
    from llama_cpp import Llama

    model_path = os.getenv("FINSAGE_GGUF", abspath(cfg.serve.gguf_path))
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"GGUF model not found at {model_path}. "
            "Export it with src.merge.merge_and_quantize or set FINSAGE_GGUF."
        )
    n_threads = cfg.api.n_threads or (os.cpu_count() or 4)
    return Llama(
        model_path=model_path,
        n_ctx=cfg.api.n_ctx,
        n_threads=n_threads,
        verbose=False,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm
    _llm = _load_llm()
    yield
    _llm = None


app = FastAPI(title="FinSage API", version="1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    history: list[list[str]] = []   # [["user msg", "assistant msg"], ...]
    max_tokens: int | None = None
    temperature: float | None = None


class ChatResponse(BaseModel):
    answer: str
    model: str


def _check_key(x_api_key: str | None):
    if cfg.api.api_key and x_api_key != cfg.api.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@app.get("/")
def root():
    return {"service": "FinSage", "model": cfg.model.name, "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok" if _llm is not None else "loading"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, x_api_key: str | None = Header(default=None)):
    _check_key(x_api_key)
    if _llm is None:
        raise HTTPException(status_code=503, detail="Model still loading")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in req.history:
        if len(turn) == 2:
            messages.append({"role": "user", "content": turn[0]})
            messages.append({"role": "assistant", "content": turn[1]})
    messages.append({"role": "user", "content": req.message})

    out = _llm.create_chat_completion(
        messages=messages,
        max_tokens=req.max_tokens or cfg.api.max_tokens,
        temperature=req.temperature if req.temperature is not None else cfg.api.temperature,
    )
    answer = out["choices"][0]["message"]["content"].strip()
    return ChatResponse(answer=answer, model=cfg.model.name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serve.api:app", host=cfg.api.host, port=cfg.api.port)
