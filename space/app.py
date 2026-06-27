"""
FinSage API — Hugging Face Space (FastAPI + llama.cpp, CPU).

Downloads the fine-tuned GGUF from the model repo at startup and serves it over a
CORS-enabled /chat endpoint so the portfolio website (any origin) can call it.

POST /chat   {"message": "...", "history": [["u","a"], ...]}  ->  {"answer": "..."}
"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import HfApi, hf_hub_download
from llama_cpp import Llama

REPO = os.getenv("HF_REPO", "AshutoshMore96/finsage-gguf")
SYSTEM = (
    "You are FinSage, a meticulous financial-analysis assistant. Answer the question "
    "directly and concisely. Do not output sentiment labels or extra tokens."
)

# Find the .gguf file in the repo (robust to its exact name) and download it.
_gguf = next(f for f in HfApi().list_repo_files(REPO) if f.endswith(".gguf"))
_model_path = hf_hub_download(repo_id=REPO, filename=_gguf)
llm = Llama(model_path=_model_path, n_ctx=4096,
            n_threads=os.cpu_count() or 2, verbose=False)

app = FastAPI(title="FinSage API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class ChatReq(BaseModel):
    message: str
    history: list = []


@app.get("/")
def root():
    return {"service": "FinSage", "status": "ok", "model": _gguf}


@app.post("/chat")
def chat(req: ChatReq):
    messages = [{"role": "system", "content": SYSTEM}]
    for turn in req.history:
        if len(turn) == 2:
            messages.append({"role": "user", "content": turn[0]})
            messages.append({"role": "assistant", "content": turn[1]})
    messages.append({"role": "user", "content": req.message})

    out = llm.create_chat_completion(
        messages=messages, max_tokens=320, temperature=0.3,
        stop=["<|im_end|>", "neutral"],
    )
    return {"answer": out["choices"][0]["message"]["content"].strip()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
