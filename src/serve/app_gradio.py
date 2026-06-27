"""
Browser chat demo for FinSage.

Backend priority:
  1. llama.cpp GGUF  — uses cfg.serve.gguf_path / FINSAGE_GGUF (CPU; works on Mac + HF Spaces)
  2. MLX             — native Apple Silicon, if a converted MLX model exists
  3. transformers    — full HF model, if available (e.g. on a GPU box)

Run:  python -m src.serve.app_gradio
"""
from __future__ import annotations

import os

import gradio as gr

from src.config_utils import abspath, load_config
from src.data.prompts import SYSTEM_PROMPT

cfg = load_config()
MAX_NEW = cfg.eval.max_new_tokens


def _to_messages(history, message):
    """Normalize Gradio history (tuples OR messages format) into chat messages."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history or []:
        if isinstance(h, dict):                      # gradio type="messages"
            if h.get("content"):
                msgs.append({"role": h["role"], "content": h["content"]})
        else:                                        # gradio type="tuples" [user, assistant]
            u, a = h
            if u:
                msgs.append({"role": "user", "content": u})
            if a:
                msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": message})
    return msgs


def _load_backend():
    """Return a callable respond(message, history) -> str."""
    # 1) GGUF via llama.cpp — what runs locally on the Mac (and on HF Spaces)
    gguf = os.getenv("FINSAGE_GGUF", abspath(cfg.serve.gguf_path))
    if os.path.exists(gguf):
        from llama_cpp import Llama
        llm = Llama(model_path=gguf, n_ctx=cfg.api.n_ctx,
                    n_threads=(cfg.api.n_threads or os.cpu_count() or 4), verbose=False)

        def respond(message, history):
            out = llm.create_chat_completion(
                messages=_to_messages(history, message),
                max_tokens=MAX_NEW, temperature=cfg.api.temperature)
            return out["choices"][0]["message"]["content"].strip()

        print(f"Backend: llama.cpp GGUF ({gguf})")
        return respond

    # 2) MLX (Apple Silicon)
    try:
        from mlx_lm import generate, load
        model, tokenizer = load(abspath(cfg.serve.mlx_model_path))

        def respond(message, history):
            prompt = tokenizer.apply_chat_template(
                _to_messages(history, message), tokenize=False, add_generation_prompt=True)
            return generate(model, tokenizer, prompt=prompt, max_tokens=MAX_NEW, verbose=False)

        print("Backend: MLX (Apple Silicon)")
        return respond
    except Exception as e:
        print(f"MLX unavailable ({e}); trying transformers.")

    # 3) transformers (full HF model, e.g. GPU box)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = abspath(cfg.serve.model_path)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto", device_map="auto")

    def respond(message, history):
        prompt = tok.apply_chat_template(
            _to_messages(history, message), tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    print("Backend: transformers")
    return respond


def main() -> None:
    respond = _load_backend()
    demo = gr.ChatInterface(
        fn=respond,
        type="messages",
        title="FinSage 📈 — Finance Reasoning Assistant",
        description="Fine-tuned Qwen2.5-3B (QLoRA + DPO). Ask about filings, valuation, "
                    "accounting, or financial reasoning.",
        examples=[
            "What is the difference between EBITDA and net income?",
            "A company has revenue of 500M and COGS of 300M. What is the gross margin?",
            "Explain what a 10-K filing contains.",
        ],
    )
    demo.launch()


if __name__ == "__main__":
    main()
