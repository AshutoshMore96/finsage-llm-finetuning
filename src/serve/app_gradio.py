"""
Browser chat demo for FinSage.

Prefers MLX on Apple Silicon (loads the quantized 4-bit MLX model -> fits 8GB M2).
Falls back to transformers on CUDA/CPU (e.g. for a Colab demo).

    python src/serve/app_gradio.py
"""
from __future__ import annotations

import gradio as gr

from src.config_utils import abspath, load_config
from src.data.prompts import SYSTEM_PROMPT

cfg = load_config()
MAX_NEW = cfg.eval.max_new_tokens


def _load_backend():
    """Return a callable respond(message, history) -> str."""
    # Try MLX (Apple Silicon) first
    try:
        from mlx_lm import generate, load
        model, tokenizer = load(abspath(cfg.serve.mlx_model_path))

        def respond(message, history):
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
            for u, a in history:
                msgs += [{"role": "user", "content": u},
                         {"role": "assistant", "content": a}]
            msgs.append({"role": "user", "content": message})
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            return generate(model, tokenizer, prompt=prompt,
                            max_tokens=MAX_NEW, verbose=False)

        print("Backend: MLX (Apple Silicon)")
        return respond
    except Exception as e:
        print(f"MLX unavailable ({e}); falling back to transformers.")

    # Fallback: transformers
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = abspath(cfg.serve.model_path)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype="auto", device_map="auto")

    def respond(message, history):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for u, a in history:
            msgs += [{"role": "user", "content": u},
                     {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": message})
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()

    print("Backend: transformers")
    return respond


def main() -> None:
    respond = _load_backend()
    demo = gr.ChatInterface(
        fn=respond,
        title="FinSage 📈 — Finance Reasoning Assistant",
        description="Fine-tuned Qwen2.5-3B (QLoRA + DPO). Ask about filings, valuation, "
                    "accounting, or financial reasoning.",
        examples=[
            "What is the difference between EBITDA and net income?",
            "A company has revenue of $500M and COGS of $300M. What is the gross margin?",
            "Explain what a 10-K filing contains.",
        ],
    )
    demo.launch()


if __name__ == "__main__":
    main()
