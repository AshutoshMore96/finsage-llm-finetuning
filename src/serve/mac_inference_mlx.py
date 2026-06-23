"""
Local inference on Apple Silicon (M2) using MLX — fast and memory-light.

Prereqs (on the Mac):
    pip install mlx-lm
    # convert the merged HF model to a quantized MLX model:
    python -m mlx_lm.convert --hf-path models/finsage-merged-16bit -q \
        --mlx-path models/finsage-mlx-4bit

Then:  python src/serve/mac_inference_mlx.py
"""
from __future__ import annotations

from src.config_utils import abspath, load_config
from src.data.prompts import SYSTEM_PROMPT


def main() -> None:
    cfg = load_config()
    from mlx_lm import generate, load

    model_path = abspath(cfg.serve.mlx_model_path)
    print(f"Loading MLX model from {model_path} ...")
    model, tokenizer = load(model_path)

    print("FinSage (MLX) ready. Type a finance question, or 'quit' to exit.\n")
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"quit", "exit"}:
            break
        if not user:
            continue
        history.append({"role": "user", "content": user})
        prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        resp = generate(
            model, tokenizer, prompt=prompt, max_tokens=cfg.eval.max_new_tokens,
            verbose=False,
        )
        print(f"\nFinSage> {resp}\n")
        history.append({"role": "assistant", "content": resp})


if __name__ == "__main__":
    main()
