"""
Build DPO preference pairs (the "data flywheel" step).

Strategy: for each finance Q&A item,
    chosen   = the gold/reference answer (high quality)
    rejected = an answer sampled from the *base* model (typically weaker / less grounded)

This produces realistic preference contrast and demonstrates how teams bootstrap
preference data without expensive human labeling.

Output schema (trl DPOTrainer format):
    {"prompt": "<chat-templated prompt>", "chosen": "...", "rejected": "..."}

Requires a GPU for generation -> run on Colab/Kaggle.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

from src.config_utils import abspath, load_config
from src.data.prompts import build_messages, render_prompt
from src.quiet import quiet, tame_generation


def load_base_model(cfg):
    """Load the base model for generating 'rejected' answers (unsloth, 4-bit)."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        dtype=cfg.model.dtype,
    )
    FastLanguageModel.for_inference(model)
    tame_generation(model)
    return model, tokenizer


def main() -> None:
    quiet()
    cfg = load_config()
    random.seed(cfg.data.seed)
    n = cfg.data.max_dpo_rows

    print("· Loading source Q&A (virattt/financial-qa-10K) ...")
    ds = load_dataset("virattt/financial-qa-10K", split="train")
    rows = [r for r in ds if str(r.get("question", "")).strip()
            and str(r.get("answer", "")).strip()]
    random.shuffle(rows)
    rows = rows[:n]

    model, tokenizer = load_base_model(cfg)

    pairs: list[dict] = []
    for i, r in enumerate(rows):
        question = r["question"].strip()
        gold = r["answer"].strip()
        context = (r.get("context") or "").strip()
        user_input = f"Context from the 10-K filing:\n{context}" if context else ""

        prompt_text = render_prompt(tokenizer, question, user_input=user_input)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
        rejected = tokenizer.decode(
            gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # Skip degenerate cases where the base model already nails it
        if not rejected or rejected.lower() == gold.lower():
            continue

        pairs.append({"prompt": prompt_text, "chosen": gold, "rejected": rejected})
        if (i + 1) % 50 == 0:
            print(f"  generated {i + 1}/{len(rows)} (kept {len(pairs)})")

    out_path = Path(abspath(cfg.data.dpo_file))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\n✅ Wrote {len(pairs)} DPO pairs -> {out_path}")


if __name__ == "__main__":
    main()
