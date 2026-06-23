"""
Download and normalize finance instruction datasets into a single JSONL of chat messages.

Output schema (one JSON object per line):
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Sources (all free on the Hugging Face Hub):
    - sujet-ai/Sujet-Finance-Instruct-177k   (broad finance instructions)
    - virattt/financial-qa-10K               (Q&A grounded in SEC 10-K filings)
    - dreamerdeo/finqa                        (numeric reasoning over tables)

Runs CPU-only — safe on the Mac.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

from src.config_utils import abspath, load_config
from src.data.prompts import build_messages


def _first(row: dict, *keys: str) -> str | None:
    """Return the first non-empty value among the given keys."""
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def load_sujet(limit: int) -> list[dict]:
    print("· Loading sujet-ai/Sujet-Finance-Instruct-177k ...")
    ds = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k", split="train")
    out = []
    for row in ds:
        instruction = _first(row, "user_prompt", "instruction", "question", "inputs")
        answer = _first(row, "answer", "output", "response")
        if not instruction or not answer:
            continue
        out.append(build_messages(instruction, answer))
        if len(out) >= limit:
            break
    print(f"  -> {len(out)} examples")
    return out


def load_financial_qa_10k(limit: int) -> list[dict]:
    print("· Loading virattt/financial-qa-10K ...")
    ds = load_dataset("virattt/financial-qa-10K", split="train")
    out = []
    for row in ds:
        question = _first(row, "question")
        answer = _first(row, "answer")
        context = _first(row, "context") or ""
        if not question or not answer:
            continue
        user_input = f"Context from the 10-K filing:\n{context}" if context else ""
        out.append(build_messages(question, answer, user_input=user_input))
        if len(out) >= limit:
            break
    print(f"  -> {len(out)} examples")
    return out


# Official FinQA release (script-free raw JSON — the HF mirrors use legacy loading
# scripts that datasets>=4 no longer supports).
FINQA_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json"


def load_finqa(limit: int) -> list[dict]:
    print("· Loading FinQA (official raw JSON) ...")
    try:
        ds = load_dataset("json", data_files=FINQA_URL, split="train")
    except Exception as e:
        print(f"  ! skipped finqa ({e})")
        return []
    out = []
    for row in ds:
        qa = row.get("qa") or {}
        question = _first(qa, "question") or _first(row, "question")
        answer = _first(qa, "answer", "exe_ans") or _first(row, "answer")
        pre = _first(row, "pre_text") or ""
        post = _first(row, "post_text") or ""
        table = row.get("table")
        if not question or not answer:
            continue
        ctx_parts = [p for p in (pre, str(table) if table else "", post) if p]
        user_input = "Financial context:\n" + "\n".join(ctx_parts) if ctx_parts else ""
        out.append(build_messages(question, answer, user_input=user_input))
        if len(out) >= limit:
            break
    print(f"  -> {len(out)} examples")
    return out


def main() -> None:
    cfg = load_config()
    random.seed(cfg.data.seed)

    cap = cfg.data.max_sft_rows
    # split the cap roughly across sources
    examples: list[dict] = []
    examples += load_sujet(int(cap * 0.6))
    examples += load_financial_qa_10k(int(cap * 0.25))
    examples += load_finqa(int(cap * 0.15))

    random.shuffle(examples)
    examples = examples[:cap]

    out_path = Path(abspath(cfg.data.sft_file))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            # ex is a list of chat messages; wrap so the column is named "messages"
            f.write(json.dumps({"messages": ex}) + "\n")

    print(f"\n✅ Wrote {len(examples)} SFT examples -> {out_path}")


if __name__ == "__main__":
    main()
