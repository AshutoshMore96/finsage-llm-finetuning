"""
Stage 4 — Evaluate base vs. fine-tuned on held-out finance Q&A.

Metrics:
  * ROUGE-L (lexical overlap with the gold answer)
  * (optional) LLM-as-judge win-rate via the Anthropic API

Writes a markdown report to outputs/eval_report.md.

Run on GPU (fast) or Mac (slow):  python -m src.eval.evaluate
"""
from __future__ import annotations

import os
from pathlib import Path

from datasets import load_dataset

from src.config_utils import abspath, load_config
from src.data.prompts import render_prompt


def load_model(model_name_or_path, cfg):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        dtype=cfg.model.dtype,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate(model, tokenizer, question, user_input, max_new_tokens):
    prompt = render_prompt(tokenizer, question, user_input=user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def rouge_l(pred: str, ref: str) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(ref, pred)["rougeL"].fmeasure


def llm_judge(cfg, question, ref, ans_a, ans_b) -> str:
    """Return 'A', 'B', or 'tie'. A = base, B = fine-tuned."""
    import anthropic
    client = anthropic.Anthropic()
    prompt = (
        f"Question:\n{question}\n\nReference answer:\n{ref}\n\n"
        f"Assistant A:\n{ans_a}\n\nAssistant B:\n{ans_b}\n\n"
        "Which assistant answer is more accurate and helpful for this finance question? "
        "Reply with exactly one token: A, B, or tie."
    )
    msg = client.messages.create(
        model=cfg.eval.judge_model, max_tokens=5,
        messages=[{"role": "user", "content": prompt}],
    )
    verdict = msg.content[0].text.strip().upper()
    return "A" if "A" in verdict else "B" if "B" in verdict else "tie"


def main() -> None:
    cfg = load_config()
    n = cfg.data.max_eval_rows

    # Held-out slice (take from the end to avoid overlap with the head used in training)
    ds = load_dataset("virattt/financial-qa-10K", split="train")
    rows = [r for r in ds][-n:]

    base_model, base_tok = load_model(cfg.model.name, cfg)
    ft_model, ft_tok = load_model(abspath(cfg.merge.merged_dir), cfg)

    base_rouge, ft_rouge = [], []
    wins = {"base": 0, "ft": 0, "tie": 0}
    rows_md = []

    for i, r in enumerate(rows):
        q = str(r.get("question", "")).strip()
        gold = str(r.get("answer", "")).strip()
        ctx = str(r.get("context", "") or "").strip()
        if not q or not gold:
            continue
        ui = f"Context from the 10-K filing:\n{ctx}" if ctx else ""

        base_ans = generate(base_model, base_tok, q, ui, cfg.eval.max_new_tokens)
        ft_ans = generate(ft_model, ft_tok, q, ui, cfg.eval.max_new_tokens)

        base_rouge.append(rouge_l(base_ans, gold))
        ft_rouge.append(rouge_l(ft_ans, gold))

        if cfg.eval.use_llm_judge and os.getenv("ANTHROPIC_API_KEY"):
            v = llm_judge(cfg, q, gold, base_ans, ft_ans)
            wins["base" if v == "A" else "ft" if v == "B" else "tie"] += 1

        if i < 5:  # keep a few qualitative examples in the report
            rows_md.append(
                f"### Example {i+1}\n**Q:** {q}\n\n**Gold:** {gold}\n\n"
                f"**Base:** {base_ans}\n\n**Fine-tuned:** {ft_ans}\n"
            )

    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    report = [
        "# FinSage — Evaluation Report\n",
        f"Held-out examples: **{len(base_rouge)}**\n",
        "| Metric | Base | Fine-tuned (SFT+DPO) |",
        "|--------|------|----------------------|",
        f"| ROUGE-L | {avg(base_rouge):.4f} | {avg(ft_rouge):.4f} |",
    ]
    if cfg.eval.use_llm_judge:
        total = sum(wins.values()) or 1
        report.append(
            f"| LLM-judge win-rate | {wins['base']/total:.2%} | {wins['ft']/total:.2%} "
            f"(tie {wins['tie']/total:.2%}) |"
        )
    report.append("\n## Qualitative examples\n")
    report += rows_md

    out_path = Path(abspath(cfg.eval.output_report))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report))
    print(f"✅ Wrote eval report -> {out_path}")
    print(f"ROUGE-L  base={avg(base_rouge):.4f}  finetuned={avg(ft_rouge):.4f}")


if __name__ == "__main__":
    main()
