"""
Stage 4 — Evaluate base vs. fine-tuned on held-out finance Q&A.

Metrics:
  * ROUGE-L (lexical overlap with the gold answer)
  * (optional) LLM-as-judge win-rate via Groq (free; reads GROQ_API_KEY from .env)

Writes a markdown report to outputs/eval_report.md.

Run on GPU (fast) or Mac (slow):  python -m src.eval.evaluate
"""
from __future__ import annotations

import os
import random
from pathlib import Path

from datasets import load_dataset

from src.config_utils import abspath, load_config
from src.data.prompts import render_prompt
from src.quiet import quiet, tame_generation


def _load_env_file() -> None:
    """Load KEY=VALUE lines from a repo-root .env into os.environ (no dependency).
    Makes GROQ_API_KEY available no matter how the script is launched."""
    for base in (Path.cwd(), Path(__file__).resolve().parents[2]):
        env_path = base / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_env_file()


def load_model(model_name_or_path, cfg):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        dtype=cfg.model.dtype,
    )
    FastLanguageModel.for_inference(model)
    tame_generation(model)
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


def judge_ready(cfg) -> bool:
    """True only if the judge is enabled AND GROQ_API_KEY is set."""
    return bool(cfg.eval.use_llm_judge and os.getenv("GROQ_API_KEY"))


JUDGE_RUBRIC = (
    "You are grading two candidate answers to a finance question against a REFERENCE "
    "answer.\n"
    "Judge ONLY factual correctness and alignment with the reference. "
    "IGNORE length and writing style — a longer, more elaborate, or bulleted answer is "
    "NOT better. A concise answer that matches the reference facts is best; an answer that "
    "adds unsupported claims is worse.\n\n"
    "Question:\n{question}\n\nReference answer:\n{ref}\n\n"
    "Answer A:\n{a}\n\nAnswer B:\n{b}\n\n"
    "Which answer is more factually correct and closer to the reference? "
    "Reply with exactly one token: A, B, or tie."
)


def llm_judge(cfg, question, ref, base_ans, ft_ans, rng) -> str:
    """Debiased LLM-as-judge via Groq. Returns 'base', 'ft', or 'tie'.

    Mitigates the two classic judge biases:
      * position bias — randomizes which answer is shown as A vs B
      * verbosity bias — rubric tells the judge to ignore length/style
    """
    from groq import Groq

    client = Groq()  # reads GROQ_API_KEY from the environment

    a_is_ft = rng.random() < 0.5            # randomize order
    a_text, b_text = (ft_ans, base_ans) if a_is_ft else (base_ans, ft_ans)
    prompt = JUDGE_RUBRIC.format(question=question, ref=ref, a=a_text, b=b_text)

    resp = client.chat.completions.create(
        model=cfg.eval.judge_model, max_tokens=5,
        messages=[{"role": "user", "content": prompt}],
    )
    v = resp.choices[0].message.content.strip().upper()

    has_a, has_b = "A" in v, "B" in v
    if "TIE" in v or has_a == has_b:        # tie, or ambiguous -> tie
        return "tie"
    chose_a = has_a and not has_b
    # map the chosen slot back to which model it was
    if chose_a:
        return "ft" if a_is_ft else "base"
    return "base" if a_is_ft else "ft"


def main() -> None:
    quiet()
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
    judge_on = judge_ready(cfg)
    judge_rng = random.Random(cfg.data.seed)   # reproducible A/B shuffling
    if judge_on:
        print(f"· LLM-judge ON (Groq: {cfg.eval.judge_model}, debiased)")
    elif cfg.eval.use_llm_judge:
        print("! LLM-judge enabled but GROQ_API_KEY not found — skipping judge.\n"
              "  Put 'GROQ_API_KEY=gsk_...' in a .env file at the repo root, "
              "or export it, then re-run.")

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

        if judge_on:
            try:
                wins[llm_judge(cfg, q, gold, base_ans, ft_ans, judge_rng)] += 1
            except Exception as e:
                # e.g. Groq 429 rate limit — stop judging but keep ROUGE + write report
                print(f"! judge stopped at example {i} "
                      f"({type(e).__name__}: {str(e)[:100]}) — continuing with ROUGE only "
                      f"({sum(wins.values())} judged so far).")
                judge_on = False

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
