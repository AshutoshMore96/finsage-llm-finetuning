"""
Fine-tune vs. RAG vs. prompt-engineering — the senior-level comparison.

Runs up to 5 approaches over a held-out finance Q&A set and reports accuracy
(ROUGE-L, token-F1), latency and a "when to use which" verdict:

  1. base_zeroshot     — base model, plain prompt            (lower bound)
  2. prompt_engineered — base model + few-shot + CoT         (no training)
  3. rag               — base model + retrieved 10-K context (no training)
  4. finetuned         — QLoRA+DPO model, plain prompt       (your model)
  5. finetuned_rag     — fine-tuned model + retrieval        (best of both)

Run on GPU:
    python -m src.compare.rag_pipeline --build      # once
    python -m src.compare.benchmark
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from src.compare import prompt_baseline
from src.compare.llm_backend import LLM
from src.compare.rag_pipeline import RagIndex
from src.config_utils import abspath, load_config


def rouge_l(pred: str, ref: str) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(ref, pred)["rougeL"].fmeasure


def token_f1(pred: str, ref: str) -> float:
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r:
        return 0.0
    common = 0
    rc = defaultdict(int)
    for t in r:
        rc[t] += 1
    for t in p:
        if rc[t] > 0:
            common += 1
            rc[t] -= 1
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def main() -> None:
    cfg = load_config()
    approaches = list(cfg.compare.approaches)
    n = cfg.compare.max_rows

    ds = load_dataset("virattt/financial-qa-10K", split="train")
    rows = [r for r in ds][-n:]

    # Lazy-load only what the selected approaches need
    base_llm = ft_llm = rag = None
    if {"base_zeroshot", "prompt_engineered", "rag", "finetuned_rag"} & set(approaches):
        print("Loading base model ...")
        base_llm = LLM(cfg.model.name)
    if {"finetuned", "finetuned_rag"} & set(approaches):
        print("Loading fine-tuned model ...")
        ft_llm = LLM(abspath(cfg.merge.merged_dir))
    if {"rag", "finetuned_rag"} & set(approaches):
        print("Loading RAG index ...")
        rag = RagIndex().load()

    mn = cfg.eval.max_new_tokens
    metrics = {a: {"rougeL": [], "f1": [], "latency": []} for a in approaches}
    samples = []  # qualitative

    for i, r in enumerate(rows):
        q = str(r.get("question", "")).strip()
        gold = str(r.get("answer", "")).strip()
        ctx = str(r.get("context", "") or "").strip()
        if not q or not gold:
            continue
        ui = f"Context from the 10-K filing:\n{ctx}" if ctx else ""
        per_q = {"question": q, "gold": gold}

        for a in approaches:
            if a == "base_zeroshot":
                res = base_llm.ask(q, ui, max_new_tokens=mn)
            elif a == "prompt_engineered":
                res = base_llm.chat(prompt_baseline.build_prompt(q, ui), max_new_tokens=mn)
            elif a == "rag":
                res = base_llm.chat(rag.build_prompt(q), max_new_tokens=mn)
            elif a == "finetuned":
                res = ft_llm.ask(q, ui, max_new_tokens=mn)
            elif a == "finetuned_rag":
                res = ft_llm.chat(rag.build_prompt(q), max_new_tokens=mn)
            else:
                continue
            metrics[a]["rougeL"].append(rouge_l(res.text, gold))
            metrics[a]["f1"].append(token_f1(res.text, gold))
            metrics[a]["latency"].append(res.latency_s)
            per_q[a] = res.text

        if i < 4:
            samples.append(per_q)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} done")

    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    write_report(cfg, approaches, metrics, samples, avg)
    try:
        write_chart(cfg, approaches, metrics, avg)
    except Exception as e:
        print(f"! chart skipped ({e})")


def write_report(cfg, approaches, metrics, samples, avg):
    nice = {
        "base_zeroshot": "Base (zero-shot)",
        "prompt_engineered": "Prompt-engineered",
        "rag": "RAG",
        "finetuned": "Fine-tuned (SFT+DPO)",
        "finetuned_rag": "Fine-tuned + RAG",
    }
    lines = [
        "# Fine-tune vs. RAG vs. Prompt-engineering — Benchmark\n",
        f"Held-out finance Q&A examples: **{len(metrics[approaches[0]]['rougeL'])}**\n",
        "| Approach | ROUGE-L | Token-F1 | Avg latency (s) | Needs training? | Needs corpus? |",
        "|----------|--------:|---------:|----------------:|:---------------:|:-------------:|",
    ]
    flags = {
        "base_zeroshot": ("No", "No"),
        "prompt_engineered": ("No", "No"),
        "rag": ("No", "Yes"),
        "finetuned": ("Yes", "No"),
        "finetuned_rag": ("Yes", "Yes"),
    }
    for a in approaches:
        tr, co = flags.get(a, ("?", "?"))
        lines.append(
            f"| {nice.get(a, a)} | {avg(metrics[a]['rougeL']):.4f} | "
            f"{avg(metrics[a]['f1']):.4f} | {avg(metrics[a]['latency']):.2f} | {tr} | {co} |"
        )

    lines += [
        "\n## When to use which (the senior takeaways)\n",
        "- **Prompt-engineering** — fastest to ship, zero training cost. Best when the base "
        "model already has the knowledge and you only need to shape format/behavior.",
        "- **RAG** — best when answers depend on *fresh or proprietary documents* (new "
        "filings, internal data). Knowledge updates by re-indexing, no retraining. Adds "
        "retrieval latency and depends on retrieval quality.",
        "- **Fine-tuning (QLoRA+DPO)** — best for *durable domain style, formatting, and "
        "reasoning patterns*; lowest per-query latency (no retrieval) and cheapest at scale. "
        "Costs an upfront training run and must be retrained to absorb new facts.",
        "- **Fine-tuned + RAG** — usually the strongest: the model speaks the domain *and* is "
        "grounded in retrieved evidence. This is the common production pattern.",
        "\n## Qualitative samples\n",
    ]
    for s in samples:
        lines.append(f"**Q:** {s['question']}\n\n**Gold:** {s['gold']}\n")
        for a in approaches:
            if a in s:
                lines.append(f"- *{nice.get(a, a)}:* {s[a]}")
        lines.append("")

    out = Path(abspath(cfg.compare.output_report))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"✅ Wrote comparison report -> {out}")


def write_chart(cfg, approaches, metrics, avg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [a.replace("_", "\n") for a in approaches]
    rouge = [avg(metrics[a]["rougeL"]) for a in approaches]
    lat = [avg(metrics[a]["latency"]) for a in approaches]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = range(len(approaches))
    ax1.bar([i - 0.2 for i in x], rouge, width=0.4, label="ROUGE-L", color="#2b8a3e")
    ax1.set_ylabel("ROUGE-L")
    ax2 = ax1.twinx()
    ax2.bar([i + 0.2 for i in x], lat, width=0.4, label="Latency (s)", color="#d9480f")
    ax2.set_ylabel("Avg latency (s)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_title("FinSage — Accuracy vs. Latency by approach")
    fig.tight_layout()
    path = abspath(cfg.compare.chart_path)
    fig.savefig(path, dpi=150)
    print(f"✅ Wrote chart -> {path}")


if __name__ == "__main__":
    main()
