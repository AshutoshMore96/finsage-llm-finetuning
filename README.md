# FinSage 📈 — Fine-Tuning an LLM into a Finance Reasoning Assistant

![FinSage running locally on an M2](finsage_demo.gif)

![Python](https://img.shields.io/badge/python-3.11-blue)
![Base model](https://img.shields.io/badge/base-Qwen2.5--3B--Instruct-orange)
![Techniques](https://img.shields.io/badge/QLoRA%20%7C%20DPO%20%7C%20GGUF-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Turn a small open LLM (**Qwen2.5-3B-Instruct**) into a specialized financial-analysis
assistant using the **full modern fine-tuning stack** — then run it on a laptop:

**QLoRA (SFT) → DPO (preference alignment) → merge → GGUF quantization → local inference.**

> **The story:** trained on a single cloud GPU (a free Colab T4 is enough; this run used an
> A100 on Vast.ai), quantized to a 1.8 GB GGUF, and deployed to run **locally on an 8 GB Apple
> M2 MacBook** — mirroring how real teams train in the cloud and deploy on the edge.

---

## 🎯 TL;DR

- **Fine-tuned** Qwen2.5-3B on SEC 10-K Q&A + finance instructions with **QLoRA**, then aligned
  with **DPO** using a self-generated preference dataset (the RLHF "data flywheel").
- **ROUGE-L doubled: 0.328 → 0.663** on a 300-example held-out set.
- Evaluated with a **debiased LLM-as-judge** (Groq): identifying + correcting verbosity/position
  bias swung the judged win-rate from a misleading 91%–9% (base) to a true ~tie (49%–46%, fine-tuned).
- Quantized to **GGUF (Q4_K_M, 1.8 GB)** → runs locally via Ollama / llama.cpp on an 8 GB Mac.
- Includes a **fine-tune vs. RAG vs. prompt-engineering** benchmark and a **FastAPI** serving
  layer (Docker-ready).

---

## 🔗 Models & artifacts (Hugging Face)

| Artifact | Repo | Use |
|----------|------|-----|
| GGUF (Q4_K_M, 1.8 GB) | [`AshutoshMore96/finsage-gguf`](https://huggingface.co/AshutoshMore96/finsage-gguf) | Ollama / llama.cpp (laptop) |
| Merged 16-bit model | [`AshutoshMore96/finsage-merged`](https://huggingface.co/AshutoshMore96/finsage-merged) | MLX / vLLM / transformers |
| SFT LoRA adapter | [`AshutoshMore96/finsage-sft-adapter`](https://huggingface.co/AshutoshMore96/finsage-sft-adapter) | resume / inspect |
| DPO LoRA adapter | [`AshutoshMore96/finsage-dpo-adapter`](https://huggingface.co/AshutoshMore96/finsage-dpo-adapter) | final aligned adapter |

### Try it in 3 commands (no training, no GPU)
```bash
hf download AshutoshMore96/finsage-gguf --include "*.gguf" --local-dir ./models/gguf
printf 'FROM ./models/gguf/Qwen2.5-3B-Instruct.Q4_K_M.gguf\n' > Modelfile
ollama create finsage -f Modelfile && ollama run finsage "What is EBITDA?"
```

---

## 📊 Results

Evaluated on **300 held-out** finance Q&A examples (never seen during training).

| Metric | Base Qwen2.5-3B | Fine-tuned (QLoRA SFT + DPO) |
|--------|----------------:|-----------------------------:|
| **ROUGE-L** (vs. gold answer) | 0.328 | **0.663** |
| LLM-judge win-rate — *naïve* | 91% | 9% |
| LLM-judge win-rate — **debiased** | 46% | **49%** (tie 5%) |

**ROUGE-L more than doubled (+102%).** The fine-tuned model produces concise, reference-aligned
answers; the base model is verbose and occasionally fabricates structure.

**The LLM-judge story is the interesting part.** A *naïve* judge favored the base model 91%–9% —
a textbook case of **verbosity + position bias** (the base's long, bulleted answers read as "more
helpful," and it was always shown first). After **debiasing the judge** (randomized A/B order +
a length-neutral, correctness-focused rubric),
the result collapses to a near-tie (**49% vs 46%**, fine-tuned slightly ahead). That ~40-point
swing shows the original "base win" was **almost entirely an artifact, not quality**.

_Honest interpretation:_ both models are similarly *factually* correct (the base already has the
facts from the provided 10-K context), so a length-neutral judge rates them close. Fine-tuning's
real, measurable win is **concise, reference-aligned output** — exactly what ROUGE-L captures and
what SFT+DPO targets. _Takeaway: LLM-as-judge scores can't be trusted until debiased — a pitfall
this project surfaces rather than hides._ (Judge: Groq `llama-3.1-8b-instant`, 300 held-out examples.)

### Example (held-out)
> **Q:** What % of the Firm's 2023 employment opportunities were filled by external candidates?
>
> **Gold:** Approximately 60%.
>
> **Base:** *"To find the percentage… Given: 60%… The remaining 40%… Final Answer: 60%"* — correct but verbose.
>
> **Fine-tuned:** *"Approximately 60% of the Firm's employment opportunities in 2023 were filled by external candidates."* ✅ concise, matches the reference.

---

## 🏗️ Architecture

```
  Finance data                Stage 1: SFT          Stage 2: alignment        Deploy
  ────────────                ────────────          ──────────────────        ──────
  Sujet-Finance-177k  ┐
  financial-qa-10K    ├──►  Qwen2.5-3B (4-bit)  ──►  + SFT LoRA  ──►  + DPO LoRA  ──►  merge ─┐
  FinQA               ┘        [ QLoRA ]            [ trl SFT ]      [ trl DPO ]            │
                                                         ▲                                  ▼
                          self-generated preference pairs ┘                    ┌── GGUF Q4_K_M (1.8GB) ─► Ollama / llama.cpp ─► 8GB M2
  (chosen = gold answer, rejected = base-model answer)                         └── merged 16-bit ────────► MLX / vLLM / HF Hub
```

Full working analysis + timeline: [`docs/PIPELINE.md`](docs/PIPELINE.md).

---

## 🖥️ Hardware & where each step runs

| Step | Where | Why |
|------|-------|-----|
| 1. Data prep | Mac **or** cloud | CPU only, light |
| 2. SFT (QLoRA) | **Cloud GPU** (free T4 works) | `bitsandbytes` 4-bit is CUDA-only — can't run on Mac MPS |
| 3. DPO alignment | **Cloud GPU** | same as above |
| 4. Merge + quantize | Cloud GPU | produces 16-bit + GGUF |
| 5. Inference + demo | **Mac M2 (local)** | GGUF Q4_K_M ≈ 1.8 GB, runs in 8 GB RAM |
| 6. Evaluation | Cloud GPU (fast) or Mac | |

> ❗ **An 8 GB M2 cannot *train* this** — `bitsandbytes` QLoRA needs an NVIDIA GPU. Train on a
> cloud GPU (free Colab/Kaggle T4, or a cheap Vast.ai box), then run inference locally.

---

## 📂 Repo structure

```
Finetuning_LLM/
├── config/config.yaml          # central config (model, datasets, hyperparams)
├── src/
│   ├── data/                   # prompts, SFT data prep, DPO preference-pair builder
│   ├── train/                  # train_sft_qlora.py, train_dpo.py  (unsloth + trl)
│   ├── merge/merge_and_quantize.py   # merge LoRA → 16-bit + GGUF
│   ├── eval/evaluate.py        # base vs fine-tuned: ROUGE-L + debiased Groq judge
│   ├── compare/                # ★ fine-tune vs RAG vs prompt-engineering benchmark
│   ├── serve/                  # FastAPI (api.py), Gradio, MLX local inference
│   └── backup_cloud.py         # push artifacts to HF Hub / GCS (Lightning/Vast-friendly)
├── docs/PIPELINE.md            # working analysis + timeline
├── tests/                      # GPU-free unit tests (pytest)  + .github/workflows/ci.yml
├── notebooks/colab_train.ipynb # one-click cloud training (auto-backs-up each stage)
├── Dockerfile, docker-compose.yml, Makefile, .env.example
└── requirements*.txt           # training (CUDA) / mac / serve
```

---

## 📊 Datasets (all free)

**SFT (instruction tuning):**
- [`sujet-ai/Sujet-Finance-Instruct-177k`](https://huggingface.co/datasets/sujet-ai/Sujet-Finance-Instruct-177k)
- [`virattt/financial-qa-10K`](https://huggingface.co/datasets/virattt/financial-qa-10K) — Q&A over SEC 10-K filings
- **FinQA** (numerical reasoning) — official raw JSON from [czyssrs/FinQA](https://github.com/czyssrs/FinQA)
  (the HF mirrors use legacy loading scripts that `datasets>=4` rejects)

**DPO (preference pairs):** generated by `prepare_dpo_data.py` — `chosen` = gold answer,
`rejected` = base-model answer (the data flywheel).

**Evaluation:** held-out split of `virattt/financial-qa-10K`; [`PatronusAI/financebench`](https://huggingface.co/datasets/PatronusAI/financebench) for extra benchmarking.

---

## 📜 License
Code: MIT. Datasets and base models retain their original licenses (check each link).
