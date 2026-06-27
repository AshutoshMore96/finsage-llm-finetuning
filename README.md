# FinSage 📈 — Fine-Tuning an LLM into a Finance Reasoning Assistant

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
- Evaluated with a **debiased LLM-as-judge** (Groq Llama-3.3-70B) after identifying verbosity +
  position bias in the naïve judge.
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

> _Set the repos to **public** on HF for these links to work for others._

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
| LLM-judge win-rate (Groq Llama-3.3-70B) | 91% | 9%\* |

**ROUGE-L more than doubled (+102%).** The fine-tuned model produces concise, reference-aligned
answers; the base model is verbose and occasionally fabricates structure.

\* The *naïve* LLM-judge favored the base model — a textbook case of **verbosity + position
bias**: the base's long, bulleted answers read as "more helpful" despite lower factual alignment
with the gold answer. Measured against ground truth (ROUGE-L) and on inspection, the fine-tuned
model is clearly better. The judge is now **debiased** (randomized A/B order + a length-neutral,
correctness-focused rubric — see [`src/eval/evaluate.py`](src/eval/evaluate.py)). _Takeaway:
LLM-as-judge scores can't be trusted until debiased — a pitfall this project surfaces rather than
hides._

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

## 🧠 Why this project is portfolio-worthy

1. A real problem where the base model underperforms (numeric / financial reasoning, 10-K Q&A).
2. The complete modern stack: **PEFT (LoRA/QLoRA)**, **preference optimization (DPO)**,
   **quantization (GGUF/MLX)**, **serving (FastAPI/Docker)**.
3. **Rigorous before/after evaluation** + a **fine-tune vs. RAG vs. prompting** comparison.
4. **Critical analysis, not just numbers** — debiased the LLM judge, documented a training-data
   artifact (see [Known limitations](#️-known-limitations--honest-notes)).
5. A working **local demo on an 8 GB laptop**, with all artifacts on the HF Hub.

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
├── docs/                       # CONCEPTS, PIPELINE (analysis+timeline), DEPLOYMENT
├── tests/                      # GPU-free unit tests (pytest)  + .github/workflows/ci.yml
├── notebooks/colab_train.ipynb # one-click cloud training (auto-backs-up each stage)
├── Dockerfile, docker-compose.yml, Makefile, .env.example
└── requirements*.txt           # training (CUDA) / mac / serve
```

> 📚 **Deep dives:** [`docs/CONCEPTS.md`](docs/CONCEPTS.md) (every technique explained) ·
> [`docs/PIPELINE.md`](docs/PIPELINE.md) (working analysis + timeline) ·
> [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) (cloud hosting).

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

## 🚀 Quickstart

### A) Train on a cloud GPU (free T4 / Kaggle / Vast)
```bash
pip install -r requirements.txt
python -m src.data.prepare_sft_data
python -m src.train.train_sft_qlora        # QLoRA SFT
python -m src.data.prepare_dpo_data        # build preference pairs (resumable)
python -m src.train.train_dpo              # DPO alignment
python -m src.merge.merge_and_quantize     # → 16-bit + GGUF
```
Or use [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) (one-click, auto-backs-up
each stage to Google Drive). Push artifacts to the Hub with `src/backup_cloud.py`.

### B) Run locally on the Mac (GGUF — recommended for 8 GB)
```bash
hf download AshutoshMore96/finsage-gguf --include "*.gguf" --local-dir ./models/gguf
printf 'FROM ./models/gguf/Qwen2.5-3B-Instruct.Q4_K_M.gguf\n' > Modelfile
ollama create finsage -f Modelfile
ollama run finsage "What is the difference between EBITDA and net income?"
```
Or via the FastAPI server: `pip install -r requirements-serve.txt && make api`.

MLX (native Apple Silicon) alternative, from the **merged** repo:
```bash
mlx_lm.convert --hf-path AshutoshMore96/finsage-merged -q --mlx-path models/finsage-mlx-4bit
python src/serve/mac_inference_mlx.py
```

### C) Evaluate (before vs after)
```bash
echo "GROQ_API_KEY=gsk_..." > .env       # free key from console.groq.com (optional judge)
python -m src.eval.evaluate              # → outputs/eval_report.md
```

### D) Fine-tune vs. RAG vs. prompt-engineering
```bash
python -m src.compare.rag_pipeline --build
python -m src.compare.benchmark          # → outputs/comparison_report.md + chart
```

### E) Serve as a cloud API
```bash
make api                                 # FastAPI on :8000 (CPU GGUF, runs anywhere)
docker compose up --build                # or containerized
```
Hosting guide (HF Spaces / Render / Fly.io / VM / vLLM-GPU): [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## ⚠️ Known limitations & honest notes

- **Trailing sentiment tokens.** The model occasionally appends a label like `neutral` —
  an artifact of the Sujet-Finance dataset mixing **sentiment-classification** tasks into the
  SFT data. Mitigated at serving time with a system prompt + stop tokens; the proper fix is to
  filter sentiment-only rows from `prepare_sft_data.py` and retrain.
- **LLM-judge bias.** The naïve judge showed verbosity + position bias (see Results). The judge
  is now debiased, but LLM-as-judge should always be treated with caution.
- **3B scale.** Small model for laptop deployment — expect occasional factual slips on hard
  numeric reasoning. See scaling note below.

---

## 🔧 Scaling up
Set `model.name` in `config/config.yaml` to `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` for stronger
results (still fits a free T4 via unsloth). The 7B Q4_K_M GGUF (~4.5 GB) is tight but runnable on
an 8 GB Mac via Ollama with a small context window.

## 📜 License
Code: MIT. Datasets and base models retain their original licenses (check each link).
