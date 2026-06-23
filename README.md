# FinSage 📈 — Fine-Tuning an LLM into a Finance Reasoning Assistant

End-to-end project that turns a small open LLM (**Qwen2.5-3B-Instruct**) into a specialized
financial-analysis assistant using the **full modern fine-tuning stack**:

**QLoRA (SFT) → DPO (preference alignment) → merge + GGUF quantization → local inference + eval.**

> **Story:** Trained for free on a Colab/Kaggle **T4 GPU**, quantized, and deployed to run
> **locally on an 8GB Apple M2 MacBook**. This mirrors how real teams train on the cloud and
> deploy on the edge.

---

## 🧠 Why this project is portfolio-worthy

1. A real problem where the base model underperforms (numeric/financial reasoning, 10-K Q&A).
2. The complete pipeline: **PEFT (LoRA/QLoRA)**, **preference optimization (DPO)**, **quantization (GGUF/MLX)**.
3. **Rigorous before/after evaluation** + an optional **fine-tune vs. RAG vs. prompting** comparison.
4. A working **local demo on an 8GB laptop** + model card pushed to the Hugging Face Hub.

---

## 🖥️ Hardware & where each step runs

| Step | Where | Why |
|------|-------|-----|
| 1. Data prep | Mac **or** Colab | CPU only, light |
| 2. SFT (QLoRA) | **Colab/Kaggle T4 (free)** | `bitsandbytes` 4-bit is CUDA-only — cannot run on Mac MPS |
| 3. DPO alignment | **Colab/Kaggle T4 (free)** | same as above |
| 4. Merge + quantize | Colab (merge) → Mac (GGUF/MLX) | |
| 5. Inference + Gradio demo | **Mac M2 (local)** | 3B 4-bit ≈ 2GB, runs in 8GB RAM |
| 6. Evaluation | Colab (fast) or Mac (slow) | |

> ❗ **Your 8GB M2 cannot train this** — `bitsandbytes` QLoRA needs an NVIDIA GPU. Use the free
> Colab notebook (`notebooks/colab_train.ipynb` instructions in this README) for steps 2–3,
> then come back to the Mac for steps 4–6.

---

## 📂 Repo structure

```
Finetuning_LLM/
├── config/config.yaml          # central config (model, datasets, hyperparams)
├── src/
│   ├── data/
│   │   ├── prompts.py          # system prompt + chat formatting
│   │   ├── prepare_sft_data.py # download + format finance SFT datasets -> JSONL
│   │   └── prepare_dpo_data.py # build (chosen/rejected) preference pairs
│   ├── train/
│   │   ├── train_sft_qlora.py  # QLoRA supervised fine-tuning (unsloth + trl)
│   │   └── train_dpo.py        # DPO preference alignment
│   ├── merge/merge_and_quantize.py  # merge LoRA, push to Hub, GGUF/MLX instructions
│   ├── eval/evaluate.py        # base vs fine-tuned, ROUGE + optional LLM-judge
│   ├── compare/               # ★ fine-tune vs RAG vs prompt-engineering
│   │   ├── llm_backend.py     #   shared generation wrapper
│   │   ├── rag_pipeline.py    #   FAISS + MiniLM retrieval baseline
│   │   ├── prompt_baseline.py #   few-shot + chain-of-thought baseline
│   │   └── benchmark.py       #   runs all approaches, writes report + chart
│   └── serve/
│       ├── api.py             # ★ FastAPI + llama.cpp (CPU GGUF) — cloud-hostable
│       ├── app_gradio.py       # browser chat demo
│       └── mac_inference_mlx.py# native Apple-Silicon inference (MLX)
├── docs/                       # CONCEPTS, PIPELINE (working analysis), DEPLOYMENT
├── tests/                      # GPU-free unit tests (pytest)
├── Dockerfile, docker-compose.yml, Makefile, .env.example
├── .github/workflows/ci.yml    # CI (lint-free smoke tests)
├── scripts/run_all.sh          # full pipeline (for cloud GPU box)
├── requirements.txt            # cloud/training deps (CUDA)
├── requirements-mac.txt        # local Mac deps (MLX / llama-cpp)
├── requirements-serve.txt      # API serving deps (FastAPI + llama.cpp)
└── data/processed, models, outputs  # artifacts (gitignored)
```

> 📚 **Deep dives:** [`docs/CONCEPTS.md`](docs/CONCEPTS.md) (every technique explained),
> [`docs/PIPELINE.md`](docs/PIPELINE.md) (working analysis + timeline),
> [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) (cloud hosting).

---

## 📊 Datasets (all free on Hugging Face)

**SFT (instruction tuning):**
- [`sujet-ai/Sujet-Finance-Instruct-177k`](https://huggingface.co/datasets/sujet-ai/Sujet-Finance-Instruct-177k)
- [`virattt/financial-qa-10K`](https://huggingface.co/datasets/virattt/financial-qa-10K) — Q&A over SEC 10-K filings
- **FinQA** (numerical reasoning) — official raw JSON from [czyssrs/FinQA](https://github.com/czyssrs/FinQA) (the HF mirrors use legacy loading scripts that `datasets>=4` rejects)

**DPO (preference pairs):** built by `prepare_dpo_data.py` (gold answer = *chosen*, base-model
answer = *rejected*), or use [`argilla/distilabel-intel-orca-dpo-pairs`](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs) as a template.

**Evaluation:**
- [`PatronusAI/financebench`](https://huggingface.co/datasets/PatronusAI/financebench)
- held-out split of `virattt/financial-qa-10K`

---

## 🚀 Quickstart

### A) Local setup on the Mac (data prep + inference)
```bash
source venv/bin/activate
pip install -r requirements-mac.txt

# 1. Prepare datasets (CPU-only, runs on Mac)
python -m src.data.prepare_sft_data
```

### B) Training on free Colab / Kaggle (steps 2–3)
1. Open a **T4 GPU** runtime (Colab: Runtime → Change runtime type → T4 GPU).
2. Clone this repo + `pip install -r requirements.txt`.
3. Run:
```bash
python -m src.data.prepare_sft_data
python -m src.train.train_sft_qlora          # QLoRA SFT
python -m src.data.prepare_dpo_data          # generate preference pairs
python -m src.train.train_dpo                # DPO alignment
python -m src.merge.merge_and_quantize       # merge + push to Hub
```
Or just run everything: `bash scripts/run_all.sh`

### C) Back on the Mac — quantize for local use & demo
```bash
# Option 1: MLX (native Apple Silicon, recommended)
pip install mlx-lm
python -m mlx_lm.convert --hf-path <your-hf-username>/finsage-qwen2.5-3b -q
python src/serve/mac_inference_mlx.py

# Option 2: GGUF via Ollama / llama.cpp (see merge_and_quantize.py output)

# Browser demo:
python src/serve/app_gradio.py
```

### D) Evaluate (before vs after)
```bash
python -m src.eval.evaluate            # writes outputs/eval_report.md
```

### E) Fine-tune vs. RAG vs. prompt-engineering (the differentiator)
```bash
python -m src.compare.rag_pipeline --build   # build FAISS index (once)
python -m src.compare.benchmark              # -> outputs/comparison_report.md + chart
```
Compares 5 approaches (base zero-shot, prompt-engineered, RAG, fine-tuned, fine-tuned+RAG)
on accuracy (ROUGE-L, token-F1) and latency, with a "when to use which" verdict.

### F) Serve it as a cloud API
```bash
pip install -r requirements-serve.txt
make api                                     # FastAPI on :8000  (CPU GGUF — runs anywhere)
# or containerized:
docker compose up --build
```
Full hosting guide (HF Spaces / Render / Fly.io / VM / vLLM-GPU): [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## 📈 Results (fill in after training)

| Metric | Base Qwen2.5-3B | + QLoRA SFT | + DPO |
|--------|----------------|-------------|-------|
| FinanceBench acc. | _ | _ | _ |
| 10-K Q&A ROUGE-L | _ | _ | _ |
| LLM-judge win-rate vs base | — | _ | _ |

---

## 🔧 Scaling up
Set `model.name` in `config/config.yaml` to `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` for stronger
results (still fits a free T4 via unsloth). The 7B 4-bit GGUF (~4.5GB) is tight but runnable on
the 8GB Mac via Ollama with a small context window.

## 📜 License
Code: MIT. Datasets and base models retain their original licenses (check each link).
