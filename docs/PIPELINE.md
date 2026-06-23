# FinSage — Working Analysis & Timeline

How the project actually executes: what runs after what, where it runs, what it produces,
and roughly how long each step takes.

---

## Dependency graph (what depends on what)

```
                         config/config.yaml  (read by every step)
                                   │
        ┌──────────────────────────┼───────────────────────────────┐
        ▼                          ▼                                ▼
[1] prepare_sft_data        [R1] rag_pipeline --build        (HF datasets, downloaded)
   data/processed/             models/rag_index/
   sft_finance.jsonl           (faiss.index + chunks.pkl)
        │                          │
        ▼                          │
[2] train_sft_qlora              │
   outputs/sft/  (LoRA adapters) │
        │                          │
        ▼                          │
[3] prepare_dpo_data  ───────────┤ (uses BASE model to generate 'rejected')
   data/processed/dpo_finance.jsonl
        │                          │
        ▼                          │
[4] train_dpo                     │
   outputs/dpo/  (aligned adapters)
        │                          │
        ▼                          │
[5] merge_and_quantize            │
   models/finsage-merged-16bit/   │
   models/...-gguf/finsage.Q4_K_M.gguf
        │                          │
        ├───────────────┬─────────┴───────────────┐
        ▼               ▼                          ▼
[6] evaluate     [7] compare/benchmark      [8] serve (api / gradio / mlx)
   eval_report.md   comparison_report.md       FastAPI :8000
                    comparison_chart.png       (uses GGUF)
```

**Hard ordering rules**
- `[2]` needs `[1]`. `[4]` needs `[2]` + `[3]`. `[5]` needs `[4]`.
- `[3]` (DPO data) loads the **base** model, so it can run any time after `[1]`, but its
  output is only consumed by `[4]`.
- `[6]`, `[7]` need `[5]` (the merged model). `[7]` also needs `[R1]` (the RAG index) for the
  `rag` / `finetuned_rag` approaches.
- `[8]` (serving) needs the GGUF from `[5]`.
- `[R1]` is independent of training — build it any time.

---

## Stage-by-stage analysis

| # | Step | Command | Runs on | Inputs | Outputs | ~Time (T4) |
|---|------|---------|---------|--------|---------|-----------|
| 1 | SFT data prep | `python -m src.data.prepare_sft_data` | Mac or cloud (CPU) | 3 HF datasets | `sft_finance.jsonl` | 3–8 min (download-bound) |
| R1| Build RAG index | `python -m src.compare.rag_pipeline --build` | Mac or cloud (CPU) | 10-K contexts | FAISS index + chunks | 3–6 min |
| 2 | QLoRA SFT | `python -m src.train.train_sft_qlora` | **GPU** | sft jsonl + base model | LoRA adapters | 30–60 min (1 epoch, 20k rows) |
| 3 | DPO data gen | `python -m src.data.prepare_dpo_data` | **GPU** | 10-K Q&A + base model | dpo jsonl | 15–30 min (3k pairs) |
| 4 | DPO align | `python -m src.train.train_dpo` | **GPU** | dpo jsonl + SFT adapters | aligned adapters | 15–30 min |
| 5 | Merge + quant | `python -m src.merge.merge_and_quantize` | **GPU** (merge) | base + DPO adapters | 16-bit model + GGUF | 5–10 min |
| 6 | Evaluate | `python -m src.eval.evaluate` | GPU (fast) / Mac | merged + base | `eval_report.md` | 5–15 min |
| 7 | Benchmark | `python -m src.compare.benchmark` | GPU | merged + base + index | report + chart | 10–25 min |
| 8 | Serve | `uvicorn src.serve.api:app` | Mac / cloud (CPU) | GGUF | live API | seconds to start |

Total first run on a free T4: **~2–3 hours** end to end.

---

## Recommended execution timeline

### Day 0 — local prep (Mac)
1. `pip install -r requirements-mac.txt`
2. `make data` (step 1) and `make rag` (step R1) — both CPU, verify data looks right.
3. Commit + push the repo to GitHub.

### Day 0/1 — training (Colab/Kaggle T4)
4. Open `notebooks/colab_train.ipynb`, select T4, `pip install -r requirements.txt`.
5. Run steps **1 → 2 → 3 → 4 → 5** (or `bash scripts/run_all.sh`).
6. Run step **6** (eval) and step **7** (benchmark). Save `eval_report.md`,
   `comparison_report.md`, `comparison_chart.png` — these go in your portfolio.
7. Download `models/finsage-merged-16bit-gguf/` (or push the model to the HF Hub).

### Day 1 — local serving + demo (Mac)
8. `pip install -r requirements-serve.txt`, drop the GGUF into `models/`.
9. `make api` → hit `http://localhost:8000/docs`. Or `python src/serve/app_gradio.py`.

### Day 1/2 — cloud hosting
10. Follow `docs/DEPLOYMENT.md` (Docker → Render/Fly.io/HF Spaces or a VM).

---

## Failure modes & what to do

| Symptom | Cause | Fix |
|---------|-------|-----|
| `bitsandbytes`/CUDA error on Mac | tried to train locally | train on Colab/Kaggle GPU |
| OOM during SFT | seq len / batch too high or 7B on T4 | lower `max_seq_length`, `per_device_batch_size`; keep 3B |
| DPO step OOM | DPO holds 2 forward passes | already batch=1; lower `dpo.max_length` |
| `train_on_responses_only` warning | chat-template markers differ | harmless; it falls back to full-sequence loss |
| GGUF export fails in step 5 | llama.cpp converter hiccup | convert later with llama.cpp, or use MLX path |
| RAG answers look generic | retrieval missed | raise `rag.top_k`, increase `max_corpus_docs` |
| API 503 | model still loading | wait for `/health` to return `ok` |

---

## What "good" looks like (success criteria)
- Fine-tuned model **beats base** on ROUGE-L / token-F1 in `eval_report.md`.
- In `comparison_report.md`, **fine-tuned + RAG** is at or near the top, and you can *explain
  the trade-offs* (the senior signal — not just the numbers).
- The API serves answers locally and in a container; `pytest` is green in CI.
