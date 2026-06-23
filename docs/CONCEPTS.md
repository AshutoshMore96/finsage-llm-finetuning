# Concepts used in FinSage (plain-English + why we use each)

This is the "explain everything" reference. Read top-to-bottom and you'll understand
every technique in the repo.

---

## 1. The model: a small open LLM (Qwen2.5-3B-Instruct)
A **large language model** predicts the next token given previous tokens. "3B" = 3 billion
parameters. "Instruct" = already fine-tuned by its makers to follow chat instructions, so we
start from a helpful assistant rather than a raw text-predictor.
**Why small:** it trains on a free T4 and, once quantized, runs on an 8GB laptop.

---

## 2. Fine-tuning vs. the alternatives
Three ways to make a base model better at *your* task:

| Technique | What changes | Knowledge source | Best for |
|-----------|--------------|------------------|----------|
| **Prompt engineering** | nothing (just the prompt) | model's pre-trained memory | quick behavior shaping |
| **RAG** | nothing (adds retrieved text) | external document store | fresh / proprietary facts |
| **Fine-tuning** | the model weights | your training data | durable style, format, reasoning |

FinSage implements **all three** and benchmarks them (`src/compare/`).

---

## 3. Supervised Fine-Tuning (SFT)
Show the model thousands of `(question → ideal answer)` pairs and train it to reproduce the
answers. This is standard supervised learning with a **cross-entropy loss** on the answer
tokens. After SFT the model "talks like a finance analyst."

- **Train-on-responses-only:** we mask the prompt tokens in the loss so the model is graded
  only on *its answer*, not on parroting the question. Cleaner learning signal.

---

## 4. LoRA — Low-Rank Adaptation (the "efficient" in PEFT)
Full fine-tuning updates all 3B weights — huge memory. **LoRA** freezes the original weights
and injects tiny trainable "adapter" matrices into each layer. Instead of learning a full
weight update ΔW (a big matrix), it learns ΔW ≈ **B·A**, where A and B are skinny low-rank
matrices. You train <1% of the parameters and get ~the same quality.

- **`r` (rank):** size of the adapter (we use 16). Bigger = more capacity, more memory.
- **`alpha`:** scaling factor for the adapter's contribution (we use 32).
- **target_modules:** which layers get adapters (attention + MLP projections).

**PEFT** = Parameter-Efficient Fine-Tuning, the umbrella term; LoRA is the most popular method.

---

## 5. QLoRA — Quantized LoRA
LoRA still loads the frozen base model in full precision (16-bit) → lots of VRAM. **QLoRA**
first **quantizes the frozen base to 4-bit** (`bitsandbytes` NF4 format), then trains LoRA
adapters on top. Result: a 7B model fine-tunes on a single 16GB GPU.

- **4-bit NF4:** "NormalFloat4," a 4-bit number format tuned for the bell-curve distribution
  of neural-net weights.
- **Why it needs CUDA:** `bitsandbytes`' 4-bit kernels are written for NVIDIA GPUs — they do
  **not** run on Apple's Metal/MPS. That's why we train on Colab, not the Mac.

---

## 6. DPO — Direct Preference Optimization (alignment)
SFT teaches *a* good answer. **Preference optimization** teaches the model to prefer *better*
answers over *worse* ones. Classic RLHF does this with a separate reward model + reinforcement
learning (PPO) — complex and unstable. **DPO** skips the reward model: given triples
`(prompt, chosen, rejected)`, it directly nudges the model to raise the probability of
`chosen` and lower `rejected`, relative to a frozen reference copy of itself.

- **`beta`:** how hard to push away from the reference model (we use 0.1). Higher = more
  aggressive, risk of degradation.
- **Where our pairs come from:** `chosen` = gold dataset answer, `rejected` = the *base*
  model's own (weaker) answer. This is the **data flywheel** — bootstrapping preference data
  without human labelers.
- **ORPO** is a newer one-stage alternative (SFT + preference in a single loss); DPO is the
  widely-adopted standard, so we use it.

---

## 7. Quantization for deployment (GGUF / MLX)
After training we **merge** the LoRA adapters back into the base weights (one standalone
model), then quantize for cheap inference:
- **GGUF + llama.cpp:** a CPU-friendly format. A 3B model at `Q4_K_M` (~2GB) runs on a $5
  cloud VM or your Mac, no GPU.
- **MLX:** Apple's native array framework; `mlx-lm` runs quantized models fast on M-series
  chips using the GPU/Neural Engine.

**Quantization** = storing weights in fewer bits (4 instead of 16) → 4× smaller, slightly
lower quality, much faster/cheaper inference.

---

## 8. RAG — Retrieval-Augmented Generation
Instead of relying on the model's memory, **fetch relevant documents at query time** and put
them in the prompt. Pipeline:
1. **Chunk** documents into passages.
2. **Embed** each chunk into a vector with an embedding model (MiniLM).
3. Store vectors in a **vector index** (FAISS).
4. At query time, **embed the question**, find the nearest chunks (**cosine similarity**),
   and **stuff** them into the prompt.

- **Embeddings:** numeric vectors where similar meanings are close together.
- **FAISS:** Facebook's fast similarity-search library (we use `IndexFlatIP` = exact inner-
  product search on normalized vectors = cosine similarity).
- **Trade-off:** always up-to-date (re-index, don't retrain) but adds retrieval latency and
  depends on retrieval quality.

---

## 9. Evaluation
You can't claim improvement without measuring it.
- **ROUGE-L:** longest-common-subsequence overlap between prediction and gold answer (lexical
  similarity).
- **Token-F1:** harmonic mean of token precision/recall vs. the gold answer.
- **LLM-as-judge:** ask a strong model (Claude) which of two answers is better — correlates
  better with human judgment than pure overlap metrics. Used as an optional win-rate.
- **Held-out set:** we evaluate on examples the model never trained on (the tail of the
  dataset) to avoid inflated scores.

---

## 10. Serving & MLOps
- **FastAPI:** Python web framework exposing `/chat` and `/health` endpoints.
- **Docker:** packages the app + dependencies into a portable container that runs identically
  on your laptop and in the cloud.
- **Health checks / API key / config-as-YAML / CI tests:** the production hygiene that
  separates a notebook from a deployable service.

---

## Glossary quick-hits
- **Token:** a sub-word unit the model reads/writes.
- **Epoch:** one full pass over the training data.
- **Gradient accumulation:** simulate a big batch on small VRAM by summing gradients over
  several mini-batches before updating.
- **Learning-rate scheduler (cosine):** smoothly decays the learning rate during training.
- **bf16/fp16:** 16-bit float formats for faster training.
- **Adapter merge:** folding LoRA deltas into base weights to get a single deployable model.
