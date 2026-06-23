"""
RAG baseline — answer finance questions by RETRIEVING relevant 10-K context, no fine-tuning.

Pipeline:  chunk 10-K contexts -> embed (MiniLM) -> FAISS index -> retrieve top-k
           -> stuff into the prompt -> let the *base* model answer.

Build the index once:   python -m src.compare.rag_pipeline --build
Then RAG is used by the benchmark (src.compare.benchmark).
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from datasets import load_dataset

from src.config_utils import abspath, load_config
from src.data.prompts import SYSTEM_PROMPT

RAG_SYSTEM = (
    SYSTEM_PROMPT
    + "\n\nUse ONLY the retrieved context below to answer. If the context is "
    "insufficient, say so explicitly."
)


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


class RagIndex:
    def __init__(self):
        self.cfg = load_config().rag
        self._embedder = None
        self.index = None
        self.chunks: list[str] = []

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.cfg.embed_model)
        return self._embedder

    # ---- build ----
    def build(self) -> None:
        import faiss
        import numpy as np

        print("· Loading corpus (financial-qa-10K contexts) ...")
        ds = load_dataset("virattt/financial-qa-10K", split="train")
        seen, corpus = set(), []
        for r in ds:
            ctx = str(r.get("context", "") or "").strip()
            if ctx and ctx not in seen:
                seen.add(ctx)
                corpus.append(ctx)
            if len(corpus) >= self.cfg.max_corpus_docs:
                break

        for doc in corpus:
            self.chunks.extend(_chunk(doc, self.cfg.chunk_chars, self.cfg.chunk_overlap))
        print(f"· {len(corpus)} docs -> {len(self.chunks)} chunks; embedding ...")

        vecs = self.embedder.encode(
            self.chunks, normalize_embeddings=True, show_progress_bar=True,
            batch_size=64,
        ).astype("float32")
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

        out = Path(abspath(self.cfg.index_dir))
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out / "faiss.index"))
        with open(out / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"✅ Saved RAG index -> {out}")

    # ---- load ----
    def load(self) -> "RagIndex":
        import faiss
        out = Path(abspath(self.cfg.index_dir))
        self.index = faiss.read_index(str(out / "faiss.index"))
        with open(out / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        return self

    # ---- retrieve ----
    def retrieve(self, query: str, k: int | None = None) -> list[str]:
        k = k or self.cfg.top_k
        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        _, idx = self.index.search(q, k)
        return [self.chunks[i] for i in idx[0] if i >= 0]

    def build_prompt(self, question: str) -> list[dict]:
        ctxs = self.retrieve(question)
        context_block = "\n\n".join(f"[doc {i+1}] {c}" for i, c in enumerate(ctxs))
        user = f"Retrieved context:\n{context_block}\n\nQuestion: {question}"
        return [
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": user},
        ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="build the FAISS index")
    args = ap.parse_args()
    if args.build:
        RagIndex().build()
    else:
        print("Pass --build to construct the index.")
