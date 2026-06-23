"""Lightweight, GPU-free tests. Run: pytest -q"""
from __future__ import annotations

import pytest


def test_config_loads():
    from src.config_utils import load_config
    cfg = load_config()
    assert cfg.project_name == "finsage"
    assert cfg.model.max_seq_length > 0
    assert 0 < cfg.dpo.beta <= 1
    assert "finetuned" in list(cfg.compare.approaches)


def test_build_messages_roles():
    from src.data.prompts import build_messages
    m = build_messages("Q?", "A.", user_input="ctx")
    assert [x["role"] for x in m] == ["system", "user", "assistant"]
    # prompt-only form omits the assistant turn
    mp = build_messages("Q?")
    assert [x["role"] for x in mp] == ["system", "user"]


def test_prompt_baseline_fewshot():
    from src.compare.prompt_baseline import build_prompt
    msgs = build_prompt("What is EBITDA?")
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    # few-shot adds paired user/assistant turns before the final question
    assert sum(1 for m in msgs if m["role"] == "assistant") >= 1


def test_token_f1_and_chunk():
    pytest.importorskip("datasets")  # benchmark/rag import datasets at module load
    from src.compare.benchmark import token_f1
    from src.compare.rag_pipeline import _chunk
    assert token_f1("gross margin is 40%", "gross margin is 40%") == pytest.approx(1.0)
    assert token_f1("", "x") == 0.0
    chunks = _chunk("word " * 600, size=1200, overlap=150)
    assert len(chunks) >= 2
