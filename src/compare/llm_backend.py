"""
Shared LLM generation interface used by the comparison benchmark.

One small wrapper around an unsloth/transformers causal LM so the RAG,
prompt-engineering and fine-tuned paths all generate the same way (apples-to-apples).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from src.config_utils import load_config
from src.data.prompts import SYSTEM_PROMPT
from src.quiet import quiet, tame_generation

quiet()


@dataclass
class GenResult:
    text: str
    latency_s: float
    n_tokens: int


class LLM:
    """Thin wrapper exposing chat() over a HF/unsloth model."""

    def __init__(self, model_name_or_path: str):
        from unsloth import FastLanguageModel

        cfg = load_config()
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=cfg.model.max_seq_length,
            load_in_4bit=cfg.model.load_in_4bit,
            dtype=cfg.model.dtype,
        )
        FastLanguageModel.for_inference(self.model)
        tame_generation(self.model)
        self.max_seq_length = cfg.model.max_seq_length

    def chat(self, messages: list[dict], max_new_tokens: int = 512) -> GenResult:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        t0 = time.perf_counter()
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        dt = time.perf_counter() - t0
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GenResult(text=text, latency_s=dt, n_tokens=int(new_tokens.shape[0]))

    def ask(self, question: str, user_input: str = "",
            system: str = SYSTEM_PROMPT, max_new_tokens: int = 512) -> GenResult:
        user = question if not user_input else f"{question}\n\n{user_input}".strip()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.chat(messages, max_new_tokens=max_new_tokens)
