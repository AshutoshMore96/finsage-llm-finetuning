"""System prompt + chat formatting shared by data prep, training, eval and serving."""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are FinSage, a meticulous financial-analysis assistant. "
    "You answer questions about financial statements, SEC filings, markets, accounting, "
    "valuation and quantitative reasoning. "
    "Think step by step for any numeric question, show the key calculation, and state the "
    "final answer clearly. If the question cannot be answered from the given information, "
    "say so instead of guessing. Be concise and precise."
)


def build_messages(instruction: str, output: str | None = None,
                   user_input: str = "", system: str = SYSTEM_PROMPT) -> list[dict]:
    """Return a chat-format message list. If `output` is None, only prompt messages
    are returned (useful for generation / DPO prompts)."""
    user = instruction if not user_input else f"{instruction}\n\n{user_input}".strip()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if output is not None:
        messages.append({"role": "assistant", "content": output})
    return messages


def render_prompt(tokenizer, instruction: str, user_input: str = "") -> str:
    """Render a generation-ready prompt string using the model's chat template."""
    messages = build_messages(instruction, output=None, user_input=user_input)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
