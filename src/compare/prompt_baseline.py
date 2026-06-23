"""
Prompt-engineering baseline — squeeze the BASE model with few-shot + chain-of-thought,
no fine-tuning and no retrieval. This is the "did you even need to fine-tune?" control.
"""
from __future__ import annotations

from src.config_utils import load_config
from src.data.prompts import SYSTEM_PROMPT

# Hand-written few-shot exemplars (domain demonstrations)
FEW_SHOT = [
    {
        "q": "A company reports revenue of $200M and cost of goods sold of $120M. "
             "What is the gross profit margin?",
        "a": "Gross profit = 200 - 120 = $80M. Gross margin = 80 / 200 = 40%.",
    },
    {
        "q": "What does a current ratio below 1 indicate?",
        "a": "A current ratio below 1 means current liabilities exceed current assets, "
             "signaling potential short-term liquidity risk.",
    },
    {
        "q": "Define free cash flow.",
        "a": "Free cash flow = operating cash flow - capital expenditures. It is the cash "
             "a company generates after funding operations and asset maintenance.",
    },
]

COT_SUFFIX = (
    "\n\nThink step by step. For numeric questions show the calculation, then give the "
    "final answer on its own line prefixed with 'Answer:'."
)


def build_prompt(question: str, user_input: str = "") -> list[dict]:
    cfg = load_config().prompting
    system = SYSTEM_PROMPT + (COT_SUFFIX if cfg.use_cot else "")
    messages = [{"role": "system", "content": system}]

    for ex in FEW_SHOT[: cfg.num_few_shot]:
        messages.append({"role": "user", "content": ex["q"]})
        messages.append({"role": "assistant", "content": ex["a"]})

    user = question if not user_input else f"{question}\n\n{user_input}".strip()
    messages.append({"role": "user", "content": user})
    return messages
