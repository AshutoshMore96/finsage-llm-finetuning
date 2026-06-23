"""
Stage 2 — Preference alignment with DPO (Direct Preference Optimization).

Loads the QLoRA SFT adapters, then optimizes them on (prompt, chosen, rejected) pairs.
DPO needs no separate reward model and is the current open-source standard.

Run on a CUDA GPU:  python -m src.train.train_dpo
"""
from __future__ import annotations

from datasets import load_dataset

from src.config_utils import abspath, load_config


def main() -> None:
    cfg = load_config()

    from unsloth import FastLanguageModel, is_bfloat16_supported, PatchDPOTrainer

    PatchDPOTrainer()  # unsloth patch for memory-efficient DPO

    # 1) Load the base model and re-attach the SFT adapters as the starting point
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        dtype=cfg.model.dtype,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        use_rslora=cfg.lora.use_rslora,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.data.seed,
    )
    # Load weights learned during SFT
    model.load_adapter(abspath(cfg.sft.output_dir), adapter_name="default")

    # 2) Preference dataset
    ds = load_dataset("json", data_files=abspath(cfg.data.dpo_file), split="train")

    # 3) DPO trainer
    from trl import DPOConfig, DPOTrainer

    dpo_args = DPOConfig(
        output_dir=abspath(cfg.dpo.output_dir),
        per_device_train_batch_size=cfg.dpo.per_device_batch_size,
        gradient_accumulation_steps=cfg.dpo.grad_accum,
        num_train_epochs=cfg.dpo.epochs,
        learning_rate=cfg.dpo.learning_rate,
        beta=cfg.dpo.beta,
        max_length=cfg.dpo.max_length,
        max_prompt_length=cfg.dpo.max_prompt_length,
        logging_steps=cfg.dpo.logging_steps,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        seed=cfg.data.seed,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,           # unsloth/peft uses the disabled-adapter as reference
        args=dpo_args,
        train_dataset=ds,
        tokenizer=tokenizer,
    )
    trainer.train()

    out = abspath(cfg.dpo.output_dir)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"\n✅ Saved DPO-aligned adapters -> {out}")


if __name__ == "__main__":
    main()
