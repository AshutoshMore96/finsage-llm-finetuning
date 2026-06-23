"""
Stage 1 — Supervised fine-tuning with QLoRA (4-bit base + LoRA adapters).

Uses unsloth (2x faster, lower memory) + trl SFTTrainer. Fits a free Colab/Kaggle T4.
Saves LoRA adapters to outputs/sft.

Run on a CUDA GPU:  python -m src.train.train_sft_qlora
"""
from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

from src.config_utils import abspath, load_config


def main() -> None:
    cfg = load_config()

    from unsloth import FastLanguageModel, is_bfloat16_supported

    # 1) Load 4-bit base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        dtype=cfg.model.dtype,
    )

    # 2) Attach LoRA adapters (this is the "QLoRA" part: LoRA on a 4-bit base)
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

    # 3) Load + format dataset using the model's chat template
    ds = load_dataset("json", data_files=abspath(cfg.data.sft_file), split="train")

    def format_example(ex):
        ex["text"] = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        return ex

    ds = ds.map(format_example, remove_columns=ds.column_names)
    splits = ds.train_test_split(test_size=cfg.data.val_split, seed=cfg.data.seed)

    # 4) Trainer
    from trl import SFTConfig, SFTTrainer

    sft_args = SFTConfig(
        output_dir=abspath(cfg.sft.output_dir),
        per_device_train_batch_size=cfg.sft.per_device_batch_size,
        gradient_accumulation_steps=cfg.sft.grad_accum,
        num_train_epochs=cfg.sft.epochs,
        learning_rate=cfg.sft.learning_rate,
        warmup_ratio=cfg.sft.warmup_ratio,
        weight_decay=cfg.sft.weight_decay,
        lr_scheduler_type=cfg.sft.lr_scheduler,
        logging_steps=cfg.sft.logging_steps,
        save_steps=cfg.sft.save_steps,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        seed=cfg.data.seed,
        dataset_text_field="text",
        max_seq_length=cfg.model.max_seq_length,
        report_to="none",   # set to "wandb" to log to Weights & Biases
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        args=sft_args,
    )

    # Learn only on the assistant's response (mask the prompt) for cleaner gradients
    if cfg.sft.train_on_responses_only:
        try:
            from unsloth.chat_templates import train_on_responses_only
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
            )
        except Exception as e:
            print(f"! train_on_responses_only skipped: {e}")

    trainer.train()

    out = abspath(cfg.sft.output_dir)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"\n✅ Saved QLoRA SFT adapters -> {out}")


if __name__ == "__main__":
    main()
