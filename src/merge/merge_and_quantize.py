"""
Stage 3 — Merge the final (DPO) LoRA adapters into the base model and export.

Produces:
  * a 16-bit merged HF model (for the Gradio demo / further conversion)
  * optional push to the Hugging Face Hub
  * optional GGUF export (for Ollama / llama.cpp on the Mac)

Run on a CUDA GPU:  python -m src.merge.merge_and_quantize
"""
from __future__ import annotations

from src.config_utils import abspath, load_config


def main() -> None:
    cfg = load_config()

    from unsloth import FastLanguageModel

    # Load base + final adapters (DPO output is the latest stage)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        dtype=cfg.model.dtype,
    )
    final_adapters = abspath(cfg.dpo.output_dir)
    model.load_adapter(final_adapters, adapter_name="default")

    # 1) Merge to 16-bit HF format
    merged_dir = abspath(cfg.merge.merged_dir)
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"✅ Merged 16-bit model -> {merged_dir}")

    # 2) Optional: push to the Hub
    if cfg.merge.push_to_hub and cfg.merge.hub_repo:
        model.push_to_hub_merged(
            cfg.merge.hub_repo, tokenizer, save_method="merged_16bit"
        )
        print(f"✅ Pushed to https://huggingface.co/{cfg.merge.hub_repo}")

    # 3) Optional: GGUF export for llama.cpp / Ollama (q4_k_m is a good Mac default)
    try:
        model.save_pretrained_gguf(
            merged_dir + "-gguf", tokenizer, quantization_method="q4_k_m"
        )
        print(f"✅ GGUF (q4_k_m) -> {merged_dir}-gguf")
    except Exception as e:
        print(f"! GGUF export skipped ({e}). You can convert later with llama.cpp.")

    print(
        "\n--- Run locally on the Mac (8GB M2) ---\n"
        "Option A) MLX (native Apple Silicon):\n"
        f"  python -m mlx_lm.convert --hf-path {merged_dir} -q "
        f"--mlx-path {abspath(cfg.serve.mlx_model_path)}\n"
        "  python -m src.serve.mac_inference_mlx\n\n"
        "Option B) Ollama with the GGUF file:\n"
        "  1. Create a Modelfile:  FROM ./finsage-merged-16bit-gguf/<file>.gguf\n"
        "  2. ollama create finsage -f Modelfile && ollama run finsage\n"
    )


if __name__ == "__main__":
    main()
