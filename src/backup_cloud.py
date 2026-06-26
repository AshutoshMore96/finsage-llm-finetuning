"""
Push training artifacts (LoRA adapters, merged model, datasets) to durable cloud
storage. Works anywhere — Lightning AI, a VM, Kaggle — not just Colab.

Backends:
  hub  -> Hugging Face Hub        (easiest; free; needs env HF_TOKEN)
  gcs  -> Google Cloud Storage    (needs env GOOGLE_APPLICATION_CREDENTIALS=key.json)

CLI examples:
  python -m src.backup_cloud hub --repo AshutoshMore96/finsage-sft --path outputs/sft
  python -m src.backup_cloud gcs --bucket my-bucket --prefix finsage --path outputs/sft

Programmatic:
  from src.backup_cloud import to_hf_hub
  to_hf_hub("outputs/sft", "AshutoshMore96/finsage-sft-adapter")
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def to_hf_hub(path: str, repo_id: str, repo_type: str = "model",
              private: bool = True, skip_checkpoints: bool = True) -> None:
    """Upload a folder to the Hugging Face Hub. Reads HF_TOKEN from the env."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(path),
        repo_id=repo_id,
        repo_type=repo_type,
        ignore_patterns=["checkpoint-*"] if skip_checkpoints else None,
    )
    print(f"✅ uploaded {path} -> https://huggingface.co/{repo_id}")


def to_gcs(path: str, bucket: str, prefix: str = "",
           skip_checkpoints: bool = True) -> None:
    """Upload a folder to a Google Cloud Storage bucket.

    Auth: set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON key, or run
    `gcloud auth application-default login` first.
    """
    from google.cloud import storage

    client = storage.Client()
    bkt = client.bucket(bucket)
    root = Path(path)
    base = root.parent
    n = 0
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        if skip_checkpoints and "checkpoint-" in str(f):
            continue
        blob_name = f"{prefix}/{f.relative_to(base)}".lstrip("/")
        bkt.blob(blob_name).upload_from_filename(str(f))
        n += 1
    print(f"✅ uploaded {n} files from {path} -> gs://{bucket}/{prefix}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Back up artifacts to cloud storage.")
    ap.add_argument("backend", choices=["hub", "gcs"])
    ap.add_argument("--path", required=True, help="local folder, e.g. outputs/sft")
    ap.add_argument("--repo", help="hub repo id, e.g. user/finsage-sft")
    ap.add_argument("--bucket", help="GCS bucket name")
    ap.add_argument("--prefix", default="finsage", help="GCS key prefix")
    ap.add_argument("--public", action="store_true", help="make the hub repo public")
    ap.add_argument("--keep-checkpoints", action="store_true")
    args = ap.parse_args()

    skip = not args.keep_checkpoints
    if args.backend == "hub":
        if not args.repo:
            ap.error("hub backend needs --repo")
        to_hf_hub(args.path, args.repo, private=not args.public, skip_checkpoints=skip)
    else:
        if not args.bucket:
            ap.error("gcs backend needs --bucket")
        to_gcs(args.path, args.bucket, prefix=args.prefix, skip_checkpoints=skip)


if __name__ == "__main__":
    main()
