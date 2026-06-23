"""Tiny helper to load config/config.yaml and resolve paths relative to repo root."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def _to_ns(obj):
    """Recursively convert dicts to attribute-access namespaces."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_config(path: str | os.PathLike | None = None) -> SimpleNamespace:
    path = Path(path) if path else CONFIG_PATH
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _to_ns(raw)


def abspath(rel: str) -> str:
    """Resolve a config path (relative to repo root) to an absolute path."""
    p = Path(rel)
    return str(p if p.is_absolute() else REPO_ROOT / p)


if __name__ == "__main__":
    cfg = load_config()
    print("Loaded config for project:", cfg.project_name)
    print("Model:", cfg.model.name)
