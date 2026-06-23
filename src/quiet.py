"""Reduce log/warning noise during generation-heavy steps (DPO data, eval, benchmark).

The big offender is transformers printing
'Both max_new_tokens and max_length seem to have been set ...' on *every* generate()
call. We fix the root cause (clear the model's default max_length) and also lower the
transformers log level so deprecation spam doesn't bury real progress output.
"""
from __future__ import annotations

import logging
import warnings


def quiet() -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass


def tame_generation(model) -> None:
    """Remove the default max_length so passing max_new_tokens doesn't warn each call."""
    try:
        model.generation_config.max_length = None
    except Exception:
        pass
