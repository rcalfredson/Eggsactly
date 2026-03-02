# project/lib/torch_utils.py

from __future__ import annotations

from typing import Any


def load_state_dict_compat(path: str, *, map_location=None) -> Any:
    """
    Load a PyTorch state_dict with best-effort compatibility across torch versions.

    - Newer torch: uses weights_only=True (safer deserialization)
    - Older torch (e.g. 1.12): falls back to plain torch.load
    """
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
