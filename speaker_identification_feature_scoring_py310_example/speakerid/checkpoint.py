from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def torch_load_compatible(path: str | Path, map_location: Any = "cpu") -> Dict[str, Any]:
    """
    Load project checkpoints with PyTorch 2.2.x.

    We intentionally do not pass the newer `weights_only` argument so the
    code remains compatible with the pinned PyTorch 2.2.2 environment.
    """
    return torch.load(str(path), map_location=map_location)
