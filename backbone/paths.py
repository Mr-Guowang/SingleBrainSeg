from __future__ import annotations

import os
import sys
from pathlib import Path

BACKBONE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKBONE_ROOT.parent
NNUNET_ROOT = Path(os.environ.get("BRAINSEG_NNUNET_ROOT", PROJECT_ROOT / "nnUNet"))

def add_project_paths() -> None:
    """Make local network code and optional nnU-Net source importable."""
    paths = [BACKBONE_ROOT]
    if NNUNET_ROOT.exists():
        paths.append(NNUNET_ROOT)
    for path in (str(p) for p in paths):
        if path not in sys.path:
            sys.path.insert(0, path)
