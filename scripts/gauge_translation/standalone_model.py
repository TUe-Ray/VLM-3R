"""Load the standalone gauge model without importing the full ``llava`` package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_MODEL_PATH = Path(__file__).resolve().parents[2] / "llava" / "model" / "cut3r_gauge_translation.py"
_MODULE_NAME = "_cut3r_gauge_translation_standalone"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _MODEL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load gauge translation model from {_MODEL_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = _MODULE
_SPEC.loader.exec_module(_MODULE)

for _name in dir(_MODULE):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_MODULE, _name)
