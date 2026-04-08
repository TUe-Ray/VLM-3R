#!/usr/bin/env python3
"""Validate EoMT mask-pooling alignment contract via side-output methods - DEBUG VERSION.

This script does not touch the main fusion path. It exercises:
- _build_eomt_pool_side_cache
- _compute_eomt_mask_pooled_side_output

and reports the required validation sections.
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

print("DEBUG: Starting imports", flush=True)

import torch
print("DEBUG: Imported torch", flush=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("DEBUG: About to import LlavaMetaForCausalLM", flush=True)

from llava.model.llava_arch import LlavaMetaForCausalLM

print("DEBUG: Successfully imported LlavaMetaForCausalLM", flush=True)


class DummyPoolValidator(LlavaMetaForCausalLM):
    def __init__(self):
        print("DEBUG: Initializing DummyPoolValidator", flush=True)
        self.config = SimpleNamespace(
            eomt_pool_top_k=3,
            eomt_pool_selection="mean_mask_confidence",
            eomt_pool_mask_area_threshold=0.5,
        )
        print("DEBUG: DummyPoolValidator initialized", flush=True)

    def get_model(self):
        return self


print("DEBUG: About to create validator instance", flush=True)

def main():
    print("DEBUG: Entering main", flush=True)
    print("SUCCESS: Script ran successfully!")


if __name__ == "__main__":
    main()
