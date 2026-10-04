#!/usr/bin/env python3
"""Smoke check that the common RecipeLauncher change preserves eager mode."""

import json
import os
from pathlib import Path

import torch
import habana_frameworks.torch as ht

if os.environ.get("PT_HPU_LAZY_MODE") != "0":
    raise RuntimeError("Run this control with PT_HPU_LAZY_MODE=0")
x = torch.arange(128, dtype=torch.float32).reshape(8, 16)
y = ((x.to("hpu") * 2 + 1) * 3 + 5).cpu()
assert torch.equal(y, (x * 2 + 1) * 3 + 5)
result = dict(status="PASS", mode="eager", bridge=ht.__file__)
(Path(os.environ["PROBE_OUT"]) / "result.json").write_text(json.dumps(result) + "\n")
print(json.dumps(result), flush=True)
