"""Inspect checkpoints generated locally by the G1 infrastructure test."""
import json
from pathlib import Path
import torch

root = Path("logs/rsl_rl/g1_infrastructure/2026-10-08_23-13-49_smoke")
first = torch.load(root / "model_0.pt", map_location="cpu", weights_only=False)
last = torch.load(root / "model_4.pt", map_location="cpu", weights_only=False)
print("CHECKPOINT_KEYS", list(last))
changes = {}
for group, values in last.items():
    if isinstance(values, dict):
        deltas = []
        for name, value in values.items():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all(), (group, name)
                old = first.get(group, {}).get(name)
                if torch.is_tensor(old) and value.is_floating_point():
                    deltas.append((value - old).abs().max().item())
        if deltas:
            changes[group] = max(deltas)
assert any(v > 0 for v in changes.values()), changes
result = {"final_iteration": last.get("iter"), "max_parameter_changes": changes,
          "final_checkpoint_bytes": (root / "model_4.pt").stat().st_size}
Path("../logs/checkpoint-verification.json").write_text(json.dumps(result, indent=2) + "\n")
print("CHECKPOINT_VERIFIED", json.dumps(result))
