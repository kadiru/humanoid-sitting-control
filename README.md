# Humanoid sitting and supported leaning

Research workspace for Unitree G1 sitting and supported leaning in IsaacLab:
rear-camera surface recognition, whole-body control, demonstrations, PPO,
contact/load estimation, and physical scoring.

This repository was first imported on 2026-11-09. The commits group existing
work by subsystem; they are **not** a reconstruction of historical Git commits.
Versioned experiment directories and their recorded timestamps, receipts, and
hashes remain the provenance record. The [reconstructed saved-history branch](https://github.com/kadiru/humanoid-sitting-control/commits/reconstructed-saved-history/)
now exposes 773 saved record groups as separate archive commits. These commits
were created now; they do not invent original authors, dates, or unsaved edits.
Archive order follows folder labels and local record numbers, not a verified
global edit chronology. Its final tree exactly matches the complete seven-commit
import (24,267 tracked files). See [reconstruction inventory](repository_metadata/reconstructed_history.json).

## Current result

The latest narrow-support calibration produced a 10.86-second supported hold,
with approximately 55% of bodyweight carried by the support. The complete trial
**failed**: one pelvis-slip sample and unresolved ankle-velocity consistency
prevented admission to a sensor-qualified return to standing. Zero complete
sit–hold–stand cycles from the requested three starts have passed.

The original development baseline remains 64/72 clean, with eight unresolved
cases and no accepted repairs. Training outcomes, model audits, and simulated
physical success are recorded separately; an audit PASS is not a repair.

- Current index: [current_successor_checkpoint.json](validated_low_seat_contact_01/current_successor_checkpoint.json)
- Latest native-readout fixture (negative reproduction result): [execution_receipt.json](native_contact_readout_20261109_v2/execution_receipt.json)
- Contact-entry preparation: [execution_receipt.json](rest_cycle_contact_fix_20261109_v1/execution_receipt.json)
- Latest sitting execution (failed): [execution_receipt.json](rest_cycle_load_validation_20261109_v1/execution_receipt.json)
- Latest recorded trial: [20-second video](rest_cycle_load_validation_20261109_v1/results/narrow_support_load_calibration.mp4)
- Load observer: [observer.py](rest_cycle_load_validation_20261109_v1/observer.py)

## Layout

- Top-level versioned directories retain the Mac control workspace and compact
  evidence from prior experiments.
- `ubuntu/` snapshots the project-owned sources from
  `/home/kadir/humanoid-sitting`, including `scripts/`, `sitting_task/`, and
  `experiments/`. It is a source snapshot, not an installed simulator.
- `repository_metadata/` records the import inventory, exclusions, dependency
  revision, source snapshots, and commit grouping.
- `repository_tools/` contains the conservative import and secret-screening tool.

## Runtime and evidence

The existing runtime is Ubuntu at `/home/kadir/humanoid-sitting`, using
`IsaacLab/.venv/bin/python`. IsaacLab is an external dependency pinned in
`repository_metadata/runtime_dependencies.json`; its checkout was clean at
import. Datasets, trajectories, policy weights, and generated solver arrays
remain in their existing locations. See [ARTIFACTS.md](ARTIFACTS.md).

Many frozen scripts contain absolute source/data paths. A fresh clone alone is
not sufficient to reproduce every experiment: restore the indexed artifacts,
the pinned simulator environment, and the recorded directory layout first.
Historical launch scripts are preserved for inspection; do not run every script
or treat a historical authorization as current permission.

The current human-approved GPU allocation is **4090 only** for this project;
the 4070 is reserved for AIWorker. Creating this repository launches no training,
simulation, CI workflow, or recurring automation.

## Commit practice

Commit each substantive code change together with its bounded validation and
outcome receipt. Keep unsuccessful outcomes intact. Push each completed commit
before starting the next logical change. Do not add credentials, environments,
bulk trajectories, or generated solver matrices to ordinary Git history.
