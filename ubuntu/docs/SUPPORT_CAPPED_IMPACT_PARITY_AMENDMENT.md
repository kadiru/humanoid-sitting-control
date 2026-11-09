# Timer-regression parity amendment before any training

October12,2026 UTC. The original strict parity gate stopped the coordinator after
the full timer regression and before training. The original source, protocol,
failed status and outputs are preserved. No failed evaluation is discarded.

Every controller input, issued action, startup sensor value and recorded robot
pose/kinematic trajectory matches the archived run bitwise. One value in the
10Hz rollout force labels differs by0.000030517578125N. Three values in the50Hz
force-label archive differ by at most the same amount. Each difference is one
float32 unit of precision at that value. All other label fields match exactly.
Every episode metric, including peak force, reward, failure flags and effort,
is unchanged except the intended duration representation and success correction.
Only case13 changes from failure to success at exactly150 supported ticks.

The recovery gate therefore allows at most one float32 ULP ONLY for archived
force labels, while retaining exact comparisons for all other fields and all
physical outcomes. This tolerance is chosen after inspecting the discrepancy;
it is explicitly a validation amendment, not a predeclared condition. It does
not alter training, reward, observations, commands or any success threshold.

The recovery coordinator is scripts/resume_support_capped_impact_01.py. It uses
the completed timer run, verifies the narrow exception, records every deviation,
and resumes the unchanged four-run protocol. Its source and this amendment are
saved in the study's recovery/ directory, separately from the original freeze.
