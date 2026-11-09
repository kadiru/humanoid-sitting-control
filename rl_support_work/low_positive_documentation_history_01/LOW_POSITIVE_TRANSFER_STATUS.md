# Low positive-slope transfer — active development, 2026-10-16

User authorized repairing remaining low-seat slipping, then giving PPO a longer control window and comparing stability and total effort against the repaired reference and current policies. Work directly over SSH to kfuUbuntu. No second coding agent. All completed studies remain unchanged.

Data: `/data/humanoid-sitting-perception/low_positive_transfer_01`. At preparation, 20,120 prior files were hash-protected. Fresh holdout: 36.5/38.5/40.5/42.5 cm × +7.5/+12.5/+17.5 degrees, each twice (24 trials), reserved before new simulation. Prior positive-slope holdout is now development data and must not be called unseen in this study. Negative slopes excluded.

Diagnosis: prior remaining low-seat slips are pelvis-seat touchdown sliding around 11.8–12.1 seconds, while feet are almost stationary and measured geometry remains available. Moving tangentially while touching the seat is implicated. This is not established as a single-factor explanation.

First physical probes `phase_only` and `phase_deeper` each retained only 2/12 stable/clean trials. Prepositioning backward before lowering produced overshoot, impact and (in some cases) housing contact. These variants are rejected. All 122 outputs are protected in `phase_probe_preservation.json`; no savings are claimed from their failed fragments.

Current persistent probes `low_positive_timing_gpu0/1` test `timing_mild` and `timing_strong`. They retain the previous reference path and endpoints while slowing the touchdown part of the five-second transfer. Frozen resting parent59117 still acts at 11/15 seconds; these are parent-on probes, not pure-reference results. Status: `probe_status_timing_mild.json` and `probe_status_timing_strong.json`.

New learning software is isolated in `feedback_posture_ppo.py` and `run_low_positive_feedback.py`: 414 sensor/reference channels plus three previous applied residuals; three symmetric hip/knee/ankle residuals, initially bounded by .004 rad and .004 rad/s; refreshed every .5 seconds, applied every .1 seconds. The repaired feedback reference and frozen resting parent remain active. Complete undiscounted Monte Carlo reward-to-go uses all37-joint physical torque-squared and explicit physical/support/slip/contact penalties; no short bootstrap. Initial proposed learning window10–20s; earlier windows require separate development validation. This is continuous bounded residual control, not unassisted sitting.

Software tests passed exact-zero dynamics, absolute/slew bounds, delayed credit assignment, finite PPO updates and exact CPU optimizer replay. Simulator preflight, final reward/budget freeze and main training have NOT started. Do not report a physical repair or learned improvement until validated. Existing paused automation has not yet been reactivated for this new study.
