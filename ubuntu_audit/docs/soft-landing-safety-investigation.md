# Sensor soft-landing investigation — 2026-10-09

No candidate promoted. All teacher/student checkpoints preserved; explicit archive `baselines/sensor-soft-landing-155-of-168` includes student, working contact estimator, original controller/evaluator, hashes and benchmark references. The working controller remains unchanged in `actions` mode. No hardware deployment, installation changes, driver changes, safety-limit changes or new large dataset.

## Mechanism evidence

Offline inspection of the 180-trial baseline found all 14 ceiling violations within 200 ms of first load-bearing contact. They are not late seated-hold force spikes. In the original clean/noisy banks, failing root downward velocities immediately preceding the peak were 0.14–0.26 m/s. Original clean failing cases first predicted contact 80–140 ms before any recorded >1 N seat force; force was zero at prediction. The estimator is partly anticipating touchdown, rather than reliably detecting actual contact.

All three clean baseline failures had local ankle-target steps around 0.002 rad, not the unrestricted 0.05 rad guard. Therefore large ankle-command steps at impact do not explain these cases by themselves. Full tracking-error, effort and target-step diagnostics are saved in `baseline_diagnostics.json`. Actual foot/balance dynamics and tracking must still be distinguished from target continuity. No evidence supports blaming weight-transfer instability for these 14 baseline spikes; separate damping and delay interventions do cause instability.

Clean-camera height estimates were 0.489–0.746 mm below simulator truth. Those small errors coexist with failures but do not establish causation: furniture and initial configuration are confounders. A matched 10 mm Gaussian perturbation of the *floor-relative height channel only* tests sensitivity, not a coherent shift of all seat coordinates; center XYZ remains unchanged. Sensor-delay ablation changes physical-equivalent q/dq/torque/IMU delay from 20 to 60 ms; it does not change motor-command delay. A separate command queue tests 20 ms command transport latency.

## Controlled experiments

` smooth ` adds a continuous knee-position-dependent ankle slew cap, down to 0.0015 rad per 20 ms, while retaining the original guards. It reduced median force but worsened the upper tail. Doubling ankle damping from 4 to 8 Nm s/rad with stiffness 160 and original torque caps substantially worsened balance; rejected. Both are optional experimental settings, never defaults. No collision geometry, anchoring, gravity or success criteria changed.

The conservative contact estimator is a 640 ms causal-window GRU trained for 1,800 short optimization steps on existing 96 teacher and 24 guarded learner trajectories. Simulator support is used only as a supervision label; negative contact classification has weight 3. Thirty trajectories from geometry-index holdouts were excluded from fitting. No evaluation-bank trajectory entered training. Original checkpoint remains untouched. This is estimator refinement, not a new actor or reinforcement-learning improvement.

## Matched quantitative results

Whole-trial force quantiles below include failures. Force ceiling remains **632.527832 N**. Each bank compares identical root/joint initial arrays, geometry and ceiling; assertions recorded in `metrics.json`. Fresh banks are held out from estimator training, but reused for evaluation, not claimed as never inspected.

| Run | Success | Falls / unintended | Peak P95 / P99 / max (N) | Maximum effort (Nm) | Minimum physical hold (s) |
|---|---:|---:|---:|---:|---:|
| soft_landing_actions_clean_01 | 33/36 | 0 / 0 | 690 / 761 / 773 | 32.37 | 7.12 |
| soft_landing_actions_noisy_01 | 32/36 | 0 / 0 | 746 / 1125 / 1224 | 31.61 | 7.54 |
| soft_landing_actions_fresh_clean_01 | 44/48 | 0 / 0 | 725 / 1091 / 1168 | 32.18 | 6.70 |
| soft_landing_actions_fresh_noisy_01 | 46/48 | 0 / 0 | 596 / 687 / 715 | 32.24 | 7.24 |
| soft_landing_smooth_clean_02 | 33/36 | 0 / 0 | 707 / 882 / 923 | 32.11 | 7.42 |
| soft_landing_damping_clean_02 | 20/36 | 4 / 5 | 2535 / 2572 / 2577 | 40.94 | 0.00 |
| soft_landing_conservative_clean_02 | 34/36 | 0 / 0 | 664 / 892 / 919 | 32.29 | 7.14 |
| soft_landing_conservative_noisy_02 | 32/36 | 0 / 0 | 754 / 1140 / 1227 | 31.63 | 7.54 |
| soft_landing_conservative_fresh_clean_02 | 44/48 | 0 / 0 | 828 / 991 / 1114 | 32.12 | 6.80 |
| soft_landing_conservative_fresh_noisy_02 | 45/48 | 0 / 0 | 623 / 704 / 728 | 32.32 | 7.24 |
| soft_landing_causal_height_02 | 35/36 | 0 / 0 | 554 / 916 / 1104 | 32.22 | 7.36 |
| soft_landing_causal_delay_02 | 12/36 | 11 / 24 | 6374 / 7047 / 7405 | 228.63 | 0.00 |
| soft_landing_causal_command_02 | 31/36 | 0 / 2 | 1158 / 1603 / 1719 | 32.79 | 3.96 |

Physical hold is independent of latched safety failures: it never converts an unsafe trial into success. Effort sample P95/P99, root descent, final stability RMS, all failure flags and original per-joint caps remain in machine-readable results. Existing effort caps are simulation benchmark caps, not verified hardware limits.

## Contact-estimation results

Positive label: sensor-time support >5% body weight (15.8 N), not infinitesimal touchdown. Metrics use threshold 0.65. Precision/recall are pooled time-sample statistics; long stable holds dominate them. First premature-positive timing and post-onset latency are reported separately so early false positives cannot masquerade as zero latency.

| Run | Precision / recall | Episodes >200 ms early | Earliest false timing (s) | Post-onset latency P95 (ms) |
|---|---:|---:|---:|---:|
| soft_landing_actions_clean_01 | 98.51% / 99.75% | 6 | -0.26 | 0 |
| soft_landing_actions_noisy_01 | 98.59% / 99.80% | 8 | -0.28 | 0 |
| soft_landing_conservative_clean_02 | 99.41% / 99.92% | 2 | -11.76 | 20 |
| soft_landing_conservative_noisy_02 | 99.42% / 99.94% | 0 | -0.14 | 20 |
| soft_landing_conservative_fresh_clean_02 | 99.17% / 99.98% | 7 | -12.30 | 20 |
| soft_landing_conservative_fresh_noisy_02 | 99.36% / 99.97% | 10 | -12.04 | 20 |

Startup false predictions still exist (including >11 s early); these need a verified sensor initialization/contact-state treatment. We did not silently zero startup torques or remove early unsafe steps. Better classification did not improve aggregate sitting success (155/168 versus 155/168), and fresh noisy success decreased. No promotion.

## Deployment boundary and remaining work

Controller/estimator see joint encoders and velocities, filtered estimated motor torque, IMU gyro/gravity, previous issued joint targets, latched RGB-D seat estimates, and sit command. Zero onboard foot-force channels. No privileged seat forces, root position/velocity, COM, ground-truth geometry or external phase clock enter their observation vector. Motion GRU retains an implicit timing schedule; actor body decoder lacks live proprioceptive feedback (ankle head has it). The authored governor is explicitly retained. The RGB-D mount and torque/IMU mappings remain unverified hardware assumptions.

The experiments identify delayed feedback as a strong failure cause and demonstrate geometry sensitivity, but do not completely partition every failure among geometry, tracking, contact anticipation and balance. A validated physical state/clearance observer and feedback-driven body descent controller remain needed. Do not increase training budget on the rejected variants. Any future candidate must improve held-out and noisy force tails without falls, unintended contacts, torque violations or relaxed success requirements.

## Artifacts and resume

- `runs/soft_landing_02/metrics.json`: distributions, effort, stability, precision/recall, timing, matching checks.
- `runs/soft_landing_02/baseline_diagnostics.json`: all 14 failing baseline cases.
- `runs/soft_landing_02/failing_examples.png` and `failing_examples.json`: preserved worst-case examples and full trace paths.
- Each evaluation folder has exact `command.json`, source, geometry audit, initial states, outcomes and compressed trajectories. All rejected models/logs retained.
- `scripts/run_soft_landing_causal_checks.sh` and `run_soft_landing_estimator_holdouts.sh`: exact executed commands. Existing output folders deliberately refuse overwrite; use new names to rerun.
- `runs/contact_estimator_conservative_05`: training arguments/metrics and experimental checkpoint.
- `runs/soft_landing_02/integrity.json`: checkpoint hash verification, storage budget and free space. This stage budget is 250 MiB, minimum free 30 GiB; no new large data collection.
