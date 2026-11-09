**Reviewed decision: do not promote this candidate.** It regressed on all three paired banks. Retain the previous policy. The pace-disabled diagnostic did not recover the old behavior, and an isolated actuator test exposed a training/execution distribution mismatch that requires a controlled correction.

[Watch the complete-body comparison video](sitting_comparison_full_view.mp4).

# Learned pace PPO pilot

Experimental checkpoint only; original deployment default remains unchanged.

| Bank | Previous success | Pace success | Previous worst impact | Pace worst impact |
|---|---:|---:|---:|---:|
| Known easier seats | 16/16 | 8/16 | 570.1 N | 781.2 N |
| Known wider seats | 5/16 | 2/16 | 1948.1 N | 3033.5 N |
| New wider holdout | 11/16 | 7/16 | 911.4 N | 999.2 N |

196,608 new transitions: 65,536 on the easier bank, then 131,072 on the mixed bank. Final checkpoint only.
Before any holdout evaluation, mixed stage 01 was interrupted: a box overlapped the initial standing robot. Stage 02 restarts from the completed easy checkpoint and uses the first eight prior TRAINING-bank wider prototypes, selected in original order rather than by success. A 192-start placement probe passes. The interrupted run is preserved and excluded.
Six paired full-horizon runs contain 96 attempts and 19,200 recorded case-frames. Refusals count as failures.
The actor/critic use 145 sensor-derived channels, including the observable pace filter. Four new unique tests plus the previous twelve pass; discovery executes twenty tests because four admission tests are imported twice.
Simulator probes confirm cold reset behavior, isolation of other robots, fresh camera admission and per-robot pace reset.

- Known easier seats: gained [], lost [2, 6, 7, 9, 12, 13, 14, 15]; matched-success leg torque +5.86% (8 pairs).
- Known wider seats: gained [], lost [0, 1, 3]; matched-success leg torque +4.30% (2 pairs).
- New wider holdout: gained [], lost [6, 7, 8, 9]; matched-success leg torque -10.47% (7 pairs).

A full-trial force peak uses all 50 Hz checks. Video curves use 10 Hz samples and may miss spikes. Robot poses/forces in the reconstruction are evaluation truth, never policy input.

- 8 geometries with two starts per bank; insufficient to establish arbitrary-surface reliability.
- Rigid initially aligned seats; no compliant cushions, navigation or hardware validation.
- Pace changes recurrent progression, not guaranteed physical vertical speed.
- Training ends at first failure; all evaluation attempts continue for 20 seconds with failures latched.
- No privileged actor/critic inputs. True poses, forces and velocity are reward/evaluation labels only.
- Pace saturation is asymmetric: positive requests cannot accelerate the scaffold.
- This pilot jointly changes action representation, exploration, curriculum and budget; it does not isolate causal contributions.
- CSV landing margins describe the pelvis center inside the true rectangle, not full contact-area coverage. Landing velocity is a 10 Hz finite difference, not an instantaneous sensor reading.

## Diagnostic after the known-set regression

With the same final learned weights, fixing pace to its original value changes known easier-set success from 8/16 to 7/16; the previous PPO scored 16/16. See [the pace-disabled comparison](pace_ablation/report.md). This is a known-set diagnostic, not a new holdout.

## Exploration-distribution diagnostic

An isolated test of the implemented filter gives mean pace 0.8632 under zero-mean Gaussian exploration (std 0.5), versus 1.0000 for deterministic mean actions. This asymmetric clipping creates a training/execution mismatch. It is a plausible contributor requiring controlled retraining, not a proven sole cause. See [the diagnostic](pace_distribution/report.md).
