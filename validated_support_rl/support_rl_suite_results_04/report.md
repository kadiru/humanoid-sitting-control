# Two supported-sitting PPO pilots

The fresh bank has 14/16 PPO successes, versus 14/16 for the baseline. Gained cases: [0]; lost cases: [14]. Holding-torque changes are conditional on the 13 matched successful attempts. This is evidence of adaptation, not established reliability superiority.

Neither checkpoint replaces the existing default controller.

Two small residual PPO pilots over a frozen sensor-based motion scaffold; rigid reachable seats behind the robot.

| Paired evaluation | Baseline success | PPO success | Baseline / PPO worst seat force (N) | Matched success leg torque change (Nm) |
|---|---:|---:|---:|---:|
| Pilot 1 training bank | 7/8 | 8/8 | 729.8 / 583.5 | -0.205 (7 pairs) |
| Pilot 1 unseen bank | 8/8 | 6/8 | 517.8 / 2318.0 | -0.056 (6 pairs) |
| Pilot 1 larger unseen bank | 23/24 | 23/24 | 633.2 / 711.7 | -0.011 (23 pairs) |
| Pilot 1 live-depth ablation | 8/8 | 6/8 | 517.8 / 2426.2 | -0.288 (6 pairs) |
| Pilot 2 known regression bank | 8/8 | 8/8 | 517.8 / 451.7 | -0.346 (8 pairs) |
| Pilot 2 fresh unseen bank | 14/16 | 14/16 | 772.1 / 651.1 | -0.305 (13 pairs) |
| Pilot 2 training bank | 14/16 | 14/16 | 784.2 / 682.4 | -0.335 (14 pairs) |

Each completed main pilot used 98,304 transitions; 196,608 total, plus a separate 512-transition plumbing smoke test and 15,360 optimized transitions in the interrupted balanced run. These are small initialization pilots, not a converged large-scale RL result.

Training reward alone does not qualify a checkpoint. The initial unseen bank becomes a known regression bank for Pilot 2. Only its new fresh bank remains untouched by the first pilot analysis.

## Measurement and limits

- No navigation, soft cushions, arbitrary surfaces, hardware validation or statistical superiority established.
- Actor and critic use sensor-derived channels. Simulator truth is used only for reward/failure labels and evaluation.
- Ideal simulated depth and IMU equivalents; existing small proprioceptive noise/delay, not calibrated hardware sensing.
- Foot force is a reward/evaluation label, not an actor input. Physical foot-pressure sensors are unverified.
- Synchronized batch resets waste samples after a latched failure; this is a bounded pilot implementation.
- The first balanced run stopped at its first reset on hidden startup geometry and is preserved. Pilot 2 counts unobservable starts as refused failures and holds posture; no resampling or truth fallback.
- Torque comparisons include only matched successful attempts; all-trial failures and impact are reported separately.
- PPO final checkpoints were fixed for evaluation. Pilot 1 failures informed Pilot 2 training coverage; that old bank is a regression test, not a fresh holdout.

Detailed per-run counts, depth-error checks, timing, checkpoint hashes, and successful-case paired comparisons are in `results.json`. The archived source and exact banks accompany every run.
