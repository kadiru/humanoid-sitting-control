# Fixed-progression PPO retention study

Both candidates started from the preserved policy with identical training seats, seed and budget. The extra pace action was removed. The parent penalty was the only difference between these two runs.

| Bank / controller | Complete successes | Worst impact |
|---|---:|---:|
| easy / Preserved policy | 16/16 | 570.1 N |
| easy / Continued PPO | 15/16 | 861.3 N |
| easy / PPO + parent penalty | 15/16 | 692.1 N |

The full-horizon parent run reproduces all twelve earlier trace arrays exactly. Each training arm performed 32,768 transitions; completed and unfinished attempts account for all of them. Parent normalizers stayed exactly fixed, initial weights were identical, optimizers started fresh, and both learned finite nonzero weight changes.

Fresh holdout used: False. Known gates determine whether further evaluation is permitted; failed arms were stopped. No deployment default changed.

The policy and critic use depth, robot kinematics and simulated onboard sensor equivalents. True poses and contact forces are evaluation/reward labels only. Gaussian joint exploration remains; this study removes the added asymmetric pace exploration.

This is one matched pair of short training runs, not proof of generalization to arbitrary surfaces. Freezing normalization and resetting the optimizer are shared changes, so comparisons to older experiments do not isolate those effects. Conditional torque results refer only to matched successful trials. If the wider gate is reached, its parent numbers reuse the preserved prior full-horizon evaluation on identical seeds and geometry.

Preserved original files/checkpoints and clean IsaacLab checkout verified. Legacy storage-check exit: 1; root free 29.64 GiB, actual output volume free 82.01 GiB. Existing files were not deleted.

![Matched experiment](comparison.png)
