# Supported-sitting RL: latest research state

Status: the matched fixed-progression retention study is complete. Both new candidates fail the predeclared retention gate and are rejected for promotion. Keep the preserved pilot-3 policy as the research parent and leave the deployment default unchanged. All simulation, training and rendering jobs have finished.

Work as the sole primary agent through Mac SSH to `kfuUbuntu`, project `/home/kadir/humanoid-sitting`; never start another coding agent on Ubuntu. The user has authorized RL research. Use tmux for long jobs and `/data/humanoid-sitting-perception` for large outputs. Preserve all code, datasets and checkpoints. No software was installed.

## Validated results

Each evaluation has eight known easier rigid geometries and two starts each. Every attempt runs the full 20 seconds with latched failures, initial standing and at least three continuous seconds of stable supported sitting required.

| Policy | Complete successes | Worst full-trial seat impact |
|---|---:|---:|
| Preserved pilot-3 parent | 16/16 | 570.12 N |
| Matched continued PPO | 15/16 | 861.27 N |
| PPO with frozen-parent KL penalty | 15/16 | 692.05 N |

Both candidates lose case 15, a 27.75 cm high rigid couch, to impact. First sampled latched failures are at 13.2 and 13.1 seconds; the original threshold is two robot body weights, about 632.53 N. The parent case has 3.08 seconds of continuous supported hold and peak force 542.03 N. Neither candidate gains a success on this bank. The 15 matched successes have conditional leg-torque changes of +3.59% and -2.90%; the latter is not an overall reliability gain.

The parent run reproduces all twelve earlier trace arrays bit for bit. Four new implementation tests pass, including exact weight/optimizer update parity against installed RSL-RL when the added penalty is zero. All completed and unfinished transitions account for the prescribed budget. There were no first-tick physical failures in the completed training attempts. Source/checkpoint preservation and a clean IsaacLab checkout are verified.

No wider-bank or fresh-holdout evaluation was run for these candidates because both failed the first gate. Seeds 51741/51743 remain unused. No longer training run or deployment change is warranted by these results.

## What changed and what the comparison isolates

Two matched runs each use 32 PPO updates x 64 steps x 16 robots = 32,768 transitions, 65,536 total. Both start from exactly the same parent actor/critic and Gaussian standard deviations, with fresh optimizers and frozen parent normalization statistics. Both use the same archived placement-validated mixed TRAINING bank from `support_rl_pace_mixed_train_02/geometry_bank.json` (SHA256 `140a25d21c971581810cb2a3b8aa2521f5e1b6ac3b2b87b896a6bb394725afcd`), geometry argument 51621, start seed 51713 and learning rate 3e-5. The bank contains eight easier prototypes and the first eight of the older wider training bank; no test shapes or success ranking enter it.

The only arm difference is coefficient zero versus one on KL(frozen parent || current), summed over the 23 Gaussian coordinates and averaged over the PPO minibatch. This soft penalty uses current training observations, not a replay of evaluation data, and is not a guarantee of preserving trajectories. Frozen normalizers and fresh optimizers are shared changes; this experiment does not isolate their effects relative to historical pilots.

The added pace action is removed entirely. Original 144 sensor channels, 23 bounded joint residuals, frozen actions-mode scaffold at progression one, reward and physical limits are retained. Gaussian joint exploration remains; no sampled joint coordinate reached the residual clipping bound during either run. Runtime actor and critic stay sensor-only. Simulator poses, contact forces and velocities remain reward/failure/evaluation labels only.

Ordinary continuation completes 44 training attempts (13 successes); the penalty run completes 40 (16 successes). These exploratory training outcomes are not test scores. Each takes about four minutes. The penalty reduces policy drift on the same 3,200 parent sensor observations: p95 correction difference 0.07369 degrees -> 0.01362 degrees; maximum 0.16779 -> 0.04208 degrees; mean KL 0.45390 -> 0.01613. This offline comparison is after training and does not simulate command slew or establish a closed-loop safety bound. Both still lose the same test case.

## Landing visibility: next concrete research task

The 48 new full attempts provide 9,600 case-frames, with 8,492 accepted current-depth estimates. Maximum accepted clearance error is 1.13038 cm and none exceeds 2 cm. Accuracy on accepted frames does not imply availability during landing.

An additional offline audit finds the current ten-channel depth estimate unavailable at the first sampled seat support above 25% of body weight in 11/16 parent, 13/16 control and 12/16 penalty trials. Case 15 is unavailable at that event in all three. The parent succeeds despite these gaps, so the audit does NOT establish missing depth as the cause of the regression. It does identify a gap in continuous relative tracking. Contact/IMU/proprioceptive features still exist; this is not loss of all sensing.

Next: capture raw depth, synchronized joint/IMU samples and per-frame rejection reasons through the lowering/contact interval, using training geometries. Stored traces currently omit rejection reasons, so do not guess the exact occlusion or fitting cause. Test whether recent geometry plus kinematics can maintain a short-term clearance estimate with explicit uncertainty when a fresh plane is unavailable; use simulator truth only for scoring. In parallel, measure how sensitive the preserved motion is to small command changes on training placements. This should precede another larger RL run or higher penalty sweep. Do not claim that this study has fixed contact robustness, continuous tracking or arbitrary-surface sitting.

## Files and reproduction

- Parent: `/data/humanoid-sitting-perception/support_rl_independent_train_04/final.pt`, SHA256 `2c0f6907196be68bce9a1ee2e6b6ef6e4c82bf0ecdff622dc55b81fe8ef58d13`.
- Control: `support_rl_retention_control_train_01/final.pt`, SHA256 `8274b4fd6a92c096734ad6937dd9dd7f30009ccb38a0479ec12ad16ed5c8b0ab`.
- Penalty candidate: `support_rl_retention_anchored_train_01/final.pt`, SHA256 `c3ae2f3a9d69766110c70bef2dff7c0f26e89ae518bdb8fc9aad4013f0dd694a`.
- New module: `sitting_task/src/humanoid_sitting/support_retention.py`; a restricted local PPO update based on installed RSL-RL 5.5.1, with zero-weight parity tests. Installed packages remain unchanged.
- Entry point: `scripts/train_support_rl_retention.py`; tests: `scripts/test_support_retention.py`; launch/gates: `run_support_rl_retention_01.sh`, `check_support_rl_retention.py`.
- Finalization: `finalize_support_rl_retention_01.sh`; report and video: `report_support_rl_retention.py`, `render_support_rl_retention.py`.
- Post-training diagnostic: `diagnose_support_landing_visibility.py`; no extra simulation or retraining.
- Parent/evaluations: `support_rl_retention_easy_{parent,control,anchored}_01`.
- Report: `/data/humanoid-sitting-perception/support_rl_retention_results_01`, including `landing_visibility/`, pose/force reconstruction video, plots and CSVs. All training and finalization exit markers are zero.
- Mac delivery: `/Users/kadir/humanoid-sitting-control/validated_support_rl_retention`.
- Protocol: `docs/SUPPORT_RL_RETENTION_PROTOCOL.md`, written before training/results. Previous pace-pilot handoff is preserved as `docs/SUPPORT_RL_PILOT_4_ARCHIVE.md`; older archives remain.

The legacy preservation checker still returns exit 1 only after verifying preservation because root free space is 29.64 GiB, below its old 30 GiB assertion. The actual `/data` destination has 82.01 GiB free and passes its separate reserve check. This warning is preserved in the report, not erased; nothing was deleted to free space.

Scope remains initially aligned rigid seats behind the robot. No navigation, deformable cushions, hardware deployment or universal-surface success is claimed.
