# Humanoid sitting

**Geometry-conditioned rigid sitting now passes 432/432 transitions**, including 288 held-out cases across chairs, benches, boxes and rigid couch shapes. Zero safety violations; peak force 616 N under the unchanged 633 N limit; shortest seated hold 6.54 s. The registered task is `Humanoid-Sit-Geometry-Conditioned-G1`.

[Progress and resume command](docs/geometry-conditioned-progress.md) · [Comparison video](runs/geometry_reference_progress_01/transitions.mp4) · [Quantitative results](runs/geometry_reference_progress_01/metrics.csv).

Geometry adaptation is primarily in the reference generator. The 80-update PPO pilot did not show a consistent improvement and remains experimental. Existing controllers and checkpoints are preserved; `baselines/geometry-conditioned-rigid-v1` saves the new scripted baseline. Approximately 39 GiB remains free.

The results below describe the preserved earlier furniture and fixed-seat baselines.

**Rigid furniture curriculum:** the current controller passed 144/144 held-out transitions across procedural chairs, benches, boxes and rigid couch-shaped seats in the medium geometry range, with zero safety violations. Peak seat force: 539 N; shortest continuous hold: 10.44 s. [Progress and resume commands](docs/furniture-progress.md). The fixed-seat baseline is preserved in `baselines/fixed-seat-v1` with checksums.

The wider geometry stage remains **unpromoted** after contact/impact failures; safety thresholds were unchanged. Four-family results reflect a scripted reference with PPO residuals; zero residuals also pass.

The results below describe the preserved fixed-seat baseline.

**The fixed-seat standing-to-sitting transition now passes scripted and residual PPO evaluation under gravity.** The floating-base G1 stands, bends with forward torso lean, reaches the seat without shin contact, and recovers to a stable seated hold. Only joint actuators provide control. Existing installations, driver, checkpoints and logs were preserved.

| Controller | Trials | Success | Falls | Unintended contact | Peak seat normal force | Minimum continuous seated hold |
|---|---:|---:|---:|---:|---:|---:|
| Scripted reference | 54 | 100% | 0% | 0% | 318.9 N | 11.54 s |
| PPO training seed 7 | 54 | 100% | 0% | 0% | 325.5 N | 11.04 s |
| PPO training seed 42 | 54 | 100% | 0% | 0% | 347.2 N | 11.02 s |
| PPO training seed 123 | 54 | 100% | 0% | 0% | 369.8 N | 10.98 s |
| Seed 7 policy, fresh initialization seeds | 54 | 100% | 0% | 0% | 331.1 N | 11.06 s |
| Preserved seated-hold policy, starts seated | 54 | 100% | 0% | 0% | 545.1 N | 20.00 s |

Each trial lasts 20 seconds. Full policies passed **216/216** trials in total. Training used three seeds, 64 environments and 40 additional PPO updates per seed (184,320 transitions total), initialized from the preserved partial-descent checkpoint. Zero-residual control already succeeds: PPO learned residual corrections around the validated scripted motion; these results do not establish an improvement over that reference or a skill learned from scratch.

## Collision diagnosis and correction

The `knee_link` collision shape includes a 150 mm cylinder along the **shin**. Backward lean swept that cylinder into the front/underside of the seat. The old ankle stiffness also allowed substantial pitch error during deep bending. [Loaded geometry](runs/g1_kinematics_geometry_03/model.json) and [trajectory analysis](runs/transition_geometry_validated_01/analysis.json) verify the corrected motion; forward kinematics agree with simulated foot positions to under one micrometre. Minimum shin clearance from the seat front is 151.6 mm.

The seat stays at its original pose: centre (-0.4, 0, 0.25) m, dimensions (0.5, 0.9, 0.1) m, top 0.30 m. The robot starts at (0, 0, 0.72) m. The reference holds standing for 2 s, preleans for 1 s, descends for 4 s, recovers upright for 2 s, then holds. Deep hip/knee targets are -1.60/1.40 rad with 0.15 rad forward lean. Final hip/knee targets are -1.25/1.40 rad. Ankle feedback uses centre-of-mass position and forward velocity until the seat bears load. Ankle gains are 160/4; the original 20 Nm ankle effort limits remain enforced.

The custom task is `Humanoid-Sit-Full-Transition-G1` in [full_transition.py](sitting_task/src/humanoid_sitting/tasks/full_transition.py). Its reference clock is independent of episode counters randomized by PPO. Policy actions are bounded ±0.08 rad joint residuals. Earlier curriculum tasks/checkpoints remain available unchanged.

## Evaluation and safety

No automatic reset can erase a failure. Contact histories cover every 5 ms physics sample. Original requirements remain: load-bearing seat contact, controlled descent, upright stable posture, no falls/pelvis-floor contact, no knee/foot seat or arm support above 10 N, downward speed at most 0.8 m/s, and at least 3 continuous seconds seated. An additional total seat normal-force limit of two body weights (632.5 N) is enforced. Reported impact is the peak resultant seat normal force, including all contacting bodies; it excludes friction.

Initialization grids vary x by ±15 mm, root height 0.715–0.725 m, pitch around -0.02 rad, lateral position ±8 mm and joint offsets ±0.01 rad, with additional small seeded pose jitter. Initial standing is checked before descent. Training/evaluation seeds are 7/42/123; fresh holdout initialization seeds are 2024/2025/2026. These are bounded aligned starts on one rigid seat, not arbitrary furniture or approach tests.

The seated-policy comparison uses its previously recorded **settled** pose and original gains. An earlier comparison using its authored reset penetrated the floor and failed the strengthened impact/contact checks; those failed results are preserved in `runs/seated_hold_policy_comparison_01`. No thresholds were relaxed. Starting seated and completing a transition are different skills; hold durations reflect that difference.

[Evaluation batch](runs/full_transition_policy_batch_01.json) · [Fresh-seed results](runs/full_transition_policy_holdout_01/results.json) · [Seated baseline](runs/seated_hold_policy_comparison_02/results.json) · [Scripted baseline](runs/full_transition_task_baseline_01/results.json) · [Comparison video](runs/full_transition_comparison_02/comparison.mp4)

## Resume without installation

```bash
cd /home/kadir/humanoid-sitting
bash scripts/run.sh python ../scripts/evaluate_full_transition.py \
  --device cuda:0 \
  --checkpoint /home/kadir/humanoid-sitting/IsaacLab/logs/rsl_rl/g1_full_transition/2026-10-09_10-08-57_seed_7/model_96.pt \
  --seeds 2024 2025 2026 --output ../runs/my_full_transition_evaluation
```

Output directories must be new. Add `--video --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'` for a nominal full-transition video. Add `--seated-baseline` and the original seated checkpoint to compare seated stability. Omit `--checkpoint` to evaluate the scripted full-transition reference.

[Exact commands](docs/commands.md), logs, rollout arrays, plots and source snapshots are retained. Selected checkpoint: `IsaacLab/logs/rsl_rl/g1_full_transition/2026-10-09_10-08-57_seed_7/model_96.pt`; selection used lowest measured impact, not better success. All three new checkpoints contain verified finite actor/critic updates and were loaded for evaluation.

Next work should expand this validated initial-state envelope conservatively before varying furniture. Approach, perception, compliant couches, Cosmos and real-robot deployment remain future stages. No longer training run is needed for the current bounded milestone.

Isaac Lab commit `be9551260b269d4d8a964b71d5849810a1237b33`; Python 3.12.15; Isaac Sim 6.1.0.0; torch 2.11.0+cu130; RSL-RL 5.5.1. PhysX/training and renderer use RTX 4090 GPU 0; RTX 4070 remains the desktop GPU. Approximately 49 GiB free after preserving the baseline and new experiments. Do not rerun setup.

[Previous milestone](docs/status-before-full-transition.md) · [Resolved packages](logs/resolved-packages.txt)
