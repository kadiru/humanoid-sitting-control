# Supported-sitting RL: latest research state

Status: the learned-pace PPO pilot is complete and rejected for promotion. It regresses on all three paired evaluation banks. Preserve its code and checkpoint as an experiment; retain the previous policy and the existing deployment default. No training job needs to keep running.

The user has authorized reinforcement-learning research. The original handoff pause is historical. Work as the primary agent through Mac SSH to `kfuUbuntu`, project `/home/kadir/humanoid-sitting`; never start another coding agent on Ubuntu. Use tmux for long jobs and `/data/humanoid-sitting-perception` for large outputs. Preserve all code, datasets and checkpoints; no software was installed.

## Validated outcome

Each bank contains eight rigid geometries and two starts each. All evaluations last the full 20 seconds, require standing at 1.52 seconds and at least three continuous seconds of stable seat support, and latch failures through the entire attempt. Simulator truth is used only for reward/failure labels and offline evaluation. Actor and critic remain sensor-only.

| Bank | Previous PPO | New PPO + pace | Previous worst impact | New worst impact |
|---|---:|---:|---:|---:|
| Known easier | 16/16 | 8/16 | 570.1 N | 781.2 N |
| Known wider | 5/16 | 2/16 | 1948.1 N | 3033.5 N |
| New wider holdout | 11/16 | 7/16 | 911.4 N | 999.2 N |

The candidate gains no successes and loses 8, 3 and 4 cases respectively. Conditional leg torque changes are +5.86%, +4.30% and -10.47% on 8, 2 and 7 matched successes. The fresh torque reduction does not compensate for lost reliability. The impact threshold is two simulated robot body weights, about 632.53 N, a project criterion rather than a hardware-certified limit.

Across 19,200 main-evaluation case-frames, 15,959 clearance estimates are accepted; the largest accepted error is 1.14393 cm and none exceeds 2 cm. Rejected observations remain explicit. This does not establish hardware accuracy or rule out effects of missing observations. The two known baseline runs reproduce every one of the 12 shared recorded arrays bit for bit, including body poses, forces, observations and corrections. The new neutral interface preserves the old behavior.

## Diagnosis and next research gate

Turning off the runtime pace adjustment with exactly the same final actor/critic weights gives 7/16 on the known easier bank, versus 8/16 with pace enabled and 16/16 for the parent. It recovers no lost case and loses case 4. The changed learned policy therefore remains degraded without the pace action; this test does not identify every training cause. This is a known-set diagnostic, not a fresh holdout.

Six lost easier cases achieved 3.9–6.3 seconds of stable support before failing around 16.9–18.8 seconds. Other losses include instability before contact. Thus reaching the seat or briefly sitting is insufficient: stable support must persist. `case_diagnostics.csv` records support/failure timing and pelvis-center placement; its geometry is evaluation truth and its times/velocity are sampled at 10 Hz.

An isolated test exposes a concrete action-distribution mismatch: with the same constant zero Gaussian mean and standard deviation 0.5, sampled actions yield average filtered pace 0.863233, while deterministic mean actions yield 1.0. The one-sided clipping only permits slowing, so exploration is biased toward slower motion. This is an actuator test, not another robot rollout; actual training means vary. It is a plausible contributor to the regression, not a proven sole cause.

Do not scale this training recipe or keep tuning the candidate. The next gate is to remove or explicitly handle the stochastic/deterministic action mismatch, ensure the learned joint policy sees deployment-like trajectories during training, and preserve the known successful sitting behavior with an explicit regression/retention check. Start from the preserved previous policy, not the regressed candidate. Validate action semantics and balanced, sustained support on known cases before using a new holdout. A further controlled study is needed to distinguish exploration, curriculum and longer-training effects.

## Implementation and training

- `support_pace.py`: actions-mode frozen scaffold with a per-robot recurrent progression setting applied before decoding. Contact inference and ankle feedback stay at 50 Hz. This setting is not guaranteed physical vertical speed.
- `train_support_rl_pace.py`: 24 PPO outputs: the original 23 joint corrections and one pace request. Actor/critic inputs are the previous 144 sensor-derived channels plus the realized pace-filter state.
- Pace target is `clip(1 + .75*clip(action,-1,1), .25, 1)`, filtered with gain .1 and maximum change .04 per 20 ms tick. It changes the next nominal target, one ordinary tick later. Independent raw Gaussian samples retain ordinary PPO log probabilities; there is no hidden correlated-noise process.
- Original +/-0.08 rad joint corrections, +/-0.05 rad total target slew, physical limits and reward are unchanged. No simulated foot-force channel enters the policy.
- Parent actor/critic weights and old normalizer channels are preserved when expanding 144/23 to 145/24. New input weights start at zero, pace output mean at zero and std at .5. Expansion uses a fresh optimizer; stage 2 resumes the complete 24-action checkpoint.
- `train_support_rl_pace_validated.py`: corrected mixed training placements and optional first-tick setup validation. The original files remain unchanged.

Accepted main training is exactly 196,608 additional transitions: 64 updates x 64 steps x 16 environments on easier prototypes, then 128 x 64 x 16 on a mixed bank. The stages complete 69 attempts (29 successes) and 201 attempts (17 successes), respectively. These exploratory training rates are not evaluation scores. Completed plus unfinished steps account for every transition in every environment. Final actor and critic are finite and the new pace head receives nonzero updates. Final checkpoint only; no checkpoint selection from holdout scores.

Parent: `/data/humanoid-sitting-perception/support_rl_independent_train_04/final.pt`
Parent SHA256: `2c0f6907196be68bce9a1ee2e6b6ef6e4c82bf0ecdff622dc55b81fe8ef58d13`
Experimental candidate: `/data/humanoid-sitting-perception/support_rl_pace_mixed_train_02/final.pt`
Candidate SHA256: `b21fd9d2db493865ab49843cf427223d8d1baf505acdee1d3f346c18da72b873`

## Preserved interruptions and checks

The first mixed-stage run (`support_rl_pace_mixed_train_01`) was interrupted because a generated deep box caused repeated first-tick wrong-body contacts before any sit action. Its files and `interruption.json` are preserved. Its partial checkpoint is not used. No holdout had been evaluated when this repair was made.

The corrected mixture uses eight newly generated easier geometries plus the first eight entries, in original order, from the prior TRAINING bank `support_rl_independent_train_04/geometry_bank.json`. They were not ranked by success and are not evaluation geometries. Exact bank, indices and source hash are archived. A 192-start physical setup probe found no first-tick physical failures; 187 starts passed camera admission and five were explicitly refused (all from the same prototype). Every prototype had accepted starts. Pre-reset failure records are also checked, so an automatic reset cannot hide initial collisions.

Sixteen unique reward/admission/reset/pace unit checks pass; discovery executes 20 tests because four admission checks are imported twice. The simulator cold-state and pace probes pass. Other robots retain their state/history on an indexed reset; the selected robot holds one ordinary tick for fresh camera data.

The preservation audit verifies all 117 original source files, five archived baseline files, three original checkpoints and a clean IsaacLab checkout. The legacy script subsequently fails its root-disk 30 GiB reserve assertion: root had about 29.77 GiB free. This warning is retained, not erased. `/data`, the actual report/video destination, had about 83 GiB free and passes the separate output-volume check. No existing files were deleted to free space.

## Artifacts and reproduction

- Accepted runs: `support_rl_pace_easy_train_01`, `support_rl_pace_mixed_train_02`.
- Main evaluations: `support_rl_pace_{easy,wide,fresh}_{prior,candidate}_01`.
- Diagnostic: `support_rl_pace_easy_neutral_01` (16 additional full attempts; 112 full attempts total including the main suite).
- Report: `/data/humanoid-sitting-perception/support_rl_pace_results_01`. Includes numerical/source audits, plots, a pose/force reconstruction, CSV case diagnostics, and `pace_ablation/` and `pace_distribution/` diagnostics.
- Preferred video: `sitting_comparison_full_view.mp4`; the original tighter framing is retained as `sitting_comparison.mp4`. Visualization poses/forces are evaluation-only truth, and force curves are 10 Hz samples with separate full-trial peaks.
- Launchers: `run_support_rl_pace_01.sh` (stopped), `run_support_rl_pace_02.sh` (completed), `run_support_rl_pace_ablation_01.sh` (completed).
- Finalization: `finalize_support_rl_pace_03.sh` (completed); earlier finalizers retain their interruption/storage-guard exits. Full-view video rendering has its own source and exit marker.
- Mac delivery: `/Users/kadir/humanoid-sitting-control/validated_support_rl_pace`.
- Protocol and amendments: `docs/SUPPORT_RL_PACE_PROTOCOL.md`; original protocol preserved separately. Previous completed handoff: `docs/SUPPORT_RL_PILOT_3_ARCHIVE.md`; pilots 1/2 remain in their archive.

No default controller changed. These are initially aligned rigid chair/bench/box/couch geometries; there is no navigation, deformable cushion model or hardware deployment. Shared-frame perception code and all earlier data remain preserved.
