# Supported-sitting RL: latest research state

Status: the independent-reset PPO pilot is complete. The new policy improves the
known regression bank from 15/16 to 16/16 full-attempt successes, but remains at
5/16 on a fresh, wider seat bank. Its successful fresh-bank sits use more leg
torque than the previous PPO. No candidate replaces the existing default.

The user has explicitly resumed reinforcement learning. The original handoffs'
PPO pause is historical. Continue as the primary agent through Mac SSH to
`kfuUbuntu`; never launch another coding agent on Ubuntu. Preserve old code,
datasets, checkpoints and environments. Use tmux for long jobs and `/data` for
large outputs. No software was installed for this work.

## What is now implemented

- `scripts/train_support_rl_independent.py`: each training robot resets on its
  own failure or 20-second horizon. Evaluation always continues for the full
  20 seconds, with failures latched. Refused and unfinished attempts are recorded.
- `support_sampling.py`: isolated motion/contact recurrent-state resets, fresh
  delayed sensor buffers and torque filters. Other robots retain their state.
- Reset robots hold their joint targets for one ordinary control tick before
  measuring startup depth. This fixes stale simulator camera transforms without
  adding global physics steps or changing neighbors' controller progression.
- Wider balanced training prototypes: height 26–35 cm, depth 40–64 cm, width
  78–114 cm, seat x -44 to -36 cm, lateral offset +/-4 cm and yaw +/-8 degrees.
  These are rigid chair, bench, box/platform and couch shapes. Geometry remains
  fixed within a run; robot starts vary independently at every attempt.
- A sensor-only 3D figure combines camera points, encoder forward kinematics,
  gravity, the detected floor and the visible sitting patch in one frame.

The actor and critic still receive the same 144 sensor-derived channels. There
are no true robot/seat poses, linear velocity, COM, physical contact labels or
furniture parameters in these inputs. Simulator truth is used for reward,
termination and evaluation only. Foot forces remain labels rather than assumed
hardware sensor inputs. The existing 20 ms proprioceptive delay/noise and 10 Hz
live-depth schedule remain. Invalid geometry is withheld, never replaced by truth.

The reward and action limits are unchanged: stable support, torso uprightness,
quiet posture, seated leg torque economy and load sharing; penalties for impact,
fast descent, effort and abrupt commands. Low torque earns the seated reward only
with actual stable seat support. Twenty-three bounded joint corrections act over
the frozen sensor05/contact02 scaffold. This is still residual PPO, not a VLA,
new DAgger collection, a learned approach skill, or a fully independent motion policy.

## Validated result

All evaluations are full 20-second attempts, with a standing check at 1.52 seconds,
three continuous seconds of stable supported sitting, and no latched failure.
The impact threshold is two robot body weights (about 632.53 N), a project metric,
not a calibrated hardware safety limit. Each bank contains eight geometries and
two starts each. These small samples do not establish statistical superiority.

| Bank | Frozen controller | Previous PPO | New PPO | Previous / new worst impact |
|---|---:|---:|---:|---:|
| Fresh wider seats | 5/16 | 5/16 | 5/16 | 1968.3 / 1948.1 N |
| Known regression seats | not rerun | 15/16 | 16/16 | 766.5 / 570.1 N |

The new PPO recovers regression case 15 and loses no successes in either bank.
On the wider bank all three controllers fail the same eleven cases. The new
candidate nevertheless has two falls and six wrong-contact flags, versus one fall
and five wrong-contact flags for the previous PPO; failure categories can overlap.
It has ten impact violations there. Thus unchanged success count does not mean
all failure behavior is unchanged or improved.

On the five matched successful fresh attempts, leg torque rises from 4.485 to
5.111 Nm (+13.97%) versus the previous PPO; it is also 4.72% above the frozen
controller. On the 15 matched regression successes it falls from 4.127 to
4.022 Nm (-2.54%). These are conditional comparisons; failed attempts are reported
separately and never used to advertise a torque saving.

Across the five valid evaluation runs, 14,006 of 16,000 recorded depth case-frames
are accepted. Maximum accepted clearance error is 1.1311 cm; zero accepted errors
exceed 2 cm. These are ideal simulated depth results, not hardware calibration.
The common-frame figure uses sensor inputs and static robot calibration only.
The video uses simulator body poses/forces for evaluation and labels that fact.

## Training and reset validation

The third main pilot continues `support_rl_diverse_train_02/final.pt` for exactly
98,304 additional transitions: 16 environments, 96 updates, 64 steps per update.
The training geometry/start seeds are 51511/51513. Actor and critic initial state
dicts exactly match the parent. Final actor and critic tensors are finite; actor
weights change by up to 0.007134 and belong to the optimizer.

It completes 160 attempts in 713.33 seconds: 25 successes and 10 startup refusals.
Training failures terminate early, so these are not held-out scores. Completed
attempts contain 93,378 steps and unfinished attempts contain 4,926, exactly
98,304 total. Every individual robot also accounts for exactly 6,144 steps.

Twelve targeted reward/admission/reset tests pass. The final simulator reset probe
shows zero changes in 24 measured state/history fields of non-reset robots, no
extra physics tick, fresh joint readings, a controller matching cold initialization
at the same GPU batch size, and a horizon ending only the selected environment.
Startup height is checked against evaluation-only truth after the ordinary hold tick.

The new policy uses little of its existing action range. On the fresh rollout,
95% of absolute corrections are below 0.2025 degrees and the maximum is 0.5690
degrees; none reaches the +/-0.08 rad bound. This run does not establish that
increasing joint limits would solve the failures. Exploration and the ability
to learn a coherent lowering motion are the next hypotheses to test.

## Camera-reset bug and preserved diagnostics

The first no-warmup implementation passed state isolation, but its startup camera
still observed stale transforms: seat estimates stayed near the default 30 cm.
The resulting `fresh_zero_03` and `fresh_prior_03` runs are preserved as debugging
evidence and excluded from the valid comparison. No main PPO training ran with
that camera bug. `support_rl_independent_interruption_03.json` records the stop;
the shell EXIT trap returned zero on termination, which was not completion.

Paired camera probes reduce maximum startup height error from 49.55 to 5.39 mm
and increase usable live samples from 6/32 to 27/32 after the one-tick hold. The
corrected experiment uses a fresh holdout seed, 51541/51543. The earlier diagnostic
bank used 51521/51523. Regression seeds 51403/51407 reuse a known pilot-2 bank.
All new comparisons share the new startup protocol; historical pilot-2 scores
are not silently reused as their baseline.

Probe 02 also preserved an earlier test failure caused by comparing different GPU
batch sizes. The final cold-controller check uses the same batch size at unchanged
tight tolerances. Probe smoke horizons are two seconds, not sitting-success scores.

## Artifacts and reproduction

Ubuntu project: `/home/kadir/humanoid-sitting`.
Large outputs: `/data/humanoid-sitting-perception`.

- Training: `support_rl_independent_train_04/final.pt`.
- Evaluations: `support_rl_independent_{fresh_zero,fresh_prior,fresh_candidate,regression_prior,regression_candidate}_04`.
- Final report/video: `support_rl_independent_results_05` (project `runs/` symlink).
- Numerical source, banks, arguments and archived hashes accompany each run.
- `scripts/run_support_rl_independent_05.sh` completed all fixed simulations.
- `scripts/finalize_support_rl_independent_05.sh` renders the final presentation.
- `docs/SUPPORT_RL_INDEPENDENT_PROTOCOL.md` describes the complete experiment.
- `docs/SUPPORT_RL_PILOTS_1_2_ARCHIVE.md` preserves the previous latest handoff.

Report version 05 has identical numerical results to 04. Its video explicitly
shows the full-trial peak separately from the 10 Hz force curve, which can miss
brief impact spikes. The full-trial peak uses all 50 Hz evaluation checks. The
20-second H.264 reconstruction shows a failed and a successful fresh case at
2x playback. It is not an RGB camera movie or a depiction of privileged inputs.

The preservation audit verifies all 117 original source files, five archived
baseline files and three known original checkpoints, and a clean IsaacLab checkout.
No original source, dataset or checkpoint was deleted or replaced. The new report
also verifies archived numerical sources, paired seeds/banks, full horizons,
checkpoint continuity and transition accounting. The original shared-frame code
remains unchanged. The small CPU startup-fit profile measures 21.7 ms/image on
recorded 320x240 inputs; it excludes rendering, physics, live depth and PPO.

## Next research gate

The evidence supports keeping the reset/camera infrastructure and retaining the
new checkpoint as an experiment. It does not support arbitrary-surface sitting
or promoting the new PPO over the default controller.

Next test a bounded learned lowering-speed action and more structured, temporally
smooth exploration, with a staged mixture of easier and harder seat configurations.
Keep meaningful limits and the same full-trial reliability criterion. Do not simply
raise joint limits: this policy did not approach them. First validate the new action
semantics in a small run, then use a larger training budget if behavior is sound.
Treat every bank analyzed here as a known regression set for future model choices;
reserve a new untouched holdout. Compare reliability and all-trial impact before
conditional torque savings. Deformable cushions and finding/approaching a seat
remain later stages; physical hardware sensing and control are not validated.
