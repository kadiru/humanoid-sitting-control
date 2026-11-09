# Learned progression PPO pilot (fixed before evaluation)

Hypothesis: a coherent learned pace request can reduce hard landings missed by
small independent joint residuals. This is an experiment, not a claim that latent
progression equals physical vertical speed or that slower is always safer.

The sensor-only actor and critic receive the existing 144 channels plus the
realized pace filter state. The 24th action requests 25–100% recurrent progression:
clip(1 + 0.75 * clip(action,-1,1),0.25,1). Zero reproduces the original scaffold.
Positive actions saturate at normal speed. A first-order gain 0.1 filter and
0.04/tick slew limit smooth IID Gaussian PPO requests. The filter state is observed;
PPO log probabilities still describe the independent raw actions actually sampled.
There is no hidden correlated noise. Pace affects the next nominal target, after
one ordinary sensor/physics tick; current joint residuals affect the current target.
Contact inference and ankle feedback remain at 50 Hz. Joint residual bounds, total
joint-command slew, physical effort limits, reward and complete-attempt criterion
are unchanged. Simulator truth stays in rewards, failures and offline measurement.

Initialize from pilot 3, explicitly expanding 144/23 to 145/24. Preserve old actor
and critic weights/normalizers, disconnect the new input initially, initialize pace
mean to zero and standard deviation to 0.5. Old joint stds are preserved. Use a fresh
optimizer for the expansion; resume the whole 24-action checkpoint for stage 2.
Do not enlarge joint corrections or choose checkpoints using held-out scores.

Preflight: real-checkpoint neutral-controller equality over history, controlled
state interpolation, pace bounds/smoothing, actor/critic functional migration,
existing reward and reset tests, simulator reset/pace probe and two-update smoke.

Fixed training: 64 updates on 16 legacy-range prototypes (seeds 51611/51613), then
128 updates on 8 legacy-range and 8 wider prototypes (51621/51623), 64 steps per
environment per update. Total 196,608 new transitions. Independent failure/horizon
resets, one normal hold tick for camera freshness, original 20-second horizon.
Final checkpoint only; no stopping based on holdout results.

Paired full 20-second evaluations (8 geometries, two starts each): known easy
regression 51403/51407; known wider regression 51541/51543; untouched wider holdout
51641/51643. Compare migrated-neutral pilot 3 against the final pace candidate.
Report all-attempt success and impact first; leg torque only on matched successes.
Every prior analyzed bank is known, not a new holdout. Refusals count as failures.
No default controller replacement, navigation, deformable seat or hardware claim.

## Placement amendment, before any holdout evaluation

The first mixed-stage run was stopped after prototype 10 (a deep box close to the
robot) repeatedly caused wrong-body contact on its very first 20 ms hold tick.
It is preserved as `support_rl_pace_mixed_train_01` with `interruption.json` and is
excluded from model selection and reported main-training transitions. The completed
65,536-step easy checkpoint remains the parent; no partial mixed checkpoint is used.

Corrected mixed stage `support_rl_pace_mixed_train_02` retains the eight new easy
prototypes. Its eight wider prototypes are the first eight entries, in original
order, from the prior TRAINING bank `support_rl_independent_train_04/geometry_bank.json`.
These are neither holdout geometries nor selected by sitting success. This avoids
introducing an untested overlapping placement while preserving wider seat variation.
The source bank hash and exact indices are archived in every corrected manifest.

Before restarting the full 128-update mixed stage, run 12 randomized full resets of
all 16 robots (192 initial placements), holding for the ordinary camera-admission
tick. Require no initial physical failure and at least one accepted visual start per
prototype. These simulator-truth checks audit training setup only; runtime actor,
critic and geometry admission still receive sensor-derived inputs only.

All six paired evaluation banks, seeds, full horizons, physical limits and selection
of the final checkpoint remain as registered above. Main accepted training still
contains 196,608 transitions. Debug-run and probe transitions are recorded separately.

## Post-training diagnostics

After the known easier-set regression, one extra full 20-second evaluation was added on that same known bank, fixing the pace output to one while preserving the final actor/critic weights. No retraining or checkpoint selection occurred. A separate isolated actuator test compares sampled zero-mean Gaussian actions with deterministic mean actions; it is not a robot performance score.
