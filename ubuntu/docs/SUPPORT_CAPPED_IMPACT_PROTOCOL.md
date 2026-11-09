# Capped-impact-only PPO comparison

Authorized October 11, 2026. Both reward arms start from the preserved 47/64
starting PPO: support_split_ppo_train_01/final.pt, SHA256
ece102b139323af18bd1c7aad6d8fb06e478f38946f07e52406a31bd4acc7dda.
Actor, critic and normalizer values are copied exactly; optimizers are fresh.

Control: original reward. Experimental: replace ONLY the original impact term
with -5 times each increase in the episode maximum of
min(max(force/bodyweight - 1, 0)^2, 1). Entire episode impact cost is bounded
between -5 and 0. Original effort, support, approach, smoothness and failure
terms remain unchanged. No phase-action, critic, architecture or gain change.

Each arm gets 32 updates x 64 steps x 16 robots = 32,768 transitions for each of
two training seeds: 52813 and 52817. Total 131,072 transitions in four runs.
Same mixed 16-seat training bank, geometry seed 51621, 3e-5 learning rate,
3 learning epochs, 4 minibatches, zero entropy/parent-KL penalties, frozen
normalizers. Actor and critic remain sensor-only. Current tracking is frozen.

The supported-hold timer uses int64 ticks, with 150 consecutive good ticks
required at 50 Hz. Broken float scores are retained as diagnostic fields.
Per-robot timer and impact history reset independently. Timer changes only
evaluation/logged success, not observations, commands, reset conditions or
the dense supported-hold reward.

Before training, six unit tests must pass and the prior new-reward policy is
re-evaluated on its novel bank (52741/52743). All command, sensor and trajectory
arrays must match the preserved run bitwise. Only its known three-second
rounding case (13) should change success: 9/16 -> 10/16. The policy under this
test is NOT the initialization for the new training runs.

Evaluate all four FINAL checkpoints, with no checkpoint or seed selection, on:

- Known bank 51403/51407 (legacy shape profile).
- Previously fresh bank 52641/52643.
- Mixed training geometries with new starts 52623.
- Previously novel bank 52741/52743.
- New broad holdout 52841/52843, declared before training.

The first four are now known regression sets; the fifth is fresh. Every attempt
runs 20 seconds with failures latched. Each policy receives the same 80 cases.
Starting-policy results on the first four banks are reused and transparently
rescored by integer ticks. Evaluate the starting policy on the new holdout.
Assert geometry and startup sensor equality in every comparison. Keep refused
starts in primary scores; any admitted-only metric is secondary.

Report results separately for each training seed and bank, pooled paired gains/
losses, full-case force statistics and failures, and effort on common successes.
Do not present the two seeds' 160 outcomes as 160 unique placements. Training
returns use different objectives and are not policy-quality comparisons.
Confirm budgets, finite weights, initial parity, normalization and source hashes.

Persistent session: support_capped_impact_01. New outputs only, under
/data/humanoid-sitting-perception/support_capped_impact_01. Automatic sequence:
tests, frozen source manifest, cost audit, timer regression, four training runs,
fresh baseline, 20 evaluations, comparison report. Stop and preserve files on
error. No automatic promotion. Expected duration about two to two-and-a-half
hours using observed simulation speeds. No software installation.
