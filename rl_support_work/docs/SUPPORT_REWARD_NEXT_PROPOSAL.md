# Proposed next PPO comparison — not launched

Question: does bounding the peak-impact penalty improve landing without losing
successful sitting? Do not change effort reward in the same experiment.

Initialize each arm from support_split_ppo_train_01/final.pt, the47/64 starting
policy. Preserve all current checkpoints. Use the validated151-input interface,
camera timing, rigid-seat bank, motor limits and nominal scaffold unchanged.

Use two objectives:

1. Existing reward control.
2. Replace only its time-integrated impact term with the incremental episode cost
   `-5 * delta(max_so_far(min(max(force/bodyweight - 1, 0)^2, 1)))`.

The complete episode impact cost is bounded in[-5,0]. Keep the existing separate
failure penalty and physical failure thresholds. Leave the old leg-economy term
unchanged. The5-unit cap is a testable scale choice, not an optimum inferred from
the current study. Check cost monotonicity, per-robot resets and timestep
independence before training. Inspect counterfactual costs on archived training
episodes, without treating that calculation as a policy-performance result.

Train32 updates per arm, with two predeclared training seeds: four runs,
131,072 transitions total. Fresh optimizers, frozen normalization and inherited
PPO settings shared between arms. Evaluate all final checkpoints; do not choose
one favorable seed. Log reward component sums, value loss and policy change.
Do not compare raw return values between unlike objectives as success metrics.

Fix the float-duration scoring defect with integer supported-hold ticks in all
prospective arms. Existing baselines can be transparently rescored from archived
durations, retaining the original scores alongside them.

The four existing evaluation banks are now known regression sets, since their
outcomes informed the diagnosis. Use them to detect lost capabilities, and
predeclare an additional fresh bank before training to check generalization.
Keep all attempts, including refused starts, in the primary physical scores.
Use common successful cases for paired torque comparisons and also show success
counts, force distributions, hold durations and late-contact failures.

If this isolates a useful impact change, run an effort-only comparison next.
If it fails, investigate temporal credit and learning to control descent timing
as separate RL experiments, rather than immediately adding multiple knobs.
Earlier pace-action and KL-retention experiments are preserved and should be
consulted before repeating either idea.
