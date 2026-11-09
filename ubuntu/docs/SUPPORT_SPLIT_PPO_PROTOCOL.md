# Independent distance observations and one bounded PPO pilot

User authorization: fix the observation interface, validate it, then run a small
PPO experiment and compare it with the current controller. This supersedes the
earlier blanket training pause; old failed whole-trial validation gates remain
recorded failures and are not retrospectively reclassified.

## Frozen design before new simulation results

Keep the first 144 sensor inputs, including the complete-patch validity bit.
Append seven independently valid clearance/normal/validity/age/budget channels.
No edge estimate is invented when the observed patch cannot be transported.
Current scalar distance can remain available. Expired, malformed or unaccepted
distance has explicit missing values. All estimates remain in the current pelvis
frame; the budget is conditional on the tracker assumptions, not a probability.

Migrate actor and critic with zero new input weights, retain old normalization,
and preserve the old first-layer matrix multiplication shape for exact initial
action parity. Freeze all normalization; use fixed physical scales on new inputs.
Keep camera correction,50Hz tracking, nominal scaffold, reward, Gaussian initial
exploration, motor gains, limits and physical success criteria unchanged.

## Prospective pilot readiness checks

- Unit tests: independent validity, stale/invalid rejection, exact parent actor
  and critic migration on CPU/GPU, gradients reaching the new input weights.
- Sensor-only replay of an archived full trial batch: unchanged old geometry,
  no usable scalar distance lost because patch edges are unavailable, no expired
  estimates marked valid; labels loaded only after reconstruction for scoring.
- Closed-loop known bank51403/start51407: migrated parent retains16/16; compare
  full trajectories with archived continuous-tracking controller.
- Before the first physical failure, accepted distance error<2cm and availability
  >=99% on these replay/regression checks. Report whole-trial and landing figures
  too; correctly withheld post-fall geometry does not by itself block this pilot.
- Indexed-reset probe: selected tracker cleared, other robots unchanged, fresh
  admission after one physics tick. Camera refresh must not advance physics.

## Training and comparison

One run:32 PPO updates x64 steps x16 robots=32768 transitions, learning rate3e-5,
3 epochs,4 minibatches, fixed normalizers, fresh optimizer, no KL penalty.
Initialize from preserved support_rl_independent_train_04/final.pt. Use the
archived mixed training geometry bank from support_rl_pace_mixed_train_02 with
training starts52613. No reward sweep, checkpoint selection or extra pilot based
on evaluation outcomes. Evaluate the final checkpoint even if it regresses.

Matched20s evaluations of the migrated frozen parent and trained final policy:
known51403/51407; new geometry52641/start52643; mixed training geometry with new
starts52623.16 robots per bank,48 trials per policy. Evaluation failures stay
latched and simulations run the full horizon. No holdout enters training.

Primary: paired successes and newly lost successes. Also report impact maximum
and upper-tail forces, supported hold, seated leg torque, foot load share,
failure categories, distance/patch availability and accepted distance error.
Report whether newly appended inputs acquired weights and affect deterministic
actions; offline ablation is sensitivity evidence, not causal success evidence.

Promotion requires improved held-out success without losing old successes or
increasing severe impacts. A small run may be inconclusive or regress: preserve
it and the current controller, do not automatically launch more training.
Scope: aligned rigid simulated seats, not navigation, cushions or hardware.

Outputs go to /data/humanoid-sitting-perception with fresh suffixes; initial
budget2GiB, preserve all old sources, data, checkpoints and simulator software.
