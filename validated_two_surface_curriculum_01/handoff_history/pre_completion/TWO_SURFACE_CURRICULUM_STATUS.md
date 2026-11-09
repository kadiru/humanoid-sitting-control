# Two-surface intermediate poses validated; gated PPO running

October13,2026 (PDT). User authorized one seat and one leaning surface, validated
intermediate poses, PPO progressively toward standing, and baseline comparisons.
Work remains in this chat through Mac SSH. No other coding agent was launched.

New root: /data/humanoid-sitting-perception/two_surface_curriculum_01.
Persistent tmux: two_surface_curriculum_01. Study: ppo_01.
Protocol: docs/TWO_SURFACE_CURRICULUM_PROTOCOL.md and ppo_01/protocol.json.

## Completed validation

- Selected the existing40cm level seat and45degree leaning surface (old cases3/7).
- Preserved114 coarse and105 refined candidates. Constructed seven stages per
  surface using joint/foot kinematics and body-pitch corrections. Selection is
  recorded in selected_path_01/selection.json before training.
- All14 selected poses passed an independent restored20second test with100%
  stationary qualifying ticks and no physical failures. Both final endpoints
  qualify as supported rest. Unsupported intermediate poses are not sit successes.
- Fixed commands retain all14 poses without physical failures;7/14 are already
  supported and satisfy the transition endpoint goal without moving.
- Reference-only motion passes9/14 starts. Sitting starts0/1 physically fail;
  leaning starts0/1/4 fail supported-rest qualification. Standing transitions
  are not validated by the static-pose result.
- Four interpolation/control tests pass. Indexed reset isolation is exact and
  does not advance physics. Fixed/reference root,q,command,geometry and all405
  observation channels match bitwise at startup onGPU0.
- Two-update smoke:2048 active transitions, finite changed actor and critic,
  initial actor output exactly zero, correct final weights loaded; both endpoints
  pass the full20second deterministic post-smoke evaluation.
- RSL continuation iteration metadata was corrected before main training: load
  the completed iteration and increment before the next block. The simulation,
  control, observation and reward definitions did not change after preflight.

## Running experiment

Two fresh seeds54911/54917 onGPUs0/1,16environments each. PPO learns bounded
residual corrections around an offline joint-command reference. The reference
stays outside trainable layers; this is reference-based control, not an
unassisted general sitting policy. Runtime uses sensors, requested commands and
control time, not live simulator geometry/root translation/contact labels.
Actor/critic both405channels;23bounded body actions; effort covers all37 joints.

Stages6→0 go from supported to standing. Each stage receives32updates and a
fresh full20second deterministic test on both surfaces. Advance only on2/2.
One additional32-update block is allowed on failure; after a second failure the
seed stops advancing. Maximum448updates/458752 transitions perseed;917504 total,
including indexed-reset warmup. The2048-transition smoke is separate.

First optimizer snapshot: updates19/18, recorded transitions19456/18432,
active19056/18032 plus400warmup each. Each has10/16 successful noisy completed
training episodes. These are not deterministic evaluation results or proof of
standing transitions. Read worker_54911.json /worker_54917.json for current gates.

Final evaluations compare constant commands, reference-only motion and each
last trained policy on all14 starting conditions. The last stage-passing policy
is also evaluated if different. Compare physical failures first, then support
success, whole20s/final10s/final5s total squared and absolute torque, RMS and
leg/upper contributions on common full-duration successes. No policy is promoted.

Verify all source/bank and prior-study hashes, actual budgets, exact loaded
weights, startup matching,20s horizons and command bounds. Preserve failed stages.
Sensor audits must distinguish availability from accuracy and include individual
traction/slip/motion failures. This is a two-surface development curriculum with
repeated conditions, not unseen-surface generalization. Torque is not energy.

Mac preflight figure:
validated_two_surface_curriculum_01/preflight/preflight_visuals_01/validated_waypoints.png.
Independent audit: preflight_verification_01.json at the remote study root.
Prior resting, effort and transition studies remain complete and unchanged.
