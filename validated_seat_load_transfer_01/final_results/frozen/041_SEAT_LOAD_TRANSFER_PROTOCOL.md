# Seat load transfer: bounded standing-start PPO pilot

This new study is separate from all earlier frozen studies. Root:
`/data/humanoid-sitting-perception/seat_load_transfer_01`.

The user requested a posture that puts weight on the seat without unwanted
contacts, followed by PPO learning to reach it from standing and comparison
with the current reference and current policies. All execution is from the
Mac through SSH. No other coding agent is used and no policy is promoted.

## Evidence before the pilot

The hip-roll housing is the unwanted-contact body. Existing crouch commands
retain leg preload. A new upright target with zero leg preload holds across
38/40/42 cm and -5/0/+5 degree seats. Eighteen independent 20-second seated
replays pass every stability tick with no physical failure. Final-five-second
vertical seat load is 73.2–74.2% bodyweight, total squared torque 91–96 Nm².
These are seated starts, not a successful standing motion. Verification and
the correction of an initial normal/vertical-load reporting label are saved
under `rest_verified_02` and `reporting_recovery_01.json`.

The selected development approach uses eight foot-preserving IK knots from
10–14 seconds and retains 30% of the old leg preload at the end. Its first
nine-surface diagnostic passes only 2/9 transitions, with one hip-housing
failure; the other failures primarily miss the conservative friction margin.
That is an explicitly difficult learning baseline, not a validated solution.
The current reference remains the stronger stability benchmark. No success
threshold or contact classification is relaxed to admit this pilot.

## Frozen pilot definition

- Two fresh policy seeds: 56811 and 56817. Zero initial residual mean. They do
  not inherit the older two-action policy into the new 23-action network.
- Each: 64 updates, 32 decisions/update, 5 physics ticks/decision, 18 envs.
  184,320 physics transitions per seed; 368,640 total, including indexed-reset
  warmup. Initial 25-tick global settling is separate (450 ticks per run).
- First 8 updates are critic-only; remaining 56 update actor and critic.
- All episode resets start standing. No supported-reset curriculum is used.
- 406 runtime channels: existing 405 sensor/reference channels plus clock.
  The actor and critic see the same channels. The reset learning mask is a
  separate storage key, excluded from both network input groups.
- Bounded residuals on 23 body joints, held 100 ms. Fourteen finger commands
  stay fixed. Absolute limits, command filtering and slew caps stay enforced.
  Exact per-joint spans are recorded in `action_contract.json`.
- Actor/critic MLPs: 64,64; no observation normalization; Gaussian initial
  standard deviation .25; fixed Adam learning rate 1e-4; four epochs/four
  minibatches; entropy coefficient 0; gamma .9975; lambda .98. Remaining
  installed runner defaults are captured in each run's serialized config.
- Active-only PPO excludes reset warmup from surrogate, KL, advantage
  normalization and value loss. Critic-only updates leave actor gradients
  `None`, preserving actor weights and Adam state. Regression results and
  the inference-tensor compatibility repair are preserved.

The reward blends from approach tracking/balance to resting between 10 and
14 seconds. Resting reward per second is
`3*good + 2*good*clip(vertical_seat_load/.65,0,1)
 - good*clip(sum_37_joint_torque_Nm_squared/150,0,4)
 - 2*invalid - clip((friction_utilization-.6)/.4,0,2)`.
It retains a small command-slew cost and a -10 first-physical-failure cost.
Episodes are finite20-second trials with no continuing-value bootstrap.
Simulation truth is used for reset construction, reward and evaluation only.
Runtime geometric plans use measured depth planes, encoders and IMU, plus an
explicit offline motion library and known horizontal placement assistance.

## Evaluation and interpretation

Development contains nine surfaces, each repeated at two placements. Eight
additional sitting surfaces were reserved before target tuning: 38.5/41.5cm
at +/-2.5 degrees and 39.5/40.5cm at +/-7.5 degrees. They are repeated twice,
never used for target tuning, reward calibration, training or model selection.
Withheld means withheld from this policy, not necessarily new to the project.

Both final full-budget checkpoints must finish before holdout evaluation.
There is no checkpoint selection or automatic promotion. Compare full20-second
standing trials against the current repaired reference, current policies56211
and56217 (each with frozen parent55211), and the new zero-residual reference.
Existing development baselines may be reused only after bank, source and
startup parity checks. Holdout baselines are freshly evaluated.

Report physical failures and ordinary stable success first. Separately report
the stronger target: full stable success, mean final5s vertical seat load>=.5BW,
and final5s total squared torque<=35% of the matched current reference.
Compare whole20s/final10s/final5s squared torque, absolute torque, instantaneous
joint RMS averaged over time, leg/upper contributions and load distribution on
common full-duration successes only. Failed fragments cannot support savings
claims. Audit measured geometry availability and errors before failures.

Verify preserved hashes, finite changed networks, full budgets, optimizer
continuity, actual final checkpoint loading, full horizons, matched starts,
constant fingers, bounded commands and the holdout barrier. Report any
cross-device numerical tolerance explicitly. Two seeds and finite rigid slabs
do not establish arbitrary furniture generalization. Torque is an effort proxy,
not electrical energy or evidence of a global optimum.
