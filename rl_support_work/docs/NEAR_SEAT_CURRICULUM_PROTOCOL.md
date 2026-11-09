# Near-seat weight transfer and progressive PPO

New isolated study: `/data/humanoid-sitting-perception/near_seat_curriculum_01`.
The user authorizes validating near-seat weight transfer without unwanted contacts or slipping, then progressively learning earlier starts and comparing stability/whole-body effort against the current reference and policies. Work is direct Mac SSH; no other coding agent. Prior studies are preserved, with 4,714 protected file hashes.

## Validated physical change

Fixed pelvis-x rotation produced shear and failed friction/slip criteria. A sensor-planned path rotating the pelvis about its inferred seat-contact point passes 18/18 full20-second near-seated trials at both six- and four-second transfer durations. No physical failure, slip-threshold, friction-margin or legal-contact violation occurs in any active physics tick. Nine development geometries (38/40/42cm × -5/0/+5 degrees), two distinct start leans (.1/.2 rad), are used here. These are genuinely different start states, not standing success. Compared with matching fixed commands, the six-second transfer reduces sampled whole-trial squared torque about20.6% and final5s22.2%, with final seat load74.9%BW. Sampling is100ms; full-tick summaries are authoritative for safety. Exact all-tick effort is recorded by the PPO harness.

The actual PPO harness separately validates two repeated placements per surface at each tested start. A deeper .35rad static candidate fails one contact case and is excluded. A bridge from the current approach's10s crouch is harder and is preserved as a failed development probe; it is not labeled a validated transition.

Runtime control accepts depth plane, joint encoders, IMU, command history and an explicit motion reference. Pelvis contact pivot is inferred from the body collision model and observed plane, not simulator contact coordinates. Existing fixed plane calibration is used. Horizontal placement/reference anchoring remains explicit assistance. No live ground-truth root pose, surface geometry, contact labels or forces enter the controller or actor/critic. Simulation truth is used for reset construction, rewards and evaluation.

## Bounded main study

Two fresh seeds57111/57117, physicalGPU0/GPU1, independent progression. Each128 new updates with32 decisions/update,5 physics ticks/decision and18 environments:368640 physics transitions per seed,737280 total including indexed-reset warmup. Blocks are[24,16,16,16,16,16,16,8]; first8 updates critic-only,120 actor-enabled. Each new process's25-tick initial settling is separate and logged. Development smoke budgets are reported separately. Checkpoint actor, critic and Adam state continue across blocks.

Start stages:5=settled.1rad lean;4=settled.2rad;3=current approach at10s;2=at8s;1=at6s;0=standing. Physical intermediate states are captured from the unchanged current reference; their logging exactly reproduces all saved reference startup/trace arrays. Stage6 is the upright rest endpoint and is not a scheduled training stage. Never substitute invalid interpolated states.

After each block, evaluate the current stage for the full20s, on18 development trials. Advance one stage earlier only if all18 have no physical failures, continuous qualifying support for final5s, zero all-tick slip/friction/legal-contact violations, and final5s mean vertical seat load>=60%BW. Otherwise stay at the same stage within the unchanged total budget. Do not force a failed policy to standing. Report achieved stages and actual standing-start training counts; the curriculum may finish before reaching standing.

Policy/critic406 channels:405 sensor/reference plus episode clock.23 bounded body-joint residuals,14 fixed fingers,100ms action duration. Same64,64 networks,Gaussian initial std.25,fixedAdam1e-4,4epochs/4minibatches,gamma.9975,lambda.98,entropy0 as prior seat-load pilot. Reset warmup is excluded from actor/critic loss, advantage normalization and KL. Disabled actor gradients remainNone. Exact contracts/configs are saved.

Reward uses motion-phase time: before10s,.5tracking+.25balance;10–14s blend to3good+2good*clip(vertical-seat-load/.65,0,1)-good*clip(sum37jointNm²/150,0,4)-2invalid-clip((friction-utilization-.6)/.4,0,2). Multiply byDT; retain command-slew penalty and-10 first physical failure. Tracking follows the new commanded reference. No continuing-value bootstrap. Existing physical criteria are unchanged. The new rolling motion is planned at the10s approach handoff, or immediately for near-seat resets, and lasts4s.

## Held-out comparison and verification

Eight surfaces were reserved before learning:38.75/41.25cm at±1.25degrees;39.25/40.75cm at±6.25degrees. Two placements per geometry. Do not train, calibrate or select checkpoints on these. Prior-study holdouts are historical, not new unseen evidence.

Both final checkpoints finish before reserved evaluation. No best-checkpoint selection or automatic promotion. Match full20s standing evaluations of current reference,current policies56211/56217 (each with parent55211),new rolling reference,and both final policies. Evaluate final policies separately from near-seat starts; do not combine those successes with standing success. Compare physical failures/stability first, then whole20s/final10s/final5s sum of squared physical Nm, absolute torque, instantaneous all-joint RMS Nm averaged over time, leg/upper contributions and load distribution on common full-duration successes only. Report geometry availability/errors before failures.

Independently check frozen/prior hashes, all update/active/warmup budgets, finite changed weights, critic-only actor identity, optimizer continuity, exact loaded checkpoints, valid reset admission/isolation, full horizons, startup parity, constant fingers and bounded commands. Gate evidence must match progression and holdout must follow the final-checkpoint barrier. Preserve execution/reporting failures with recovery manifests and fresh output directories.

Two seeds and nearby rigid slabs do not establish arbitrary-surface generalization. Torque is an effort proxy, not electrical energy or a global optimum. Reference-assisted residual PPO is not unassisted motion discovery.
