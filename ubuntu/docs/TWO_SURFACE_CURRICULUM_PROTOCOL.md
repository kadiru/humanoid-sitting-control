# Two-surface validated-waypoint curriculum

Authorized October13,2026: construct intermediate poses for one seat and one
leaning surface, train PPO progressively toward standing starts, and compare
stability and joint effort with the existing baselines. All earlier studies
remain preserved. New root: /data/humanoid-sitting-perception/two_surface_curriculum_01.

## Scope and prerequisites

Use the existing40cm level seat and45degree leaning surface (old training cases3
and7). Two candidate searches produced14 selected poses, seven per surface.
Every selected state passed an independent restored20second stationary test,
including unchanged contact, impact, joint-limit and torque criteria; final poses
also qualify as supported rest. Unsupported intermediate poses are NOT sitting
successes. Selection predates all policy training.

A frozen offline joint-command path connects the poses, with cubic interpolation
and2seconds per segment. Zero residual follows this path. The initial development
reference succeeds from9/14 starts, with failures from standing. This is a
reference-based residual policy, not unassisted general control. The reference
has no live access to true surface or robot pose. Runtime inputs are joint/torque
and IMU equivalents, measured depth geometry with kinematic tracking, the
requested reference joint commands and the control clock. Actor and critic both
receive405 channels (three histories of111 sensor channels plus23 reference
commands and normalized phase). No current privileged contact, root translation,
true geometry or pose-bank reset state is fed to the actor/critic.

Control23 body joints with absolute bounded residuals: ±.01rad hip/ankle roll,
±.05rad shoulder/elbow and±.025rad other controlled joints. Reference remains
outside trainable layers. Targets use tanh, .1 smoothing, .002rad/tick slew and
.03rad physical command margin. The14 uncontrolled commands remain fixed at the
initial values. All37 joint torques contribute to effort. Gains/limits unchanged.

Four pure control tests, reset isolation, constant-command retention, a2-update
optimizer smoke and deterministic post-smoke endpoint retention must pass before
the main study. A GPU0 reference replay provides a matched baseline. Training
does not require the unlearned reference to reach support from standing.

## Reward and evaluation

At20ms: positive reward combines2*good*(.75+.25/(1+total_squared_torque/B)),
.5*exp(-20*mean(joint_error_to_current_waypoint²)) and
.25*exp(-10*root_speed²-2*angular_speed²), with .02*mean((command_delta/.002)²)
subtracted. All terms are multiplied by dt; first physical failure costs5.
Positive tracking/balance terms are zero after physical failure. Warmup reward
is zero. B is a fixed reward scale1500Nm² sitting /280Nm² leaning; it is not a
measured energy model or a reported comparison denominator. Root velocity and
pose-bank joint targets are reward labels only.

Success: full1000 control ticks (20seconds), no latched physical failure, final
5seconds continuously qualifying support. Motion before contact is not required
to satisfy a seated-rest gate. Physical failure definitions and contact labels
are imported unchanged. Log individual legal-contact, traction, slip, body-speed
and tilt diagnostics rather than only their combined qualification.

Compare fixed commands, the frozen reference with zero residual, and learned
residuals from exactly matched starts. Fixed standing is a stability baseline,
not a successful sit. Fixed supported endpoints are a resting baseline, not a
matched transition-energy baseline. Report whole20s, final10s and final5s total
squared torque, absolute torque, instantaneous37-joint RMS averaged over time,
and leg/upper contributions. Effort claims require common full-duration successes.

## Bounded adaptive curriculum

Fresh seeds54911 /54917, GPUs0 /1,16environments each (repeated two surfaces),
64steps/update. Stages descend6,5,4,3,2,1,0;6 is already supported and0 standing.
Each stage receives32updates, followed by a fresh deterministic20second check on
both surfaces at that starting stage. Advance only on2/2 successes. If it fails,
continue another32updates at the same stage and check again. After a second
failure stop that seed's curriculum; do not force advancement or extend budgets.
Each32-update block is32768 transitions including indexed-reset warmup.
Maximum448updates and458752 transitions per seed;917504 total. Actual totals
depend on the recorded gates. The separate2048-transition smoke and evaluations
are excluded. Active/warmup counts are explicit. No learning-rate/reward tuning
or checkpoint change based on the final comparison.

128x128ELU actor/critic, zero actor output initialization, std.1, learning rate
3e-5 adaptiveKL .003, entropy.0001,3epochs,4minibatches,gamma.998,lambda.97,
no observation normalization. Later blocks resume full optimizer and weights.

Record every stage/block/checkpoint and gate. Final evaluation on all14 starting
conditions is performed for each seed's last trained checkpoint; if different,
also evaluate the last checkpoint that passed a stage. Report both, with no
automatic promotion. All gates use these same two development surfaces; none is
an unseen-surface generalization test. Baselines and final tests run onGPU0.

## Integrity and reporting

Freeze new runtime sources, bank and protocol before main training. Preserve and
check earlier study hashes. Verify changed finite actor/critic, cumulative
budgets, correctly loaded final policies, exact startup arrays,1000tick horizons,
command bounds/constant uncontrolled commands and unchanged geometry. Measure
sensor errors and availability before physical failure. Retain every failed
candidate and original development/reference run. Run jobs persistently in tmux.
No second coding agent. Report sitting/leaning and start-stage outcomes separately.
Two seeds on two finite rigid surfaces cannot establish arbitrary-surface or
hardware capability. Torque is an effort proxy, not electrical energy.
