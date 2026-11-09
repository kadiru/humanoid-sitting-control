# Direct whole-body effort minimization from supported postures

October 12, 2026 (PDT). The user clarified that reducing total joint effort is
the purpose of sitting/leaning and authorized training directly for that goal.
This separate experiment preserves all previous studies. It learns posture
adjustment from supported starts, not the standing-to-support transition.

## Objective

Minimize the sum of squared applied torque, in Nm², across all 37 actuated
joints while retaining physically valid supported rest. Report summed absolute
Nm, instantaneous all-joint RMS Nm, leg/upper-body contributions and load
distribution as additional measures. Torque is an effort proxy, not a measured
electrical-energy model. Gains, torque limits, geometry and contact labels are
unchanged; the policy cannot reduce motor gains or disable joints to fake savings.

For each training geometry, B is the fixed-posture mean sum of squared torque
over the final ten seconds of its matched 20-second baseline. On each 20 ms tick:

`reward = .02 * max(0, 1/(1 + sum(tau²)/B) - .001*mean((command_delta/.002)²)) * good - 5*first_failure`

Among valid steady postures, lower total torque strictly improves reward; there
is no independent posture-matching bonus or explicit angle target. B rescales
the learning signal across conditions without changing the torque-minimizing
posture in a condition. The small command-change cost is zero at rest. A fall,
invalid contact, slip, insufficient support or excessive motion receives no
effort-saving credit. Existing supported-surface physical labels define `good`.

## Controller and observations

The remembered starting motor command is retained outside the neural network.
The policy requests an absolute offset from it, using `span*tanh(action)`.
Spans are .04 rad for hip/ankle roll, .08 rad ankle pitch, .30 rad shoulder/elbow,
and .15 rad for other controlled joints. A .1 low-pass and .002 rad per-tick slew
limit smooth commands, retaining .03 rad margin from physical joint limits.
There are 23 controlled body joints; the other 14 motor commands stay constant
and their torques remain included in the objective. Zero action holds the
reference exactly or returns toward it after an offset. Persistent action bias
cannot integrate indefinitely. There is no privileged failure-triggered command
freeze during full evaluation.

Actor and critic receive the same 402 channels: three histories of measured
joint states, measured torque, IMU gravity/gyro, previous motor command, rear
depth plane with kinematic propagation, and remembered reference command.
No simulator body translation, true surface pose, contacts, case identifier,
goal pose or reset stage is supplied to the networks. Truth remains confined
to reset construction, reward and evaluation labels. Existing depth tracking
has a stationary-feet assumption; errors and unavailable observations must be
reported during valid motion, not hidden behind its validity flag.

## Validation and development record

Four focused controller/reward tests pass. Indexed reset isolation passes.
Zero-action fixed posture passes 13/13 training and 16/16 held-out conditions,
all for 20 seconds. One held-out outline endpoint had been borderline in an
earlier endpoint validation; it stays in this test and happens to pass this
matched fixed baseline. Earlier results are not relabeled.

Initial exploration std .30 kept all 13 upright but caused one forbidden
contact and only two full qualifying rests. That probe and its original source
are preserved. Initial std was reduced to .15 before any full training run;
repeat results are preserved in `exploration_probe_02`. Random exploratory
failures are learning data and retain their original penalty/labels. The
original orchestration's all-random-exploration-success launch gate was removed
after inspection; this is documented separately from physical success criteria.
The full study must additionally pass a finite two-update optimizer smoke test
and inspect its deterministic full-duration rollout. Development smoke uses
3,328 transitions and is separate from the study budget.

## Frozen learning pilot

Two fresh seeds 54511 and 54517, physical GPUs 0 and 1, respectively. Each gets
256 PPO updates × 64 steps × 26 environments = 425,984 transitions; 851,968 total.
Initial global 0.5 s settling is excluded. Indexed reset 25-tick warmup is included
and reported separately. All resets are to supported endpoints; there is no
in-episode teleportation or mixed standing curriculum.

PPO uses 128–128 ELU actor/critic, zero actor output initialization, initial
Gaussian std .15, learning rate 1e-4 with adaptive KL target .005, five epochs,
four minibatches, entropy .001, gamma .998 and lambda .97. Observation
normalization is disabled. No checkpoint selection on held-out results and no
automatic policy promotion. Sources, banks, fixed baselines and protocol are
hashed before the two runs. Changes after freezing require preservation and a
documented execution/report repair; research definitions cannot silently change.

## Comparison

Each identical final policy runs 20 seconds on 13 training geometries and 16
held-out geometries, with the same initial states, geometry, seed, batch size
and evaluation GPU as its fixed baseline. Success is unchanged: no latched
physical failure, at least 95% qualifying ticks and the final two seconds
continuously qualifying. Continue the policy for the full horizon after a
failure; do not count pre-failure low-torque fragments as savings.

Compare whole-trial and final-ten-second effort on common successful trials,
by seed, sitting versus leaning, height, slope and outline. A 10% reduction with
unchanged success is a useful result threshold, not an assertion of a global
optimum. There is no automatic freeze-on-low-torque rule in this experiment;
steady low-torque poses already maximize the objective.

The held-out bank includes intermediate 27.5/32.5/37.5 cm level seats, 10/30-degree
slopes, narrower elliptical/chamfered outlines and 75-degree back support.
These configurations are withheld from this policy's training, but have been
used in prior project evaluations. Two seeds reuse a finite rigid bank. This
does not establish arbitrary-surface, cushion, vertical-wall, hardware or
standing-transition performance.

Data root: `/data/humanoid-sitting-perception/effort_rest_01`.
Full study: `ppo`, persistent tmux `effort_rest_ppo_01`.
Operate from the Mac through SSH; no additional coding agent or software reinstall.
