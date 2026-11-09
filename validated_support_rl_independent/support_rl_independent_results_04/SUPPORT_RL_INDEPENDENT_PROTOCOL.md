# Independent-reset supported-sitting PPO pilot

This experiment continues the second PPO checkpoint for a fixed 98,304 additional
transitions. It changes training sample handling and geometry coverage. The
reward, frozen sensor05/contact02 motion scaffold, 144 sensor-derived actor and
critic inputs, 23 residual actions, joint slew and effort limits remain unchanged.
There is no software installation or hardware deployment.

## Attempt lifecycle

Training ends an individual attempt on the first physical failure, a sensor-derived
startup refusal, or the 20-second finite horizon. The simulator resets only that
environment. Kinematics are forwarded without stepping physics for its neighbors. The reset
robot holds for one ordinary 20 ms control tick; its first camera observation is
taken afterward, when PhysX/RTX has published the new scene transforms. That robot's motion/contact recurrent states, delayed
sensor history, filtered torque, previous command, depth observer, and counters
are reinitialized. Unavailable startup geometry never receives a simulator-truth
replacement and remains a counted refusal. There is no hidden resampling loop.

Every simulation tick advances each non-reset robot's controller once. The next
observation for a reset robot contains the new attempt's first measurements.
The first torque-filter sample is zero because the simulator may retain the old
applied-torque cache until a physics step. Remaining sensor noise/delay is retained.

Evaluation always runs all robots for the entire 20 seconds. Failure labels latch
and cannot be erased by a successful-looking posture later in the attempt.
Success requires a standing check at 1.52 seconds, at least three continuous
seconds of stable supported sitting, and no failure throughout the full attempt.
Evaluation resets do not interrupt failed trajectories early.

The new startup protocol omits the old five global physics settling steps. This
allows a consistent indexed reset that cannot advance neighbors. All comparisons
in this pilot use the new protocol; historical pilot-2 numbers are not silently
reused as its baseline. Both successful and failed attempts are retained. The
budget-cut partial attempts are reported separately and cannot count as successes.

## Fixed experiment design

- Parent checkpoint: `support_rl_diverse_train_02/final.pt`.
- Training: 16 parallel robots, 16 fixed geometry prototypes, 96 updates of 64
  steps per robot, geometry seed 51511, robot/training seed 51513.
- Geometry ranges: height 0.26–0.35 m, depth 0.40–0.64 m, width 0.78–1.14 m,
  seat x -0.44 to -0.36 m, lateral offset -0.04 to 0.04 m, yaw -8 to 8 degrees.
  Training covers four independently permuted levels per dimension. Robot starts
  are randomized on each attempt. Geometry shapes remain immutable within a run.
- Families: rigid chair, bench, box/platform and couch shape. No compliance.
- Fresh evaluation: eight independently sampled geometries, two starts each,
  geometry seed 51541 and start seed 51543. Compare zero residual, the previous
  PPO, and the fixed final new PPO checkpoint. This is a small holdout, not a
  statistical reliability demonstration.
- Regression evaluation: previous pilot-2 fresh bank, now explicitly a known
  regression set; seeds 51403 and 51407, eight geometries and two starts each.
  Compare previous and new PPO under the same new startup protocol.
- No intermediate checkpoint selection, reward tuning, or held-out geometry
  copying into the training bank. The complete launcher is fixed before results.

Simulator pose, velocity, contact and furniture parameters are used for reward,
failure labels and offline evaluation only. The actor and critic retain the
existing sensor-only input contract. Foot force is not added as an assumed
hardware sensor. Ideal depth/IMU and the virtual camera mount are not hardware
calibration. The +/-0.08 rad residual over a frozen nominal motion still limits
what sitting strategies this policy can represent.

## Validation

The indexed-reset probe compares the untouched robots' pose, velocities, joints,
controller/contact memory, sensor delay/filter state, geometry, command, counters,
depth features and observation vector. All differences must be below 1e-6 and
no physics tick may occur. The restarted robot must match a cold controller at
the same GPU batch size, receive fresh joint readings, and reset alone at its
horizon. Accepted depth clearance is also checked against evaluation-only truth.

Probe 01 passed the initial isolation/accounting checks. Probe 02 added a cold
controller comparison with a different GPU batch size and stopped at a maximum
latent difference of 0.00005245. Probe 03 matches batch size at the same tight tolerances and passes isolation.
A subsequent broad-bank diagnostic caught stale startup camera transforms. The
main experiment was stopped before training; its two preliminary evaluation runs
are preserved as diagnostics, not valid reliability scores. Probe 04 additionally
checks measured seat height after a one-tick pose hold, using truth only in the test. All earlier logs/source snapshots remain preserved.
Twelve targeted reward, admission and reset tests pass. Probe runs use a short
two-second horizon and are engineering smoke tests, not sitting success scores.

All training transitions must equal completed-attempt steps plus unfinished
steps, both in total and independently for every robot. The reporter checks full
evaluation horizons, paired banks/seeds/source hashes, finite changed actor
weights, optimizer membership, and archived source hashes. It reports all-trial
impact and failures before conditional torque changes on matched successes.

The persistent launch is `scripts/run_support_rl_independent_05.sh`; its predecessor
03 stopped at probe 02 before any main training or evaluation. Outputs remain
under `/data/humanoid-sitting-perception` with fresh names. No existing checkpoint
is overwritten or promoted automatically.

The first broad diagnostic bank used seeds 51521/51523. Its startup estimates
remained near the default 0.30 m despite true 0.26–0.35 m geometry. The corrected
experiment uses a fresh holdout seed and re-runs all baseline comparisons.
