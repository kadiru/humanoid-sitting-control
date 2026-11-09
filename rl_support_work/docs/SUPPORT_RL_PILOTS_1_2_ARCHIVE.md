# Supported-sitting RL: latest research state

Status: two bounded PPO pilots and all paired evaluations are complete. The
broader second policy reduces seated leg torque, but does not improve success
rate on its fresh test bank. Neither candidate replaces the existing default
controller. These are completed 20-second attempts, not early-reset counts.

## Direction authorized by the user

The latest user instruction resumes reinforcement learning and asks for progress
toward sitting on a reachable surface behind the robot. This supersedes the old
PPO pause in the original handoffs. Preserve those handoffs, source, datasets and
checkpoints. Work directly over `ssh kfuUbuntu`; do not launch another coding
agent on Ubuntu. Long jobs run under persistent tmux sessions.

PPO now learns sensor-driven corrections to all 23 body-joint targets. It remains
a residual policy over the frozen sensor05/contact02 motion scaffold. There is no
new VLA, no new DAgger collection, and no privileged teacher in the action path.
The frozen scaffold's nominal motion remains a strong restriction on the behavior.
This is a first learning stage, not arbitrary-surface sitting from scratch.

## What was added

- `sitting_task/src/humanoid_sitting/support_objective.py`: a supported-sitting
  reward with stable seat support, torso uprightness, low motion, leg torque
  economy, seat load share, and penalties for impact, descent and abrupt commands.
- `sitting_task/src/humanoid_sitting/support_depth.py`: current-scan depth/FK/IMU
  clearance measurement, plane normal and visible patch features. It does not
  require global map registration. Ambiguous measurements are marked unavailable.
- `scripts/train_support_rl.py`: bounded RSL-RL PPO training and full-attempt
  paired evaluation, with source snapshots and fixed final-checkpoint selection.
- `scripts/train_support_rl_diverse.py`: a second training bank with balanced
  coverage of dimensions and yaw; the objective and action contract stay fixed.
- `support_readiness.py` and `scripts/train_support_rl_guarded.py`: explicit
  sensor-derived refusal for unobservable startup geometry. It holds the starting
  joint targets; unknown geometry is zero/invalid, the sit command is zero,
  task reward is zero, and the refused attempt remains in the denominator.
- Saved-bank evaluation, live-depth input ablation, comparison reports, and a
  simulator reconstruction video that explicitly includes failures.

The original shared-frame observer is unchanged. Its SHA-256 remains
`daec7a1c769c68579cebfafe79d2063cb19660ca6e09ced705f460218f9fcf5c`.
Its prior full 26-condition evaluation remains separate from the faster local
clearance measurements added for these RL pilots.

## Meaning of the reward

The user's human analogy is implemented as a physical outcome: most weight rests
on the seat, the torso stays upright and quiet, and holding the posture requires
less leg torque. The feet may carry stabilizing load. There is no requirement that
the legs do zero work or that a specific hip motor does most of the work.

Low torque earns the seated economy reward only when the pelvis/thighs are
actually supported, the pelvis is within the seat bounds, the torso is upright,
motion is small and the trial has not failed. Falling or leaning on the wrong
body part cannot earn that reward. Five targeted reward tests pass.

About 75% seat load is a soft preference, not a success threshold. Full-attempt
success requires a standing start, at least three continuous seconds of supported
stable sitting, and no latched fall, wrong contact, pelvis-floor contact, excessive
impact/descent or original effort-limit violation during the entire 20 seconds.
The impact threshold is two robot body weights (~632.53 N), a project evaluation
threshold, not a calibrated hardware safety limit.

## Sensor and ground-truth boundary

Actor and critic both receive 144 sensor-derived channels: the existing delayed
proprioceptive/initial-depth packet, the frozen controller's proposed joint
targets, current depth geometry, and a learned contact/load estimate.

The rear-downward virtual depth camera is 128x96 and sampled every 100 ms of
simulation time. Its geometry processing uses synchronized encoders and projected
gravity. The separate old proprioceptive packet retains its 20 ms delay and small
noise. Visible patch size/center are not complete furniture bounds.

Neither network receives simulator root position, linear velocity, furniture
parameters, COM, contact forces, or true floor/seat clearance. Simulator truth is
used for training reward/failure labels and offline evaluation only. Inherited
environment observations/rewards are not used by PPO; its wrapper supplies the
independent sensor-only observation and the new reward.

Foot forces are currently evaluation/reward labels, not actor inputs. Real foot
pressure-sensor availability has not been verified. The virtual camera mounting,
ideal depth and IMU equivalents, and torque proxy are not hardware calibration.

## Experiment design and known results

Each main pilot uses 98,304 transitions. Pilot 1 has four randomly sampled
geometries, eight parallel environments and 192 updates. Pilot 2 has 16 balanced
geometries, 16 environments and 96 updates. Both use 64 steps per environment per
update, a fixed 3e-5 learning rate, three PPO epochs and no entropy bonus. These are
small initialization pilots, not a claim of convergence.

The correction is bounded to +/-0.08 rad before total-command slew limiting;
total target changes are bounded to +/-0.05 rad per 20 ms step, then original
joint and actuator limits apply. The correction can bypass the scaffold's tighter
late ankle slew rule, which remains a limitation of the pilot action design.

Known completed comparisons:

| Evaluation | Frozen baseline | PPO pilot 1 |
|---|---:|---:|
| Training bank | 7/8 | 8/8 |
| First unseen bank | 8/8 | 6/8 |
| Larger unseen bank | 23/24 | 23/24 |

Pilot 2 final checkpoint, using the guarded startup admission in both controllers:

| Evaluation | Frozen baseline | PPO pilot 2 | Baseline / PPO worst impact | Seated leg torque change on matched successes |
|---|---:|---:|---:|---:|
| Known regression bank | 8/8 | 8/8 | 517.8 / 451.7 N | -7.25% (8 pairs) |
| Fresh unseen bank | 14/16 | 14/16 | 772.1 / 651.1 N | -6.90% (13 pairs) |
| Training bank | 14/16 | 14/16 | 784.2 / 682.4 N | -7.46% (14 pairs) |

On the fresh bank, PPO recovers baseline failure case 0 but loses baseline success
case 14; case 15 fails with both controllers. All these failures are impact-limit
violations. Pilot 2 has no falls in these 40 deterministic evaluations, but this
does not establish hardware safety or arbitrary-surface generalization. The fresh
holding-torque comparison is 4.416 Nm baseline versus 4.111 Nm PPO, conditional on
the 13 attempts successful with both. Failed cases are not included in that torque
average and are reported separately above.

The main completed pilots each use 98,304 transitions. The broader stochastic
training run passes 65/96 complete attempts, including one refused startup. Its
actor weights change by up to 0.010459, remain finite, and belong to the PPO
optimizer. Training success is not a deterministic held-out evaluation.

Across the 13 evaluation rollouts, the current depth observer accepts 30,964 of
33,600 recorded case-frames; maximum accepted clearance error is 0.7462 cm, and
none exceeds the 2 cm engineering allowance. These are ideal simulated depth
results, not hardware accuracy. The saved-depth regression separately accepts
100/102 samples with maximum error 0.0648 cm; all six total blackouts are withheld.

Pilot 1's two lost successes were starts on the same previously unseen bench,
with falls, wrong contact and high impact. Removing only the new live-depth
channels still gives 6/8 with those two failures. Thus there is no demonstrated
success benefit from live depth in that ablation. Accurate geometry alone did not
prevent controller failure. The ablation is not a retrained blind-policy study.

Pilot 1's training bank happened to cover only about -0.26 to +2.08 degrees yaw,
and no depth greater than 0.522 m. Pilot 2 covers yaw levels -5, -1.7, +1.7, +5
degrees and depths 0.41, 0.47, 0.53, 0.59 m, with independently permuted width and
height levels. No failed held-out geometry was copied into training. Coverage is
a hypothesis being tested, not a proved sole cause of the first failures.

The old unseen bank is a known regression test for Pilot 2. Its fresh holdout uses
eight new geometries and two starts each. An additional paired evaluation on all
16 training geometries tests whether the new coverage was learned at all.

The first balanced run (`support_rl_diverse_train_01`) stopped at its first reset:
a randomized starting pose hid part of the landing patch and the old startup
detector raised `ValueError`. This was an unhandled unavailable measurement, not
a justified reason to invent geometry or discard that sample. Its files and log
are preserved. The guarded restart (`support_rl_diverse_train_02`) retains those
attempts as refused failures. This is a command hold, not a separately validated
active-balance or hardware safety controller. Four new targeted tests pass, including exact valid
sensor-stream equivalence and preventing a refused command from changing targets.
The guarded baseline and guarded PPO use identical admission logic in all three
Pilot 2 paired evaluations.

## Reproduction and preservation

Ubuntu root: `/home/kadir/humanoid-sitting`.
Large outputs: `/data/humanoid-sitting-perception/support_rl_*`.
Completed runs get symlinks under the project `runs/`; logs and exit files are
under `logs/`. Never reuse an output name; scripts refuse existing directories.

Launchers run through the existing `scripts/run.sh` environment without installs:

- `run_support_rl_pilot_01.sh`: Pilot 1 and its initial paired evaluations.
- `run_support_rl_ablation_01.sh`: Pilot 1 live-depth ablation.
- `run_support_rl_extended_01.sh`: Pilot 1 larger unseen bank.
- `run_support_rl_diverse_01.sh`: Pilot 2, regression, fresh baseline and PPO.
- `run_support_rl_balanced_eval_01.sh`: Pilot 2 matched training-bank evaluation.
- `finalize_support_rl_suite_01.sh`: report, video, preservation audit and decode.
- `run_support_rl_guarded_02.sh`: the authoritative restarted Pilot 2, three
  baseline/PPO pairs, final suite report/video, preservation audit and decode.
  The earlier queued launchers exited when the unguarded training failed; their
  nonzero exit records remain preserved and do not represent completed tests.
- `finalize_support_rl_suite_03.sh`: the completed-result report, reconstruction
  video, decode check and source/rollout audit. The guarded launcher completed all
  simulations but its first report stopped on equivalent source-path spellings
  (`IsaacLab/../scripts/...` versus `scripts/...`). The reporter now normalizes
  path spellings while retaining every recorded content hash. This was a report
  comparison issue; no training or evaluation was rerun or altered for it.
- `finalize_support_rl_suite_04.sh`: final presentation version with less crowded
  3D axis labels. Numerical results are unchanged from version 03. The exact
  rendering source is included in the delivered directory.

Final result directory: `support_rl_suite_results_04`. It contains `results.json`,
`report.md`, `comparison.png`, `training_diagnostics.png`,
`sitting_comparison.mp4`, four representative stills, video metadata and a bundle
audit. The 40-second video is a simulator pose/force reconstruction for evaluation,
with a measured depth clearance readout and 2x playback. It is not a camera movie
or a depiction of privileged policy inputs. It explicitly shows the first PPO
bench fall and the second PPO's remaining fresh-case impact failure.

Preservation audit: all 117 pre-existing source files, five archived baseline
files, and three known existing checkpoint hashes match; the IsaacLab checkout
remains clean. No software was installed. The bundle audit verifies current and
archived numerical source hashes for all 13 evaluation runs and both completed
training runs, final checkpoint hashes, and all 33,600 recorded evaluation
case-frames. Nine targeted reward/admission tests pass. All prior perception
datasets and trained models remain in place.

The initial trained PPO remains at `support_rl_train_01/final.pt`; Pilot 2 writes
`support_rl_diverse_train_02/final.pt`. No existing checkpoint is overwritten.
Every run records arguments, geometry bank, numerical-source hashes, and frozen
model hashes. Training verifies finite changed actor weights and optimizer
membership. The output suite records final checkpoint hashes.

## Next research gate

Compare complete-trial success and all-trial impact first. Compare leg torque
only on matched successful trials and report lost successes separately. Do not
promote a candidate just because its training reward or selected successful-case
torque improved.

Once a reliable learned policy is established, expand the curriculum in stages:
larger reachable seat-height/offset/heading changes; sensor noise, dropout and
latency; contact/friction and body-parameter variation; compliant cushions; then
reducing dependence on the nominal motion. Finding and approaching a seat is a
separate later skill. A surface must be reachable and support the robot's weight.

Before a much larger training campaign, replace synchronized failed-state sampling
with properly isolated per-environment resets and profile/batch depth processing.
The current 10 Hz sensor schedule is simulation time, not a real-time claim.
Failure states continue physically for diagnostics, but remain terminal to PPO
until the batch ends, wasting part of the current pilot sample budget.

The immediate next learning objective is eliminating impact failures across a
wider randomized curriculum while preserving the measured torque benefit. Merely
increasing the current checkpoint's reward or selecting attractive replays is not
enough. Wider surface heights and deformable cushions also require progressively
relaxing the frozen nominal-motion constraint; a +/-0.08 rad correction around the
old motion cannot represent every feasible sitting strategy.
