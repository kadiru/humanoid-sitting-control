# Sensor-feedback PPO for supported sitting

This document describes the canonical pilot design. See
`SUPPORT_RL_LATEST.md` for the subsequent broader training bank, explicit startup
refusal handling, completed comparisons, and authoritative reproduction launcher.

The user's latest instruction explicitly resumes RL work. This supersedes the
historical training pause in the original handoffs; those handoffs and all prior
controllers/checkpoints remain preserved. The primary agent runs Ubuntu commands
through SSH from the Mac, with no additional Codex agent.

## First bounded experiment

Train an RSL-RL PPO policy that supplies corrections for all 23 body joint targets
on top of frozen sensor actor05, contact estimator02, and their existing actions
mode governor. This is **residual RL over an imitation-trained motion scaffold**,
not a new end-to-end policy or sitting learned from scratch. The scaffold uses no
privileged teacher at inference. The new PPO head receives live feedback for all
its outputs, addressing the old body's lack of live proprioceptive feedback.

The first bank contains four rigid furniture geometries (chair, bench, box, and
couch) sampled with seed 51031 and two independently jittered starts per geometry.
The independent evaluation bank uses geometry seed 51103 and start seed 51107.
These remain initially aligned, reachable seats behind the robot. Neither bank
tests finding furniture, walking, large heading changes, cushions, unstable
furniture, physical deployment, or arbitrary surfaces.

PPO is initialized to zero correction so its deterministic initial behavior
matches the frozen scaffold exactly. Bounds: +/-0.08 rad correction, +/-0.05 rad
total command change per 20 ms control step, original joint and actuator limits.
The new head may change the old governor's proposed target within these bounds;
it is not constrained to preserve the governor's tighter late ankle slew rule.

## Observation and information boundary

Actor and critic both use the same 144 sensor-derived channels:

| Slice | Meaning |
|---|---|
| 0:109 | Existing delayed encoder/velocity/estimated torque/IMU/previous action/initial depth geometry/sit packet |
| 109:132 | Current target proposed by the frozen sensor controller |
| 132:142 | Current depth-plane clearance, normal in pelvis frame, visible patch centroid/spans, validity, and age |
| 142:144 | Frozen learned contact probability and load estimate |

No true root translation/linear velocity, COM, fixture geometry, simulated seat
force, or foot-force channel enters either network. Projected gravity and gyro
are ideal simulated IMU equivalents with the existing small sensor noise; this
does not validate a hardware IMU filter. Applied actuator torque is an estimated
motor-torque proxy, not a hardware-calibrated measurement.

Simulator body poses, ground-truth geometry and contacts are used explicitly as
reward/termination labels and offline evaluation. A real seat is not assumed to
report force. Foot forces are useful privileged labels/diagnostics; their real
hardware availability remains unverified and they are absent from actor inputs.

The separate `CurrentSupportDepth` module reads only current depth, camera
calibration, encoder FK, IMU gravity and a floor-to-seat separation measured from
startup depth. It fits the current scan directly in a gravity-aligned pelvis
frame. It needs no global position or map registration. Floor/seat identity is
checked using the existing same-scan association rule. Body points are removed
using calibrated FK collider bounds. Ambiguous/poorly conditioned measurements
are withheld. The visible patch centroid/spans are NOT complete furniture bounds,
and this module does not replace the full shared-frame alignment estimator.

The new rear-facing downward virtual camera is 128x96, measured every 100 ms.
Encoder and gravity readings used to process a depth frame are from that same
simulation instant. The existing proprioceptive packet is delayed by 20 ms.
Timing is simulation time; wall-clock processing measurements are reported
separately and are not a proof of real-time hardware execution.

## Objective and human analogy

The desired outcome is a supported, quiet seated posture. The human observation
about reducing muscular effort motivates lower motor torque, but the robot has
different joints and actuators. No particular hip or torso motor is forced to do
most of the work, and feet may retain useful stabilizing load.

- Approach the seat-relative target and establish pelvis/thigh support.
- Sustain a valid upright, low-velocity seated hold.
- Prefer about 75% seat load as a soft objective, leaving useful foot support.
- Reward lower leg torque only while actual supported, upright, stable sitting
  is present. Low torque during standing, free fall or invalid contact earns no
  seated economy reward.
- Penalize impact, rapid descent, overall effort and abrupt command changes.
- A latched physical failure cancels further task reward and success eligibility.

Reward weights and thresholds are explicit in `support_objective.py`; no reward
uses future sensor measurements. Rewards describe post-action outcomes, while
policy observations describe the pre-action state, as required for RL.

## Evaluation and training accounting

Each attempt lasts 20 seconds without an automatic mid-trial reset. Failure flags
are latched across the whole attempt. Rewards after failure are zero and the PPO
done mask stays true until the synchronized batch reset. Sensors and physics
continue normally for diagnostic recording; no privileged failure flag is packed
into the observation or used to alter the issued action. This bounded pilot's
synchronized failed-state sampling is less efficient than independent resets.

Success requires standing at 1.52 s with both feet loaded, at least three seconds
of uninterrupted supported upright stable sitting, and no fall, pelvis-floor
contact, unintended arm/knee/foot seat contact, excessive descent, impact over
two body weights, or effort-limit violation at any time. Seat-local finite-area
bounds vary with geometry; comparisons in this phase are paired and should not
be pooled with the differently initialized historical 180-trial result.

Reported diagnostics include full-trial maximum impact, seated leg RMS torque,
remaining foot load, live-depth availability/error, and wall-clock latency.
Holding-torque reduction is secondary to successful complete sitting; it is not
a calibrated electrical-energy measurement. Small pilot counts
do not establish statistical superiority or generalization to arbitrary seats.

## Existing validation before the longer pilot

- Five reward loophole tests pass.
- Two-update smoke run: 512 transitions, finite actor parameters, maximum parameter
  change 0.00066359, optimizer confirmed to include actor parameters.
- Saved-depth regression: 102 sampled case-frames, 100 accepted, maximum accepted
  clearance error 0.0648 cm; all six all-missing-depth checks withhold measurement.
- Zero-correction baseline: 7/8 complete attempts succeed; one impact violation.
  1,482/1,600 evaluation depth samples accepted, maximum accepted error 0.3366 cm.
- In those seven successful baseline cases, mean leg RMS torque is 8.10 Nm while
  standing (1–2 s), 10.27 Nm during lowering (5–10 s), and 5.08 Nm when settled
  (17–20 s). In the settled interval the seat carries 77.4% of body weight and
  the feet 22.6%. These are selected successful-case diagnostics, not RL gains.

Longer pilot: 192 PPO updates x64 steps x8 environments = 98,304 transitions.
The final checkpoint is fixed in advance for evaluation; no selection using
the independent furniture bank. Old controllers and checkpoints remain defaults
unless a separate promotion is justified by results.

## Reproduction

Run from `/home/kadir/humanoid-sitting` with `bash scripts/run.sh`. Use new output
names; scripts refuse to overwrite run directories. The persistent launcher is
`scripts/run_support_rl_pilot_01.sh`, running in tmux session `support_rl_pilot`.
Outputs are under `/data/humanoid-sitting-perception/support_rl_*`, with symlinks
under `runs/` after completion. Logs and exit status are under `logs/`.

Preserve the original shared-frame observer unchanged, including its full
26-condition validation. Further work should broaden reachable geometry,
introduce compliance and disturbances, evaluate sensor delay/calibration changes,
and progressively reduce dependence on the frozen scaffold after PPO shows a
measurable benefit. A surface must be reachable and able to support the robot;
"any surface" is a research direction, not a validated capability.
