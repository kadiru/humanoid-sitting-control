# Deployment observation audit

The preserved DAgger checkpoint is `baselines/dagger-student-v1/best.pt`. It generates full targets independently of teacher IK/reference calls, but is **not deployment-ready**. Its original 99-dimensional vector and effective structured-network inputs are audited below. The baseline now has 432/432 successful trials across banks 10514, 10515, and fresh bank 24001; the initial 288/288 checkpoint bytes remain unchanged.

| Original indices | Channel | Physical G1 availability / remaining privilege |
|---|---|---|
| 0:3 | exact root body linear velocity | Requires a validated state estimator; absent from audited low-level HG message. Body-X is used by the balance head. |
| 3:6 | root body angular velocity | IMU gyro is available; mounting frame must be calibrated. Ignored by selected structured trajectory head. |
| 6:9 | projected gravity | Can derive from onboard fused IMU orientation; exact simulator orientation was used previously. Ignored by selected head. |
| 9:46 | 37 relative joint positions | Model contains 23 body and 14 hand joints. Body encoders are available; hand transport/mapping has not been verified. Ignored by selected head. |
| 46:83 | 37 joint velocities | Same body/hand distinction. Ignored by selected head. |
| 83:89 | exact seat-relative center XYZ and depth/width/thickness | Authored simulator geometry. Selected head uses Z to reconstruct exact seat top height. Direct depth/width masked; actual front edge enters separately. |
| 89:91 | exact world seat yaw sine/cosine | Requires perception and a calibrated frame on hardware. Used. |
| 91:92 | exact root height | Not a raw HG channel. Used with relative seat Z to reconstruct exact seat height. |
| 92:93 | elapsed clock /20 | Available as a computer clock, but constitutes explicit temporal dependence. Used. |
| 93:95 | exact initial front-edge clearance/Y alignment | Uses seat bank and exact initial root XY. Used. Requires perception/registration physically. |
| 95:96 | exact mass-weighted COM minus feet X | Simulator body positions/masses. Used by explicit classical balance head; potentially estimable from encoders/IMU and a calibrated model, but not available as raw telemetry. |
| 96:97 | exact upward pelvis/thigh seat load /body weight | Filtered simulator contact force; no verified physical equivalent. Used to disable balance correction. |
| 97:98 | time since exact load threshold crossing | Simulator contact plus clock. Used. |
| 98:99 | exact load-crossing flag | Simulator contact privilege. Used. |

The old student's balance law remains a bounded COM/velocity correction with gains distilled from the teacher. Its action independence must not be conflated with independence from privileged state or explicit timing.

## Strict experimental interface: 109 channels

`humanoid_sitting.deployment.SensorPacket` is a typed tensor contract with no simulator imports. Unknown fields are not accepted. `SimSensorInterface` reads only joint position/velocity, applied motor torque as a noisy low-pass torque-estimate surrogate, body gyro, and projected gravity as a noisy IMU surrogate. It cannot read root pose/linear velocity, COM, contact forces, furniture bank, or simulation clock. Measured seat geometry comes from a separate renderer-backed RGB-D process.

| Indices | Channels |
|---|---|
| 0:23 | body joint encoder positions, radians |
| 23:46 | body joint encoder velocities, radians/s |
| 46:69 | estimated motor torques, Nm; causal low-pass and noise in simulation |
| 69:72 | IMU gyro in calibrated body frame |
| 72:75 | IMU-derived gravity direction |
| 75:98 | previous commanded body joint targets |
| 98:101 | estimated seat surface center XYZ in the initial gravity-aligned body frame |
| 101:102 | seat surface height relative to the depth-estimated floor |
| 102:104 | visible free seating depth/width, not hidden full furniture bounds |
| 104:106 | estimated heading sine/cosine; symmetric rectangles remain ambiguous modulo pi |
| 106:108 | perception confidence and validity |
| 108:109 | sit command |

**Zero foot-feedback channels.** The inspected `unitree_hg::LowState` declares motor states and IMU but no foot-force field. Do not substitute quadruped `unitree_go` fields or simulator foot contacts. Estimated motor torque is not a direct contact-force measurement.

The actor is a GRU producing 23 absolute body joint targets, with no teacher reference, explicit elapsed clock, scripted COM balance, or contact-force channel. Recurrent memory can still learn temporal behavior; lack of a clock input does not prove purely reactive control. The initial pilot used a 0.01-rad/control-step command guard; subsequent pilots use 0.05 rad/control-step. Both retain every original success threshold, collision requirement and joint/effort limit. This guard is experimental, not a verified physical velocity limit. Fourteen simulated hand targets are parked at their initial values and are not learned outputs or input channels.

RGB-D estimation uses only rendered depth, intrinsics, fixed camera-to-body extrinsics, and IMU gravity. Floor and seat height are inferred from point clouds. No camera world pose, root pose, authored seat geometry, semantic mask, or family label is used by the estimator. A static estimate is latched at the sit command for an already aligned, anchored seat; there is no approach, localization, or live visual servoing claim. Camera capture uses simulator poses only to render the sensor, just as IMU/motor sensor emulation uses the simulated dynamics.

## Hardware evidence and commissioning limits

Pinned SDK audit: commit `63096d0ac0c5d2dec9d6e0c22cd5233410ca2f36`; headers and provenance saved in `runs/deployment_interface_01/sdk_audit`. The official [LowState](https://github.com/unitreerobotics/unitree_sdk2/blob/63096d0ac0c5d2dec9d6e0c22cd5233410ca2f36/include/unitree/idl/hg/LowState_.hpp), [MotorState](https://github.com/unitreerobotics/unitree_sdk2/blob/63096d0ac0c5d2dec9d6e0c22cd5233410ca2f36/include/unitree/idl/hg/MotorState_.hpp), and [IMUState](https://github.com/unitreerobotics/unitree_sdk2/blob/63096d0ac0c5d2dec9d6e0c22cd5233410ca2f36/include/unitree/idl/hg/IMUState_.hpp) support this channel inventory. The official [G1 example](https://github.com/unitreerobotics/unitree_ros2/blob/master/example/src/src/g1/lowlevel/g1_low_level_example.cpp) distinguishes 23/29-DoF layouts and serial/parallel ankle indexing.

The task asset is 23 body plus 14 hand DoFs, not automatically equivalent to every G1 variant. Proposed SDK slot mapping is recorded in `deployment.py`; torso/waist and elbow-roll/wrist-roll frame/sign correspondence requires physical checks. IMU frame, quaternion convention, actuator mode, torque calibration, hand parking, camera extrinsics and actual sensor availability remain unverified on hardware. The simulated rear-facing camera mount is proposed equipment; the stock forward-facing camera cannot be assumed to see a seat behind an aligned robot. No hardware motor-command publisher has been created or run.

## Recurrent geometry variant

`deployment_geometry_recurrent_04` uses a geometry/command-only recurrent head for body targets and a learned neural ankle head using the 109-channel packet plus recurrent latent state. There is no authored trajectory, inverse kinematics, COM law, force threshold or externally supplied phase. However, the recurrent body head has no proprioceptive feedback: it learns a geometry-conditioned temporal trajectory and depends on the number of inference calls since initialization. The ankle head supplies learned sensor feedback. This is a strict-input partial milestone, not a wholly reactive controller; memory-reset and sensor-delay ablations measure the remaining dependency.

Existing simulation effort ceilings (300 Nm for many body/hand joints and 20 Nm for ankles) are preserved for benchmark continuity. They are **not a verified physical G1 actuator envelope**. Hardware torque limits, PD gains, motor-mode semantics, frames/signs, IMU latency and sensor extrinsics must be established for the actual variant before any motor-command deployment. No hardware rollout has been attempted.

The depth estimator now rejects small surfaces and requires at least 75% observed coverage in a 20 × 24 cm aligned landing patch. This is a conservative perception admission gate for the aligned-seat milestone, not a calibrated uncertainty model or complete obstacle checker. Invalid estimates refuse the sit command; there is no ground-truth fallback or claim of a validated autonomous standing fallback. Estimates are latched at command start; moving furniture and mid-transition visual disturbances remain outside this experiment.

## Corrected status preprocessing (checkpoint 05)

Confidence-only substitutions in checkpoint 04 changed body targets by up to 1.81 rad because confidence was constant in training. Checkpoint 05 keeps every learned weight unchanged and zeros normalized confidence/validity in motion heads. The interface still packs 109 channels; **107 are effective neural inputs** (98 physical/controller channels, eight estimated geometry channels, one command). Validity gates admission upstream; confidence is diagnostic and is not a calibrated uncertainty bound. No geometric admission gate or physical success criterion was relaxed. Clean output replay is bitwise equal on 24,000 packets. Body motion retains implicit recurrent timing; ankle feedback remains learned.

Only simulation sensor equivalents and pre-captured, matched initial RGB-D estimates have been executed. No live physical DDS/vision integration or physical G1 rollout has been validated.
