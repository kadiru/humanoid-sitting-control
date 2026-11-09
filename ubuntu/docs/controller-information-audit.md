# Controller information audit

Audit of selected contact-v12 code and the 129-input residual PPO actor. The immutable baseline is `baselines/geometry-conditioned-rigid-v1`; its 87 recorded hashes were verified before these changes. All 432 successful baseline evaluations remain intact.

## Scripted teacher

At initialization, the teacher reads exact simulator-authored **seat top height, depth, width, world X, world Y, and world yaw**, plus the initial robot root X/Y. Height retargets deep/seated leg IK from the exported G1 joint frames. Depth, X and cosine(yaw) determine the projected front edge and a bounded seated fore-aft correction. Y determines a bounded hip-roll alignment correction. Yaw supplies hip-yaw targets. **Width is cached but does not change motion targets.** Furniture family, backrest, armrest and cushion geometry are not consulted by the motion generator; universal raised-arm clearance is assumed within the validated ranges.

During motion, the teacher uses a resettable elapsed clock, first pelvis/thigh upward seat-load contact time, current filtered pelvis/thigh seat support, actual body masses and center-of-mass positions, ankle body positions, root world X velocity, default joint positions and joint limits. These are privileged state/model inputs, not visual estimates. A shared eight-second descent, contact-triggered recovery, bounded contact search, and COM feedback determine targets. Only configured actuators apply forces; the root is free under gravity. No geometry-specific object tuning is performed.

## Residual PPO actor: exact concatenated observation

Indices are zero-based half-open intervals. There is no configured scaling, clipping or noise on these observation terms; corruption is disabled.

| Indices | Information |
|---|---|
| 0:3 | root linear velocity in robot body frame |
| 3:6 | root angular velocity in robot body frame |
| 6:9 | projected gravity in robot body frame |
| 9:46 | 37 joint positions minus fixed seated goal joints (not current reference) |
| 46:83 | 37 joint velocities relative to default velocities (zero) |
| 83:120 | previous 37 normalized residual actions |
| 120:123 | seat rigid-body center minus robot root position, XY rotated into seat frame; Z is center-relative, not seat-top-relative |
| 123:126 | seat depth, width, thickness (0.1 m) |
| 126:127 | elapsed clock / 9, saturated at 1 |
| 127:129 | sin(world seat yaw), cos(world seat yaw) minus 1 |

The actor does not directly observe seat world height/XY, family, back/arm geometry, load contact, COM error, reference targets, or initial alignment anchors. Nevertheless its action path always adds the teacher's full target trajectory: `target = teacher + 0.08 * clipped_actor_action`, clipped to original joint limits. This is a residual controller, not a standalone sitting policy. The old 127-input checkpoint has two zero-padded yaw input columns; original learned weights were preserved.

## Initial standalone student: 99 explicit inputs

The student outputs **37 absolute joint targets in radians**, with original joint-position and effort limits. Its action implementation imports no teacher, calls no IK and adds no scripted trajectory.

| Indices | Information |
|---|---|
| 0:9 | body-frame linear/angular velocity and projected gravity |
| 9:46 | joint positions minus initial configured default joints |
| 46:83 | joint velocities |
| 83:89 | same numerical seat-center relative position and dimensions as above |
| 89:91 | same numerical world-yaw features |
| 91:92 | robot root height above environment origin |
| 92:93 | elapsed clock / 20, unsaturated |
| 93:95 | initial front-edge world-X clearance and seat-Y alignment |
| 95:96 | mass-weighted COM world X minus mean foot world X |
| 96:97 | upward pelvis/thigh seat support divided by body weight |
| 97:98 | time since first support >0.25 body weight / 20; zero before contact |
| 98:99 | first-support-observed boolean |

These student inputs deliberately expose teacher feedback and contact memory to avoid an unobservable imitation target. They remain privileged numerical geometry/model state: successful imitation would not itself demonstrate perception-guided sitting. Timer and geometry ablations quantify dependence. The reference generator is entirely absent from every standalone evaluation, including the normal student test. RGB-D geometry substitution and removal of privileged state require separate measured experiments. Approach and alignment remain outside this sitting milestone.

## Closed-loop diagnosis and structured student revision

The first 99-input MLP imitated teacher-state targets with 0.0029 rad validation RMSE but failed 144/144 closed-loop trials. It learned joint-state correlations that produced unstable feedback outside the teacher distribution. An initialization bug also left its initial alignment cache at the environment's sampled reset pose instead of the explicitly written benchmark pose. Benchmark initialization now refreshes that cache before observations are read.

The selected revision uses a neural trajectory head with ten inputs: reconstructed seat top height, depth, width, world-yaw features, initial front clearance/Y alignment, elapsed time, contact age, and contact-observed boolean. It does **not** use current joint position/velocity to generate that trajectory. A separate balance head distills the teacher's known COM/velocity law from demonstration labels, clips its ankle correction at 0.35 rad, and disables it after seat loading. The fitted gains recover the teacher's 8 and 1.5 values; this is an explicitly shared classical balance primitive, not a claim that RL discovered balance. Body-X velocity approximates world-X for the small validated yaw range. No reference trajectory, IK, furniture-family branch or geometric schedule is evaluated at inference. Contact/time remain inputs; absence of the original teacher does not mean independence from timing. The PPO refinement updates the neural head, while the fitted balance gains remain fixed buffers.

The structured student is therefore a learned geometry-conditioned trajectory with contact timing and retained model-based balance, not yet a fully learned perceptual feedback controller. Ablations remove the timer, geometry, or balance head independently. An additional old-PPO ablation applies its residual around a fixed standing offset, removing the teacher trajectory and balance feedback completely.

The expanded student masks direct width/depth features in its trajectory head: width is unused by the teacher, and depth already enters through initial front-edge clearance. This prevents spurious dependence on quantities redundant for the demonstrated action. Teacher labels collected at learner states (DAgger) are explicitly separate from applied learner targets. A shadow teacher may compute labels during collection only; it never calls `apply_actions`. Production student evaluation instantiates only `StudentJointAction`, with no teacher IK/trajectory attributes, and writes `teacher_absence_verification.json`.
