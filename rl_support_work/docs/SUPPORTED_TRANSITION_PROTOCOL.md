# Supported sitting and leaning transition PPO pilot 01

Authorized task: validate supported postures across heights/slopes, then learn to reach support from standing; effort is secondary. This is a new task, not continuation of the failed effort optimization studies. Prior studies and checkpoints remain untouched.

## Physical task and evidence

Data root: `/data/humanoid-sitting-perception/supported_surfaces_01`.
13 training goals: level seats at25/30/35/40cm, tilted sitting goals at20/45deg, inclined pelvis-supported leaning at45/60deg with several knee configurations. Every training goal passed a full20-second hold under the original physical limits. Separate75deg back-supported postures also passed; three are reserved for unseen evaluation. Vertical90deg candidates did not pass the traction margin and are not claimed as validated goals.
16 heldout geometries: unseen10/30deg sitting slopes, intermediate27.5/32.5/37.5cm level heights, narrower elliptical/chamfered outlines for sitting and leaning, and three75deg back-support configurations. These geometry cases are never supplied to training. One changed-outline endpoint passed98.7% of qualifying ticks but failed the final continuous2-second requirement; retain it in the heldout test and report the limitation rather than removing it.

The original nominal standing command tipped forward. A separate45-case motor-preload calibration produced five physical survivors; case7 (2.28deg final tilt) supplies the starting state. Full20-second standing checks passed29/29 across the complete train/test geometry bank, with constant commands. This is initialization, not a runtime balance oracle. New bank03 preserves bank01, bank02 and all failed probes.

Six reset candidates interpolate encoders/orientation and align feet between standing and validated support. A78-case short physical/sensor probe rejected25 dynamically failing cases and additional inaccurate observations. The recorded admissibility mask is frozen before PPO. Choose the nearest admissible blend to the requested curriculum stage; low seats currently have only standing and fully supported resets, a material exploration limitation. All13 standing and supported endpoints are admissible. No simulator teleport occurs inside an episode.

## Runtime observation and action contract

333 channels =3 frames of23 joint positions,23 velocities/10,23 measured torques normalized by fixed effort limits, gravity3 and gyro3/3 from IMU, previous absolute joint command23, and measured support-plane features13. Depth:128x96 rear pelvis-mounted camera at5Hz. Encoder/IMU-only plane propagation at50Hz assumes approximately stationary feet for at most0.5s; disagreement or expiry invalidates the bridge. Temporal plane association uses only the preceding measured plane. Common-mode foot slip remains an estimator limitation. Downward-view plane selection is excluded because it can select the slab's front face.

Neither actor nor critic receives simulator root translation, contact forces, target pose, true surface geometry, material, case index, curriculum phase or reset metadata. Saved targets and simulator contact/body state are used only for reset construction, rewards and validation. Initial robot commands are necessarily known actuator state, not an additional goal input. Successful execution does not use privileged safety gating; failures are latched for scoring and full-horizon evaluation continues policy execution.

Policy outputs23 absolute joint targets represented around one public nominal pose with fixed per-joint scales. Clamp to physical position limits with0.03rad margin; command lowpass0.2 and slew0.015rad/20ms. Other joints retain their reset command. Actor mean is analytically initialized to preserve the previous command; it is not a motion trajectory or teacher. Standard deviation0.05. Network128ELU128ELU for actor/critic. No observation normalization.

## Reward and success

Per20ms: `dt*(0.5*exp(-6*q_error-12*root_position_error)+3*qualifying_support-0.05*clipped_effort*qualifying_support-0.02*normalized_command_change)`, minus5 on first physical failure. q_error is mean squared error to the validated goal divided by fixed action scales; root error is squared metres. Effort is normalized squared torque divided by0.01 then capped at1 for reward only. Its maximum0.05/s penalty is1/60 of the support reward. Uncapped metrics remain available; torque is a proxy, not measured energy.

A successful transition lasts the full20seconds, has no latched physical failure, and ends with5 uninterrupted seconds of qualifying supported rest. Report reset index separately: support-reset successes are not standing-transition successes. Physical failures retain original fall, forbidden-contact>10N, impact>2BW, torque and joint-limit definitions. Qualifying support also requires permitted finite-face contact, load>=0.05BW, slip<=0.02m/s, speed<=0.1m/s, gyro<=0.25rad/s, tilt<=40deg and friction utilization<=0.90 at declaredmu0.6. The torso and fixed `logo_link` rear shell are valid back-support patches only on their rear face; chest/head/arms/hip-pitch housing contacts remain forbidden. No old-study contact label changes.

## Frozen pilot and comparison

Two independent seeds54211 and54217,224 updates each,64 rollout steps,26 environments (two per training surface):372736 transitions/seed,745472 total including indexed-reset warmup ticks. Log active and warmup counts separately; initial0.5s global settling is outside the budget. Initial independent optimizer smoke test2updates×64×13=1664 transitions is development validation, not part of these two fresh runs.

Updates0–31 request supported reset;32–63 request0.9blend;64–95:0.75;96–127:0.5;128–159:0.25;160–223:standing only. Earlier stages also use20% standing resets. Invalid intermediate resets are replaced with the closest prevalidated blend, never chosen from learned-policy outcomes. Exact actual reset index is logged per episode.

PPO: learning rate1e-4 fixed,3epochs,4minibatches,gamma0.998,lambda0.97,entropy0.001. Checkpoints every32updates and final. No extra training or changing reward/budget/contracts once launched. PhysicalGPU0 andGPU1 use persistent tmux processes; no other coding agent.

Evaluate each identical final checkpoint separately on all13 training and16 heldout geometries from standing, full20seconds. Compare physical failures, final continuous support and reach rate with the constant-command standing baseline. Report sitting and leaning separately, then each surface/height/slope. The fixed supported-posture measurements are endpoint reference effort, not a matched trajectory energy comparison. Two seeds and finite rigid slabs do not establish arbitrary-surface, compliant-cushion, full-wall, approach/alignment or hardware capability. Do not promote a policy automatically.

Independent verification must include frozen source/bank hashes, finite changed weights, exact budgets, expected checkpoint loaded, full durations, matched initial states/surfaces and unchanged old studies. Preserve failed sources/results and record execution repairs. Save results/plots to the Mac and inspect them before reporting.
