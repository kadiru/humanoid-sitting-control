# Landing distance tracker: validation protocol

No RL training, optimizer updates, deployment changes or installation. The preserved pilot-3 policy supplies motion in evaluation mode. The new tracker runs passively and is not an actor input. Synchronous camera acquisition changes the camera samples available to the frozen original actor; do not claim trajectory parity with historical runs.

## Development and diagnosed causes

Use the archived mixed TRAINING geometry bank, seed 51813, to diagnose and develop. The original capture, the rejected long-bridge prototype, and ineffective render/pump synchronization experiments remain preserved. Do not include development results as held-out evidence.

In this installed IsaacLab revision, eager camera captures occur on a physics-substep schedule, and `force_recompute=True` does not make an undued sensor due. At controller reads, the captured frame can be 95 ms old. Explicitly request a current exposure with a temporary zero camera period, restore the period, and assert matching sensor capture/read timestamps with no physics steps or joint changes during capture. Leave installed IsaacLab unchanged.

The original depth estimator requires a broad floor patch in the same image as the seat. Preserve the measured startup floor/seat separation and propagate the floor using joint FK and IMU gravity. Associate a conditioned current seat plane with this reference. A uniquely matching recent seat plane can retain its identity as the feet unload. When the seat is briefly occluded but the floor remains visible, use the current floor plane and previously observed rigid seat height for at most 0.6 s since the seat itself was last seen. For this horizontal-floor scope, fit its offset with normal supplied by gravity; require the observed floor tilt below 0.5 degrees. Budget 8 mm base error, three RMS/sqrt(effective samples), horizontal lever arm times sine(observed tilt + 0.25 degrees IMU/calibration allowance), 5 mm floor-to-seat mapping error and 5 mm/s of occlusion age. This engineering budget is not a calibrated statistical confidence interval and does not validate sloped floors. Current floor identity must be unique within 3 cm of its recent observed position; it does not depend on foot contact. Between exposures, use FK for at most 100 ms, charge the full inferred foot-height shift to the engineering uncertainty budget, and withhold estimates above 2 cm budget. Two feet moving together do not establish contact. No simulator pose, floor/seat label, or contact force enters the tracker.

## Frozen validation set

After development, archive the source hashes before starting these runs. Do not tune on the results.

1. New starts on the development bank: same mixed training shapes, start seed 51823, 16 robots. This tests starts, not unseen shapes.
2. Known easier regression bank: geometry seed 51403, start seed 51407, 8 geometries x 2 starts. This is a regression set, not a fresh holdout.
3. Fresh broader bank: geometry seed 51841, start seed 51843, 8 geometries x 2 starts. No selection by sitting success.

All runs last 20 seconds, including motion after latched physical failure. Score all 48 attempts. Report every unavailable estimate and every error above 2 cm. Simulator truth is loaded only by the separate scorer after sensor-only outputs have been saved.

## Checks and release criteria

- Verify source/checkpoint preservation and clean installed IsaacLab Git state.
- All accepted clearance errors below 2 cm; p95 below 1 cm.
- At least 95% availability overall and in the landing interval on each bank. Landing interval is first sampled seat support >25% body weight minus 1 second through plus 0.5 second. Also report availability at that event and longest gaps per case.
- Compare the old and new algorithms on the same synchronized images; compare with historical unsynchronized development data separately.
- Run the tracker online at 50 Hz; reproduce its output bit for bit from sensor-only recording for one validation capture.
- Report actual processing latency. No claim of a hard real-time guarantee from Python timing.
- Replay fresh validation sensors with encoder noise 0.001 rad, gravity noise 0.001 (renormalized), gyro noise 0.005 rad/s, and depth noise 0.002 m, fixed random seed 51899. Report all accepted errors and coverage; this is a stress test, not a hardware noise model.
- Replay a 0.6-second camera blackout from 11.8 to 12.4 seconds. Require stale estimates to be withheld after the 100-ms bridge limit and reacquisition when measurements become usable. Do not hide unavailability by carrying a stale distance.
- Unit tests cover FK propagation, asymmetric and common-mode foot movement, expiry, reset/time gaps/invalid inputs, surface identity, explicit acquisition timing and schedule restoration on capture failure.

A failed criterion blocks promoting the new tracker as validated for controller use. Preserve failures; diagnose on development data and use new fresh seeds if parameters change after holdout inspection. This experiment does not validate arbitrary furniture, sloped/deforming seats, navigation or hardware operation.
