# Shared-frame perception audit — 2026-10-10

This work follows the Mac and Ubuntu handoffs. Further touchdown tuning and PPO are paused. No historical controller, checkpoint, dataset, environment, or Isaac Lab source was modified. New files form an independent observer and evaluation path.

## What the existing implementation actually did

- `scripts/capture_deployment_geometry.py` renders a **single settled startup** depth image from a proposed pelvis-mounted camera: position `(0.05, 0.18, 0.55)` m, rear/down optical axis proportional to `(-1, 0, -1.7)`, 320×240, ROS optical convention. It uses depth, fixed extrinsics, intrinsics, and simulated IMU gravity. This proposed mount has not been verified on physical G1 hardware.
- `scripts/seat_depth_estimator.py:estimate_body_frame` levels the image using gravity, estimates a floor-height histogram, and fits a horizontal seating rectangle. Its `surface_center` is in a floor-shifted initial leveled body frame, while `position_initial_body_level` retains the initial body-origin height. These are different Z origins. Width/depth and backrest heuristics are restricted to the old aligned-start furniture bank.
- `sim_deployment_sensor.py` latches that geometry. `deployment.py` packs 109 channels; actor05 masks confidence/validity, leaving 107 effective channels. There is **no live root translation, seat-map registration, current pelvis-to-seat transform, or tracked covariance**. The main body decoder advances recurrently, while the ankle decoder receives live feedback.
- Early `capture_seat_rgbd.py` / `evaluate_seat_perception.py` experiments use simulator camera-world poses. They do not establish sensor-only moving registration. The strict startup capture does avoid those poses.
- `analyze_transition_geometry.py:G1Model` reconstructs joint transforms from the loaded USD export, but its caller supplies a world root pose. The new runtime instead computes all FK in pelvis coordinates and estimates the floating-base pose separately.
- The project root is not a Git repository. Isaac Lab is clean at `be9551260b269d4d8a964b71d5849810a1237b33`; the verified environment is Python 3.12.15, Isaac Sim 6.1.0.0, torch 2.11.0+cu130. RTX4090 is the simulation device. No installation is needed.
- Existing Ubuntu Codex processes were idle; no additional agent was started. All new simulations/replays use named persistent tmux sessions with logs and exit files.

## Information boundary

`shared_frame.py` has no simulator imports. Its update packet contains depth, intrinsics, calibrated camera mount, named-order encoders, gravity, integrated gyro rotation, sensor/receipt times, and an explicit reset epoch. It does not accept root/body/seat world poses, geometry-bank metadata, contact forces, COM, actor phase, or reset positions.

The body calibration uses joint frames and collider vertices exported from the actual loaded G1 model. Exported initial body poses and masses are ignored. All 37 simulated joint positions are used in the model's recorded order (including the 14 parked hand joints); the preserved actor still uses its separate 23-body interface. The pelvis is the model's root, so `T_P_pelvis = I`; other body transforms come from encoder FK.

`capture_shared_frame.py` uses the unchanged sensor05/contact02 controller solely to produce moving observations. New geometry is not fed into the actor in this milestone. `record_truth()` is a separate recorder; it reads privileged poses into `truth.npz`. `run_shared_frame.py` reads only `sensors.npz` and static calibration, never the geometry manifest or truth file. `evaluate_shared_frame.py` opens truth only after estimation, to measure errors and annotate visualizations. A `--track-online` capture invokes the same observer during simulation without feedback into the controller.

## Frame and timestamp contract

All transforms are right-handed homogeneous matrices. `T_A_B` maps B coordinates into A: `p_A = T_A_B @ [p_B,1]`. Distances are metres, angles radians, time seconds; quaternion interchange is XYZW.

- C: ROS optical frame, x right, y down, z forward.
- P: pelvis root, x forward, y left, z up in the nominal posture.
- O: local odometry frame; initially gravity aligned, XY origin at the initial pelvis projection, Z=0 inferred from the observed floor. Global XY and heading are gauge freedoms, not absolute localization.
- S: estimated seat frame; origin at the visible support's oriented bounding-center on the fitted plane, or the center inferred by an accepted physical-contour ellipse fit; z follows the upward normal. X follows the shorter rectangle/ellipse axis. Orientation is **modulo 180°**; near-square shapes also flag axis ambiguity. No automatic front/back resolution is claimed.

`T_O_C = T_O_P @ T_P_C`; `T_S_P = inverse(T_O_S) @ T_O_P`. Body links and collision geometry use `T_O_link = T_O_P @ FK_P_link(q)`.

Depth, q, and gravity refer to the same capture instant. Gyro increments cover the preceding capture interval, composed in body coordinates; the capture integrates consecutive 20 ms gyro samples. A real adapter must interpolate encoder/IMU history to image exposure time. The observer rejects >5 ms timestamp skew, nonmonotonic packets, >300 ms missing intervals, negative transport delay, and >250 ms old packets. An epoch change clears the map and all anchors. Outputs are explicitly timestamped at capture time; no prediction to receipt time is implied. 60 ms delayed tests preserve measurement-time synchronization.

## Estimation and limits

1. Unproject axial depth using camera intrinsics and calibrated extrinsics. Remove robot returns using conservative FK-oriented collider bounds with a 14 mm margin. This can discard valid near-body scenery; it avoids putting moving robot points into a static map. It is not an exact optical silhouette mask.
2. Integrate gyro rotation, correct tilt from gravity, and use an explicitly conditional two-foot kinematic translation prior. The prior uses encoder FK and geometrically observed floor proximity, **not contact-force labels**. It assumes stationary support; simultaneous foot slip is unobservable without independent scene information.
3. Register each scan against the local map with robust gravity-constrained point-to-plane ICP **and point-to-line depth contour constraints**. Depth jumps require valid pixels on both sides; image borders are not physical edges. Straight contours retain their along-line ambiguity. Report the four-DOF information rank (yaw plus XYZ), with angular increments scaled to displacement at a 0.5 m lever arm. Null-space directions retain the prior instead of inventing measurements. Floor-only depth cannot determine XY/yaw. Planar point normals are regularized to their fitted plane so independent depth noise cannot masquerade as lateral structure. Reject poor overlap or large residuals.
4. Fit floor/seat planes and retain observed connected support cells at 25 mm resolution. A convex hull is a display/axis aid only; footprint admission uses eroded occupancy and retains holes. Rectangular supports use a minimum-area rectangle axis. A curved support uses an ellipse only when physical depth contours fit within 6 mm RMS and span at least 200 degrees, occupy at least seven of twelve angular sectors, avoid parameter bounds, and have sufficient anisotropy to define an axis. This permits a model-inferred center but does not fill unknown support cells. A rejected area-moment heuristic biased partially visible rectangular seats; its results remain saved in clean05/noisy04/occlusion04. The first ellipse fit accepted two parallel noisy chair edges; its failure is preserved in noisy05. The 20×24 cm footprint gate is an axis-aligned heuristic in O, not a complete G1 sit-affordance proof. Missing or occluded extents are not silently treated as usable support.
5. Report pelvis pose, offset, yaw-axis error, pelvis-collider clearance to the plane, and a separate pelvis/thigh-proxy clearance over finite observed support. A negative shin-to-plane value is **not** a collision with the seat: observed obstacle distances and support overlap are separately reported. Unobserved obstacles and compliance remain unknown.
6. Retain the static seat map through short losses, update the pelvis from available gyro/kinematics, increase a conservative positional drift budget, and mark the result stale after one second without seat evidence. That drift budget is explicitly **not calibrated covariance**. Motion of an entirely occluded seat cannot be detected. Valid geometry is not authorization for motion.

The minimum signed plane distance over an entire thigh can be negative while the thigh is in front of the seat. Therefore the main `clearance_m` is the pelvis collider's signed minimum plane distance; `support_proxy_plane_clearance_m` includes whole pelvis/thigh proxies; `overlap_clearance_m` requires their projection to overlap observed support. None is a simulated force or a prediction of deformable contact.

## Reproduction

Run from `/home/kadir/humanoid-sitting`. Use fresh output directories; scripts refuse to overwrite results. No `setup.sh` or training command is needed.

```bash
bash scripts/run.sh python ../scripts/test_shared_frame.py

bash scripts/run.sh python ../scripts/capture_shared_frame.py \
  --count 6 --seconds 16 --output ../runs/shared_frame_capture_NEW --device cuda:0 \
  --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'

OPENBLAS_NUM_THREADS=1 bash scripts/run.sh python ../scripts/run_shared_frame.py \
  --capture ../runs/shared_frame_capture_NEW --output ../runs/shared_frame_clean_NEW

OPENBLAS_NUM_THREADS=1 bash scripts/run.sh python ../scripts/evaluate_shared_frame.py \
  --capture ../runs/shared_frame_capture_NEW --tracking ../runs/shared_frame_clean_NEW --video
```

Use replay `--mode noisy` for 5 mm Gaussian depth noise, 10% dropout, and 60 ms receipt delay; `--mode occlusion` for 0.5 s and 2 s total depth losses plus 60 ms delay. These are synthetic stress conditions, not a calibrated sensor model. Add capture `--track-online --count 1` to execute the observer during moving simulation.
