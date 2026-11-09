# Current perception handoff — 2026-10-10

Continue via `ssh kfuUbuntu`, project `/home/kadir/humanoid-sitting`. The Mac agent remains the primary agent. Do not start another agent remotely. Preserve all original data/checkpoints and intermediate experiments. No PPO or soft-landing tuning is authorized by this research phase.

Read `docs/shared-frame-observability-results.md` for the latest measured results, supplementing the earlier `docs/shared-frame-results.md` and `docs/shared-frame-audit.md`. The original handoff documents remain unchanged.

## Current implementation

`sitting_task/src/humanoid_sitting/shared_frame.py` is sensor-only: depth, static intrinsics/extrinsics, encoder FK, integrated gyro, and gravity. Runtime never reads the truth archive or geometry manifest. Current tracker SHA-256: `8738ee3d281505a99551c92175b0b5a5f725d43db4413c9fc0482684f5c72409`.

New behavior:

- `direct_clearance_m` measures pelvis collider distance to the current seat plane, independently of unobserved odometry directions. It is `None` without a current associated plane.
- `quality.quantities` reports separate clearance, forward, lateral, and yaw flags with reasons. Alignment dimensions require current observed motion directions and physical boundary evidence. Full-seat extent is never inferred solely from a cropped image boundary. Ellipse center/axis is explicitly a model hypothesis.
- `valid` now means all observed-patch alignment gates pass; `legacy_valid` archives the former scalar-budget rule. It is not a controller enable signal. `clearance_m` remains the map prediction, distinct from current measurement.
- Prolonged depth loss triggers a bounded ±25 cm XY reacquisition search using physical contours. Ambiguous/underconstrained matches are rejected; map fusion waits for reacquisition.
- Map truncation no longer cuts off negative-X geometry. Support-height points are prioritized, with remaining points sampled across the spatial extent. Maximum map size remains 40,000 points/12,000 edges.
- `capture_shared_frame.py --camera-profile rear_down` mounts an additional observer camera at `[-0.22, 0, 0.55]` in the pelvis frame, looking down. The original camera still supplies the frozen controller's startup geometry. Static `T_P_C` is stored in the sensor archive; replay validates that it is a single rigid transform. Old recordings retain the legacy calibration fallback.

## Canonical validated artifacts

- Physics replay: `shared_frame_quality_clean_04`, `shared_frame_quality_noisy_04`, `shared_frame_quality_occlusion_03`; input `shared_frame_capture_01`.
- Sensor-level adversarial regressions: `shared_frame_stress_04` / `shared_frame_stress_capture_01` and `shared_frame_holdout_02` / `shared_frame_holdout_capture_01`. The second bank exposed a bug and is now a regression set, not an untouched holdout.
- Actual online bench: `shared_frame_quality_live_02`; exact replay `shared_frame_quality_live_replay_02`.
- Controlled camera comparison: `shared_frame_rear_down_capture_02` / `shared_frame_rear_down_02`, and `shared_frame_camera_comparison_02`. Added-camera RGB is blank; valid axial depth is the observer input and final visualized channel. RGB capture remains to be fixed separately.
- Truth-free fixture check: `shared_frame_quality_sensor_only_02`; contract `shared_frame_quality_contract.json`.
- Aggregate audit: `shared_frame_observability_audit_final`. Videos: `shared_frame_quality_visuals` and `shared_frame_quality_visuals_camera_02` (depth shown for the added camera).
- Preservation: `shared_frame_quality_integrity.json`; 117 original files, archived baseline, checkpoints, and Isaac Lab unchanged. Unit-test log: `logs/shared_frame_quality_tests.log` (21 passed).

Original view: clearance available 966/966 clean and noisy frames; clean/noisy clearance P95 0.110/0.770 cm. Missing depth: all 150 samples abstain. Strict full alignment availability remains only 43/966 clean, because seat extents or motion directions are missing. Blind 12/16 cm slides recover on the first returning depth frame, with 0.040/0.141 cm lateral error. Current-quality flags have no frame violations of the provisional 2 cm clearance, 3 cm offset, 5 degree axis-yaw gates in the five named regressions.

The controlled new-camera bench trial is the main new positive result: identical measured root trajectory to the original-camera trial, full alignment 0/161 → 119/161 (74%), clearance 161/161, forward 160/161, lateral 129/161, yaw 160/161. Accepted P95 errors: clearance 0.058 cm, forward 0.277 cm, lateral 0.290 cm, yaw 0.083 degrees. Online and replayed poses agree exactly. This is one physics scene, not validation across all six surface families or of a physical mounting bracket.

## Preserve and do not promote

All earlier variants remain for provenance. Early quality versions accepted blind lateral predictions; the first noisy 180-degree-turn bank exposed coordinate-sorted map cropping. `shared_frame_rear_down_capture_01` / `shared_frame_rear_down_01` is a rejected camera comparison: changing the old controller's startup view caused a rapid descent outside the tracker regime. The second camera trial fixes experimental isolation and reproduces the original trajectory exactly. Do not summarize the first probe as a successful view experiment or hide its failure.

## Next research step

Validate the rear vertical observer view over the full six-surface bank, depth noise/dropout, occlusion/reacquisition, and fresh geometry. Keep the original controller camera separate and weights frozen. Evaluate physical occlusion and finite support, not only an infinite-plane clearance. There is no perception-driven alignment/descent controller yet. Hardware calibration, probabilistic uncertainty, hard real-time behavior, fast fall tracking, moving furniture, and arbitrary relocalization remain unvalidated. The analytical motion fixtures are not physical turning/slip controllers.

Use new output directories; replay scripts refuse overwrites. Launch long simulations through persistent tmux sessions. No additional package or driver installation is needed.
