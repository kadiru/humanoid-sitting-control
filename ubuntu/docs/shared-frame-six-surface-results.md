# Six-surface depth perception: validation and contour association fix

The rear vertical observer camera substantially improves measured pelvis-to-seat alignment on benches and boxes. Testing all six surfaces exposed a shape-association bug on the rounded box; it is fixed and the failing recordings are retained. Chair and couch boundaries remain incomplete, so this is not a perception-driven controller release.

The original controller camera, controller weights, physics, and trajectories were held fixed. No PPO or soft-landing tuning was performed. Runtime uses only depth, static calibration, encoder kinematics, gyro increments, and gravity. Simulator truth supplies offline error labels.

## Results after the fix

| Condition | All alignment gates | Clearance available | Clearance P95 (mm) | Forward P95 (mm) | Lateral P95 (mm) | Yaw P95 (deg) | Threshold violations |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 602/966 | 966/966 | 0.58 | 3.12 | 2.86 | 0.116 | 0 |
| noisy | 640/966 | 966/966 | 3.40 | 5.03 | 3.94 | 0.366 | 0 |
| occlusion | 501/966 | 816/966 | 0.54 | 3.12 | 2.98 | 0.129 | 0 |
| original | 43/966 | 966/966 | 1.10 | 3.43 | 2.71 | 0.426 | 0 |

P95 errors apply only where each quantity is accepted. Every rejected estimate and its error remains in the run files. Limits are 2 cm clearance, 3 cm forward/lateral, and 5 degrees axis yaw; these are provisional simulation criteria, not calibrated probabilities. Noisy means 5 mm Gaussian axial noise, 10% independent missing pixels, and 60 ms receipt latency. Occlusion removes all depth from 6–6.5 s and 8–10 s. Higher availability in the noisy run reflects sensitivity of the heuristic edge/observability gates; it is not evidence that noise improves sensing.

| Surface | Original camera, clean | New camera, clean | New camera, noisy | New camera, occlusion |
|---|---:|---:|---:|---:|
| chair / rect | 0/161 | 0/161 | 0/161 | 0/161 |
| bench / rect | 0/161 | 124/161 | 160/161 | 98/161 |
| box / rect | 0/161 | 159/161 | 160/161 | 134/161 |
| rigid_couch / rect | 0/161 | 0/161 | 0/161 | 0/161 |
| bench / chamfered | 0/161 | 160/161 | 160/161 | 135/161 |
| box / ellipse | 43/161 | 159/161 | 160/161 | 134/161 |

Clearance is distance from the pelvis collision geometry to the currently observed seat plane. The finite observed support cells are retained separately; alignment availability is not a guarantee that a complete sitting trajectory fits.
Clean all-frame P95 body-position error: 0.306 cm; pelvis-position error: 0.241 cm. These use only the initial XY/yaw gauge alignment, then compare the reconstructed body to truth offline.

## Failure found and repaired

Contours were associated by plane height before the shape fit. Six unrelated edges at the same height as the seat contaminated the rounded support contour. The robust ellipse optimizer found a plausible model, but its all-edge residual rejected it. The fallback minimum-area rectangle then reported a polygon-facet axis as reliable.
At sensor t=2.0 s, the reported axis changed from -0.04° to -5.12° after removing the unrelated edges; the accepted ellipse residual is 2.38 mm. The fix associates edges with the selected connected support component using a 3 cm proximity gate before shape fitting. It uses no furniture labels or truth.

| Condition | Before: violating quantity/frame pairs | After |
|---|---:|---:|
| clean | 190 | 0 |
| noisy | 103 | 0 |
| occlusion | 165 | 0 |

A frame can contribute both yaw and lateral violations in this table. The pre-fix runs remain in `shared_frame_rear_down_six_*_01`. The corrected runs use suffix `_02`. The revised observer SHA-256 is `50000e1a059f380d033f9435194ccf7101c4cd6a8418ce81dca085e1cb53fbb7`.

## Validation and preservation

- 22 CPU tests passed, including a new disconnected-contour regression. Existing tests still reject image-crop boundaries, ambiguous straight-edge reacquisition, and blind common-foot slips.
- All 150 fully missing-depth frames rejected both clearance and alignment. Reacquisition after depth returns at 10 s occurred at: case 0: 10.7 s, case 1: 10.0 s, case 2: 10.0 s, case 3: 10.4 s, case 4: 10.0 s, case 5: 10.0 s.
- Original and added-camera trajectories match exactly in all six physics scenes (root position and orientation). The controller inputs and frozen weights were preserved.
- A fresh six-scene physics capture with the revised observer matched all 966 replayed frames. Every consumed sensor/calibration array was exactly equal before reusing the replay; maximum pose/clearance difference was 0, with identical nested quality values.
- Preservation check: 117 original source files, archived baselines, checkpoints, and Isaac Lab remained unchanged. Free Ubuntu disk at verification: 31.40 GiB.

| Prior regression | Clearance / forward / lateral / yaw accepted | Above-limit accepted samples |
|---|---:|---:|
| shared_frame_quality_clean_05 | 966 / 96 / 244 / 290 | 0 |
| shared_frame_quality_noisy_05 | 966 / 97 / 336 / 336 | 0 |
| shared_frame_quality_occlusion_04 | 816 / 85 / 246 / 257 | 0 |
| shared_frame_stress_05 | 461 / 26 / 338 / 338 | 0 |
| shared_frame_holdout_03 | 444 / 78 / 244 / 244 | 0 |

The two analytical motion banks remain regressions, including the second bank previously used to diagnose map cropping. They are not untouched holdouts or physical walking/slip controllers.

## Remaining work

The chair has no confirmed rear seat extent; the couch also lacks confirmed side extents and a seat axis. Their motion registration can be fully constrained while seat geometry remains incomplete. The current boundary extractor primarily keeps depth-discontinuity contours on the seat plane. Boundaries where the seat meets a backrest or armrest need explicit obstacle/plane-junction evidence, or additional views; do not relax the existing gates to mask this gap.
Next implement and validate these supported seat boundaries, then freeze the observer again and test fresh heights, dimensions, offsets, and yaws plus extrinsic-calibration errors. Do this before enabling perception-driven alignment. No perception-driven descent is enabled. The virtual mount has not been physically verified; the added-camera RGB channel remains blank and all visuals use its valid depth channel.

## Artifacts

- `runs/shared_frame_six_validation_02`: report, per-surface measurements, and depth comparison. Final 24-second common-frame video with revised caption spacing: `runs/shared_frame_six_visuals_03/six_surface_shared_frame.mp4`.
- `runs/shared_frame_component_diagnostic_01`: before/after contour association figure.
- `runs/shared_frame_component_regression_01`: five prior regression audits.
- `runs/shared_frame_six_live_contract_02.json` and `runs/shared_frame_six_integrity_03.json`: live/replay and preservation evidence.
- `scripts/validate_shared_frame_six.sh` preserves the initial experiment; `scripts/validate_shared_frame_component_fix.sh` reproduces the corrected benchmark in new output directories.
