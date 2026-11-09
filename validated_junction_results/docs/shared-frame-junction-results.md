# Measured seat/wall junctions: validation results

The observer now uses bounded intersections of measured vertical surfaces and the measured seat plane to confirm seat boundaries. It does not need furniture labels, model dimensions, simulator poses, or a new controller.

**Fresh-geometry validation failed:** accepted offsets exceed 3 cm, and the noisy run also has clearance and yaw outliers. Boundary recovery passes the existing-scene regressions, but this observer is not ready to authorize alignment control. The frozen fresh-bank failures are retained, not tuned away.

## Availability on existing scenes

| Surface | Before: clean alignment | After: clean | After: noisy | After: occlusion |
|---|---:|---:|---:|---:|
| chair / rect | 0/161 | 160/161 | 160/161 | 128/161 |
| bench / rect | 124/161 | 124/161 | 160/161 | 98/161 |
| box / rect | 159/161 | 159/161 | 160/161 | 134/161 |
| rigid_couch / rect | 0/161 | 113/161 | 160/161 | 83/161 |
| bench / chamfered | 160/161 | 160/161 | 160/161 | 135/161 |
| box / ellipse | 159/161 | 159/161 | 160/161 | 134/161 |

## Accepted errors and availability

| Bank / condition | All alignment gates | Clearance available | Clearance P95 (mm) | Forward P95 (mm) | Lateral P95 (mm) | Yaw P95 (deg) | Violating quantity/frames |
|---|---:|---:|---:|---:|---:|---:|---:|
| existing_rear/clean | 875/966 | 966/966 | 0.581 | 6.197 | 2.793 | 0.297 | 0 |
| existing_rear/noisy | 960/966 | 966/966 | 3.398 | 6.460 | 4.881 | 0.349 | 0 |
| existing_rear/occlusion | 712/966 | 816/966 | 0.535 | 6.227 | 2.899 | 0.298 | 0 |
| existing_original/clean | 43/966 | 966/966 | 1.098 | 3.427 | 2.712 | 0.426 | 0 |
| existing_original/noisy | 57/966 | 966/966 | 7.703 | 6.478 | 9.861 | 1.589 | 0 |
| existing_original/occlusion | 43/966 | 816/966 | 1.086 | 3.438 | 9.454 | 1.464 | 0 |
| fresh_geometry/clean | 722/966 | 847/966 | 0.563 | 27.611 | 28.566 | 0.227 | 46 |
| fresh_geometry/noisy | 826/966 | 848/966 | 3.738 | 33.164 | 35.315 | 0.663 | 152 |
| fresh_geometry/occlusion | 584/966 | 696/966 | 0.541 | 27.659 | 8.477 | 0.230 | 15 |
| shared_frame_junction_stress_01 | 26/486 | 461/486 | 0.000 | 11.522 | 0.556 | 0.067 | 0 |
| shared_frame_junction_motion_01 | 78/486 | 444/486 | 2.263 | 10.345 | 2.030 | 0.245 | 0 |

Errors are conditional on each quantity being accepted; all rejected-frame errors remain available. Provisional per-frame limits: 2 cm clearance, 3 cm forward/lateral, 5° axis yaw. Noisy inputs add 5 mm Gaussian axial noise, 10% missing pixels, and 60 ms latency. Occlusion removes all depth at 6–6.5 s and 8–10 s. Flags remain heuristic, not calibrated probabilities.

The fresh-bank maxima show failures that P95 alone can hide:

| Condition | Clearance max (cm) | Forward max (cm) | Lateral max (cm) | Yaw max (deg) |
|---|---:|---:|---:|---:|
| clean | 0.123 | 4.201 | 3.578 | 0.696 |
| noisy | 3.291 | 3.694 | 7.207 | 7.513 |
| occlusion | 0.118 | 3.360 | 4.048 | 0.318 |

Across the 11 conditions, 9,666 case/frame samples were evaluated. The eight existing regressions have zero threshold violations; the three fresh conditions have 46, 152, and 15 violating quantity/frame pairs. A frame can contribute more than one violation. Higher noisy availability reflects sensitivity of heuristic gates, not evidence that noise improves sensing. Remaining clean couch rejections in the existing bank are mainly unobserved lateral motion, even when its seat boundaries are confirmed.

## Fresh geometry bank

| Surface | Depth × width × height (m) | Seat x/y (m) | Yaw (deg) | Clean / noisy / occluded alignment |
|---|---|---|---:|---|
| chair / rect | 0.58 × 0.94 × 0.34 | -0.42 / -0.025 | -10.0 | 160/161 / 160/161 / 135/161 |
| bench / rect | 0.46 × 0.9 × 0.26 | -0.37 / 0.035 | 12.0 | 96/161 / 99/161 / 72/161 |
| box / rect | 0.56 × 0.8 × 0.35 | -0.43 / -0.03 | -12.0 | 105/161 / 114/161 / 74/161 |
| rigid_couch / rect | 0.58 × 1.1 × 0.33 | -0.42 / 0.025 | 9.0 | 42/161 / 133/161 / 37/161 |
| bench / chamfered | 0.5 × 0.9 × 0.28 | -0.38 / -0.03 | -11.0 | 160/161 / 160/161 / 135/161 |
| box / ellipse | 0.57 × 1.0 × 0.325 | -0.41 / 0.03 | 10.0 | 159/161 / 160/161 / 131/161 |

The geometry JSON was recorded before capture; the same observer hash was used for every run. This is a small new simulation bank, not evidence of arbitrary geometry generalization. The analytical motion banks remain prior regressions.

## Geometry evidence and verification

- Vertical surfaces are fitted from depth-cloud points/normals with bounded deterministic plane proposals. Each junction requires at least 24 wall points, ≤8 mm fit RMS, ≥12 cm wall span, visible height above 8 cm, and samples within 5 cm of seat height. Only locally supported intersection samples within 3 cm of observed seat points are retained. No unseen span is filled. These are geometric tolerances, not proof of physical attachment.
- Junction samples qualify the seat boundary; they never enter the odometry registration map. Existing pelvis/seat poses and direct clearance are unchanged: {'T_O_P': 0.0, 'T_S_P': 0.0, 'direct_clearance_m': 0.0}.
- 27 tests passed, including measured wall support, rotated junctions, floating-wall rejection, distant-wall rejection, and preservation of gaps.
- Fresh six-scene online/replay check (966 frames): {'T_O_P': 0.0, 'T_S_P': 0.0, 'direct_clearance_m': 0.0, 'quality_exact_match': True}.
- No PPO, training, soft-landing tuning, or perception-driven actuation was introduced. Original source/checkpoint preservation is separately verified.

## Practical limits and next step

The scene map assumes static surfaces. A near-base point within 5 cm is evidence for a bounded geometric intersection, not a certified contact seam. Clearance is to the observed seat plane; finite support holes and body geometry remain separate. The virtual camera mount and ideal encoder/IMU equivalents are still simulation assumptions.
The fresh-bank failures include accumulated-map pose errors of 5.7–7.5 cm in inspected chair/bench frames. A shifted map can retain low local registration residuals and reset its engineering allowance, while its seat center shifts by a different amount. Accurate current-plane clearance therefore coexists with wrong map-relative offsets. Current-frame boundary checks reject the incomplete shape in the inspected failing dimensions; this is diagnostic evidence, not a deployed fix. See `runs/shared_frame_junction_drift_diagnostic_01/diagnostic.json`.
Next fix map consistency and recovery during partial visibility or fast motion, and distinguish current relative measurements from map predictions. Revalidate the preserved failures without relaxing error limits. Then test extrinsic/IMU calibration errors and finite body-support margins before an alignment-only controller; keep descent disabled.

Machine-readable measurements: `runs/shared_frame_junction_results_02/results.json`. Final visualizations show observed junctions in cyan. All original recordings and prior observer snapshots are retained.
