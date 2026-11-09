# Reference-map integrity: validated results

Evaluated 14,013 case/frame samples across 17 conditions. Accepted errors exceeded the unchanged per-frame limits in **0 quantity/frame pairs**. These are conditional accuracy results; rejected predictions remain logged and may drift substantially.

## Change and measurement contract

Pose registration now uses an immutable first depth scan. Newly fused geometry cannot alter that reference. Matching checks surface-normal agreement and the residual at the final pose; aborted solver steps cannot be accepted. Runtime registration requires over 60% correspondence coverage and under 8 mm plane RMS. Geometry fusion requires all four gravity-constrained pose directions to be observed. These are engineering gates chosen while diagnosing the retained failures, not calibrated probabilities. The evaluation limits remain 2 cm clearance, 3 cm forward/lateral, and 5 degrees axis yaw.

After 0.5 s without accepted tracking, or a depth blackout, a bounded contour-based recovery search uses the fixed reference. Repeated searches during tracking loss are spaced 0.5 s apart; large local corrections can trigger an immediate check. Ambiguous or insufficient matches remain withheld. A stationary-foot prior alone cannot certify tracking. The display map may continue to show predictions, but rejected poses are not fused.

Current-plane clearance is evaluated separately from map-relative offsets. Plane extrapolation uses 6 cm effective spatial samples, a 5 mm noise floor, and an 8 mm model allowance; ill-conditioned or excessively extrapolated fits are withheld. Low fit residual on a thin strip is insufficient. This allowance is heuristic, not a statistical guarantee.

## Accuracy and availability

| Bank / condition | Alignment accepted | Clearance accepted | Max clearance (cm) | Max forward (cm) | Max lateral (cm) | Max yaw (deg) | Violations |
|---|---:|---:|---:|---:|---:|---:|---:|
| existing_rear/clean | 680/966 | 966/966 | 0.065 | 0.748 | 0.456 | 0.318 | 0 |
| existing_rear/noisy | 721/966 | 959/966 | 0.595 | 0.894 | 1.304 | 0.826 | 0 |
| existing_rear/occlusion | 539/966 | 813/966 | 0.065 | 0.748 | 0.456 | 0.318 | 0 |
| existing_original/clean | 17/966 | 750/966 | 0.111 | 0.331 | 1.329 | 2.518 | 0 |
| existing_original/noisy | 12/966 | 880/966 | 1.351 | 0.618 | 1.246 | 4.635 | 0 |
| existing_original/occlusion | 17/966 | 648/966 | 0.110 | 0.331 | 1.307 | 2.065 | 0 |
| known_failures/clean | 621/966 | 844/966 | 0.123 | 0.463 | 0.951 | 0.237 | 0 |
| known_failures/noisy | 594/966 | 843/966 | 0.486 | 0.802 | 0.460 | 0.728 | 0 |
| known_failures/occlusion | 384/966 | 696/966 | 0.118 | 0.463 | 0.951 | 0.237 | 0 |
| plane_identity_failures/clean | 434/966 | 620/966 | 0.130 | 0.437 | 0.362 | 0.250 | 0 |
| plane_identity_failures/noisy | 454/966 | 619/966 | 0.528 | 0.637 | 0.753 | 0.762 | 0 |
| plane_identity_failures/occlusion | 313/966 | 538/966 | 0.130 | 0.437 | 0.362 | 0.250 | 0 |
| new_geometry/clean | 355/483 | 480/483 | 0.060 | 0.346 | 0.463 | 0.947 | 0 |
| new_geometry/noisy | 368/483 | 480/483 | 0.524 | 0.649 | 0.622 | 0.434 | 0 |
| new_geometry/occlusion | 234/483 | 408/483 | 0.060 | 0.346 | 0.463 | 0.947 | 0 |
| stress | 0/486 | 461/486 | 0.005 | unavailable | 0.291 | 0.090 | 0 |
| motion | 58/486 | 436/486 | 0.317 | 0.675 | 0.224 | 0.294 | 0 |

Noisy inputs add 5 mm axial noise, 10% missing pixels and 60 ms delay. Occlusion removes all depth for 6–6.5 s and 8–10 s. Every blackout frame withholds alignment and clearance. Higher noisy availability can reflect heuristic gate sensitivity; it is not a sensing advantage.

## Tradeoff against the previous observer

| Bank / condition | Previous alignment | Current alignment | Previous violations | Current violations |
|---|---:|---:|---:|---:|
| existing_rear/clean | 875/966 | 680/966 | 0 | 0 |
| existing_rear/noisy | 960/966 | 721/966 | 0 | 0 |
| existing_rear/occlusion | 712/966 | 539/966 | 0 | 0 |
| known_failures/clean | 722/966 | 621/966 | 46 | 0 |
| known_failures/noisy | 826/966 | 594/966 | 152 | 0 |
| known_failures/occlusion | 584/966 | 384/966 | 15 | 0 |
| plane_identity_failures/clean | 434/966 | 434/966 | 233 | 0 |
| plane_identity_failures/noisy | 456/966 | 454/966 | 235 | 0 |
| plane_identity_failures/occlusion | 313/966 | 313/966 | 188 | 0 |

Availability falls when the original view no longer constrains pose. This fixes false confidence by both improving reference consistency and refusing unsupported estimates. It does not establish uninterrupted tracking. The first, incomplete fixed-reference candidate remains in runs ending `integrity_*_01`; its noisy failures are preserved.

## New geometry: frozen observer test

| Case | Surface | Depth / width / height (m) | X / Y (m) | Yaw (deg) | Clean / noisy / occluded alignment |
|---|---|---|---|---:|---|
| 0 | chair / rect | 0.55 / 0.96 / 0.32 | -0.41 / 0.01 | -8.0 | 159/161 / 160/161 / 58/161 |
| 1 | rigid_couch / rect | 0.56 / 1.02 / 0.29 | -0.39 / -0.018 | 13.0 | 77/161 / 77/161 / 59/161 |
| 2 | box / ellipse | 0.49 / 0.94 / 0.31 | -0.405 / 0.016 | -9.0 | 119/161 / 131/161 / 117/161 |

The geometry file was fixed before capture. The same observer source hash was used in all 17 conditions and the new online capture: `16aef05dcc09da997a9cd8da7483e2295bffce6e21101a3bcf0e9f6760c5a4b6`. This is one small additional simulation bank, not arbitrary-scene generalization.

## Verification and limits

- 35 focused CPU tests passed, including immutable-reference isolation, refusal to fuse blind frames, incompatible normals, broad versus thin/collinear plane support, plane identity after loss, ambiguous support pairs, geometry, timestamps and runtime input contracts.
- Online/replay agreement on all 483 new-bank samples: {'T_O_P': 0.0, 'T_S_P': 0.0, 'direct_clearance_m': 0.0, 'quality_exact_match': True, 'registration_exact_match': True}.
- A separate replay directory containing only sensors.npz reproduces all 18 checked outputs exactly; runtime reads no truth or geometry manifest.
- Original sources, checkpoints and Isaac Lab are checked separately in the preservation record. No controller tuning, PPO or perception-driven actuation was introduced.

Per-case rejection durations, recovery counts, accepted P95 and maximum errors, all-emitted errors and compute timings are in results.json. Computation was measured during concurrent validation jobs; this is not a real-time benchmark. The new capture uses 10 Hz sensor timestamps while simulation may wait for processing.

The first reference limits field-of-view coverage and recovery. During tracking loss, the lowest sufficiently broad observed plane is treated as floor; this assumes a visible flat floor and does not resolve arbitrary platforms or repeated levels. Static surfaces, ideal encoder/IMU equivalents and the virtual camera mount remain assumptions. Finite support holes and whole-body collision margins are separate from plane clearance. Runtime quality is heuristic. Alignment/descent control remains disabled.

The preceding candidate passed 11 prior conditions but failed the next six-surface bank: 654 clearance violations from lost plane identity and two noisy forward violations. Those frozen failures remain archived. The corrected observer uses registered height only when registration is trusted; otherwise it requires a visible lower floor and the learned floor-to-seat separation in the same scan. A lone plane or ambiguous pair is rejected. Local corrections above 4 cm or 0.06 rad require multi-hypothesis recovery before fusion. The six-surface bank is now regression data.

Next develop validated reference/keyframe transitions to recover coverage without reintroducing map contamination, then test camera/IMU calibration bias and execution latency before adding alignment control. Preserve this bank as regression data; reserve another untouched bank for subsequent validation.

Video: runs/shared_frame_integrity_visuals_03/map_integrity.mp4. Full new artifacts are on /data/humanoid-sitting-perception, linked from project runs/.
