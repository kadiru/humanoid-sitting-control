# Frozen observer: six-surface validation

Frozen estimator and frozen controller, six previously seen simulated surface geometries. Runtime uses sensor archive only. Truth is offline. Quality budgets are heuristics; no control authorization, hardware calibration or untouched geometry generalization claim.

| Condition | Complete alignment | Clearance | Forward | Lateral | Yaw | Accepted threshold violations |
|---|---:|---:|---:|---:|---:|---:|
| clean | 602/966 | 966/966 | 640/966 | 771/966 | 800/966 | 0 |
| noisy | 640/966 | 966/966 | 640/966 | 800/966 | 800/966 | 0 |
| occlusion | 501/966 | 816/966 | 540/966 | 638/966 | 668/966 | 0 |
| original | 43/966 | 966/966 | 96/966 | 244/966 | 290/966 | 0 |

Thresholds: 2 cm clearance, 3 cm forward/lateral, 5 degrees axis yaw. Errors are evaluated per accepted frame; all rejected-frame errors are retained.

| Surface | Original alignment | New clean | New noisy | New occlusion |
|---|---:|---:|---:|---:|
| chair / rect | 0/161 | 0/161 | 0/161 | 0/161 |
| bench / rect | 0/161 | 124/161 | 160/161 | 98/161 |
| box / rect | 0/161 | 159/161 | 160/161 | 134/161 |
| rigid_couch / rect | 0/161 | 0/161 | 0/161 | 0/161 |
| bench / chamfered | 0/161 | 160/161 | 160/161 | 135/161 |
| box / ellipse | 43/161 | 159/161 | 160/161 | 134/161 |

## Accepted-error P95

| Condition | Clearance (cm) | Forward (cm) | Lateral (cm) | Yaw (degrees) |
|---|---:|---:|---:|---:|
| clean | 0.058 | 0.312 | 0.286 | 0.116 |
| noisy | 0.340 | 0.503 | 0.394 | 0.366 |
| occlusion | 0.054 | 0.312 | 0.298 | 0.129 |
| original | 0.110 | 0.343 | 0.271 | 0.426 |

Live/replay agreement: `{'T_O_P': 0.0, 'T_S_P': 0.2674723620890509, 'quality_flags_identical': True, 'same_observer_version_as_online_capture': False, 'scope': 'Revised shape association: odometry must agree; relative seat pose and quality can change. This is not live validation of the revised shape fit.'}`.
Depth blackout abstention: `{'samples': 150, 'alignment_accepted': 0, 'clearance_accepted': 0}`.

## Rejected measurements and finite support

- clean: clearance: {}; forward: {'motion_direction_unobserved': 6, 'physical_extent_unconfirmed': 324, 'seat_axis_ambiguous': 2}; lateral: {'motion_direction_unobserved': 82, 'physical_extent_unconfirmed': 164, 'seat_axis_ambiguous': 2, 'unobserved_motion_budget': 20}; yaw: {'motion_direction_unobserved': 6, 'axis_unconfirmed': 163, 'seat_axis_ambiguous': 2}.
  Unsupported observed cells: {'n': 966, 'p95': 0.014975041597337771, 'max': 0.019736842105263157}.
- noisy: clearance: {}; forward: {'motion_direction_unobserved': 6, 'physical_extent_unconfirmed': 322}; lateral: {'motion_direction_unobserved': 6, 'physical_extent_unconfirmed': 161}; yaw: {'motion_direction_unobserved': 6, 'axis_unconfirmed': 161}.
  Unsupported observed cells: {'n': 966, 'p95': 0.02810304449648712, 'max': 0.029239766081871343}.
- occlusion: clearance: {'no_current_seat_plane': 150}; forward: {'motion_direction_unobserved': 167, 'physical_extent_unconfirmed': 324, 'seat_axis_ambiguous': 2, 'pose_reacquisition_required': 113, 'seat_observation_expired': 77, 'unobserved_motion_budget': 28}; lateral: {'motion_direction_unobserved': 245, 'physical_extent_unconfirmed': 164, 'seat_axis_ambiguous': 2, 'pose_reacquisition_required': 113, 'seat_observation_expired': 77, 'unobserved_motion_budget': 58}; yaw: {'motion_direction_unobserved': 167, 'axis_unconfirmed': 163, 'seat_axis_ambiguous': 2, 'pose_reacquisition_required': 113, 'seat_observation_expired': 77, 'unobserved_motion_budget': 5}.
  Unsupported observed cells: {'n': 966, 'p95': 0.015, 'max': 0.019736842105263157}.

## Experimental isolation

Offline differences between the original and added-camera root trajectories:

- Case 0 (chair): max position 0 cm; max rotation 0 degrees.
- Case 1 (bench): max position 0 cm; max rotation 0 degrees.
- Case 2 (box): max position 0 cm; max rotation 0 degrees.
- Case 3 (rigid_couch): max position 0 cm; max rotation 0 degrees.
- Case 4 (bench): max position 0 cm; max rotation 0 degrees.
- Case 5 (box): max position 0 cm; max rotation 0 degrees.

## Limitations

The six geometries are existing regression scenes. The camera mount is simulated and its RGB channel remains unavailable; depth is valid and is shown. Clearance measures the current seat plane; finite support occupancy is measured separately. Observed support admission does not certify the whole body or a sitting trajectory. Partial view, occlusion, and unobserved motion can still make alignment unavailable. No PPO, controller tuning, or perception-driven actuation was introduced.
