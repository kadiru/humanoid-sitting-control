# Observability and measurement availability audit

Offline errors use the separate truth archive. Runtime quality uses only sensor geometry and kinematics. Directional allowances are engineering heuristics, not calibrated probabilities.

| Run | Old valid | Current clearance | Forward | Lateral | Yaw | All alignment gates |
|---|---:|---:|---:|---:|---:|---:|
| shared_frame_quality_clean_04 | 460/966 | 966/966 | 96/966 | 244/966 | 290/966 | 43/966 |
| shared_frame_quality_noisy_04 | 562/966 | 966/966 | 97/966 | 336/966 | 336/966 | 57/966 |
| shared_frame_quality_occlusion_03 | 450/966 | 816/966 | 85/966 | 246/966 | 257/966 | 43/966 |
| shared_frame_stress_04 | 364/486 | 461/486 | 26/486 | 338/486 | 338/486 | 26/486 |
| shared_frame_holdout_02 | 347/486 | 444/486 | 78/486 | 244/486 | 244/486 | 78/486 |

Errors conditional on a quality flag are diagnostic, not an independent calibration set. Forward/lateral frame thresholds are 3 cm, clearance 2 cm, and axis yaw 5 degrees. The main `valid` flag now requires all alignment gates; the former rule is archived as `legacy_valid`. Inspect `quality.quantities` for each measurement and its reasons.
