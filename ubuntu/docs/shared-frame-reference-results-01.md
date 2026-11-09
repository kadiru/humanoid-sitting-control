# Directly anchored reference views: results

15,945 case/frame samples; 5 accepted quantity/frame violations; 534 frames used reference consensus. Samples across sensor modes are correlated.

## Method

The original scan remains immutable. At most eight separately captured views can be admitted, only from a full-rank pose registered directly to the original scan. Reverse registration must also be full rank, exceed 60% overlap, have under 8 mm plane RMS and close within 1 cm / 0.015 rad. Admissions are at least one second apart and require 3 cm translation or 0.08 rad rotation from the last admitted view. Each admitted array is copied and read-only. Views obtained from the bank cannot admit new views; graph depth is bounded. The display map is never a registration target.

The original reference is tried first. A fallback requires at least two views whose independently fitted poses all agree within 1 cm / 0.015 rad. Large corrections and lost tracking use the existing bounded recovery checks. A successful fallback carries another 8 mm engineering registration allowance. These consistency checks are not statistically independent measurements or calibrated probabilities.

## Accuracy and availability

| Condition | Before pose | After pose | Gained / lost | Bank matches | Clearance | Max clearance cm | Max forward cm | Max lateral cm | Max yaw deg | Violations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rear_clean | 680 | 873/966 | 193 / 0 | 153 | 966 | 0.065 | 0.748 | 0.456 | 0.318 | 0 |
| rear_noisy | 721 | 894/966 | 173 / 0 | 131 | 959 | 0.595 | 0.894 | 4.274 | 0.848 | 5 |
| rear_occlusion | 539 | 688/966 | 149 / 0 | 81 | 813 | 0.065 | 0.748 | 0.456 | 0.318 | 0 |
| original_clean | 17 | 17/966 | 0 / 0 | 0 | 750 | 0.111 | 0.331 | 1.329 | 2.518 | 0 |
| original_noisy | 12 | 12/966 | 0 / 0 | 0 | 880 | 1.351 | 0.618 | 1.246 | 4.635 | 0 |
| original_occlusion | 17 | 17/966 | 0 / 0 | 0 | 648 | 0.110 | 0.331 | 1.307 | 2.065 | 0 |
| fresh_clean | 621 | 622/966 | 1 / 0 | 1 | 844 | 0.123 | 0.463 | 0.951 | 0.237 | 0 |
| fresh_noisy | 594 | 650/966 | 56 / 0 | 1 | 844 | 0.486 | 0.802 | 0.460 | 0.728 | 0 |
| fresh_occlusion | 384 | 493/966 | 109 / 0 | 6 | 696 | 0.118 | 0.463 | 0.951 | 0.237 | 0 |
| unseen_clean | 434 | 469/966 | 35 / 0 | 25 | 620 | 0.130 | 0.437 | 0.362 | 0.250 | 0 |
| unseen_noisy | 454 | 498/966 | 44 / 0 | 31 | 620 | 0.528 | 0.637 | 1.049 | 0.762 | 0 |
| unseen_occlusion | 313 | 370/966 | 57 / 0 | 6 | 538 | 0.130 | 0.437 | 0.362 | 0.250 | 0 |
| holdout_clean | 355 | 394/483 | 39 / 0 | 35 | 480 | 0.060 | 0.346 | 0.463 | 0.947 | 0 |
| holdout_noisy | 368 | 397/483 | 29 / 0 | 29 | 480 | 0.524 | 0.649 | 0.622 | 0.434 | 0 |
| holdout_occlusion | 234 | 339/483 | 105 / 0 | 18 | 408 | 0.060 | 0.346 | 0.463 | 0.947 | 0 |
| stress | 0 | 0/486 | 0 / 0 | 0 | 461 | 0.005 | unavailable | 0.291 | 0.090 | 0 |
| motion | 58 | 61/486 | 3 / 0 | 3 | 436 | 0.317 | 0.675 | 0.224 | 0.294 | 0 |
| validation_clean | new capture | 392/644 | — / — | 8 | 559 | 0.141 | 0.477 | 0.799 | 0.780 | 0 |
| validation_noisy | new capture | 528/644 | — / — | 3 | 557 | 0.518 | 0.758 | 0.619 | 0.859 | 0 |
| validation_occlusion | new capture | 329/644 | — / — | 3 | 466 | 0.141 | 0.477 | 0.687 | 0.780 | 0 |

Limits remain 2 cm clearance, 3 cm forward/lateral, 5 degrees axis yaw. Noisy replay adds 5 mm axial noise, 10% missing pixels and 60 ms delay. Occlusion removes depth at 6–6.5 s and 8–10 s; all quantities are withheld throughout each blackout. Raw rejected predictions remain in the artifacts and can drift.

## Verification and scope

Three additional offline evaluation tests pass. The evaluator now inventories metrics across all frames instead of assuming the first frame contains a seat. Explicit no_seat outputs count as unavailable for every quantity and remain in availability denominators. Saved runtime outputs were not altered. Evaluation-only replay reproduces all 483 previously recorded error dictionaries exactly; the original evaluator and failure logs are retained.

Observer SHA-256: `90be43b47e03ce0c61d05cc192f5ae4ba04a343a4fab2c834b1db5ef42447e34`. All 38 focused tests pass, covering immutable reference storage, direct-anchor admission, rejection of a single view or disagreeing fits, preservation of the added allowance, and the prior geometry/identity/timestamp contracts. All prior datasets and failures are retained.

Sensor-only isolation replay: {'samples': 540, 'exact_match': True, 'bank_matches': 1, 'fixture_files': ['sensors.npz']}. The fixture directory contains no truth or geometry manifest.

The original-camera and analytical fixtures include incompletely observed seat extents; reference transitions cannot invent hidden boundaries. Static surfaces, a visible flat floor, ideal encoder/IMU measurements and the virtual camera mount remain assumptions. This phase has no perception feedback to control, no PPO and no soft-landing tuning.

Compute times were measured under concurrent regression load, not as a real-time benchmark. Per-case rejection durations, recovery timing, admission reasons and accepted/raw errors are in results.json.

New geometry was captured after source freeze. Online/replay comparison: {'samples': 644, 'T_O_P': 0.0, 'T_S_P': 0.0, 'direct_clearance_m': 0.0, 'quality_exact': True, 'registration_exact': True}.
