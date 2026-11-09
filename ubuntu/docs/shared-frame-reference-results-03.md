# Directly anchored reference views: results

17,877 case/frame samples; 0 accepted quantity/frame violations; 637 frames used reference consensus. Samples across sensor modes are correlated.

## Method

The original scan remains immutable. At most eight separately captured views can be admitted, only from a full-rank pose registered directly to the original scan. Reverse registration must also be full rank, exceed 60% overlap, have under 8 mm plane RMS and close within 1 cm / 0.015 rad. Admissions are at least one second apart and require 3 cm translation or 0.08 rad rotation from the last admitted view. Each admitted array is copied and read-only. Views obtained from the bank cannot admit new views; graph depth is bounded. The display map is never a registration target.

The original reference is tried first. Every fallback match must also have its smallest scaled Hessian eigenvalue above 10 and above 0.5% of its largest eigenvalue (condition number below 200). This engineering margin rejects modes near the original rank cutoff; matching two references cannot repair a shared unobservable direction. A fallback requires at least two views whose independently fitted poses all agree within 1 cm / 0.015 rad. Large corrections and lost tracking use the existing bounded recovery checks. A successful fallback carries another 8 mm engineering registration allowance. These consistency checks are not statistically independent measurements or calibrated probabilities.

A rectangular patch now requires both opposing extent pairs before claiming the seat short axis. One visible pair can indicate a line direction but cannot distinguish the short axis of an incomplete rectangle or a cropped curve. Seat-frame offsets also require that axis to be confirmed. A sufficiently covered fitted ellipse remains an explicit shape hypothesis.

## Accuracy and availability

| Condition | Before pose | After pose | Gained / lost | Bank matches | Clearance | Max clearance cm | Max forward cm | Max lateral cm | Max yaw deg | Violations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rear_clean | 680 | 873/966 | 193 / 0 | 153 | 966 | 0.065 | 0.748 | 0.456 | 0.318 | 0 |
| rear_noisy | 721 | 882/966 | 161 / 0 | 119 | 959 | 0.595 | 0.894 | 1.305 | 0.848 | 0 |
| rear_occlusion | 539 | 673/966 | 134 / 0 | 80 | 813 | 0.065 | 0.748 | 0.456 | 0.318 | 0 |
| original_clean | 17 | 17/966 | 0 / 0 | 0 | 750 | 0.111 | 0.331 | 0.311 | 0.141 | 0 |
| original_noisy | 12 | 12/966 | 0 / 0 | 0 | 880 | 1.351 | 0.618 | 0.424 | 0.198 | 0 |
| original_occlusion | 17 | 17/966 | 0 / 0 | 0 | 648 | 0.110 | 0.331 | 0.311 | 0.141 | 0 |
| fresh_clean | 621 | 621/966 | 0 / 0 | 0 | 844 | 0.123 | 0.463 | 0.951 | 0.237 | 0 |
| fresh_noisy | 594 | 650/966 | 56 / 0 | 1 | 844 | 0.486 | 0.802 | 0.460 | 0.728 | 0 |
| fresh_occlusion | 384 | 477/966 | 93 / 0 | 4 | 696 | 0.118 | 0.463 | 0.951 | 0.237 | 0 |
| unseen_clean | 434 | 468/966 | 34 / 0 | 24 | 620 | 0.130 | 0.437 | 0.362 | 0.250 | 0 |
| unseen_noisy | 454 | 496/966 | 42 / 0 | 29 | 620 | 0.528 | 0.637 | 0.753 | 0.762 | 0 |
| unseen_occlusion | 313 | 369/966 | 56 / 0 | 5 | 538 | 0.130 | 0.437 | 0.362 | 0.250 | 0 |
| holdout_clean | 355 | 394/483 | 39 / 0 | 35 | 480 | 0.060 | 0.346 | 0.463 | 0.947 | 0 |
| holdout_noisy | 368 | 397/483 | 29 / 0 | 29 | 480 | 0.524 | 0.649 | 0.622 | 0.434 | 0 |
| holdout_occlusion | 234 | 339/483 | 105 / 0 | 18 | 408 | 0.060 | 0.346 | 0.463 | 0.947 | 0 |
| stress | 0 | 0/486 | 0 / 0 | 0 | 461 | 0.005 | unavailable | unavailable | unavailable | 0 |
| motion | 58 | 58/486 | 0 / 0 | 0 | 436 | 0.317 | 0.675 | 0.120 | 0.134 | 0 |
| extension_clean | additional regression | 392/644 | — / — | 8 | 559 | 0.141 | 0.477 | 0.799 | 0.780 | 0 |
| extension_noisy | additional regression | 528/644 | — / — | 3 | 557 | 0.518 | 0.758 | 0.619 | 0.859 | 0 |
| extension_occlusion | additional regression | 329/644 | — / — | 3 | 466 | 0.141 | 0.477 | 0.687 | 0.780 | 0 |
| shape_clean | additional regression | 275/322 | — / — | 31 | 322 | 0.058 | 0.518 | 0.344 | 0.898 | 0 |
| shape_noisy | additional regression | 282/322 | — / — | 23 | 322 | 1.287 | 0.679 | 1.245 | 0.248 | 0 |
| shape_occlusion | additional regression | 205/322 | — / — | 3 | 272 | 0.058 | 0.518 | 0.344 | 0.898 | 0 |
| validation_clean | new capture | 271/322 | — / — | 36 | 316 | 0.063 | 0.346 | 0.454 | 0.234 | 0 |
| validation_noisy | new capture | 304/322 | — / — | 29 | 315 | 0.720 | 0.680 | 0.634 | 0.241 | 0 |
| validation_occlusion | new capture | 206/322 | — / — | 4 | 266 | 0.063 | 0.346 | 0.454 | 0.234 | 0 |

Limits remain 2 cm clearance, 3 cm forward/lateral, 5 degrees axis yaw. Noisy replay adds 5 mm axial noise, 10% missing pixels and 60 ms delay. Occlusion removes depth at 6–6.5 s and 8–10 s; all quantities are withheld throughout each blackout. Raw rejected predictions remain in the artifacts and can drift.

## Verification and scope

Three additional offline evaluation tests pass. The evaluator now inventories metrics across all frames instead of assuming the first frame contains a seat. Explicit no_seat outputs count as unavailable for every quantity and remain in availability denominators. Saved runtime outputs were not altered. Evaluation-only replay reproduces all 483 previously recorded error dictionaries exactly; the original evaluator and failure logs are retained.

The first reference-view candidate (90be43b47e03ce0c61d05cc192f5ae4ba04a343a4fab2c834b1db5ef42447e34) accepted five lateral errors above 3 cm on the noisy rear-camera couch, reaching 4.274 cm. The smallest Hessian modes were only about 4.8–5.6 despite reference agreement. All outputs and its report remain in runs ending reference_*_01 and shared-frame-reference-results-01.md. It was not promoted. The added conditioning gate was fixed before this complete re-evaluation; error limits were not changed.

The conditioned candidate (8389184975985911fa77cea888ddca3b1c4ac6c673503c207d2c65427e3643c9) removed the five noisy lateral failures, but its next two-surface bank revealed a partial-ellipse startup yaw error of 11.234 degrees, accepted once in clean and once in occluded replay. Overall relative pose was already withheld; the individual yaw flag was incorrect. Those two failures are retained in candidate _02. The final shape fix leaves all pose and clearance numeric outputs exactly unchanged on the 23 regression conditions and corrects measurement eligibility. The final two-surface capture is a separate post-freeze test.

Observer SHA-256: `daec7a1c769c68579cebfafe79d2063cb19660ca6e09ced705f460218f9fcf5c`. All 41 focused tests pass, covering immutable reference storage, direct-anchor admission, rejection of a single view or disagreeing fits, preservation of the added allowance, and the prior geometry/identity/timestamp contracts. All prior datasets and failures are retained.

Sensor-only isolation replay: {'samples': 540, 'exact_match': True, 'bank_matches': 0, 'fixture_files': ['sensors.npz']}. The fixture directory contains no truth or geometry manifest.

The previous clean couch still loses tracking at 7.8 s (rank 3, 56% overlap with the initial scan), falling to 21% overlap at 8.0 s. Its rejected raw pose has 11.4 cm error at 8.0 s. The recovery search currently varies XY with a fixed height prior; reference transitions alone do not resolve this gap. Next investigate height recovery from an explicitly associated current floor/seat plane, preserving the identity guards. The original-camera and analytical fixtures include incompletely observed seat extents; reference transitions cannot invent hidden boundaries. Static surfaces, a visible flat floor, ideal encoder/IMU measurements and the virtual camera mount remain assumptions. This phase has no perception feedback to control, no PPO and no soft-landing tuning.

The extension and shape banks contain the previous four- and two-surface candidate captures and are now regression data. The final new capture contains two new surfaces and uses the frozen corrected observer. Compute times were measured under concurrent regression load, not as a real-time benchmark. Per-case rejection durations, recovery timing, admission reasons and accepted/raw errors are in results.json.

New geometry was captured after source freeze. Online/replay comparison: {'samples': 322, 'T_O_P': 0.0, 'T_S_P': 0.0, 'direct_clearance_m': 0.0, 'quality_exact': True, 'registration_exact': True}.

## Effect of requiring a confirmed coordinate axis

On the 23 identical-input regressions, the final axis correction changes eligibility only; all pose and direct-clearance values exactly match the conditioned candidate. Partially bounded seats now withhold additional coordinate-dependent quantities.

| Quantity | Available before axis correction | Available after |
|---|---:|---:|
| clearance | 13941 | 13941 |
| forward | 9275 | 9162 |
| lateral | 10004 | 9021 |
| yaw | 10224 | 9128 |
