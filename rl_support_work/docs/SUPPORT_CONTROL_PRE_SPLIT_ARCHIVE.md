# Humanoid sitting: current verified state

Status (October 11, 2026): the original camera-timing sitting regression is repaired on the known bank, using an **experimental 50 Hz geometry-input adapter with frozen policy weights and gains**. Paired fresh validation is complete, but the broader readiness gate fails. **No RL, no policy promotion, no deployment-default changes. All jobs are finished.**

Continue as the sole primary agent on this Mac via SSH to `kfuUbuntu`, project `/home/kadir/humanoid-sitting`. Do not start another Codex agent. Use tmux for long simulations and `/data/humanoid-sitting-perception` for outputs. Preserve old code/data/checkpoints; nothing was installed or deleted.

## What caused the regression

A four-way 16 x 20 s timing comparison isolates startup and live camera acquisition. Historical timing reproduces 16/16 and all 12 archived trajectory arrays exactly; all 200 archived 144-channel actor packets also match. Corrected startup only, live only, and both each yield 15/16, losing case14 to impact. Case14 peaks are respectively 455.21, 697.77, 714.39 and 729.20 N. Both-corrected traces exactly match the prior passive-tracker capture. The regression is a frozen-controller observation-distribution sensitivity, not recording overhead or passive tracking.

Corrected startup changes case14 seat height by -5.219 mm and body-relative seat z by +1.772 mm. Initial geometric motion targets change immediately. The live feedback pattern and validity also change. Do not claim one scalar alone explains the closed-loop force difference.

## Implemented and tested repair

The camera remains genuinely current (`support_camera.acquire_current_camera`), with zero capture timestamp age and no extra physics step. The original distance tracker remains unchanged, SHA256 `80a05f6cb26e07e54acc539faa4c699205fd68a07e29a623639906ec7a46ffd0`.

A first complete-patch adapter at 10 Hz improves case14 impact to 669.33 N but remains 15/16: rejected as a full repair, preserved in `support_tracking_control_dev_01`.

The selected experimental adapter is `support_continuous_features.ContinuousPatchFeatures`. It carries a currently observed seat point cloud into the moving pelvis frame using foot FK and IMU orientation every 20 ms. New depth arrives every 100 ms. The actor's existing 10 geometry channels receive current clearance/normal/patch center/spans, validity and image age. No simulator world pose or contact force enters this calculation. Prediction expires at 100 ms; the entire inferred 3D translation, foot disagreement and plane correction consume a <2 cm engineering budget. Common-mode foot motion remains unobservable; this budget is conditional and is not a statistical guarantee. Hidden furniture bounds are not invented.

Continuous feature source SHA256: `b3f69d32ecb6f6be0553e52533b03509ac2153fdf0da3143427d25c7dc36af69`. Seventeen tests pass: 8 original tracker, 4 measured-feature contract, 5 continuous-motion/expiry tests. Sensor-only replay reconstructs all 15,984 consumed post-warm-up geometry packets exactly. Original parent, nominal student, contact estimator, gains, action limits and physical success criteria remain fixed.

## Validation results, including failures

| Bank | Corrected-camera baseline | Tracked controller | Worst force baseline → tracked |
|---|---:|---:|---:|
| Known regression 51403/51407 | 15/16 | 16/16 | 729 → 521 N |
| Fresh shapes/starts 51941/51943 | 10/16 | 10/16 | 3486 → 3662 N |
| Mixed training shapes, new starts 51923 | 9/16 | 9/16 | 1753 → 1951 N |

Case14 is repaired: 729.20 → 362.51 N, supported hold 6.28 s, no failure flags. No formerly successful case is lost in either fresh paired bank. But worst impacts increase among already-failing cases. Do not promote this as uniformly safer.

All 160 completed investigation attempts run the full 20 s, including post-failure intervals. Candidate source/settings were frozen before the two fresh banks were inspected. Those seeds are now consumed. Geometry banks and initial encoder/IMU states match bitwise across each paired comparison.

| Bank | Distance all-frame availability | Distance landing availability | Complete actor geometry landing availability | Worst accepted distance error | Worst transported-point error |
|---|---:|---:|---:|---:|---:|
| Known | 99.91% | 99.67% | 82.73% | 1.74 mm | 9.95 mm |
| Fresh | 93.66% | 92.60% | 72.45% | 6.90 mm | 6.68 mm |
| New starts | 96.18% | 93.17% | 73.93% | 7.25 mm | 7.29 mm |

No accepted distance or transported probe error exceeds 2 cm. The whole pipeline still fails coverage. The complete patch input is withheld more often than scalar distance. Known-bank feature refusals: 225 patch-motion-budget events and 15 distance-unavailable events, plus initial warm-up commands. Fresh case10 first physical failure is 8.98 s; a 10 s tracking gap begins at 10.02 s with torso tilt 83.8 degrees. Case12's long gap also begins after physical failure. These post-hoc diagnostics do not remove failed frames from the primary scores.

Fresh failed cases remain 9,10,12,13,14,15. New-start failed cases remain 7,8,9,10,12,13,15. Cases7/10 in the new-start bank have no latched physical failure flag but fail the unchanged 3 s supported-hold requirement; case7 improves from 1.34 s to 2.82 s, still a failure. No startup refusals occur in these three candidate banks.

Timing is simulation cadence, not proven hardware throughput: per-robot geometry update median 1.84 ms, p95 37.17 ms including image frames. No noisy-camera closed-loop or hardware validation was added in this study; the original noisy/delayed proprioception emulator remains unchanged.

## Files and reproduction

Canonical remote report: `/data/humanoid-sitting-perception/support_control_validation_results_01`.
Mac delivery: `/Users/kadir/humanoid-sitting-control/validated_support_control/results` (REPORT.md, results.json, plots, success/failure videos, source freeze, scores, provenance and hashes).

- Protocol: `docs/SUPPORT_CONTROL_TIMING_PROTOCOL.md`.
- Frozen sources: `/data/humanoid-sitting-perception/support_control_frozen_01` (25 files + manifest).
- Timing factorial: `scripts/capture_support_control_timing.py`, `run_support_control_timing_01.sh`; outputs `support_control_timing_{none,startup,live,both}_02`. The initial suffix-01 recorder failed on a list/tensor API error before completing its first step and remains preserved.
- Runtime adapters: `sitting_task/src/humanoid_sitting/support_tracking_features.py` and `support_continuous_features.py`.
- Active input entry points: `scripts/capture_support_tracking_control.py` (10 Hz) and `capture_support_continuous_control.py` (50 Hz).
- Selected development capture: `support_continuous_control_dev_01`; replay: `support_continuous_replay_dev_01`.
- Validation launcher: `scripts/run_support_control_validation_01.sh`, completed `logs/support_control_validation_01.exit` =0. Do not rerun into existing directories.
- Validation captures: `support_control_validation_{fresh,starts}_{baseline,candidate}_01`.
- Scoring: `score_support_landing.py`, `score_support_continuous_features.py`, `score_support_actor_geometry.py`.
- Report: `report_support_control_validation.py`; diagnostics: `audit_support_control_timing.py`, `audit_support_control_losses.py`.
- Each full capture saves `controller.npz` (50 Hz 144-channel packets/actions/nominal/commands), raw depth/encoders/IMU, separate evaluation labels, exact fed feature audits, camera timestamps and source hashes.

Original checkpoints and original shared-frame source are hash verified. IsaacLab remains clean at `be9551260b269d4d8a964b71d5849810a1237b33`. Output volume has about 78 GiB free; the previously existing 30 GiB root reserve remains unmet (about 29.1 GiB). No deletion was used to clear it.

## Next work

Keep RL paused and the preserved pilot-3 defaults. Address complete-geometry loss during fast motion, with a separately frozen experiment that can request a new depth image before the kinematic error budget expires. Separating distance confidence from patch-shape confidence may also be necessary at the actor boundary. Then address remaining balance/collision failures and increased impact severity. Reuse scene extraction to reduce CPU cost without changing timestamp semantics. Any modified candidate needs new validation seeds; do not tune to 51941/51943 or 51923 and call them fresh again.

Scope remains aligned rigid horizontal surfaces in simulation. This does not establish navigation, arbitrary furniture orientation, soft cushions, moving hidden seats, real hardware, or universal sitting. Earlier landing-only work is preserved in `docs/SUPPORT_LANDING_PRE_CONTROL_ARCHIVE.md` and `docs/SUPPORT_LANDING_STATUS.md`; earlier RL retention results in `docs/SUPPORT_RL_RETENTION_ARCHIVE.md`.
