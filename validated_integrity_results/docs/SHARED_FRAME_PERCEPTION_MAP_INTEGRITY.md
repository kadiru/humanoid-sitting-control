# Current perception handoff — map integrity and plane identity

Use `ssh kfuUbuntu`, project `/home/kadir/humanoid-sitting`. The Mac agent is primary. Do not launch another agent remotely, tune PPO/soft landing, replace checkpoints, delete recordings or reinstall packages. No perception-driven actuation exists.

Read `docs/shared-frame-integrity-results.md` and `runs/shared_frame_integrity_results_03/results.json`. Previous handoff is archived as `docs/SHARED_FRAME_PERCEPTION_PRIOR_INTEGRITY.md`. Original user handoffs remain unchanged.

## Final frozen implementation

Observer `sitting_task/src/humanoid_sitting/shared_frame.py` SHA-256: `16aef05dcc09da997a9cd8da7483e2295bffce6e21101a3bcf0e9f6760c5a4b6`.

Inputs remain depth, static calibration, encoder FK, body gyro increments and gravity. Simulator truth and geometry manifests are evaluation-only. The unchanged sensor05/contact02 controller generates simulation motion, with no observer feedback.

- Registration uses the immutable first depth scan. Reference arrays are copied and read-only; fused display-map points never become the pose reference.
- Matching requires surface-normal agreement (absolute dot >0.9). Residual/coverage are recomputed at the returned pose; aborted large solver steps cannot be accepted. Runtime coverage must exceed 0.6 and plane RMS must be below 8 mm.
- Fusion and seat-map updates require accepted rank-4 registration, except initialization. Partial registration may update observed directions but cannot extend map geometry.
- After >=0.5 s without accepted tracking, or a depth blackout, bounded contour recovery searches the fixed reference. Repeated lost-tracking attempts are at least 0.5 s apart. Local corrections above 4 cm or 0.06 rad require immediate multi-hypothesis recovery before fusion. Ambiguous/insufficient matches remain rejected.
- Clearance has a separate identity check. Trusted registration can match reference height; otherwise the same scan must contain a sufficiently broad lower plane and a unique upper plane separated by the initially observed floor-to-seat height. A lone plane after tracking loss cannot declare itself a seat. The lowest broad plane is treated as floor under the flat-floor assumption; arbitrary platforms/repeated levels remain unvalidated.
- Plane extrapolation uses 6 cm effective spatial cells, a 5 mm noise floor, 3x linear-fit uncertainty expression, and an 8 mm model allowance. Sparse/rank-deficient fits or allowance >=2 cm are rejected. This is an engineering heuristic, not calibrated probability.

New diagnostics include `registration.reference`, `fusion_allowed`, `aborted_step`, `large_local_correction`, recovery state and `seat_plane_association`. Direct clearance remains a current-plane quantity; offsets/yaw and body/obstacle distances are map-relative. Rejected pose predictions can still drift and must not be treated as measurements.

## Validated final results

35 focused CPU tests pass: `logs/shared_frame_integrity_tests_04.log`. Tests cover incompatible normals, immutable-reference isolation from a corrupted display map, refusal to fuse blind scans, broad/thin/collinear plane conditioning, lost plane identity, ambiguity, geometry, timestamps and sensor-only contracts.

All **17 final conditions / 14,013 case-frame samples** have **zero accepted quantity/frame violations**, with unchanged limits: 2 cm clearance, 3 cm forward/lateral, 5 degrees axis yaw. Samples include repeated clean/noisy/occluded replays, not independent trials. Conditional accuracy does not establish continuous tracking.

The final untouched bank has three surfaces (chair, couch, ellipse). Clean/noisy/occluded alignment availability: 355/368/234 of 483. Maximum accepted clearance error is 0.524 cm, forward error 0.649 cm, lateral error 0.622 cm, yaw error 0.947 degrees across those conditions. Clean alignment coverage is 73.5%; the couch is accepted on only 77/161 frames. Source was frozen before capture and not adjusted on these results.

All 483 new online/replay outputs agree exactly for pelvis pose, relative pose, direct clearance including missing values, nested quality and registration diagnostics. A separate directory containing only sensors.npz reproduces 18 checked outputs exactly. Runtime reads no truth or geometry manifest.

Known original failure bank clean/noisy/occluded alignment: 621/594/384 of 966 (previously 722/826/584 with 213 violations). Existing rear camera: 680/721/539 (previously 875/960/712). Original camera: 17/12/17. Analytical stress has no full alignment frames; analytical motion has 58. See the report for clearance availability, per-quantity maxima/P95 and per-case rejection durations. Do not hide the availability cost.

## Preserved development failures

- Before-change source: `runs/shared_frame_integrity_source_before_01`, original junction observer hash cb9da561… . Original junction reports remain.
- Anchor-only candidate `shared_frame_integrity_*_01`: 4 clean and 59 noisy violations on the known bank; noisy forward error reached 46.54 cm. Not promoted.
- Candidate `_02`, source b6a86fd2…: passed 11 prior conditions but failed a new six-surface bank with 654 clearance violations and two forward violations. Lost translation let floor points satisfy a predicted seat height, yielding about 28 cm false clearance. A wrong local match also contaminated the support map. Full failed report: `docs/shared-frame-integrity-v2-results.md`; results/video `_02` remain.
- Candidate `_02` source is also archived in `runs/shared_frame_integrity_source_before_03`. The formerly new six-surface bank is now a known regression set for final `_03`, not untouched validation.

`scripts/diagnose_shared_frame_plane.py` separately shows why a low 3.1 mm plane residual on an 81-point strip did not justify clearance: extrapolation bound was about 9.5 cm before the model allowance.

## Canonical final artifacts

All final replay names end `_03`:

- `shared_frame_integrity_fresh_{clean,noisy,occlusion}_03`: original known failures from `shared_frame_junction_fresh_capture_01`.
- `shared_frame_integrity_rear_{clean,noisy,occlusion}_03`, `shared_frame_integrity_original_{clean,noisy,occlusion}_03`: existing captures.
- `shared_frame_integrity_stress_03`, `shared_frame_integrity_motion_03`: prior analytical fixtures.
- `shared_frame_integrity_unseen_{clean,noisy,occlusion}_03`: newly retained plane-identity failures, capture `shared_frame_integrity_unseen_capture_02` (six cases). Despite filename, this is now regression data.
- `shared_frame_integrity_holdout_capture_03` and `shared_frame_integrity_holdout_{clean,noisy,occlusion}_03`: untouched three-case bank for final source.
- Audits: `shared_frame_integrity_audit_03`, `shared_frame_integrity_holdout_audit_03`.
- Consolidated results/source snapshot: `shared_frame_integrity_results_03`.
- Final 24 s video: `shared_frame_integrity_visuals_03/map_integrity.mp4`.
- Preservation: `runs/shared_frame_integrity_preservation_03.json`.

Final holdout geometry: `experiments/shared_frame_integrity_holdout_geometry_03.json`, SHA-256 `fd04464594c9f9a5ae2d0dc8c00a4a9b4cde2f80f6f9ac956dda8f35e576a5b1`. Geometry/source hashes and exact commands are retained in manifests.

New large artifacts are on `/data/humanoid-sitting-perception`, linked from project runs/. No old data was removed; root remains above the 30 GiB floor. The sensors-only fixture uses suffix `_02` and its final replay `_03` directly under that data directory. Mac delivery: `validated_integrity_results/`.

Launchers: `validate_shared_frame_integrity_v3.sh` (14 regressions), `validate_shared_frame_integrity_holdout.sh` (new capture + three replays), `finalize_shared_frame_integrity_v3.sh` (report/video/preservation). Long jobs run in tmux with exit files under logs/. Fixed destinations must be changed for reproduction; do not overwrite prior outputs.

## Next priority

Recover tracking coverage with verified reference/keyframe transitions and adequate depth-view coverage, without allowing uncertain poses to alter a trusted reference. The first view cannot constrain arbitrary movement; some late trajectories remain unknown even with depth present. Keyframes alone cannot recover translation from a featureless plane. Inspect raw errors, rejection durations and recovery outcomes as well as accepted-error maxima.

Then test camera/IMU calibration bias, finite whole-body/support margins and runtime latency before alignment control. Current compute timings were collected during concurrent jobs; 10 Hz sensor timestamps do not establish real-time execution. Both successful and failed perception runs remain; no controller, PPO, software stack or checkpoint was changed.
