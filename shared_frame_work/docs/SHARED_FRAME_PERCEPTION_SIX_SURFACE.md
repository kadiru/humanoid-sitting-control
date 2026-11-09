# Current perception handoff — six-surface validation, 2026-10-10

Continue through `ssh kfuUbuntu`, project `/home/kadir/humanoid-sitting`. The Mac agent remains primary; do not start another agent remotely. Preserve code, data, checkpoints, and unsuccessful experiments. No PPO or soft-landing tuning. No installation is needed.

Read `docs/shared-frame-six-surface-results.md` first. Earlier reports and the preceding handoff (`docs/SHARED_FRAME_PERCEPTION_PRIOR_SIX_SURFACE.md`) remain available. Original user handoff documents are unchanged.

## Current observer

`sitting_task/src/humanoid_sitting/shared_frame.py` SHA-256: `50000e1a059f380d033f9435194ccf7101c4cd6a8418ce81dca085e1cb53fbb7`.

Runtime consumes only depth, static calibration, encoder FK, gyro increments, and gravity. Truth and furniture labels are offline evaluation inputs only. Frames, directional quality, direct plane clearance, bounded reacquisition, and map retention are documented in the preceding handoff. `valid` means observed-patch alignment gates pass, not controller authorization; clearance has its own flag.

New fix: associate contours with the selected connected support component before shape fitting. Same-height edges farther than 3 cm from that component are excluded from the fit. Six unrelated edges had caused a good ellipse fit to be rejected, followed by a confidently biased polygon-facet rectangle axis. A new regression covers this. All 22 tests pass; no thresholds were relaxed.

## Canonical artifacts

- Added-camera six-surface capture: `shared_frame_rear_down_six_capture_01`. Pre-fix replays: `shared_frame_rear_down_six_{clean,noisy,occlusion}_01`; audit `shared_frame_six_validation_01`. Retain these failed results: 190/103/165 violating quantity/frame pairs, all on the rounded box.
- Corrected replays: `shared_frame_rear_down_six_{clean,noisy,occlusion}_02`; audit `shared_frame_six_validation_02`. Final video/layout: `shared_frame_six_visuals_03`.
- Fresh corrected online physics: `shared_frame_rear_down_six_live_02`. `shared_frame_six_live_contract_02.json` proves every consumed sensor/calibration array exactly matches the earlier capture before its corrected replay is reused. All 966 poses, clearances, and nested quality values agree exactly.
- Prior regressions updated: `shared_frame_quality_clean_05`, `shared_frame_quality_noisy_05`, `shared_frame_quality_occlusion_04`, `shared_frame_stress_05`, `shared_frame_holdout_03`. Audit: `shared_frame_component_regression_01`. The last two are analytical regressions, not fresh holdouts or physical motion controllers.
- Diagnostic: `shared_frame_component_diagnostic_01`. Recorded estimate at 2 s: axis −0.04° → −5.12°, accepted ellipse residual 2.38 mm.
- Tests: `logs/shared_frame_component_fix_tests.log`. Preservation: `shared_frame_six_integrity_03.json`; 117 original files, baseline/checkpoint hashes, and clean Isaac Lab unchanged. About 31.4 GiB free before final video export.

The observer camera is at pelvis `[-0.22, 0, 0.55]`, looking down. The original camera still supplies the frozen controller's startup geometry. All six original/new camera root trajectories match exactly. Added-camera RGB remains blank; valid axial depth is used in the final visuals. Hardware mounting is unvalidated.

## Results

Clean full alignment: original camera 43/966 → new view 602/966. Per surface: chair 0/161; rectangular bench 124/161; rectangular box 159/161; couch 0/161; chamfered bench 160/161; rounded box 159/161. Current plane clearance: 966/966.

Accepted clean P95: clearance 0.58 mm, forward 3.12 mm, lateral 2.86 mm, yaw 0.116°. With 5 mm axial noise, 10% dropout, and 60 ms latency: alignment 640/966; P95 3.40 mm, 5.03 mm, 3.94 mm, 0.366°. Higher noisy availability reflects heuristic-gate sensitivity, not a benefit from noise or calibrated confidence.

All three corrected camera conditions and all five prior regressions have zero accepted errors above provisional 2 cm clearance, 3 cm offset, 5° axis-yaw limits. The new occlusion run rejects all 150 missing-depth samples; reacquisition after depth returns at 10 s takes 0–0.7 s. The geometry bank was used to fix the bug and is now regression data; no untouched generalization claim is justified.

Clean all-frame P95 maximum body-position error: 3.06 mm; pelvis-position error: 2.41 mm, with initial XY/yaw gauge alignment only. Clearance is to the observed seat plane; finite occupancy/holes are separate and do not certify a full-body landing or sitting trajectory.

## Next priority

Recover seat boundaries where the seat meets vertical geometry. The chair lacks confirmed rear extent; the couch also lacks side extents and axis. Registration can be rank four while geometry stays incomplete. The current extractor primarily admits depth-discontinuity contours on the seat plane. Seat/backrest and seat/armrest junctions need additional measured geometric evidence. Inspect vertical surfaces and plane intersections or additional views; do not infer hidden dimensions from furniture labels or relax quality gates.

Then freeze the observer and test fresh heights, dimensions, offsets, yaws, and extrinsic-calibration errors before perception-driven pelvis alignment. No alignment/descent controller was added. Fast falls, moving furniture, arbitrary relocalization, hardware calibration, and hard real-time behavior remain unvalidated.

Use unique output directories and persistent tmux simulations. The orchestration scripts `validate_shared_frame_six.sh` and `validate_shared_frame_component_fix.sh` have fixed output names; inspect before reproducing. Preserve all failed runs, including the earlier first camera probe that changed controller inputs, separately from validated results.
