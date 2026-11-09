# Current perception handoff — measured junctions and failed fresh-bank test

Use `ssh kfuUbuntu`, project `/home/kadir/humanoid-sitting`. The Mac agent is primary. Do not launch another agent remotely, tune PPO/soft landing, replace checkpoints, or reinstall packages. Preserve all successful and failed experiments.

Read `docs/shared-frame-junction-results.md` first. Previous handoff: `docs/SHARED_FRAME_PERCEPTION_PRIOR_JUNCTIONS.md`; previous six-surface summary: `docs/SHARED_FRAME_PERCEPTION_SIX_SURFACE.md`. Original user handoffs remain unchanged.

## Current source and scope

Observer `sitting_task/src/humanoid_sitting/shared_frame.py` SHA-256: `cb9da561abdad085a24cbedc914a19db1e134b06cfd2b3b5c5b1201ff2847683`. Runtime inputs remain depth, static calibration, encoder FK, gyro increments, and gravity. No simulator truth or fixture labels enter the observer. Controller and weights stayed frozen. No perception-driven actuation exists.

New `support_junctions` fits near-vertical surfaces from the observed map and intersects them with the observed seat plane. It requires ≥24 wall points, ≤8 mm RMS, ≥12 cm wall span, samples above 8 cm and within 5 cm of seat height. Intersection samples need local low/high wall evidence plus observed seat points within 3 cm. Unseen spans, floating surfaces, and distant walls are rejected. This is bounded geometric evidence, not proof of physical attachment.

Junctions supplement boundary evidence only; they do not enter ICP or change existing pose estimates. Seat results expose `junctions` and `junction_evidence`; replay saves sampled junctions in `tracks.npz`. Quality records `junction_count` and original `depth_contour_sides`. The body/floor/seat frame conventions and separate clearance flag remain unchanged.

## Completed validation

27 CPU tests pass, including five new geometry contracts: measured boundary, yaw equivariance, floating-wall rejection, distant-wall rejection, and visible-gap preservation. Log: `logs/shared_frame_junction_tests_01.log`.

Existing rear-camera clean alignment: 602/966 → 875/966. Chair 0→160/161; couch 0→113/161. Couch geometry is now confirmed, but its remaining clean rejections are mostly unobserved lateral motion. Existing noisy alignment is 960/966; do not interpret higher noisy availability as improved sensing. Existing clean/noisy/occlusion and the five prior regressions have zero accepted errors above provisional thresholds.

Existing clean accepted P95: clearance 0.581 mm, forward 6.197 mm, lateral 2.793 mm, yaw 0.297°. Existing before/after `T_O_P`, `T_S_P`, and direct clearance are exactly unchanged: the improvement comes from new observed boundary evidence.

Canonical runs:

- `shared_frame_junction_rear_{clean,noisy,occlusion}_01`, input `shared_frame_rear_down_six_capture_01`; audit `shared_frame_junction_six_validation_01`.
- `shared_frame_junction_original_{clean,noisy,occlusion}_01`, input `shared_frame_capture_01`.
- `shared_frame_junction_stress_01`, `shared_frame_junction_motion_01`; prior analytical regression fixtures. Audit `shared_frame_junction_regression_01`.
- `shared_frame_junction_audit_01`: sensor-only chair/couch point/normal/edge snapshots.

## Fresh geometry FAILED — not ready for control

The new geometry bank was fixed before capture in `experiments/shared_frame_junction_fresh_geometry_01.json`: six surfaces, heights 0.26–0.35 m, altered dimensions/XY, yaw −12° to +12°. Capture supports simulator-only `--geometry-bank`; the replay still never reads geometry metadata.

Capture: `shared_frame_junction_fresh_capture_01`; replays `shared_frame_junction_fresh_{clean,noisy,occlusion}_01`; audit `shared_frame_junction_fresh_audit_01`. The frozen observer was not tuned on this bank. All 966 online/replay poses, direct clearances (including unavailable values), and nested quality values match exactly.

Fresh clean/noisy/occlusion alignment flags: 722/826/584 of 966. These are NOT successful alignment results: accepted quantity/frame violations total 46/152/15. Across all 11 conditions there are 9,666 case/frame samples and 213 violating quantity/frame pairs, all in this fresh bank.

Maximum accepted fresh errors (clearance / forward / lateral cm, yaw degrees):

- Clean: 0.123 / 4.201 / 3.578 / 0.696.
- Noisy: 3.291 / 3.694 / 7.207 / 7.513.
- Occlusion: 0.118 / 3.360 / 4.048 / 0.318.

Do not characterize the observer's flags as reliable across these geometries, nor enable an alignment controller from them. P95 alone hides some outliers.

## Diagnosis and immediate next work

`runs/shared_frame_junction_drift_diagnostic_01/diagnostic.json` compares map predictions with current-frame patch estimates at selected failures. The inspected chair/bench failures have pelvis-map errors of 5.7–7.5 cm and seat-map center shifts around 2.5–3.3 cm. Registration can then have low residuals against its already shifted map and reset the directional allowance to 8 mm. Current depth may no longer confirm the full boundary, but old map evidence still qualifies offsets. Current-plane clearance can remain accurate independently. The noisy bank additionally exposes clearance/yaw outliers; those need geometric uncertainty/association checks, not just an XY correction.

Next fix map consistency, fusion/relocalization during partial visibility or fast motion, and confidence that improperly survives tracking errors. Distinguish direct current-frame relative measurements from map predictions. Do not simply relax error limits or change the motion controller. Current-frame checks in the diagnostic reject the inspected incomplete dimensions, but this has NOT been integrated as a fix. The fresh bank is now known failure data and should be a regression set for subsequent changes; reserve another fresh bank for generalization.

After those failures are resolved, test camera/IMU calibration bias and finite whole-body support margins before perception-driven alignment. Keep descent disabled.

## Delivery and preservation

Machine-readable consolidated results: `runs/shared_frame_junction_results_02/results.json`. Final 24 s video: `runs/shared_frame_junction_visuals_02/seat_wall_junctions.mp4`; cyan points show measured junctions and fresh clips are explicitly labeled failed validation. Local delivery on Mac: `validated_junction_results/`.

Preservation check: `runs/shared_frame_junction_integrity_01.json`; original 117 source files, baseline/checkpoint hashes, and clean Isaac Lab remain unchanged. Capture/replays use unique directories. The first final report generation failed on `None` clearances; its source is retained in `shared_frame_junction_results_01`; corrected results are `_02`. This was a reporting failure, not a simulation failure. The first video is retained; final `_02` clarifies failed validation and quality flags.

All original data and observer snapshots remain. Added-camera RGB is still blank; depth is valid and shown. Long simulations use tmux. Inspect fixed output names in orchestration scripts before reproducing; existing result directories must not be overwritten.
