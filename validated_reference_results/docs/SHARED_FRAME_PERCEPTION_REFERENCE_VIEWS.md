# Current perception handoff — verified reference views

The Mac agent is primary. Use `ssh kfuUbuntu`, project `/home/kadir/humanoid-sitting`. Do not launch another agent, tune PPO/soft landing, replace checkpoints, delete recordings or reinstall software. No observer output drives control. Original user handoffs remain unchanged. The preceding validated handoff is `docs/SHARED_FRAME_PERCEPTION_PRIOR_REFERENCES.md`.

## Frozen observer and result

`sitting_task/src/humanoid_sitting/shared_frame.py` SHA-256:
`daec7a1c769c68579cebfafe79d2063cb19660ca6e09ced705f460218f9fcf5c`.

Read `docs/shared-frame-reference-results-03.md` and `runs/shared_frame_reference_results_03/results.json`. All **26 conditions / 17,877 case-frame samples** have **zero accepted quantity/frame violations** at unchanged limits: 2 cm clearance, 3 cm forward/lateral, 5 degrees axis yaw. Samples across replay modes are correlated, not independent trials. Invalid predictions remain logged and can drift substantially. The older evaluator's aggregate gates include invalid predictions; use the per-quantity accepted-error audit to assess eligibility.

41 observer tests and three offline evaluator tests pass. Logs: `logs/shared_frame_reference_tests_04.log` and `logs/shared_frame_reference_evaluation_tests_01.log`. All final regression, live-capture, sensor-only and finalization exit files are zero.

## What changed

- The immutable initial scan remains the first pose target. Up to eight additional immutable views can be admitted only from full-rank poses registered directly to it. Reverse registration must also pass 60% overlap, 8 mm RMS, full rank and a 1 cm / 0.015 rad cycle check. Admissions are at least one second apart, with 3 cm or 0.08 rad view change. A bank-derived pose cannot admit another view; no recursive reference chain or fused-map target exists.
- Fallback requires at least two separately captured views whose fitted poses all agree within 1 cm / 0.015 rad. Each fallback fit needs its smallest scaled Hessian eigenvalue above 10 and above 0.5% of the largest. This rejects weak modes that can appear observable through noisy normals. Large corrections and lost tracking retain the bounded multi-hypothesis recovery checks. The extra reference costs another 8 mm engineering allowance. These are heuristic consistency gates, not calibrated probabilities or independent measurements.
- A minimum-area rectangle cannot identify the seat short axis from one opposing boundary pair. Both extent pairs are required; a sufficiently covered fitted ellipse remains an explicit shape hypothesis. Forward/lateral coordinates also require the axis to be confirmed. Clearance remains separately qualified. Numeric pose and direct-clearance outputs exactly match the preceding conditioned candidate on all 23 identical-input regression conditions; this last fix changes eligibility only.
- Offline evaluation now handles a first frame with no seat. Missing quantities stay unavailable and remain in denominators. Metrics are inventoried across all frames. Re-evaluating 483 older samples reproduces every original error dictionary exactly. Saved runtime outputs were not modified.

Inputs remain depth, static calibration, encoder FK, gravity and gyro-equivalent increments. Runtime reads no truth archive or geometry manifest. Ideal encoder/IMU equivalents and the virtual camera mount remain assumptions. The sensor-only fixture contains just `sensors.npz`; 540 outputs match exactly, including reference admission diagnostics (that subset has no accepted fallback frames). All 322 outputs from the final live capture match replay exactly for poses, direct clearance, quality and registration.

## Availability and remaining gaps

| Saved condition | Previous relative-pose availability | Current |
|---|---:|---:|
| Rear camera, clean | 680/966 | 873/966 |
| Rear camera, noisy | 721/966 | 882/966 |
| Rear camera, occluded | 539/966 | 673/966 |
| Previous three-surface bank, clean | 355/483 | 394/483 |
| Previous three-surface bank, occluded | 234/483 | 339/483 |

Across the original 17 conditions, availability rises from 5,801 to 6,743 of 14,013 frames, with no previously accepted full-pose frame lost. The original camera still has only 17/966 clean and 12/966 noisy full-pose frames; the analytical stress fixture has none. Hidden boundaries cannot be reconstructed by reference switching alone.

The axis correction deliberately reduces individual-quantity availability on the 23 matched regressions: forward 9,275→9,162; lateral 10,004→9,021; yaw 10,224→9,128. Clearance stays 13,941. Do not hide this cost or interpret a relative-pose availability flag as being aligned or ready to land. Yaw remains an axis modulo 180 degrees.

The final previously untouched bank has a couch and ellipse. Clean/noisy/occluded relative-pose counts are 271/304/206 of 322; accepted maxima across modes are 0.720 cm clearance, 0.680 cm forward, 0.634 cm lateral and 0.242 degrees yaw. Higher noisy availability reflects heuristic sensitivity, not better sensing. All blackout frames withhold all quantities.

The prior clean couch still loses tracking at 7.8 s: rank 3, 56% overlap; at 8.0 s overlap is 21% and the rejected raw pose has 11.4 cm error. Recovery currently searches XY with a fixed height prior. Keyframes alone do not solve this height/overlap loss. Timings were measured under concurrent load; the largest run P95 was about 1.275 s per update. Sensor timestamps at 10 Hz do not establish real-time execution.

## Preserved failures

- Before this phase: `runs/shared_frame_reference_source_before_01`, source `16aef05d…` (the prior map-integrity observer).
- Candidate `_01`, source `90be43b4…`: five accepted lateral errors on the noisy rear-camera couch, max 4.274 cm. Multiple references agreed in a weak direction. Full 20-condition results and video remain. Source is also in `runs/shared_frame_reference_source_before_02`.
- Candidate `_02`, source `83891849…`: fixed those five failures, then a new partial ellipse produced an accepted 11.234-degree yaw error at frame 1 in clean and occluded modes. Overall pose was already withheld. Full 23-condition results and video remain. Source is also in `runs/shared_frame_reference_source_before_03`.
- `runs/shared_frame_reference_evaluator_before_01` preserves the old offline tools. An initial geometry file used unsupported `chamfer` instead of `chamfered`; capture `_01` stopped before simulation. Evaluator failures from no-seat startup were repaired separately, with original logs retained. Neither was a robot/estimator success.

## Canonical artifacts and reproduction

All final tracking run names end `_03`: `shared_frame_reference_{tag}_03`.

- `rear`, `original`, `fresh`, `unseen`, `holdout`, each with `_clean`, `_noisy`, `_occlusion`, reuse the earlier integrity captures and are all regression data.
- `extension_{mode}` uses `shared_frame_reference_validation_capture_02` (four cases), geometry file `_02`.
- `shape_{mode}` uses `shared_frame_reference_validation_capture_03` (two cases), geometry file `_03`; this is the retained startup-axis failure bank.
- `validation_{mode}` uses `shared_frame_reference_validation_capture_04` (two new cases), geometry file `_04`. Geometry SHA-256: `5f982436b15fb22b12797054f5d991f3f50d6cd580b19133a5e0b71bc97b53e7`.
- `stress` and `motion` reuse their earlier analytical captures.
- Results/source: `runs/shared_frame_reference_results_03`; audits: `runs/shared_frame_reference_audit_03` and `runs/shared_frame_reference_validation_audit_03`.
- Video: `runs/shared_frame_reference_visuals_03/reference_tracking.mp4`, verified H.264, 1400×900, 10 fps, 360 frames / 36 seconds. It includes successful recovery, remaining loss, both development failures and fresh geometry. `coverage.png` is in the results directory.
- Preservation: `runs/shared_frame_reference_preservation_03.json`: all 117 original files, five baseline files and three checkpoints unchanged; Isaac Lab clean; root 30.678 GiB free. New artifacts remain on `/data/humanoid-sitting-perception`, linked from project runs/. No old datasets were removed.
- Launchers: `validate_shared_frame_references_03.sh` (23 regressions), `validate_shared_frame_reference_validation_04.sh` (new capture and three replays), `validate_shared_frame_reference_sensor_only_03.sh`, `finalize_shared_frame_references_03.sh`. Reports/plots use `_v3.py`; video uses `_03.py`. Fixed destinations must be changed for reproduction. Use persistent tmux processes.
- Mac delivery: `validated_reference_results/`. Authoring mirror: `shared_frame_work/`.

## Next work

Investigate height recovery using an explicitly associated current floor/seat plane before the XY search. Preserve the same-scan floor-to-seat separation and ambiguity guards: predicted world height alone previously confused floor and seat. Never fuse a height seed without subsequent verified registration. The couch failure above is the primary saved case; all 26 conditions are now regressions, so reserve new geometry after the next source freeze.

Profile serial latency and reduce repeated expensive failed searches without relaxing validity. Then test camera/IMU calibration bias and finite whole-body/support margins before any perception-driven alignment or descent control. Keep controller training and tuning stopped.
