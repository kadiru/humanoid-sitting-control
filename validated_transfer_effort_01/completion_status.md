# Complete: unseen-surface effort PPO did not consistently improve — October 14, 2026 PDT

Both independent studies finished all authorized training and matched full-20-second evaluations. No additional training or policy promotion. Work continues directly from this Mac through SSH to kfuUbuntu; never start another coding agent. Ubuntu study: `/data/humanoid-sitting-perception/transfer_effort_01`. Frozen definitions: `docs/TRANSFER_EFFORT_PROTOCOL.md` and `ppo/{sitting,leaning}/protocol.json`.

## Verified result

Reserved holdout per task: eight geometries, twice repeated at simulator placements =16 trials, all starting standing. The corrected reference and every parent/final/selected policy have exactly the same holdout success and physical-failure case sets. Sitting:4/16 success,12 physical failures; leaning:16/16 success,0 physical failures. Sitting failures involve forbidden contacts and/or impacts, not latched falls, torque limits or joint limits. Successful sitting holdout geometries are only41cm at−2.5/+2.5degrees, twice each. Leaning succeeds across the tested62.14–68.14cm slab top-center heights and37.5–52.5degree slopes. This does not establish arbitrary-surface generalization.

Final full-budget whole20s / last10s / last5s squared-torque changes versus corrected reference on common full-duration successes:

| Task/seed | Success | Whole20s | Last10s | Last5s | Whole20s vs parent |
|---|---:|---:|---:|---:|---:|
| Sitting55611 |4/16|−1.037%|−1.675%|−1.425%|−1.293%|
| Sitting55617 |4/16|+1.532%|+2.889%|+3.245%|+0.623%|
| Leaning55711 |16/16|+1.018%|+1.252%|+1.288%|+2.476%|
| Leaning55717 |16/16|+6.992%|+3.887%|+3.932%|+2.654%|

All four final policies increase total absolute torque on holdout. Final leaning leg squared torque rises1.23%/8.14%, outweighing small upper-body savings. Sitting55611 also loses development stability:8/18 reference successes fall to6/18, with new forbidden contacts in both40cm/−5degree placements. Sitting55617 retains8/18; both leaning seeds retain18/18.

All development-selected policies are block0 (32 new updates); no parent fallback. Their whole20s holdout squared-torque changes vs reference are sitting−0.126%/+1.085%, leaning−0.960%/+4.854%. The apparent best leaning selection55711 is0.469% worse than its own parent; its reference advantage predates this training. Selection filters trained candidates for stability then effort; the parent is only a fallback, not an always-competing candidate. Do not mistake selection for improvement or deployment. No repeatable learned effort benefit was demonstrated.

## Validation and evidence

Independent reruns into `ppo/{sitting,leaning}/verified_02` pass and reproduce both original verification JSONs exactly. Each checks82 frozen hashes and the same436 protected prior hashes, finite changed actor/critic, optimizer continuity, actual checkpoint loads, exact matched physical/sensor starts, full20s horizons, constant fixed/finger commands, bounded learned commands, active/warmup/unfinished accounting, and development-only selection. Holdout was reserved before assessment and excluded from calibration/training/selection.

Budget:128 new updates/147456 transitions per seed,589824 total =573359 active +16465 indexed-reset warmup. Active/warmup by seed:55611=142610/4846;55617=142749/4707;55711=144000/3456;55717=144000/3456. Separate development smoke4608 transitions. No extra training.

No missing tracked geometry in audited valid motion before failure. Across development/holdout, max offset error2.212mm sitting/.959mm leaning; max normal error.359/.091degrees. The contact failures persisted with tracking available.

Mac final package: `/Users/kadir/humanoid-sitting-control/validated_transfer_effort_01/final_results`. Original1290 SHA256 entries verified; supplemental review has `REVIEW_SHA256.json`. `findings.html` gives the conclusion, original `report.html` and `comparison.png` are preserved, and CSVs provide whole20s/last10s/last5s squared/absolute/RMS and leg/upper effort by seed/bank/surface versus corrected reference AND parent. Independent evidence is in `independent_review`. Plot and both HTML reports visually inspected. Local preview: http://127.0.0.1:8796/final_results/findings.html. Supplemental report and evidence also copied to Ubuntu `final_export_01` without altering original exports. Completion follow-up `finish-resting-ppo-comparison` is PAUSED after the verified completion review.

No savings from failed fragments. Sitting effort comparisons include only4 successful trials (2 geometries repeated twice). Two seeds reuse finite nearby rigid surfaces. Runtime405 sensor/reference channels and23 bounded body-joint residuals remain unchanged. This is reference-assisted PPO, not independent motion discovery. Sum of squared Nm is an effort proxy, not electrical energy or a global optimum.

Next research direction, not launched: make sitting contacts safe as measured height/slope changes; for leaning, require any new effort candidate to beat both its parent and the corrected reference while retaining support. Preserve prior studies and frozen definitions; do not simply extend this unsuccessful training.
