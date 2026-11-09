# Complete: sitting contact repair and small sitting/leaning effort PPO comparison — October 14, 2026

The user's authorized task is complete: diagnose/fix sitting contact and impact failures, investigate leaning effort regression, validate, and compare small PPO runs against corrected references and starting policies. Work was direct Mac SSH to kfuUbuntu; no other coding agent. Prior code, data, checkpoints, failed probes and failed admissions remain preserved. No policy promotion or additional training is pending. The existing completion automation remains paused.

Project: `/home/kadir/humanoid-sitting`. Study root: `/data/humanoid-sitting-perception/contact_effort_repair_01`. Frozen protocols are `docs/CONTACT_EFFORT_REPAIR_PROTOCOL.md` (leaning) and `docs/REPAIRED_SITTING_EFFORT_PROTOCOL.md` (sitting). Read `docs/CONTACT_EFFORT_REPAIR_DIAGNOSIS.md` for evidence and limits.

## Verified sitting result

The repaired reference and both repaired parents pass16/16 reserved holdout trials with0 physical failures. Old corrected reference and old parent55211 pass4/16 with12 physical failures; old parent55217 passes2/16 with14 physical failures. Holdout is eight geometries repeated twice:38.5/41.5cm crossed with±3.75degrees,37.5/42.5cm at0degrees, and40cm at±6.25degrees. All start standing and run the full20 seconds. Success requires no physical failure plus final5 continuous seconds of qualifying support.

Both final sitting policies pass16/16 holdout with0 physical failures and18/18 development. Both use the SAME frozen55211 parent under the repaired reference. The55217 repaired parent failed two development friction-margin trials (16/18); its failed initial admission is preserved and it remains a comparison baseline. New learning was not initialized from it.

Holdout squared-torque changes, whole20s / final10s / final5s:
-56211 vs repaired reference:−4.200% /−7.972% /−12.372%; vs repaired55211 parent:−3.812% /−7.259% /−11.617% (16 common successes).
-56217 vs repaired reference:−2.157% /−4.097% /−6.186%; vs repaired55211 parent:−1.761% /−3.354% /−5.378% (16 common successes).
Both finals qualify under the frozen development-only selection rule; neither uses parent fallback. Holdout was opened only after both seeds and selection were frozen.

The contact repair has a substantial effort cost. Repaired reference vs old reference on their4 common holdout successes: whole20s+42.533%, final5s+145.777%. Final56211 still costs+36.621% whole20s/+116.364% final5s vs old reference;56217 costs+39.500%/+131.076%. Comparisons against BOTH old and repaired starting policies are in the report/CSVs. Never call the repaired sitting solution lower-effort than the old successful posture.

Final5s foot loads: repaired reference~87.3% body weight; final56211=82.93%, final56217=85.45%. Corresponding learned seat vertical loads17.07%/14.55%. The support gate permits only5% support load, so passing does not imply relaxed sitting. It remains a supported crouch with substantial leg work. Whole-body effort means squared physical torque over37 joints; absolute torque, instantaneous joint RMS averaged over time, and leg/upper-body metrics are also reported.

The repair uses measured plane height/slope, encoder/IMU foot-preserving IK, rearward transfer above the seat before lowering, and bounded ankle feedback from a foot-anchored pelvis-x estimate. Current physical/traction/contact/impact/joint criteria were not relaxed. Runtime never reads live simulator truth. Horizontal seat placement remains explicit offline reference assistance, not autonomous furniture discovery.

Sitting verification: original `sitting_ppo_01/verified_01` and fresh `verified_02` pass; main verification JSONs match exactly.35 frozen hashes and1546 prior hashes pass (the latter includes the completed leaning study). Each seed40 updates=8 critic-only+32 actor-enabled,11520 policy decisions,115200 physics transitions=112500 active+2700 indexed-reset warmup;4500 unfinished active transitions explicitly accounted,0 pending warmup. Total230400 transitions. Initial global0.5s settling is outside this rollout budget. No learning smoke for sitting. Finite changed actor/critic, zero-output child initialization, actor identity through critic warmup, exact loaded checkpoint, full horizons/starts/surfaces, bounded commands/fixed fingers, independent sensor-only path reconstruction, and exact first12s equality to repaired55211 parent all pass.

Repaired-reference holdout geometry confidence is false at6/3200 sampled instants (11.7–11.9s for both40cm/−6.25degree placements); numeric geometry remains finite, and none occurs above the specified motion thresholds. Maximum valid plane-offset error2.187mm, normal error0.387degrees. Foot-anchored pelvis-x truth audit RMS0.0165mm/max0.3214mm under ideal simulated sensors and stationary-foot assumptions; not hardware accuracy. Final-policy and all baseline audits are in `geometry_audit.json`.

## Verified leaning result

Final56111 passes16/16 reserved trials, lowering whole20s squared torque3.889% vs corrected reference and2.457% vs own parent55311; final5s savings8.016%/7.495%. Final56117 passes only2/16; both finals have0 physical failure flags, but56117 exceeds the friction safety margin and loses continuous stable support. Its apparent effort changes on only2 common successes are+15.142% whole20s vs reference/+10.591% vs parent, not a savings claim. Development56111=18/18;56117=1/18. Selection retains56111 final and uses unchanged parent55317 fallback for56117. The two leaning seeds have different frozen parents, unlike sitting; not a controlled random-seed-only comparison.

Leaning budgets: each8 critic-only+32 actor-enabled updates,115200 transitions=112950 active+2250 indexed-reset warmup;230400 total. Separate preflight46080 transitions. Exact first12s approach identity holds. Independent `execution_recovery_01/verified_02` and `verified_03` pass identically (27 frozen+1192 prior hashes). Evaluation-only missing holdout calibration labels and symlink checkpoint aliases were narrowly repaired with preserved failed originals and manifests; no training/actor/action/criteria/metrics changes or extra training. Do not modify this completed study: its files are protected by sitting's prior hashes.

Diagnosis: previous leaning training raised leg torque through worse knee/ankle posture. Prescribed persistent coordinated perturbations demonstrated lower-effort stable alternatives. The new learner uses two coordinated adjustments held200ms after12s with a longer horizon and critic warmup, but multiple choices changed together; no single-cause ablation. Failed56117 straightens knees~0.0183rad and increases ankle commands~0.0193rad; mean final5s friction utilization~0.921 vs parent~0.362. A disposable optimizer-state diagnostic proves zero actor gradients do not prevent Adam momentum movement; full-population KL adaptation also includes masked approach samples. This is documented as a future learning correction, not silently applied or established as the sole regression cause. The successful56111 final actually has the larger saved learning rate.

## Separate prescribed unloading probe

`postcontact_unloading_probe_01` and its preserved review are exploratory reference-only work, no additional PPO. One upright continuation passes8/9 development geometries and reduces whole20s/final5s effort3.02%/9.64% on those8 common successes, increasing seat load11.08%→24.76%. The ninth has forbidden contact. Another passes4/9 and raises effort. Neither is validated or promoted; neither affected training, selection or holdout. Small placement/startup differences are recorded. Original plot plus corrected layout both preserved. No more probing/training is running.

## Artifacts and next direction

Remote combined export: `/data/humanoid-sitting-perception/contact_effort_repair_01/final_export_01`.
Mac combined export: `/Users/kadir/humanoid-sitting-control/validated_contact_effort_repair_01/final_results`.
All341 SHA256 entries verified after copy. Includes combined findings.html, sitting/leaning reports, raw evidence, per-height/slope outcomes, all effort/load comparisons, independent validation, research checkpoints, source snapshots, and interactive sitting replay. Combined HTML, sitting comparison PNG, prior leaning plot/report, unloading plot, and final PPO replay on reserved38.5cm/−3.75degree and40cm/+6.25degree surfaces were visually inspected. Local server8797 (exec session4923) serves the package; final report URL `http://127.0.0.1:8797/final_results/findings.html`. Browser tab5 is the deliverable. Do not rerun completed training or exporters into existing directories.

The next physical objective is a legal resting posture that transfers substantially more weight off the feet, then a safe transition into it. Address contact clearance and the documented optimizer phase-masking issue before proposing another study. These results cover nearby rigid slabs with repeated placements and two child runs per task, not arbitrary furniture, electrical energy savings, or a global torque minimum. No policy was promoted.
