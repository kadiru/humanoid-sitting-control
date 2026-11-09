# Active: sitting repair validated; PPO running. Leaning comparison complete — October 14, 2026

Latest user task: fix sitting contact/impact failures, diagnose leaning effort regression, validate, and run a small comparison against the corrected reference and starting policies. Continue directly through Mac SSH to kfuUbuntu; never start another coding agent. Preserve all existing sources/data/checkpoints. No policy promotion.

## Sitting: current work

New root: /data/humanoid-sitting-perception/contact_effort_repair_01.
The selected sensor-driven sitting repair (sitting_repaired_profile.json, transfer15_pitch55) passes9/9 development geometries, then18/18 matched full20s reference trials. It uses measured plane height/slope, foot-preserving IK, a forward torso lean to transfer weight above the seat before lowering, and encoder/IMU foot-anchored pelvis-x feedback. Geometry is13 measured features; horizontal placement is explicit offline-reference assistance. Runtime never reads live simulator truth. Contact/impact/fall/joint/torque/traction criteria remain unchanged.

Original corrected reference and original starting policies each succeeded8/18. With repaired control: reference18/18, parent55211=18/18, parent55217=16/18. All repaired trials have0 physical failures. The two55217 failures are38cm/+5deg placements exceeding the unchanged friction safety margin (~.915 vs limit.9), without slip/fall. The initial both-parent admission failed and is preserved in sitting_preflight_01/status.json. Before any training, admission_02 explicitly excluded55217 as initializer; it remains an unchanged benchmark. Both new child seeds use the same validated55211 parent. Exact zero-child startup/every trace/results equality passed for BOTH parents, including the failed benchmark. Independent sensor-only path reconstruction, full-duration checks, reset isolation and1192 prior hashes passed.

Important cost: on the8 trials the old reference also completed, the repaired reference uses53.68% MORE whole20s squared torque and225.88% MORE final5s torque. Repair success is not an effort improvement. Parent55211 repair similarly costs52.73%/219.68%. Report controller gains and learned gains separately.

Persistent tmux sitting_repaired_ppo_01 now trains/evaluates the fixed study on GPU0. Read docs/REPAIRED_SITTING_EFFORT_PROTOCOL.md and sitting_ppo_01/protocol.json/status.json. Seeds56211/56217 each get40 total updates (8critic-only +32actor-enabled),16 decisions/update,18envs,10physics ticks/decision,115200 transitions/seed including indexed-reset warmup (230400 total). Two child actions held200ms after12s; original parent still updates20ms. Actor/critic406channels, parent405. There is no extra learning smoke. Zero-learning preflight is evaluation only. Initial global0.5s settle is outside the rollout budget. Actor advantages are zero outside resting; actor-enabled update count does not imply every update has a nonzero actor gradient.

Do not modify frozen sitting/leaning code, selected profile, geometry, rewards, budgets or criteria. Study compares full-budget finals against old corrected reference, both original policies, repaired reference and both repaired policies. Selection requires stability plus whole20s AND final5s torque improvement against repaired reference and own repaired55211 parent; otherwise explicit55211 fallback. Both seeds/selection finish before the16 reserved sitting holdout trials (8 geometries twice), which remain unused by sitting repair/training. Worker automatically verifies. Rerun independently into a fresh output, inspect per-seed/per-height/slope success then common-success effort, export report/plots to Mac and inspect before final response. Prepared study scripts are in rl_support_work/scripts and Ubuntu scripts.

## Leaning: verified complete

New leaning study leaning_ppo_01 has finished. Each seed56111/56117:40total updates,32actor-enabled,115200 physical transitions =112950active+2250indexed reset warmup;230400total. Separate16-update preflight46080 transitions. Parents55311/55317 remain frozen; first12seconds of ALL final development and holdout traces match the corresponding parent exactly.

Reserved holdout: reference and both parents16/16; final56111=16/16, final56117=2/16. All0 physical failures. Seed56117 remains in contact without slip but violates the friction safety margin; averaged final5s utilization~.921 vs parent~.362. Knee commands straighten~.0183rad and ankle commands rise~.0193rad. Seed56111 slightly flexes knees and lowers ankle command~.00487rad, reducing friction utilization and effort.

Final holdout whole20s / last10s / last5s squared-torque changes:
-56111 vs corrected reference: −3.889% / −5.686% / −8.016%; vs own parent: −2.457% / −5.156% / −7.495% (all16 common successes).
-56117 vs corrected reference: +15.142% / +28.184% / +39.666%; vs own parent: +10.591% / +26.858% / +38.261% (only2 common successes).
Development56111=18/18;56117=1/18. Selection uses final56111 and parent fallback for56117. No policy promoted. One successful seed is not a repeatable two-seed improvement.

Execution-only recovery preserved originals: unseen labels were absent from the frozen development reward-scale table. Evaluation-only missing labels use the pooled development scale for unused evaluation reward; actor/actions/criteria/metrics/training unchanged. Original failed output remains. A symlink recovery view required verification-only path alias resolution; exact same file/SHA/tensor/optimizer/iteration checks retained. See leaning_ppo_01/execution_recovery_01. Fresh verified_02 and independent verified_03 pass and their main verification JSONs are identical (27frozen +1192prior hashes). No extra training.

Remote final export leaning_export_01. Mac validated_contact_effort_repair_01/leaning_results contains103 SHA-verified files, report, plots, evidence and research-only checkpoints. comparison.png and HTML visually inspected. Browser tab5 (repairReportTab, browser3) at http://127.0.0.1:8797/leaning_results/findings.html. Local Python report server is exec session4923; failed background attempt was harmless. Tab5 marked deliverable this turn. Diagnostics at validated_contact_effort_repair_01/diagnosis are visually inspected; report plotting first field-name failure preserved remotely.

## Remaining task

Finish the authorized sitting PPO comparison, independently verify, create combined succinct report with both leaning results and sitting repair/learning results, copy verified artifacts to Mac, inspect plots/replay and update handoffs. Do not restart or extend leaning training. The old completion automation remains PAUSED; no new automation was created. Continue the current task rather than claiming completion.

Torque is a proxy, not electrical energy or a global optimum. These are two nearby-slab studies with repeated placements, reference assistance and two seeds, not arbitrary furniture generalization. Failed probes, admission failure and all original studies remain preserved.
