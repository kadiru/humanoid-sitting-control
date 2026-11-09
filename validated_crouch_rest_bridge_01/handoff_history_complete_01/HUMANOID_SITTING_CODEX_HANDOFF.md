# Active: crouch-to-rest bridge validated on six surfaces; PPO running — October 15, 2026

Read local `rl_support_work/CROUCH_REST_BRIDGE_STATUS.md` and Ubuntu `docs/CROUCH_REST_BRIDGE_STATUS.md` plus frozen `docs/CROUCH_REST_SAFE_PPO_PROTOCOL.md`. Root `/data/humanoid-sitting-perception/crouch_rest_bridge_01`; main `ppo_safe_surfaces_01`. Work directly through SSH, never another coding agent.

Reference repair passes12/12 clean trials on six admitted geometries; three excluded geometries remain unresolved stress tests. On those12 trials, whole20s/final5s squared torque falls35.50%/84.17% vs current reference. Controller-repair gains, not PPO. First10s approach preserved exactly. Two fresh64-rollout PPO seeds58111/58117 run independently in tmux `crouch_rest_ppo_58111/58117`;368640main transitions total. Every reset starts standing; actor control begins15s and advances earlier only after separate strict validation. Both16/32-update gates retain11/12, so no earlier control yet. No new admitted physical failures; transient traction-margin crossing is the remaining gate failure.

49frozen/8705prior hashes. Persistent `crouch_rest_finalize_01` verifies twice and exports; same-chat completion heartbeat ACTIVE every30minutes. Do not change frozen study, add training or promote a policy. Preflight SHA-verified/visually inspected report: `validated_crouch_rest_bridge_01/preflight_results`, localhost8800. Final results pending. This active study supersedes historical paused/no-work statements below; earlier results remain valid and unchanged.

---

# Complete: near-seat transfer and progressive PPO — October 15, 2026 PDT

Read `docs/NEAR_SEAT_CURRICULUM_STATUS.md` and frozen `docs/NEAR_SEAT_CURRICULUM_PROTOCOL.md` on Ubuntu. Work directly through Mac SSH to `kfuUbuntu`; never start another coding agent. Root: `/data/humanoid-sitting-perception/near_seat_curriculum_01`. All authorized runs/evaluations and the reviewed comparison are complete. No extra training or policy promotion. Completion automation finish-resting-ppo-comparison is PAUSED; no training or monitoring is pending.

Both fresh seeds57111/57117 completed128updates, but neither progressed to standing starts:0 standing-training transitions. Both passed stages5/4 (settled .1/.2rad), then spent88updates atstage3 (actual10s approach crouch). Finalstage3 stable success4/18 and5/18; strict0/18 each;2forbidden-contact failures each at42cm/−5°, starting0.18s/0.16s into transfer. No curriculum gate was bypassed.

Final standing holdout (8reserved geometries repeatedtwice): currentreference and currentPPO56211/56217 each16/16,0physical failures; rollingreference1/16,4physical; new57111=0/16,4physical;57117=8/16,4physical. Strict whole-trial no-slip/friction/contact counts0/16 and1/16 for newpolicies. Development standing0/18 and9/18,4/2physical failures, versuscurrentbaselines18/18. The newpolicies' physical failures are forbidden contacts. Holdout failures occur at39.25/40.75cm with−6.25°; other misses concern continuous support/friction.57117 first contacts unwanted geometry at9.52s on40.75cm/−6.25°, before rolling starts at10s: residual actions change an approach phase not trained from.

On57117's8common successful holdout trials only, whole20s/final10s/final5s summed37joint torque² changes: vsreference−39.330%/−79.833%/−88.211%; vs56211−36.457%/−77.956%/−86.416%; vs56217−37.893%/−78.913%/−87.379%. Final5s seat load78.81%BW vsreference10.78%; torque²102.88 vs872.68Nm²; average instantaneous joint RMS1.667 vs4.855Nm. Legs−97.58% butupperbody+70.27%vsreference. These savings do not offset stability loss or establish clean transfers; only1holdout trial isstrict.57111hasnosuccesses and no valid savings claim. No common holdout successes between57117androllingreference, so no paired effort claim there.

Near-seat final retention:both18/18strict,0physical failures. Compared with identical-start rollingreference, whole20s torque² INCREASES7.23%/21.76%; final5s+7.70%/+21.76%, drivenby leg torque. The physically validated rollingreference itself stillpasses18/18,~75%BW seatload, with2.67%/35.13% final5s savings versusholding the two nearstartposes. Those are reference-motion gains, not learned standing gains.

Verification01/02 plusfreshindependent03 pass with identicalJSON.30frozen/4714prior hashes unchanged, exact startups and finalcheckpointloads, finitechangedactor/critic, criticwarmactoridentity, Adamcontinuity/steps, full20s horizons, fixedfingers and bounded-commandcode/100ms slew checks pass. Main737280transitions=681632active+55648indexedwarmup;7200initialsettlingexcluded;5760smokeseparate. Eachseed368640:57111active340672/warm27968;57117active340960/warm27680. Unfinished/pendingwarmup perblock accounted. No standing training. Both finals preceded holdout. Savednearstartupobs precede rolling-planconstruction; don'toverstate their identity asfirstactorinput.

Newpolicy measuredplane remainsavailable beforefailure in allstandingevals; worstoffset2.19mm,normal.381°, foot-anchoredroot-xerror up to6.75mm withfootmotion. Runtime406sensor/referencechannels,23boundedresiduals,14fixedfingers; truthreset/reward/evaluationonly; explicitreference/horizontalplacementassistance remains. Two seeds/repeatedrigidslabs aren't arbitrary-surface generalization; torqueproxyisnotenergy/globaloptimum.

Mac package:`validated_near_seat_curriculum_01/final_results` (605originalSHA256entriesverified; supplemental REVIEW_SHA256.json). Remote:`final_export_01`. Freshcheck:`ppo_01/verified_03`. Report:http://127.0.0.1:8799/final_results/review.html; originalfindings,comparison/curriculumplots andmatchedreplay preserved. Plots/report/replay visually inspected. Proposed nextresearchonly: smaller validated bridge, smooth preload release, and preserve the successful approach while training the newtransition. Neither is a validated fix; noadditionaltraininglaunched.

---

Historical records follow; completion above supersedes their active status.

# Active: validated near-seat transfer; progressive PPO running — October 14, 2026 PDT

Work directly through Mac SSH to `kfuUbuntu`, project `/home/kadir/humanoid-sitting`; never start another coding agent. Read `docs/NEAR_SEAT_CURRICULUM_PROTOCOL.md` and `docs/NEAR_SEAT_CURRICULUM_STATUS.md`. New root: `/data/humanoid-sitting-perception/near_seat_curriculum_01`.

The rolling-contact reference passes18/18 near-seated 20s trials at both6s and4s movement durations with zero all-tick physical, unwanted-contact, slip or friction-margin violations. Four-second transfer reaches~75% vertical bodyweight on the seat. Relative to holding the identical start, final5s squared torque falls2.67% from the .1rad start and35.13% from the .2rad start. These are reference-motion gains from near-seated starts, not learned standing gains. Static holding also passes18/18. Sensor/model-inferred contact pivot allows horizontal pelvis motion during rotation, avoiding fixed-x shear. Failed probes remain preserved.

Two fresh progressive PPO seeds57111/57117 are running independently in tmux `near_seat_ppo_57111` (physicalGPU0) and `near_seat_ppo_57117` (physicalGPU1). Each128updates/368640transitions;737280total includes indexed-reset warmup, plus7200 initial settling ticks across16processes excluded from training budget. Separate smoke5760transitions. First8updates critic-only,120actor-enabled; optimizer state continues across blocks. Stages5/4 are settled .1/.2rad starts,3/2/1 are actual current-reference10/8/6s states,0standing. All18 development trials must pass strict no-slip/contact/friction plus>=60% seat weight before advancing. Do not bypass gates or add budget. Both first two gates passed18/18 and reachedstage3 (actual10s approach crouch). Full standing reference remains3/18, so standing success is unresolved.

Frozen:30 input hashes,4714 protected prior hashes. Runtime406sensor/reference channels,23bounded residuals,14fixed fingers; truth only reset/reward/evaluation. Preserve frozen rewards/contracts/budget/criteria. Holdout8reserved geometries repeatedtwice opens only after both final checkpoints. Final20s standing comparisons: currentreference,currentPPO56211/56217,rollingreference,final57111/57117; near-final retention evaluated separately. No policy promotion or checkpoint selection.

Persistent tmux `near_seat_finalize_01` waits for both workers, independently verifies twice, then exports `final_export_01`. Read `finalization_status.json`. If reporting/execution fails, preserve originals and document narrow recovery; never silently alter training. Copy final export to `validated_near_seat_curriculum_01/final_results`, verify SHA256, inspect plots/replay, update both handoffs and give an honest comparison including actual reached stages and actual standing-training count.

Near-seat validation already SHA-verified and visually checked at Mac `validated_near_seat_curriculum_01/validation_results`; report http://127.0.0.1:8799/validation_results/findings.html. Same-chat completion automation finish-resting-ppo-comparison is ACTIVE every30minutes, quiet during normal progress. Pause after final comparison. Persistent simulator workers/finalizer continue independently of this turn.

---

Historical records below; this active study takes precedence.

# Latest completed study: seat-load target and standing PPO — October 14, 2026

Read `docs/SEAT_LOAD_TRANSFER_STATUS.md` and `docs/SEAT_LOAD_TRANSFER_PROTOCOL.md` on Ubuntu. Work directly through Mac SSH to `kfuUbuntu`; do not start another coding agent. Root: `/data/humanoid-sitting-perception/seat_load_transfer_01`.

The seated target passed 18/18 full20s tests with no unwanted contacts, 73–74% vertical bodyweight on the seat and 91–96 Nm² total squared torque. The standing-transition pilot did not preserve reliability: reserved current reference/current policies56211/56217 each16/16; new reference0/16; new PPO56811=0/16 and56817=1/16. Each new arm has4 forbidden-contact failures; other misses mostly concern friction margin. Development new PPO0/18 and2/18 versus current18/18. Geometry remained available; foot motion challenges the stationary-foot kinematic estimate.

One successful reserved trial has 30.5% lower whole-motion squared torque and83.5% lower final5s torque versus current reference, but this is not generalizable evidence and does not offset stability loss. Static seated retention is separate from reaching it. No policy was promoted. No training or new automation is pending; existing completion automation remains paused.

Two64-update fresh seeds completed368640 total physics transitions; first8critic-only,56actor-enabled. Both independent verifications pass with identical JSON, exact startups, final checkpoint identity, budgets and52frozen/3341prior hashes. Full result package: remote `final_export_02`; Mac `/Users/kadir/humanoid-sitting-control/validated_seat_load_transfer_01/final_results`. Report/replay: `http://127.0.0.1:8798/final_results/findings.html`. Export01's CSV error and all probes/failed smokes remain preserved.

Proposed next research: establish a valid near-rest-to-rest transfer with low tangential foot forces and housing clearance, then progressively move starts earlier; do not extend the unchanged all-standing pilot or tune on its holdout. The broader goal remains unresolved.

---

Historical completed studies follow; the latest result above takes precedence.

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

---

## Preserved earlier handoff

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

---

## Preserved earlier handoff

# Active: sitting contact repair and independent leaning effort PPO — October 14, 2026

Current user authorization: diagnose/fix sitting contact and impact failures and the prior leaning effort regression; validate, then run a small comparison against corrected reference and starting policies. Work directly through SSH to kfuUbuntu, never another agent. Preserve all prior work.

New data: `/data/humanoid-sitting-perception/contact_effort_repair_01`. Read `docs/CONTACT_EFFORT_REPAIR_PROTOCOL.md` (frozen leaning definitions) and `docs/CONTACT_EFFORT_REPAIR_STATUS.md` (live status). Protected prior manifest covers1192 hashes. The previous transfer effort study is complete and unchanged.

Sitting remains UNVALIDATED. Original reference replay matches every saved startup/trace array exactly (8/18 successes). Height/slope tracking is available; contact/impact failures follow excessive rearward weight transfer during the stage4→5 segment. Simple plane adaptation, damping, lowering, and clearance braking did not pass all nine development geometries. All probes preserved. A replacement front-contact path passes offline foot-preserving IK and is being tested in `sitting_front_contact_probe_07` onGPU0. Do not claim a repaired controller or start sitting PPO before physical validation. Reserved new holdout remains unused for repair selection.

Leaning: controlled persistent posture perturbations established lower-effort stable motions (diagnostic knee/hip/ankle synergy, not learned PPO). New child PPO retains frozen parent policy and original physical command bounds, learns two coordinated posture adjustments at200ms after12s; reward directly targets total squared37-joint torque with unchanged support/failure criteria. Zero-child replay is exact. Critic warmup preserves actor;16-update preflight passes18/18 and all checks. Independent two-seed study `leaning_ppo_01` runs in tmux `leaning_effort_comparison_01` onGPU1: seeds56111/56117, each8critic-only+32actorupdates,115200physics transitions (230400total), preflight46080separate. First development checkpoint56111 passes18/18 and reduces whole-trial effort~0.45% vsparent; this is preliminary, not a holdout result. Both seeds finish before holdout. Selection must improve both whole20s and final5s vs BOTH reference and parent while retaining stability; otherwise parent fallback. Final checkpoints reported independently of selection. Do not modify frozen leaning files or add training. No policy promoted.

Completion requires independent fresh verifier, full20s matched starts/checkpoints/command/budget/source checks, success and physical failures before paired effort comparisons, holdout results separated from development, reports/plots copied to Mac and visually inspected. Torque is an effort proxy; two seeds and nearby rigid surfaces do not establish arbitrary-surface behavior. Existing completion automation is PAUSED; active work continues in this chat.

---

## Preserved prior completion

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

---

## Preserved previous handoff (superseded by completion above)

# Active: unseen-surface assessment complete; effort PPO training — October 14, 2026 PDT

User authorized unseen heights/slopes, then reducing total joint effort without losing stability versus corrected reference. Work directly from this Mac through SSH to kfuUbuntu; never start another coding agent. Prior studies remain unchanged. New root: /data/humanoid-sitting-perception/transfer_effort_01. Read docs/TRANSFER_EFFORT_PROTOCOL.md.

## Verified transfer assessment

Original motion and standing state unchanged; surface top-center height ±2 cm and pitch ±5 degrees, crossed into nine conditions per task (one known anchor + eight new geometries), twice repeated =18 trials. Sitting base 40 cm/0 degrees; leaning base slab top-center 65.136973 cm/45 degrees.

Corrected reference and both parent PPO policies: sitting8/18 overall (known2/2, unseen6/16); leaning18/18 overall (known2/2, unseen16/16). Sitting reference and parent55211 have10 physical failures; parent55217 has8 physical failures plus2 unsupported failures. All leaning trajectories have0 physical failures. Sitting low38 cm seats fail impact/forbidden-contact criteria;40 cm/+5deg and42 cm/−5deg also expose contact/impact limits. Ending in a low-effort seated pose does not erase earlier physical failure. Fixed standing remains physically safe18/18 for each task, but does not reach support.

New generic harness exactly reproduces frozen old startup and every trace array on the unchanged geometry. Indexed reset isolation, exact matched physical/sensor starts, actual parent checkpoint loading, full20s horizons, fixed/finger commands, bounded residuals, geometry audits and436protected hashes pass. No missing tracked geometry in audited valid pre-failure motion; worst offset2.173mm sitting /0.918mm leaning, normal error below0.346degrees. The observed failure boundary is mechanical, not missing seat-distance input. No physical criteria relaxed.

Mac evidence: validated_transfer_effort_01/assessment_results. All70SHA256 entries verified; height/slope plot and lower-seat failure / higher-steeper leaning replays visually inspected. Report: http://127.0.0.1:8796/assessment_results/report.html; replay beside it. Server session99566; browser tab3 deliverable. This is PRE-training transfer evidence, not a result of the new effort PPO.

## Running effort study

Independent persistent tmux transfer_effort_sitting_01(GPU0) and transfer_effort_leaning_01(GPU1) execute scripts/start_transfer_pipeline.py, which already verified assessment and optimizer preflight. Two-update smoke: sitting8/18, leaning18/18, preserving all successful reference/parent conditions with no additional physical failures; exact starts and finite changed weights pass. Smoke4608transitions total is separate. Six reward/selection behavioral tests pass; log reward_selection_tests.log.

First study seeds55611 and55711 are actively updating; second seeds55617 and55717 queued independently. Each is initialized from the corresponding prior final policy (sitting55211/55217;leaning55311/55317), retaining actor/critic/optimizer continuity. Four32-update blocks all start standing,18envs,64steps:128new updates/147456transitions perseed including indexed-reset warmup,589824total. Read ppo/{sitting,leaning}/status.json and protocol.json plus pipeline/preflight statuses; active/warmup counts are separate. No shared curriculum gates.

Reward directly favors lower sum of squared physical Nm over all37joints, scaled by successful corrected-reference final10s per surface. For failed reference conditions, pooled successful-reference torque is only a reward scale, never a failed-fragment savings baseline. Reward makes effort relevant while retaining support bonus, physical failure cost, late unsupported penalty and unchanged contact criteria. Runtime405sensor/reference channels, bounded23body actions and fixed fingers remain unchanged. Truth is labels/reward/reset only, not live actor/controller geometry.

Every32updates evaluate the complete development grid. A candidate must retain all conditions successful under corrected reference OR its parent, and add no physical-failure condition. Select using development stability first then mean squared torque on corrected-reference successes. Always report the full-budget final checkpoint separately; a parent fallback is explicit, not a learned improvement or automatic deployment.

Reserved before assessment: eight distinct holdout geometries per task, twice repeated =16trials: height±1cm × slope±2.5degrees, height±3cm at anchor slope, and slope±7.5degrees at anchor height. Never used for training/calibration/checkpoint choice. Both seeds finish and selections freeze before holdout evaluation. Baselines: corrected reference and both parent policies; evaluate final and selected checkpoints as defined. The original development grid is now training data, not an unseen final evaluation.

Frozen sources/reward/budgets/geometry/contracts must remain unchanged while running. Necessary execution/report repairs require preserved originals and a recovery manifest. No extra training or policy promotion. All previous studies protected. Torque is an effort proxy, not energy/global optimum; two seeds and nearby rigid slabs do not establish arbitrary-surface generalization.

## Completion

scripts/verify_transfer_effort_study.py verifies each worker automatically. Re-run independently into fresh verified_02 at completion; inspect source/prior hashes, optimizer continuity, exact checkpoints/starts, full20s horizons, active/warmup and unfinished accounting, constant fingers/fixed commands, bounded residuals, geometry errors/availability, selection eligibility and heldout separation. Compare failures/success first, then whole20s/last10s/last5s squared/absolute/RMS torque and leg/upper contributions on common full successes against corrected reference AND parent, by seed/height/slope/bank. Report final and selected/fallback policies distinctly.

Persistent tmux transfer_effort_export_01 runs scripts/finalize_transfer_effort.py --wait; after both verifiers pass it produces final_export_01 with SHA256.json. Copy to Mac validated_transfer_effort_01/final_results, verify hashes, inspect plots/results, update handoffs, report honestly, then pause the completion follow-up. The same-chat completion automation finish-resting-ppo-comparison is ACTIVE every30minutes for this study, quiet during normal progress. No new task or agent.

---

Previous completed study:

# Verified complete: independent sitting and leaning PPO — October 14, 2026 PDT

All four final policies pass 28/28 full 20-second trials, including 4/4 standing starts, with zero physical failures. Each also passes 4/4 at every starting stage 0–6. The corrected reference passes the same 28/28 per surface. Old reference: sitting 20/28 with 8 physical failures and 0/4 standing; leaning 11/28 with no physical failures but deficient support/traction and 0/4 standing. Fixed posture qualifies 8/28 sitting and 20/28 leaning, with no physical failures; fixed standing is not a transition controller.

Controller repairs, rather than PPO, explain the main improvement. Sitting uses IMU ankle balance feedback; leaning uses coordinated knee/hip/ankle flexion only for approach stages 0–4. Original physical criteria remain unchanged. The initial leaning smoke regression and all prior studies are preserved.

PPO whole-20-second squared torque relative to corrected reference, on all 28 matched common successes:
- Sitting 55211: −0.23%; sitting 55217: −0.08%.
- Leaning 55311: +0.52%; leaning 55317: +0.87%.
Absolute torque and average instantaneous all-joint RMS increased for all four policies. Standing-only squared torque changed +0.05% / +1.80% for sitting, and −1.43% / +4.66% for leaning. There is no consistent learned effort advantage, including in the final 10/5-second windows.

Corrected reference versus old reference reduces whole-trial squared torque 11.09% sitting (20 common successes), 12.79% leaning (11). PPO versus fixed posture reduces it 15.84–16.04% sitting (8 common successes), 2.12–4.02% leaning (20). These case sets differ; gains relative to fixed/old motion include controller changes and cannot be credited entirely to PPO. No savings computed from failed fragments.

Completed separate seeds 55211/55217 sitting and 55311/55317 leaning: 128 updates each, stages [4,2,0,0] in 32-update blocks. Final 64 updates train standing. Exactly 524,288 transitions = 512,000 active + 12,288 indexed-reset warmup; development smoke 6,144 separate. No shared curriculum gate. Leaning first-stage training episodes failed the support criterion without physical failures; later stages and final deterministic policies passed.

Both original verifiers and independent fresh verified_02 reruns pass with exactly identical verification JSON. Frozen source hashes (37 sitting, 38 leaning), 148 distinct prior artifacts, finite changed actor/critic, optimizer continuity, exact final checkpoint loading, exact physical/geometry/sensor startups, full horizons, constant fixed/finger commands, bounded learned/reference commands and sampled slew checks passed. The original 270-file export SHA256 manifest verified on Mac; derived review artifacts have a separate manifest. No frozen source, criterion, reward, budget or geometry was changed during completion.

No missing measured geometry in audited valid-motion/pre-failure samples. Maximum plane-offset error 2.107 mm sitting / 0.670 mm leaning, normal error below 0.363 degrees overall. Runtime actor/critic use 405 sensor/reference channels and 23 bounded body-joint residuals; simulator truth supplies labels/rewards/reset construction, not runtime geometry. Final standing-start seat support carries about 58–61% bodyweight vertically, leaning support about 16%; feet carry the rest.

Evidence: /data/humanoid-sitting-perception/split_support_curriculum_02/final_export_01.
Mac: /Users/kadir/humanoid-sitting-control/validated_split_support_02/final_results.
Open findings.html for conclusions and all seed/stage/window/torque comparisons; motion_replay.html for matched old/reference/PPO movement; effort_by_stage.csv for full grouped metrics. comparison.png, sitting/leaning standing replays and the final findings page were visually inspected. Local preview: http://127.0.0.1:8795/final_results/findings.html.

Two known rigid surfaces (40 cm level seat, 45-degree leaning surface), seven known start states repeated at four simulator placements, and two training seeds per task do not establish arbitrary-surface generalization. No disturbances or unseen surfaces tested. Reference-assisted residual PPO is not unassisted motion discovery. Torque is an effort proxy, not energy or proof of a global minimum. No policy promoted; no further training started. The completion automation finish-resting-ppo-comparison is now PAUSED.

Future work, not launched: test the stable motion on new heights/slopes/approach offsets, then target effort reductions beyond the corrected reference while retaining support. Work directly through SSH from Mac; never start another agent on Ubuntu.

---

Historical snapshot (superseded):

# Active: separate sitting and leaning PPO — October 14, 2026 PDT

Both corrected references and latest2-update learned smokes pass14/14 full20s
with exact starts and no physical failures. Controller repairs cut whole-trial
squared torque11.1% sitting /10.2% leaning vs oldreference on common successes.
These are controller gains; final PPO comparison remains pending.
Separate persistent GPU0/GPU1 workers train2seeds per task,128updates each,
including64updates fromstanding (524288study transitions total).
Sitting55211 finished training; remaining training/evaluations are queued.
Read docs/SPLIT_SUPPORT_STATUS.md, SPLIT_SUPPORT_PROTOCOL.md and
SPLIT_LEANING_REPAIR.md (local mirror rl_support_work/docs).
Remote root: /data/humanoid-sitting-perception/split_support_curriculum_02.
Completion follow-up ACTIVE; final export/copy/visual review still required.
Mac preflight evidence: validated_split_support_02/preflight_report_01.
Prior studies preserved; no policy promoted or arbitrary-surface claim.

---

# Active: independent sitting and leaning — October 14, 2026 PDT

Contact fixes validated: sitting reference14/14 vs old10/14; leaning14/14 vs old6/14.
Both corrected references succeed from standing2/2; no physical failures.
This is controller repair, not learned PPO improvement. Matched fixed/reset/optimizer
preflight is running independently on GPUs0/1. Read SPLIT_SUPPORT_STATUS.md and
SPLIT_SUPPORT_PROTOCOL.md in remote docs or the local rl_support_work/docs mirror.
New data: /data/humanoid-sitting-perception/split_support_curriculum_02.
Previous frozen studies/checkpoints remain intact (148 distinct hashes).
Planned separate policies each receive128 updates, including64 fromstanding,
with matched reference/fixed comparisons. No generalization or policy-promotion claim.

---

# Latest: two-surface curriculum verified complete — October 13, 2026 PDT

Fourteen intermediate/end poses passed 20-second retention. Two gated PPO seeds
completed 294,912 transitions (128/160 updates), stopping at leaning traction
failures before standing training. Final support success: reference 9/14;
last PPO 6/14 and 9/14, with 3 physical failures each versus reference 2.
One seed leaned from standing in the final test; neither sat from standing.
Static pose feasibility is established; reliable transitions remain incomplete.

Seated effort fell ~19% versus fixed posture but only ~1.6–2.0% versus the reference
motion on the respective common successes. Leaning effort increased versus fixed.
A gate/final leaning outcome changes with layout/GPU; do not claim robustness.
No policy promoted. Frozen sources and prior studies preserved and verified.

Read docs/TWO_SURFACE_CURRICULUM_STATUS.md (Mac mirror: rl_support_work/docs).
Remote evidence: /data/humanoid-sitting-perception/two_surface_curriculum_01/final_export_01.
Mac report: validated_two_surface_curriculum_01/final_results/findings.html.
Next proposed: separate surface curricula and improve contact-safe weight transfer.
No new training is running or authorized by this handoff alone.

---

> **Two-surface curriculum running — October13,2026:** Independently validated14/14 stationary intermediate poses for a40cm seat and45deg leaning surface. Reference-only motion passes9/14 starts; standing transitions remain unsolved. Fixed retention, reset isolation, exact startup matching and2-update PPO smoke(2/2 endpoint successes) pass. Fresh seeds54911/54917 now train with32-update blocks and2/2 gates before moving starts toward standing. Read TWO_SURFACE_CURRICULUM_STATUS.md /PROTOCOL.md and remote two_surface_curriculum_01/ppo_01 worker files. Preserve all prior studies; no other agent and no policy promotion. Earlier snapshots are history.

---

> **Direct effort PPO complete — October 12, 2026:** Fixed posture29/29; final PPO54511=11/29 and54517=15/29, with0/12 leaning in each. No falls; one forbidden-contact case per seed. On common successful trials, whole20s total squared torque increased65.46% /55.58%, mostly in the legs. Isolated40cm savings did not generalize. Both full budgets, exact starts/checkpoints,105 frozen and61 prior hashes pass. Steep75° perception errors persist separately. Read EFFORT_REST_STATUS.md and validated_effort_rest_01/final_results/review.html. No policy promoted or extra training; standing transitions remain unsolved. Earlier snapshots below are history.

---

> **Direct effort-minimization PPO running — October 12, 2026:** The user made total joint-effort reduction the primary objective. A separate policy learns bounded posture changes from validated sitting/leaning starts, with the stable motor reference outside the neural network. Fixed posture29/29 and two-update learned smoke13/13 full20s pass. Two fresh256-update seeds54511/54517 now train on GPUs0/1 with matched final torque/stability comparisons queued. Read EFFORT_REST_STATUS.md and EFFORT_REST_PROTOCOL.md. No final effort-saving claim yet; standing transitions remain unsolved. Previous experiments remain complete and preserved. Earlier snapshots are history.

---

> **Standing-to-support PPO complete — October 12, 2026:** Both 224-update seeds scored 0/13 training and 0/16 unseen surfaces; all 58 policy trials physically failed within 0.62–0.80 seconds, although rollouts continued for the full 20 seconds. Fixed standing remained stable in 29/29. Supported endpoint poses are validated, but reaching them is not learned. Command drift appears after the first PPO update; pre-failure tracking remains available but reaches 36 mm error. Budgets, exact startup/checkpoint matching and all 36 new/46 prior hashes pass. Read the current SUPPORTED_SURFACES_STATUS.md and validated_supported_surfaces_01/final_results/review.html. No policy promoted or additional training; completion monitor paused. Earlier snapshots below are history.

---

> **Standing-to-support PPO running — October12,2026:** Supported targets across level seats and slopes through75deg validated; corrected standing reset passed29/29 full20-second tests. Two fresh224-update PPO runs now train reaching support from standing, with effort secondary. Automatic evaluations reserve16 unseen configurations. Read rl_support_work/docs/SUPPORTED_SURFACES_STATUS.md and the frozen SUPPORTED_TRANSITION_PROTOCOL.md. No learned standing-transition or unseen-surface success claimed yet. Existing completion monitor reactivated for this pilot. Earlier snapshots are history.

---

> **Active standing-to-support task — October 12, 2026:** Supported sitting/leaning transition task active, October 12 2026. User authorized validated supported goals across heights/slopes, then standing-to-support PPO with effort secondary and unseen surfaces. Work directly via Mac SSH; no other agent. Completed prior bounded/resting/capped studies remain unchanged. New data root /data/humanoid-sitting-perception/supported_surfaces_01. New arbitrary-plane geometry/contact/depth code: scripts/run_supported_postures.py, prepare_supported_postures.py, sitting_task/src/humanoid_sitting/supported_surface.py. Posture validation01 found32/76; independent restored bank32/32 full20s. Includes level/tilted sitting and pelvis-supported leaning through60deg. Rear-camera768/768 samples: normal p95 .169deg, plane offset p95 1.49mm in those supported poses only. Down camera sometimes selects front face; excluded from planned actor. Back-shell logo_link is fixed to torso and its rear face is now a legitimate support patch only in NEW task; no previous labels changed. Vertical/75deg back-supported poses stable but fail original conservative traction reserve; not successes. Running preserved preload probe (108 candidates) on GPU1, after corrected coordinator path with failed attempt preserved. Transition reset/environment implementation in progress; no new transition optimizer run yet. See new status/protocol as authored; don't interpret stage_complete as overall task completion. Existing monitors remain paused until new training is ready.

---

> **Bounded resting PPO complete — October 12, 2026:** Both trained policies and fixed posture passed56/56 full20-second conditions. No physical failures, but normalized whole-body effort increased0.28%/2.59% by seed, including worse effort on all tested heights. Optional hold never activated and had no effect. Budgets, checkpoints, startup matching and all46 frozen hashes independently verified. No policy promoted or extra training started. Read the latest status and validated_resting_bounded_02/final_results/review.html. Completion monitor paused; all previous outputs preserved. Earlier snapshots below are history.

---

> **Bounded resting PPO pilot active — October 12, 2026:** Old-policy contact failures traced to hip-pitch housings on the seat and the right hip-roll joint limit. New bounded reference offsets retain fixed posture exactly in40/40 full20-second trials plus16/16 with the optional hold rule enabled; reset isolation and repeated resets pass. A separate two-seed64-update study compares fixed posture, PPO and PPO with measured-effort hold over56 conditions, including16 fresh states. Read the latest status and docs/RESTING_BOUNDED_PROTOCOL.md; persistent tmux resting_bounded_ppo_02 uses GPU1. The existing resting completion monitor is active for this new study. Prior studies and all checkpoints are preserved. Earlier snapshots below are history.

---

> **Resting PPO comparison complete — October 12, 2026:** Fixed posture40/40; all four learned policies0/40. Effort reward lengthened qualifying rest but produced no successful policy or validated effort saving. All budgets, checkpoint/startup matching and source preservation verified. See the latest status and validated_resting_posture_01/final_results/review.html. Both resting and capped-impact studies are complete, their monitors paused, and no policy promoted. Earlier snapshots below are preserved history.

---

> **Capped-impact comparison complete — October 12, 2026:** All four32-update runs and matched evaluations verified. Starting policy53/80; old→capped35→36/80 and54→55/80 by seed. Successful-trial peak forces increased in both seeds; the pooled effort benefit did not generalize to the new holdout. No policy promoted. Read the latest status and verified report. The seated-resting study remains separate under its own completion monitor. Earlier status snapshots below are preserved history.

---

> **Active seated-resting PPO study — October11,2026:** Seated resets and fixed-posture baselines validated. PPO is running on physical GPU1 across25/30/35/40cm seats, with two seeds and matched stability/effort arms. Read the latest status. The separate capped-impact comparison continues on GPU0; both checkpoints and histories are preserved.

---

> **Active capped-impact study — October11,2026:** Integer timer fix validated. Four32-update PPO runs (old/capped reward, two seeds) and matched80-case evaluations are queued in persistent Ubuntu tmux. Read [latest status](rl_support_work/docs/SUPPORT_RL_LATEST.md). Same-chat completion follow-up is active; earlier completed/paused states below are history.

---

> **Completed matched RL study — October11,2026:** Starting PPO47/64; old-reward continuation44/64; new reward42/64 (43 after a reproduced timer-rounding correction). New reward did not help overall; impact penalty was overscaled. All jobs finished. Read [latest status](rl_support_work/docs/SUPPORT_RL_LATEST.md) and its diagnosis/next-experiment proposal. Earlier active-job and pause instructions below are history.

---

> **Active RL study — October 11, 2026:** Two matched128-update PPO runs are queued in persistent Ubuntu tmux, followed by automatic evaluation. First real optimizer updates verified. Perception is frozen; earlier RL pause instructions below are history. Read [latest status](rl_support_work/docs/SUPPORT_RL_LATEST.md).

---

> **Current verified state — October11,2026:** Independent distance interface validated. One PPO pilot completed:35/48→37/48 successes, no lost successes, but worse impacts on two banks; checkpoint preserved, not promoted. All jobs finished. Read [latest status](rl_support_work/docs/SUPPORT_RL_LATEST.md). The earlier handoff below is preserved history.

---

> **Current handoff — October 11, 2026:** Camera-timing regression repaired experimentally with 50 Hz kinematic geometry feedback: original bank 16/16; fresh validation 10/16 and 9/16. Broader gates still fail; RL is paused and defaults unchanged. All jobs finished. Read [latest verified state](rl_support_work/docs/SUPPORT_RL_LATEST.md). Continue through Mac SSH as the sole primary agent.

---

# Humanoid Sitting — ChatGPT → Mac Codex handoff

**Prepared:** 2026-10-09 (America/Los_Angeles)  
**Purpose:** Transfer the high-signal technical context and decisions from an extensive ChatGPT conversation to a **new Codex session running locally on the Mac mini**, which will control the Ubuntu GPU workstation over **passwordless SSH**.  
**Status:** Historical account based on screenshots/transcripts supplied in ChatGPT. **The files, code, checkpoints, metrics and active processes on Ubuntu are the source of truth. Verify them before any further modifications.** This is a detailed *handoff summary*, not a verbatim export of every ChatGPT message.

## 0. IMPORTANT — latest user decision supersedes all previous next-step requests

**STOP iterating on the soft-landing contact-force limiter, estimator thresholds, damping or PPO simply to improve the simulated seat-impact metric.** The user has explicitly decided to stop because the research has been circling the same issue without solving spatial perception and relative control.

**NEW PRIMARY TASK:** Build and validate a **shared 3D representation of the seat/floor and the robot's own body**, using deployment-realistic 3D sensing and proprioception. The robot needs to know, in a consistent coordinate frame, **where its pelvis/hips are relative to a candidate sitting surface**, so it can align and lower itself using closed-loop visual/geometric servoing. This geometric state-estimation capability must work before optimizing additional sitting-policy training. Preserve all previous successful controllers and research artifacts.

User's desired functional sequence:

1. Perceive candidate sitting support surfaces (chair, couch, bench, box, table where physically feasible), floor, edges and obstacles with onboard depth/point-cloud sensing (and RGB if available).
2. Fit usable surface geometry and determine what is sittable; do not rely simply on furniture category labels.
3. Register the observed furniture and robot pelvis/feet/kinematics **in the same reference frame** with calibration, robot forward kinematics and IMU/odometry.
4. Estimate and track pelvis-to-seat **height/clearance**, longitudinal/lateral offset, orientation, support region, and uncertainty, including when the surface leaves the camera view as the robot turns.
5. Visual-/geometric-servo the robot to an appropriate approach/alignment pose; then control its descent relative to the estimated seat surface, not to a hard-coded joint-angle schedule.
6. Use robot-available proprioceptive feedback for contact/weight-transfer inference and stabilization; don't require a real chair to report contact forces.
7. Train a student sitting policy only once its inputs are meaningfully grounded in actual onboard sensors and a validated shared 3D frame; maintain privileged simulator state **for teachers, debugging and evaluation only**.

### Exact initial next milestone (do not expand scope yet)

**In Isaac Sim, using simulated depth camera and/or LiDAR measurements plus robot state/kinematics, reconstruct the usable seat and floor, compute pelvis pose in the SAME 3D frame, display/record the result, and demonstrate accurate pelvis-to-seat relative tracking across robot/sensor motion.** Use Isaac Sim ground truth exclusively as a metric for evaluating the estimator; do not feed it into the estimator or deployed actor.

For example, with a gravity-aligned local odometry frame `O`, estimated seat frame `S`, and pelvis/hip frame `H`:

```
T_O_S = estimate_from_depth_point_cloud_and_odometry(...)
T_O_H = T_O_base * FK(base_to_pelvis; joint_encoders)
T_S_H = inverse(T_O_S) * T_O_H
```

Conventions for transform direction must be documented and tested (the symbols above are illustrative); distinguish pelvis/root frame from physical hip contact surface. Useful outputs include signed pelvis-contact-proxy clearance to the seat plane, closest approach to a usable support polygon, yaw error, lateral/forward offset, and confidence/covariance. Correctly model sensor extrinsics and the camera's field of view, occlusions, delayed frames and floor-relative body pose. Avoid using visual ground-truth poses, simulator contact forces, or global reset metadata in runtime estimates.

**Exit criteria:** for multiple chairs/benches/couches/boxes (including geometries excluded from fitting), report seat/floor reconstruction errors, pelvis-to-surface clearance errors in centimeters, lateral/yaw pose errors, frame timestamp consistency, occlusion robustness, and live 3D visualizations showing sensor cloud + fitted planes/surfaces + robot meshes/joint skeleton + pelvis/seat frames. Produce a short video/recording where robot moves and the estimated relative transform updates correctly. Agree a reasonable metric tolerance based on depth-sensor noise and the target joint/contact geometry rather than choosing arbitrary thresholds.

Do not start expensive RL, training new policies, Cosmos, or hardware deployment as part of this first milestone.

---

## 1. Execution environments and intended workflow

### Mac mini — coordinating Codex

- Local user account: `kadir`.
- Mac's local Ethernet IP previously observed: `192.168.68.53` (`en0`, wired 1 Gbps). Wi-Fi/other interface also had `192.168.68.56`, but route to Ubuntu selected `en0`.
- Passwordless SSH is **verified working** from the Mac using the alias:

  ```bash
  ssh kfuUbuntu
  ```

  The SSH alias resolves to `kadir@192.168.68.60`. `scp kfuUbuntu:... ~/Downloads/` should work without prompting for the Ubuntu account password.
- New desired workflow: **Start one Codex agent locally on the Mac**, with this handoff file in its working directory. **That agent should execute work on Ubuntu through SSH**. There is *no need to start a second Codex agent on Ubuntu*. Avoid independent simultaneous agents editing the same repo.
- The Mac Codex agent may execute non-interactive commands such as:

  ```bash
  ssh kfuUbuntu 'hostname; nvidia-smi; df -h /'
  ssh kfuUbuntu 'cd ~/humanoid-sitting && git status --short'
  ssh kfuUbuntu 'cd ~/humanoid-sitting && cat README.md'
  ```

  Verify the correct repository location before assuming Git root. For long-running GPU jobs, arrange **remote `tmux` sessions or another supervised persistent process** and monitor remote logs/exit statuses. A persistent tmux *terminal* is not by itself a continuously autonomous agent; Codex still needs bounded tasks and clear continuation/stop criteria.
- The user previously expressed a preference not to approve every command. This requires deliberate permission configuration for the **Mac Codex session**; `--yolo` removes sandbox/approval safeguards and grants broad access to the Mac user account. Use only with explicit understanding of risks, avoid `sudo` or unrelated host edits, and prefer bounded command scope. Do not bypass safety-critical testing or alter existing system configs simply because command approvals are disabled.

### Ubuntu GPU workstation — execution target

- Hostname: `kfuHomeUbuntu`.
- User: `kadir`; IP: `192.168.68.60`.
- OS: Ubuntu 22.04.5 LTS; GLIBC 2.35; kernel seen as 6.8.0-138-generic.
- CPU: Intel Core i9-14900F, 32 logical CPUs.
- RAM: ~62 GiB.
- GPU 0: **GeForce RTX 4090 (24 GiB)**, intended for simulation/training.
- GPU 1: **GeForce RTX 4070 (12 GiB)**, serving the desktop.
- NVIDIA driver reported: **580.178.04** (CUDA supported by driver 13.0). No driver change was needed for validated runs. Treat future compatibility complaints as diagnosable, not a reason for blind upgrade.
- Early root free space ~86 GiB, falling over many experiments to **~33–35 GiB free by latest reports**. Recheck before downloads/data capture. `/data` was previously reported with ~85 GiB free but mount permissions/status should be verified afresh; do not assume writable. Many datasets, videos and caches are in the project.
- An earlier globally visible Isaac Sim 4.5 Python 3.10 installation predated the new project. Preserve old environments and drivers.
- Current project set up Python **3.12.15**, **Isaac Sim 6.1**, **Isaac Lab `release/3.0.0`** (a pre-release line at project start), **RSL-RL**; exact installed versions, revision, package locks and command syntax must be checked in the local checkout, logs, and `uv.lock`.
- Project root: **`/home/kadir/humanoid-sitting`**.
- Likely structure includes `IsaacLab/`, `sitting_task/`, `scripts/`, `docs/`, `runs/`, `logs/`, `datasets/`, `baselines/`, `cache/`, `tools/`.
- Most existing scripts used `bash scripts/run.sh ...` as the wrapper for the local pinned Isaac Lab runtime. Review `README.md`, `docs/commands.md` if present, existing source and their latest saved runtime commands, rather than inventing CLI flags for current Isaac Lab.
- Existing Ubuntu Codex CLI session was completed and can be closed using `/exit`; no need to preserve the old interactive agent session after handoff, but preserve files, logs, checkpoints, and active remote processes if any.

### Previous access blocker (solved)

The original Ubuntu Codex CLI sandbox hid `/dev/nvidia*` and DNS, despite the host working. It was fixed by permitting execution outside its sandbox: an elevated `nvidia-smi` worked and `git ls-remote https://github.com/isaac-sim/IsaacLab.git HEAD` worked. This was **not** a host network or NVIDIA driver failure. Mac-to-Ubuntu SSH has likewise been tested and succeeded. If Mac Codex reports network failure, check its own sandbox and SSH execution environment before changing the Ubuntu workstation.

---

## 2. Scientific goal and longer roadmap

**Ultimate goal:** a humanoid robot should be able to visually perceive a realistic sitting surface (chair, couch, bench, box, or feasible table), decide whether/where it can safely sit, approach and align itself, and lower its hips onto the surface with controlled contact and stabilization. First in NVIDIA Isaac Sim/Isaac Lab, later robust enough for physical humanoids. Cosmos may later help with image/appearance/data diversity; it cannot replace physical contact modeling or reliable 3D registration.

- Initial robot: **Unitree G1 simulated model**, currently apparently a 23-body-joint interface (verify exact model/joint mappings). Actual real G1 configuration **not yet chosen**.
- Physical sensor assumptions: RGB/depth cameras and/or LiDAR if fitted; joint encoders and joint velocity estimation; IMU; motor torque estimates if supported by selected hardware; previous action history. **Do not assume skin pressure sensors, seat-integrated force sensors, or foot pressure sensors exist.** Verify actual hardware variant and sensor topics before real deployment.
- A real seat/couch does **not** report seat contact forces. Simulated force sensors may be used as privileged training labels and safety evaluation, *never* as mandatory runtime inputs. Proprioception-based contact inference is uncertain and must have failsafes.
- Surface geometry is the transferable abstraction; chair/couch labels are secondary. Cushion deformation/compliance and dynamic load capacity require separate modeling and real-world measurement.
- Aligning to furniture in the room is **not demonstrated yet**: prior success tests started near pre-aligned seating surfaces with small randomized initial poses. Start with shared 3D frames/relative pose tracking, then alignment, then closed-loop descent, then real-device sensor adapter and transfer.
- Do not treat apparent PPO reward or a scripted successful trajectory as proof of visual understanding or learned self-positioning.

---

## 3. Chronology of experiments and major verified/reported outcomes

Below are important numbers **as reported by Codex in screenshots and snippets**. They are not independently reprocessed here. Verify in existing JSON files and logs and retain exact definitions, scenario banks, matched seeds, contact/force thresholds and checkpoints. Different test-bank percentages are often not comparable.

### A. Infrastructure and initial sitting

1. Isolated Isaac Sim 6.1/Isaac Lab 3.0 environment installed on Ubuntu. G1 physics and PPO verified on RTX 4090; Python 3.12.15. Custom `sitting_task` package installed in editable mode. Project script wrapper created.
2. A first fixed-seat PPO run completed 100 iterations and produced `model_99.pt`, but **0/736 successful sits** in initial evaluation; G1 fell before reaching seat. Collision/body contact and pose issues were diagnosed. Initial video playback/rendering occasionally slow or required explicitly selecting `--viz kit`.
3. Verified a physically supported scripted seated pose for **~28 seconds of stable hold in a 30-second simulation**; seat supported ~76% of weight in that test, feet remainder (reported, not direct hardware measurement).
4. Short PPO curricula reached seated hold **64/64**, near-seat descent **64/64**, partial descent **63/64**. Full standing-to-sitting was initially failing because shin/lower-leg collider swept underneath seat.
5. Scripted reference + residual PPO managed complete full transitions after forward lean/hip-knee timing/stronger ankle tracking; fixed-seat reference tests **216/216** with zero falls/unintended contacts and minimum hold >10 sec. Scripted demonstration was not independent learned sitting.

### B. Geometry-conditioned scripted control

6. Controller succeeded within smaller/medium furniture ranges: small slabs **108/108**, medium slabs **108/108**, medium rigid families (chairs/benches/boxes/couch shapes) **144/144**, but wider perturbations **127/144** with unwanted contacts and high peak forces. Several failures involved hands/thumbs hitting couch or shin-seat geometry.
7. A stronger **geometry-conditioned scripted baseline** achieved **432/432** transitions, including **208 held-out scenarios**, zero reported falls/unintended contacts/force or torque violations within its test bank, peak seat force ~616 N with then-633 N comparison threshold, max descent speed ~0.189 m/s, minimum seated hold ~6.54 s. Preservation regression bank **177/177**. These are constrained, simulated, mostly aligned starts; this does not prove unseen physical surface adaptation.
8. Geometry-adaptive PPO residual experiments sometimes regressed. Earlier same-bank comparison: preserved reference **144/144**, geometry-scripted **95/144**, geometry+PPO **105/144**, hidden-geometry PPO **100/144**. Later 80 PPO updates failed to produce reliable residual improvement and caused force violations. The scripted controller was retained as robust teacher/baseline.

### C. Imitation learning / DAgger and sensor-based student

9. Teacher generated demonstrations (robot states/actions, reference trajectories, seat/furniture geometry, simulated contacts, outcomes); supervised behavior cloning and **DAgger (dataset aggregation)** trained a student predicting **full joint target actions** rather than mere residual on a hard-coded teacher action. A learned student reported **288/288**, later **432/432** in tested simulation banks (no teacher trajectory or IK in action path); but ablations revealed important recurrent-timing and shared-balance dependencies. The frozen DAgger student is a useful comparison, not proof of hardware-ready autonomy.
10. Built a simulated depth/point-cloud surface estimator and deployment-oriented actor observation interface. Early prototype reportedly ~**109 numeric channels**, but that is historical and **may not be the current active input dimension**. Feeds derived geometry, joint encoder positions/velocities, estimated torques, IMU, previous actions, sit command and temporal context. **Not a raw-RGB end-to-end vision-to-motor network.** It is essential to inspect the actual code for exact vector layout, input normalization, recurrent architecture, and indirect ground-truth leakage.
11. A depth-based seat/floor estimation pipeline reported valid surface estimates in **96/96** simulated starts, including some robot occlusion, but unverified against real RGB-D/LiDAR or hardware mounting/calibration. Simulated contact forces/COM/root velocity were removed from deployment actor inputs; retained for privileged label generation and evaluation. Foot load sensor was not assumed.
12. First strict deployment student: **86/144** (59.7%) on fresh furniture, with **58 force-limit failures**. Additional independent corrected-student tests were **15/36** clean and **21/36** noisy, zero falls/unintended contacts but impacts ~1.4 kN. Delay/recurrent dependencies remained.

### D. Soft landing — current preserved baseline; now paused

13. Smoothing especially abrupt **ankle joint-target changes** substantially improved successful/safe sitting versus previous student on matched banks:

   | Test | Old student | Soft-landing baseline | Reported max force after |
   |---|---:|---:|---:|
   | Clean | 15/36 | **33/36** | ~773 N |
   | Noisy | 21/36 | **32/36** | ~1224 N |
   | New furniture clean | 37/48 | **44/48** | ~1168 N |
   | New furniture noisy | 36/48 | **46/48** | ~715 N |
   | **Total** | **109/168** | **155/168** | Existing comparison ceiling 633 N |

   No falls/unintended contacts reported on those matched tests. Nonetheless, some contact peaks remained above the 633 N simulated benchmark threshold.
14. **All 14 baseline force-limit violations occurred at first seat contact**; the primary issue was impact before a reactive contact detector could respond. A new learned contact classifier reported ~99.41% precision / 99.92% recall against simulator labels, but **did not consistently improve the overall soft-landing controller**, with test-bank regressions. Other candidate braking/damping strategies lowered medians but worsened upper-tail force or balance. Delayed sensor/motor commands exposed fragility, including very large simulated force spikes under stress tests. No new candidate was promoted; the **155/168 baseline and all checkpoints remain preserved**.
15. Latest final report (screenshot 2026-10-09 ~23:04 PDT) said: preserved **155/168**, all 14 violations at first contact, new estimator improved classification but not safety, worse extreme tails in some cases, and no new controller promoted. Roughly **33–34 GiB** free on root and a larger amount preserved in project files. User explicitly decided to **STOP these iterative soft-landing/force-limit experiments** and refocus on shared 3D pose estimation / visual servoing.

### Limitations to retain when reporting results

- A 100% success rate in a bounded Isaac Sim geometry bank **does not establish** robust perception from hardware cameras, unseen real-world furniture, autonomous approach or safe real-hardware interaction.
- A scripted teacher, reference-only controller and DAgger student are separate systems. Keep identity/ablation labels accurate.
- Force/torque statistics are simulated metrics; the 633 N limit is a **project regression threshold**, not verified safe hardware rating. Ground-truth contacts are **not** a sensor on a real chair.
- Correlation of some geometry-input ablations with better PPO scores was modest; don't imply learned geometric understanding without controlled ablations.
- The latest student is not yet physically deployed and must not be represented as hardware-ready.

---

## 4. Architecture as currently understood; inspect and verify actual code

### Perception

- Isaac Sim RGB-D/depth sensor/point cloud pipeline feeding an estimated seating surface. Past screenshot referenced `scripts/seat_depth_estimator.py` and `scripts/capture_deployment_geometry.py`, as well as `datasets/deploy_rgbd_confirmation_05` (verify paths).
- Historical prototype estimated seat/floor height, support polygon, surface centroid, bounding box/extent, and may have an assumed camera pose. Camera extrinsics, body occlusion and real-hardware field of view are unvalidated. Current user-requested task is a **rigorous common-frame robot/seat representation**.
- Do not feed direct `scene['seat'].data.root_pos_w` or `robot.data.root_pos_w` to runtime; those are allowed for error metrics and privileged labels only. Be especially cautious about preprocessing that silently uses global simulator ground truth or reset geometries.

### Low-level motor observations and actions

- Actor apparently gets encoder positions/velocities, estimated motor torques, IMU, previous-action values, sit command, derived geometric features, and temporal/recurrent state. Verify *exact* current input fields, dimensions, frames, normalization and whether a history window or GRU/LSTM/other recurrent module is active.
- Student predicts full joint target commands. A separate command conditioner/rate limiter and motor control loop implement joint-level constraints, currently particularly smoothing ankle commands.
- Separate privileged ground-truth physics for safety evaluation: seat normal force, contact classification, precise body COM/root, body/world positions, impacts, friction and collisions.
- Timers/balance primitives were removed from later deployment student but recurrent timing still apparently influences performance. Audit for hidden leaks and hard-coded phase progression.

### Training methods already attempted

- RL PPO using Isaac Lab/RSL-RL for initial velocity task, custom fixed-seat and partial-descent curricula, and later residual policy refinements. Most PPO residuals didn't consistently outperform scripted reference on furniture.
- Behavior cloning + DAgger: teacher provides joint targets for states reached by student. Later DAgger student produces full joint targets from observations and history, with no reference teacher in direct action path.
- Supervised contact-state/seat-surface estimators trained or validated against synthetic ground-truth labels. Engineering control changes (e.g. ankle command slew-rate limits) provided major improvements that should not be falsely attributed to neural weight learning.
- Cosmos not yet introduced. Do not begin Cosmos until shared spatial perception and action alignment are validated.

---

## 5. Project resources, evidence and suggested discovery commands

**Ubuntu project root**: `/home/kadir/humanoid-sitting`

Known paths from screenshots (presence must be verified):

- `README.md` — project status and commands.
- `docs/commands.md`, `docs/geometry-conditioned-progress.md`, `docs/deployment-observation-audit.md`, `docs/deployment-progress.md`, `docs/soft-landing-progress.md`, `docs/soft-landing-safety-investigation.md` — some may have different actual names; use `find`.
- `sitting_task/src/humanoid_sitting/tasks/env.py` (historical), `.../tasks/furniture.py`, `.../tasks/geometry_feedback.py`, `.../deployment.py`, `.../soft_landing.py` — inspect actual layout.
- `scripts/run.sh`, `scripts/setup.sh`, evaluation/capture/training scripts; do not rerun setup unless missing environment.
- `baselines/geometry-conditioned-rigid-v1/` (historical screenshot), preserved teacher/student checkpoints; do not overwrite.
- `IsaacLab/logs/rsl_rl/...` — trained PPO artifacts.
- `runs/geometry_reference_progress_01/transitions.mp4` — scripted geometry-adaptive comparison.
- `runs/full_transition_comparison_02/comparison.mp4` — early reference/seat-hold comparison.
- `runs/furniture_progress_01/families.mp4` — furniture family comparison.
- `runs/soft_landing_01/paired_transition.mp4` — paired soft-landing example (~973 N vs ~512 N, not representative of all).
- `runs/soft_landing_02/` — final soft-landing safety findings, metrics, failing examples, reports, hashes.
- `runs/deployment_interface_01/`, `runs/deployment_status_fresh_05/` and `datasets/deploy_rgbd_confirmation_05/` — sensor-based student/camera work; inspect existence.

First use *read-only* commands from Mac Codex:

```bash
ssh kfuUbuntu 'hostname; nvidia-smi; df -h / /data; free -h'
ssh kfuUbuntu 'cd ~/humanoid-sitting && ls -la && ls docs scripts sitting_task runs datasets 2>/dev/null'
ssh kfuUbuntu 'cd ~/humanoid-sitting && find docs -maxdepth 2 -type f | sort | tail -80'
ssh kfuUbuntu 'cd ~/humanoid-sitting && sed -n "1,180p" README.md'
ssh kfuUbuntu 'ps -eo pid,stat,etime,cmd | grep -E "isaac|python|codex|train|tmux" | grep -v grep | head -60'
```

Use existing logs/checkpoint manifests to determine real current baseline. Verify file hashes when provided. Record relevant source revisions and avoid concurrent writes by multiple agents.

---

## 6. Handoff task to execute first on Mac Codex

**Stage 0: Confirm connectivity and protect existing work (read-only).**

- Verify `ssh kfuUbuntu` works without password and execute `nvidia-smi`, `df -h`, inventory project files/processes, check status docs; avoid modifying existing environments, baselines, checkpoints or Ubuntu drivers.
- Determine whether old Codex/Sim jobs are still active before starting new commands. If user already closed Ubuntu Codex, no need to restart it.
- Summarize actual versions, paths, current git status, disk budget and canonical best baseline.

**Stage 1: Audit frames, sensing and geometry pipeline (small, bounded).**

- Inventory precise G1 simulated sensor mounts/extrinsics, available depth image or point cloud, frame IDs, coordinate conventions, floor segmentation logic, actor observations, pelvis/hip frame FK chain, and body-relative seat transforms. Identify use of Isaac Sim ground-truth in any predictor/runtime path; restrict to tests/supervision.
- Define a consistent SE(3) transform tree from RGB-D/LiDAR `camera` → robot base → gravity-aligned local odom → seat frame; compute pelvis frame from robot joint states/kinematics.
- Provide explicit contract for frame direction, timestamps, covariance/uncertainty and transform composition. Use unit tests with synthetic known transforms, yaw rotations, robot base motion and pose resets.

**Stage 2: Implement shared 3D representation and visualize (NO RL training).**

- Obtain actual depth data from simulated onboard-mounted sensor; optionally integrate consecutive scans into robot-centered/gravity-aligned local map using odometry and calibration.
- Segment floor and accessible roughly horizontal seat support regions; compute plane, support polygon, front edge, seat normal, height above floor, uncertainty. Distinguish visually indistinguishable narrow unsafe supports, cushion compliance unknown, furniture backs/armrests.
- Convert robot body/pelvis and seat estimates into same frame and calculate signed clearance from appropriate hip/pelvis contact proxy to the seat top, lateral/fore-aft pose error, yaw error, shin/foot and back/arm clearance estimates.
- Demonstrate relative pose tracking as robot moves and seat becomes partly/fully occluded, using temporal mapping where possible. Compare sensor-only estimates to simulator ground truth *solely offline for quantitative error*.
- Generate a short visualization/video plus numeric report on several held-out furniture models. Use bounded evaluation/time/caches and avoid large datasets.

**Stage 3: Only after Stage 2 validated — closed-loop alignment.**

- Define simple visual/geometric servo control toward a feasible pre-sit pose using the estimated relative transform. Incorporate latency/uncertainty and controlled stop/abort criteria.
- Do not initiate a new sitting-policy/soft-landing PPO retraining loop until this is reliably tracking in the same frame.

**Exit and reporting discipline:** after each bounded milestone, write concise markdown results with commands, source change summary, actual measured errors and videos. Preserve prior baselines. If system is blocked, report exact error and propose limited fix, not silent sweeping changes or expensive parameter sweeps.

---

## 7. Suggested instruction to the newly started Codex agent on Mac

> Read `HUMANOID_SITTING_CODEX_HANDOFF.md` completely. You are running on the **Mac mini**, not on Ubuntu. Your execution target is the Ubuntu host `kfuUbuntu` via already-configured passwordless SSH, project `/home/kadir/humanoid-sitting`. Do **not** run a second Codex instance on Ubuntu; use SSH commands yourself and remote tmux for persistent GPU jobs. First verify SSH/GPU/disk and inspect the real project code, reports and checkpoints, reconciling any mismatch with this historical handoff. The user has STOPPED further seat-impact/PPO iteration. Implement the new priority: consistent **3D robot–seat registration** using simulated onboard depth/point-cloud observations, floor geometry, robot forward kinematics and odometry/IMU. Construct and visualize the seat and pelvis in the SAME coordinate frame and track relative position/orientation/clearance as the robot moves. Privileged simulator poses are allowed only as verification labels. Do not train new sitting policies until this capability is quantified. Preserve all existing checkpoints, baselines, environments, safety constraints and drivers. Work in bounded stages, monitor storage, and report measured results/videos and what to do next.

## 8. Notes on the intended tool architecture

- Starting **Codex locally on Mac** gives it direct access to Mac shell functions and (with appropriate permissions) to `ssh kfuUbuntu`, and from there command execution on Ubuntu. This **does not** make the original web ChatGPT conversation directly control Mac/Ubuntu; user wants to transfer working context and continue in Codex.
- No need for nested Codex on Ubuntu. It would add another approval/context boundary and risk conflicting edits. Reuse the old Ubuntu Codex's *code and artifacts*, not the old agent instance.
- To persist GPU jobs, use remote `tmux` sessions/daemonized logging. For a truly continuing multi-milestone agent, local Mac Codex itself must remain running (or be orchestrated in bounded scheduled invocations). Do not claim that SSH, tmux or a single Codex run automatically means 24/7 task continuation.
- **Latest user instruction is authoritative:** stop unnecessary sitting optimization loops; implement common-reference-frame spatial perception and relative body/seat visual servoing first.
