## 2026-10-17 — offline contact/command checks through93; bounded ankle-path95 running

Physical repair remains unresolved. current_retention_09 is still the best verified full-development controller:64/72 clean,70 stable,8 slipping,2 housing contacts,all60 original successes plus4 gains retained. Preserve all64. No physical episode, optimizer update, fresh holdout or policy promotion ran in65–94. The complete-attempt video remains current; no recorded motion was added.

CURRENT bounded OFFLINE worker: tmux low_seat_landing_ankle_95, launched2026-10-17T10:12:49Z. scripts/refine_low_seat_landing_ankle_95.py writes landing_ankle_path_95/results.json; log landing_ankle_95.log. On successful exit the same tmux runs audit_low_seat_landing_ankle_96.py into landing_ankle_path_audit_96, log landing_ankle_audit_96.log. Do not duplicate/restart. Exact scope/source in landing_ankle_revision_95.json. This diagnoses right ankle-pitch saturation during flattening/landing in91v2: add a smooth temporary0–15degree world-pitch shaping term during load shift/swing, released during approach, with EXACT unchanged terminal endpoint. Jointly optimize the3 existing path variables under their prior bounds. At most40unique63-knot coarse evaluations/10iterations plus one115-knot dense check. Every foot/support/clearance/joint/effort/command/rate/material/physical criterion remains unchanged; tilt<=38deg model/40deg physical. No blind gain/arm/rate sweep, no simulator/PPO/holdout. Initial coarse10deg shaping reduced foot error but still failed geometry/commands/time; wait for final and independent audit before conclusions. No automatic physical launch even after an audit or model pass.

Completed65–81: spatial/coupled stepping improved sampled geometry, but static ankle-command limits and timing blocked admission. Endpoint72 at1mm pelvis gap passes static commands with3.443mm clearance. Wholepath75/77 pass sampled geometry and static command bounds but require14.29984/13.72568s for their specific command polylines at0.15rad/s, failing13.5s allowance. Path80 increased landing crouch bound, but its dense105knots fail one foot-orientation check and timing13.62270s. Audits76/79/81 pass arithmetic, NOT those model gates. Do not round small failures into a pass. All originals/recoveries preserved.

Force allocation82 holds path77 geometry, total foot-load schedule and endpoints fixed, minimizing its command arc. Tightened HiGHS numerical solve in v2 satisfies UNCHANGED residual checks and gives13.48205s. This only optimizes ASSUMED foot-corner forces. Independent83 replays all105 nodes by finite-difference virtual work. Critically, the ACTUAL inherited clamp/filter order (desired joint margin.03,alpha.1,incrementcap.003/20ms) rejects its inverse-filter plan:47 target-clipped ticks,largest command mismatch.010353rad,end.003977rad in CPUfloat32 command-only replay. No GPU/physics replay or execution. No filter/gain/rate change. Original82 assertion/source/log preserved in force_schedule_recovery_82; v2 tightened solver accuracy to1e-10, did NOT enlarge check tolerances.

Contact84 adds zero-gap approach followed by explicitly assumed0.1BW normal seat loading; positive-gap nodes carry ZERO seat force.31knots fail3mm housing clearance,minimum2.358963mm. Independent85 replays34 records, force balance and work. Stance86 optimizes a rearward lifted-foot endpoint/yaw and hits120mm bound at2.954543mm, still FAILED. Independent87 passes arithmetic. Bounded89 extends only that internal step-distance bound to140mm; endpoint best3.731076mm clearance,command margin.030456rad,terminal tilt within unchanged38/40deg. Independent90 checks3 endpoint records. This is zero-gap static equilibrium under zero/assumed.1BW normal seat load, NOT sensed contact or an executable step. No foot was dragged/teleported; no endpoint-only physical launch is admitted.

Full path91v2 to endpoint89 is COMPLETE and REJECTED:115knots (17shift,48swing,40approach,10assumed load),24failed geometry knots57–80,foot position5.489656mm/orientation2.370861deg,command margin.029375rad below.03,minimum housing clearance3.731079mm. Specific command-polyline time15.45344s atunchanged.15rad/s fails13.5s allowance. Positive-gap modeled seat force remainszero. Independent contact_path_audit_v2_92 passed all115 geometry/virtual-work/force/bounded-target/fixed-command calculations while keeping geometry/command/time gatesFALSE. This is NOT physical admission.

Original91 had a genuine OFFLINE planner bug: it reused a mutated swing-target rotation as the next IK base, accumulating orientation across COM solves. Independent92 correctly FAILED. contact_path_recovery_91 preserves original source/results/logs/audit failure. v2 anchors each solve to the immutable prescribed yaw plus the unchanged8deg airborne tilt bound; one corrected115-knot check, no optimization/physics rerun. Original91 and92 remain failed. New93 diagnosis finds RIGHT ANKLE PITCH at its existing-.649rad planning reserve as the foot returns flat; left foot FK stays exact through that segment. This motivates bounded95, not a rate/physical-limit relaxation. Endpoint joint roll reserves and modeled loads are also reported; limited failed paths are not impossibility proofs.

No new effort result. Verified09 equals frozen61111 on common complete clean pairs; versusselected62111@8 on20old-development common-clean cases,whole20torque²+4.38%,final5+25.45%. Do not imply savings from model commands or failed fragments. Future comparisons still require all37joints whole20/final10/final5 torque²,absolute,time-mean instantaneousRMS,leg/upper and strict subsets only on common complete clean trials.

Local progress_report_88 copied/hashverified14original entries+2supplemental review hashes; comparison/filter plots and HTML/table inspected. progress_report_93 copied/hashverified8entries; full_path.png and findings/table inspected.93 includes all-knot active-limit diagnosis and preserved planner failure. Latest URL http://127.0.0.1:8807/progress_report_93/findings.html. Local independent83/90/92 and endpoint89 manifests verified. These reports are interim DEVELOPMENT/OFFLINE evidence, not physical repair/PPO/unseen/final completion. Use95/96 statuses when they arrive; reports93/88 do not claim to include that pending result.

All24,242 prior-study files and2,789 protected earlier evidence files rechecked intact in prior_recheck_during_contact_path_91.json. completed_probes_preservation_94.json protects completed65–93 additions and original failure/recovery files, excludes active95 output/log. Running95/96 sources and revision are separately frozen. Mutable root/docs status updated; earlier studies unchanged.

User-requested complete video is already delivered and ready: /Users/kadir/humanoid-sitting-control/validated_low_seat_contact_01/all_attempts_video_v2_59/all_attempts.mp4,58.4s H264,all32 sources,696complete20s trials(405success/291failure),72recorded diagnostic prefixes,72aborted case attempts without saved motion explicitly disclosed. all_attempts_video_audit_60 verifies complete inventory,462first-event samples and hashes. Offline schematic, recordedXYZW/bodyCOM transforms;0.1s poses bracket.02s events. Counts are historical repeated/rejected attempts,not best-controller accuracy. No new motion65–94, so preserve this video; update full coverage after any future physical/PPO trial.

Physical/PPO prerequisite unchanged: remove all8 remaining failures and retain all64 current clean conditions in matched full20s tests across72development trials BEFORE smallPPO63111/63117. Fresh24 holdout remainsclosed through repair,normalization,training,selection. No materials,slip,forbidden-body ORillegal-support union,terminal40deg/legal-seat-load>=.05BW,stable-support orphysical criteria may be relaxed. Runtimeonlycurrent sensors/reference/staticmodel; truthreset/reward/evaluationonly. No automatic promotion. Reference-assisted rigid-slab simulation is not unassisted discovery,realrobot validation or sitting/leaning anywhere; leaning has not broadened. Taskunfinished; automationACTIVE. Pause onlyafter actualrepair,newPPOcomparison andupdatedcomplete subminute video delivered.


Earlier preserved handoffs follow.

Latest completed physical probe — 2026-10-17T06:21:21.383303+00:00

The sixteenth candidate tracking_headroom_31 is COMPLETE and REJECTED:16/24 stable,14clean,5slipping,8forbidden housing contacts/physical failures,17conservative warnings,zero illegal-patch cases. No clean case gained or lost versus bestnormal3s on this24-case isolating bank. Six housing failures remain added relative to bestnormal3s; all eight targeted failure rows remain unresolved. No full72case retention, PPO, fresh holdout or promotion followed. No workers remain.

Revision31 explicitly changed only applied command-rate cap on six measured-selected joint17 approach rows during8<=t<15:0.20rad/s instead of0.15, then original0.15 from15s. Nominal path,filter0.1,feedback,parent,materials and physical criteria unchanged. It was a PRE-TRAINING control revision, not an execution fix. Runtime low_seat_tracking_headroom_31.py imports the unchanged17 planner. Every20ms command recorded:1000steps,2100selected env-ticks,maxstep0.004000008rad,float32 bound passed. Frozen417input/action/reward/parent/planner/startfirst8/fingers/zerooptimizer audits passed. Full20s unselected exactness remainsFALSE; maxbodycomponentdifference0.620mm,q0.000962rad. Old17 failed exact/physical gates stayfalse. Do not call the new overall gate passed.

Tracking improved but did not repair clearance: sampled hip reference-command gaps near targeted contacts fell from0.052–0.063 toabout0.017–0.020rad,estimated inward error to2–4mm. Contacts movedlater to14.92–15.04s but allsix stillfailed. Sensor-based unloading waszero beforeoldjoint17contacts,contradicting prematureunloading as their explanation. Two new36/+15episodes hadvalid=false,geometryage0.8s beforecontact; otherfailedrowsfresh. Do not claimallfailureshadcurrentgeometry. No solecauseproof; more rate alone is not supported as a repair. Next planning must address model clearance, dynamic support/tracking and loss of measured geometry, not repeat blind gain/lean/arm sweeps.

Auditorv1 incorrectly searched thumb/index/middle names rather than the model's numbered14finger names. Original source/log/status preserved in tracking_headroom_audit_recovery_31. v2 retains the same bit-exact14finger and command/input limits and passes software checks; tracking_headroom_audit_recovery_status_31.json supersedes original historical status failure. No physics rerun. Reportv1 incorrectly assumedfreshgeometrybeforeeveryfailure; report_recovery_32 preserves source/log/partialreport. v2 records missing values/lastfresh time withouttruth/replayed-sensor substitution. No criteria changed.

Offline28: two earlierstarts1/4s,20mmhover,0.5shold,normaldescent,0.5ssettle:all8testedpaths metgeometryscreens butfailedfinishbefore15s under0.15rad/s. Offline29early1sanchorendpointsearch bestclearances7.08/3.70/6.95/5.66mm;onlyfirstreached7mmresearchreserve. Neitherboundedmodelsearchprovesimpossibility or physicalsafety. command_tracking_diagnosis_30 coversall48priorrecordedepisodes. No simulation or PPO inthese diagnostics.

Best verified full72 controller remains current_retention_09:64clean,70stable,8slip,2housing;all60originalsuccesses+4gainsretained. Preserveall64.50margin-warningcases,22strictclean. Remaining36/+15,+20;37/+16.25;36.5/+17.5,twiceeach. Thedeepest36/+20remainsunrepaired. New31common14cleanpairs vsnormal3s whole20torque²+0.00194%,final5+0.01018%;tinyincreases,noeffortsaving. Earlier09vsselected62111@8 common20olddev:whole20+4.38%,final5+25.45%. No failed-fragment savings or newPPOgains.

progress_report_v2_32 copied locally;16 original SHA entries and supplemental original/review hashes verified;comparison.png,tracking.png,findings.html/review.html visually inspected includingefforttable. URLhttp://127.0.0.1:8807/progress_report_v2_32/review.html. All24,242priorstudyfiles and2048prior-evidencefiles unchanged; completed_probes_preservation_32 now2186files. Newprobe24000active+600initialsettling,zerooptimizer. Sources/reservedbanksverified. Mutablehandoffsupdated;automationACTIVE,taskunfinished.

PPO63111/63117 must wait untiltargetedfailuresremovedandall64successfulconditionsretainedinmatchedfull20s72developmenttests. Fresh24reservedholdoutremainsCLOSED throughrepair,normalization,trainingandselection. No promotion,thresholdrelaxation or mainbudgetextension. Reference-assistedsimulation/rigidslabsandtwoseedsdonotestablishsittingorleaninganywhere;leaninghasnotbroadened.

Earlier entries follow.

Latest diagnostic update — 2026-10-17T05:51:05.716534+00:00

Physical repair remains unresolved. Best verified current_retention_09 still retains all60 original successes plus4 added successes:64/72 clean,70stable,8slipping,2housing-contact cases. All64 remain the prerequisite retention set. The fifteenth candidate joint_path_17 remains REJECTED,16/24stable/14clean/6slip/8housing; its exact-preservation and physical gates remain false. No new full20s repair trial, PPO, fresh holdout or policy promotion occurred in this diagnostic round. No physical/model worker remains.

A meaningful measurement diagnosis is now complete: sensor_prefix_v4_24 captured two same-GPU8.4s prefixes. BOTH exactly replayed their original full trace prefix arrays (time,body,q,command,geometry,reference,torque). Until8.0s unselected depths were identical. At8.2s unselected q/gravity/body/camera pose/intrinsics stayed EXACT, but rendered depth differed by up to3.814697µm and plane features by1.490116e-6. At8.4s depth validity edges also differed; maximum finite depth difference remained3.814697µm. Independent18-case plane refits EXACTLY reproduced both recorded geometries. This locates the first divergence in rendered measurements, before unselected control or motion changes; the internal renderer mechanism is not established. Do not relabel the original failed full20s preservation gate as passed or substitute truth/replayed sensors. Audit: sensor_prefix_v4_24/independent_audit.json. No tolerances relaxed.

Static pose-render attempts22 are INCONCLUSIVE because attached-camera transforms/support fits did not match written poses. Preserve all four sources/logs/results. Short physical prefix recovery24 preserved: wrong assertion expecting420 simulator steps rather than1680×5ms; v3 capture-write indentation failed atstep1; an audit was attempted prematurely before changed inputs finished. Completedv4 pair20160active+1200settling, zero complete20s episodes/optimizer/holdout. Failed diagnostic attempts add at most20280active+1800settling; overall upper bound40440active+3000settling. These are separate diagnostic budgets, not main training or repaired trials. diagnostic_recovery_27.json records everything.

Offline sensed_clearance_23 shows measured plane+encoders warn below5mm before all2 best-normal and8 rejected-path housing contacts, but only0.08–0.20s before, with6/14 clean cases also warning. Near-event model-to-label errors roughly2mm.5mm is a diagnostic line, not a physical threshold or certified safety filter. All48 recorded episodes are included;0.1s poses only bracket0.02s events.

Offline bounded screen25 uses ACTUAL saved7s IMU/encoder/plane inputs and exact frozen bridge inertial model. With fixed feet, previous fixed−0.35rad arm offset, pitch<=0.9rad,5mm model CoP/15mm patch limits, three local searches pergeometry found best tested housing clearances6.85/3.46/6.70/5.41mm for36/+15,36/+20,37/+16.25,36.5/+17.5. None reached the7mm research reserve motivated by recorded tracking error. This is not a proof of impossibility or a changed physical pass criterion.

Offline screen26 minimizes proposed symmetric sagittal foot displacement within±60mm under the same model support constraints and7mm reserve. It found hypothetical rearward18.0/22.5mm endpoints for36/+15 and37/+16.25 (7.24mm clearance), but none in the bounded searches for36/+20 or36.5/+17.5. These endpoints have NO stepping/whole-path/rate/physical certificate and MUST NOT be executed as dragged feet or teleported endpoints. No new physical candidate was launched. Next work should address the limited support/clearance envelope and dynamically feasible approach/contact planning, with explicit versioned scope and measured tracking/clearance feedback. Do not repeat blind lean/gain/arm sweeps, or presume the largest nominal endpoint clearance removes slipping.

progress_report_27 is copied locally to validated_low_seat_contact_01/progress_report_27;12 original SHA entries verified. clearance_diagnosis.png,depth_comparison.png and findings.html visually inspected including model table. URL http://127.0.0.1:8807/progress_report_27/findings.html. It is interim DEVELOPMENT diagnosis, not PPO/unseen/final results. Previous effort remains mixed; against selected62111@8 on20 common clean old-development trials, verified09 whole20 torque²+4.38%,last5+25.45%; no new complete-episode effort claim.

All24,242 initial prior-study hashes and1,810 completed prior-evidence hashes rechecked intact in prior_recheck_after_diagnostics_27.json. completed_probes_preservation_27.json protects2048 files. Fresh24 holdout remains CLOSED. Preserve all64 successes and remove all8 target slip/two housing cases in matched full20s72development tests BEFORE small PPO63111/63117. Keep the documented8update/seed uninterrupted budget and full independent audits only after physical admission. Automation stays ACTIVE; do not pause or claim completion. Sitting/leaning anywhere and real-robot capability remain unestablished.

Earlier entries follow.

Latest verified development result — 2026-10-17T05:18:26.837167+00:00

The fifteenth physical candidate, joint_path_17, is COMPLETE and REJECTED: 16/24 stable, 14 clean, 6 slipping and 8 forbidden hip-housing-contact/physical-failure cases. It added six housing failures, gained no clean cases and retained all 14 best-probe clean outcomes. Lower slip counts in newly failed contact episodes are not successes. The best verified full-development controller remains guarded retention09: all 60 original successes retained, 64/72 clean, 70 stable, 8 slipping and 2 housing-contact cases. Preserve all 64 established clean conditions. No PPO, fresh holdout or promotion; no physical workers remain.

Revision17 explicitly began the selected six approaches at8s and ended15s with a nominal rate-feasible joint path; the deepest36cm/+20deg kept previous routing. The independent v4 audit completed but overall passed=false: actual417 inputs/actions, frozen parent, rewards, zero optimizer, 24000 active/600 initial-settling ticks, exact starts/first8s, planner replay and bounds passed, while unselected full20s exact preservation FAILED. Measured geometry first diverged8.2s, before unselected command/body changes after10s; cause remains unresolved. Maximum unselected body-pose component difference2.08mm and q0.00265rad. No tolerance or criterion was relaxed. Original status_joint_path_17.json records the first offline-audit termination; joint_path_audit_recovery_status_17.json explains the completed rejected result.

Audit recovery preserved both erroneous reconstruction attempts: use originalfloat32 runtime inputs and exact frozen bridge inertial model, not newly recorded COM metadata. These corrections passed the unchanged2e-7 planner tolerance; they did not repair the real preservation or physical failures. No physics was rerun. Recorded mesh diagnosis shows new housing contacts14.60–14.70s; selected roots were12–15mm inward and3.4–4.6mm below plan around14.6s. Fresh geometry was available. Nominal clearance was not a dynamic certificate; 0.1s pose samples bracket0.02s events. Diagnose tracking/sensed clearance and unselected measurement isolation before another versioned bounded repair, without simulator-truth/replayed-sensor substitution.

Offline screens15/16/18/19/20 did not run additional physics or learning. The deepest-seat endpoint/whole-path optimization still fails the model clearance/rate combination; even velocity matching requires0.150119rad/s>0.15. Do not round it into admission. All original8 revision17 source hashes and all1659 protected prior-probe files rechecked intact. Initial24,242 prior-study hashes passed in prior_recheck_during_joint_path_17.json. completed_probes_preservation_21.json protects1810 completed evidence files.

progress_report_21 copied to validated_low_seat_contact_01: all18 original SHA entries and4 supplemental review hashes verified, comparison/housing plots and review.html visually inspected. URL http://127.0.0.1:8807/progress_report_21/review.html. This is interim development/controller evidence, NOT new PPO/unseen results. Paired whole20/final10/final5 squared/absolute/RMS/leg/upper effort and strict subsets use common complete clean episodes only. The rejected candidate's tiny paired differences are not an overall gain. Best retention09 still has mixed effort: vs selected62111@8 old development, whole20 torque²+4.38%,final5+25.45% on20 common clean cases. Automation remains ACTIVE; physical repair and requested PPO comparison are unfinished.

Earlier entries follow.

Latest closure — 2026-10-17 04:36 UTC

The counterbalance probe is COMPLETE and REJECTED as a solution:22/24 stable,14clean,10slip,2housing contacts, no added physical case/no lost best-normal clean case, but none of eight targeted failures fixed. Full actual417 input/parent/reward/budget/bound audit passed with zero actor/critic optimizer steps; these are software checks, not physical success. counterbalance_diagnosis_13 shows only1.5–3.3mm less backward root error at12s, still26.65mm error and housing contact11.98s on the deepest case. Conditional retention14 recorded not_admitted and launched nothing. No physical job, PPO or fresh-holdout evaluation is running from this round.

Verified guarded retention09 remains60/60 original successes retained,64/72 clean,8slip/2housing. All24,242 prior files rechecked unchanged; completed_probes_preservation_14 protects1,659 completed files. Preserve all64 established successes and remove the remaining eight before PPO. progress_report_14 (31 original SHA entries), supplementary review and comparison plot are locally verified/visually checked at localhost8807/progress_report_14/review.html. It covers all14 physical probes, not new PPO/unseen results. Effort tradeoff is explicit: versus selected62111 on20 common clean old-development trials, whole20 torque² +4.38%, final5 +25.45%; common-clean effort matches current61111. counterbalance_export_13 contains the new complete audit/result bundle. Automation remains ACTIVE with updated current instructions; no completion or promotion claim.

Earlier entries follow.

Latest verified retention — 2026-10-17 04:32 UTC

current_retention_09 completed at04:27:12. Its guarded best3s landing plus frozen early61111/resting59117 retains ALL60 original union successes and is clean in64/72 development trials. known_old:22stable/20clean/4slip/2housing contacts; known_transfer:24stable/22clean/2slip/0contacts; known_recent:24stable/22clean/2slip/0contacts. Four previously unresolved trials improved:37cm/+11.25 and36.5cm/+12.5, two placements each. Eight slips remain:36cm/+15,+20;37cm/+16.25;36.5cm/+17.5. Both housing contacts remain36cm/+20. This is a partial physical repair, NOT PPO admission or full repair.

Independent current_policy_independent_audit_10.json passed frozen414-channel actor/checkpoint/action checks, exact physical/sensor startups and first10s, initial=final(no optimizer), finite arrays, sampled command slew and sensor/finger bounds. prior_recheck_after_retention_10.json rechecked all24,242 original prior hashes; completed_probes_preservation_10.json protects1,582 completed probe/retention files. retention_geometry_diagnosis_10.json confirms fresh measured geometry before every remaining failure; pose samples0.1s only bracket0.02s contact events. Margin warnings remain separate (50/72 warnings;22 strict clean trials).

progress_report_10 copied to validated_low_seat_contact_01, all28 original SHA entries verified. Its comparison.png is byte-identical to visually checked round08; retention.png and supplemental review.html were visually checked. review_audit.json separately hashes3 supplemental files. URL http://127.0.0.1:8807/progress_report_10/review.html. retention_details_10 contains432 by-geometry/controller comparisons, current geometry audit and offline screens; all6 SHA entries verified locally. These are development/controller/existing-policy results, not new PPO or unseen results. Common-clean effort exactly matches current61111. Against selected62111 at8s on20 common clean OLD-development trials, whole20 torque-squared is4.38% HIGHER and last5 is25.45% HIGHER. Report this tradeoff; do not claim an effort win from the retention result.

Offline screens rate_clearance_screen_10.json, coupled_path_screen_11.json and coupled_pose_screen_12.json tested geometry/rate feasibility, with NO physical simulation or learning. The sampled larger-lean paths did not jointly meet the chosen model clearance/rate/foot-support screens within the unchanged approach and final5s support requirement. Some slower schedules also finish too late; they were not simulated. These limited model searches do NOT prove physical impossibility. The3mm model clearance target is an offline research screen, not a changed physical failure threshold.

CURRENT physical job low_seat_counterbalance_13 launched04:26:45 onGPU1. Variant normal_counterbalance_13 tests a DIFFERENT hypothesis: symmetric shoulder/elbow pitch reference offsets -0.35rad over3.5s, only measuredheight<38cm and positive slope>14deg. Best guarded normal3s pelvis/leg pose path and all feedback gains are unchanged; leg static feedforward is recomputed for arm mass distribution. Actual first10s and unselected plans remain exact. These are four reference-arm changes, not learned residual actions;14 fingers remain fixed. Model screen counterbalance_screen_13.json passed24 cases (8 active), max arm target rate0.149802rad/s, predicted forward COM shift~9.33mm. Housing clearance is NOT enlarged, so this is a physical hypothesis, not a certificate. Each24 full20s trials,24000active+600initialsettling,zero optimizer/freshholdout. Frozen59117 remains11/15s. Original sameGPU1 baseline reused explicitly; current early pulse is omitted in this isolating probe. Runtime low_seat_counterbalance_controller.py/run_low_seat_counterbalance_probe.py; launcher launch_low_seat_counterbalance_13.py; read status_normal_counterbalance_13.json and counterbalance_probe_13.log. Do not duplicate/restart. Launcher runs outcomev2/sensor audits and independent validate_low_positive_feedback_v3 plus explicit arm-reference bounds/replay into counterbalance_independent_audit_13.json. Audit pass is not physical admission.

Conditional low_seat_counterbalance_retention_14 waits for completed, independently audited probe13. It only launches a new72-case current-policy comparison if all8 targeted probe rows are clean, all14 clean best-normal3s probe cases remain clean, and no physical failure remains. If not, it records not_admitted in current_retention_status_14.json and launches NO physics. If admitted, run_low_seat_current_counterbalance_probe.py combines frozen early61111 and resting59117; launch_low_seat_counterbalance_retention_14.py evaluates all72 known cases onGPU0,72000active+1800settling,zero optimizer/freshholdout. It checks retention of all64 now-validated successes, including the original60. No PPO entry point exists in this launcher. Full independent actor/source/geometry/report audits of14 are still required if it runs.

No new PPO or fresh holdout has started and no policy is promoted. The reserved24 fresh trials remain closed. All remaining targeted slip/contact failures must be removed, and all established successes retained, before preparing/freezing the small63111/63117 PPO comparison. Completed low_positive_transfer_01 and all prior studies remain read-only. Automation stays ACTIVE until the requested new comparison is actually delivered.


Historical entries follow.

Latest — 2026-10-17 04:06 UTC: the slower/moderate lean is COMPLETE and REJECTED:16/24stable,14clean,4slipping,8housing-contact/physical failures;6 added physical cases0/1/6/7/10/11. Do not use it for PPO. housing_geometry_diagnosis_08.json compares recorded body transforms to model meshes around first contacts; actual model clearance crosses below the plane and FK/body poses agree to micrometre scale. Pose samples bracket0.02s events at0.1s resolution, so no exact-event clearance certificate or sole-cause proof is claimed. Sensor geometry was available.

Original current_retention_07 is complete:56/60 existing successes retained,60/72clean. Besides the nominal10deg boundary regression(known_old2/3), nominal39cm/+11.25deg measured38.9498cm and crossed the<39cm selector, losing known_transfer8/9. These are physical regressions, not audit errors. All13 probes and original retention outputs are preserved in completed_probes_preservation_08.json (1162files).

CURRENT GPU0 low_seat_current_retention_08 (launch04:00:36) tests the>10.5deg measured-slope guard on all72cases. Firstbank now20/20union successes retained,20/24clean,22stable,4slip,2housingcontacts; the two nominal10deg losses are recovered. Remainingbanks running. low_seat_current_retention_09 is queued until08completes, then repeats72cases with BOTH>10.5deg and<38cm guards, retaining the separate40cm/+20deg branch. Config normal_landing3s_guarded_09.json; revision selector_revision_09.json; runner launch_low_seat_current_retention_v3.py. This targets the actually unresolved36/36.5/37cm seats and leaves established39cm cases unchanged. Each72case run72000active+1800initialsettling, zero optimizer/freshholdout. Read current_retention_status_08/09.json; do not duplicate/restart. The full60case retention gate is not yet passed by the revised controller.

low_seat_round_audit_08 waits original07,guarded08 and rejectedorientation08, then verifies frozen414-channel actor inputs/actions/checkpoints, first10s/starts, initial=final(nooptimizer), physical tick labels, sampled command bounds and sensor/finger audits. It exports progress_report_08. It does NOT audit future09 or launch PPO. After09 completes, independently run equivalent actor/physical/retention checks into a fresh output, and create an updated report without overwriting08. Remaining targeted failures must still be removed before PPO even if all60current successes are retained.

progress_snapshot_09 is copied locally, all25original SHA256 entries verified. comparison.png andretention.png visually inspected; formatting-only review.html/review_audit.json separately hash-verified and visually checked at http://127.0.0.1:8807/progress_snapshot_09/review.html. This snapshot includes all13physicalprobes and originalretention07, explicitly labels08/09pending, and is NOT a PPO/unseen result. Prior reports remain unchanged. No new PPO/freshholdout or promotion.

Latest — 2026-10-17 03:55 UTC: timing probes complete and independently audited. normal_landing2s_06=22stable/12clean/12slip/2housing contacts; normal_landing3s_06=22stable/14clean/10slip/2housing contacts on24 difficult probe cases. Both retain the same-GPU baseline8 clean cases and22 stable cases, with no added physical failures. Three-second landing gains37cm/+11.25,36.5cm/+12.5 and40cm/+20 (two placements each). This is a partial physical improvement, NOT full repair or PPO admission. All24,242 prior hashes rechecked intact; completed_probes_preservation_06.json protects885 files across the12 completed probes.

CURRENT: low_seat_current_retention_07 onGPU0 evaluates the best3s controller with frozen early61111(6–7s pulse) and parent59117(11/15s) across all72 now-development trials. First bank exposed two lost36cm/+10deg cases: measured10.05145deg crossed the old>10deg selector. Original results remain preserved. selector_revision_08.json changes ONLY this measured routing boundary to>10.5deg, not physical thresholds. low_seat_current_retention_08 waits for07 to finish, then independently repeats all72 cases with the guarded configuration. Each retention run72000active+1800initialsettling, no optimizer/freshholdout. Do not duplicate/restart. Both reuse hashed same-GPU current-policy baselines explicitly. Read current_retention_status_07/08.json and current_retention_07/08/retention.json.

ParallelGPU1 low_seat_slow_orientation_08 tests a separately documented slower/moderate posture change:0.55rad pitch over3s, then2s normal descent, only measuredheight<39cm/slope>14deg; other rows use guarded3s landing. First10s, parent, feedback and applied command slew0.15rad/s remain unchanged. One24case20s physical probe,24000active+600initialsettling,no optimizer. Runtime low_seat_slow_orientation_controller.py; wrapper/launcher run_low_seat_slow_orientation_probe.py and launch_low_seat_slow_orientation.py. The launcher runs fresh outcome/sensor audits automatically. Status status_normal_slow_orientation_08.json. The model path passes IK/first-command/finite checks, but target knee rates still reach0.173rad/s and housing clearance remains only0.94mm at36cm/+20deg; this is not a geometric or physical certificate. Recorded large-rotation tracking lag is consistent with command-rate limits, not a sole-cause proof.

Timing diagnosis now timing_diagnosis_v2_06.json, using exact>10N forbidden-contact events and per-case measured-plane errors/availability. The original v1 diagnostic mistakenly used any nonzero forbidden-body force to timestamp the first event; originals preserved in timing_diagnosis_recovery_01. Physical outcome summaries and thresholds were already correct. Offline orientation-screen v1 hit a missing optional model-margin field on an unselected preserved plan; source/failure preserved in slow_orientation_screen_recovery_01. v2 explicitly reports that unavailable field asnull, with no simulation or threshold change.

No PPO or fresh holdout has started. All60 existing successes must be retained AND targeted slipping/housing failures removed before the small PPO comparison. No policy promoted.

Latest update — 2026-10-17 03:35 UTC: coordinated-pose variants also rejected (15/12stable,9/8clean,14slip,4contacts each). CURRENT low_seat_timing_0/1 test normal_landing2s_06 / normal_landing3s_06,24full20s each; new timing-only revision of the best normal controller. Audit-only low_seat_timing_audit_06 waits; no PPO/holdout.757 rejected-probe files preserved. Canonical LOW_SEAT_CONTACT_STATUS.md and normal_timing_revision_06.json. Earlier snapshots historical.

Latest update — 2026-10-17 03:27 UTC: eight completed physical repair candidates remain rejected. Normal descent gained four clean cases but retained slips/housing contacts and lost stable support; balance-handoff changes worsened contacts. Two bounded coordinated-pelvis probes run in low_seat_task_0/1 (normal_task_mild_05, normal_task_firm_05). No PPO or fresh holdout. See rl_support_work/LOW_SEAT_CONTACT_STATUS.md and task_revision_05.json. All24,242 prior files rechecked intact;623 rejected-probe files preserved. All60 current successes remain a required gate. Earlier snapshots below are historical.

Latest update — 2026-10-17 02:50 UTC: four new physical candidates rejected; none fixed low-seat slips/housing contacts. Two normal-approach probes run in tmux low_seat_normal_0/1 (normal_descent_03 and hover_rotate_normal_03). No new PPO or fresh holdout. Canonical status: rl_support_work/LOW_SEAT_CONTACT_STATUS.md; remote low_seat_contact_01/status.json. Preserve all60 current successful conditions. Same-chat continuation automation ACTIVE. Earlier snapshots below are historical.

ACTIVE NEW WORK — low_seat_contact_01. The previous comparison below is complete; the new user request is ongoing.

# Low-seat contact repair status

New user authorization follows the completed low_positive_transfer_01 study. This is a separate experiment; previous results and policies remain unchanged.

Prepared low_seat_contact_01, verified 24,242 protected files, reserved 24 fresh holdout trials, and recorded all 60 previously successful conditions across the three now-development banks. Twelve cases remain unresolved across every current policy. No new training or fresh holdout evaluation has started.

Recorded diagnosis found about 0.6 mm predicted housing clearance on the 36 cm/+20 degree endpoint. Near the main low-seat failures, the pelvis is approximately 6–10 mm below and 7–21 mm farther inward than the planned root trajectory. Fresh geometry is available. These observations motivate damping and clearance experiments; they do not prove a sole cause.

The first two 24-case matched comparisons are complete and rejected. Both baselines had 22/24 stable, 8/24 clean, 16 slipping and two housing-contact cases on this deliberately difficult probe bank. Damping alone retained those outcomes without fixing the failures. The forward-clearance candidate lost 12 stable cases and added eight physical-failure cases (10/24 stable, 8/24 clean, 10 contacts/physical failures). Exact startups and first ten seconds passed independent audits. Bound/finger checks passed; estimator errors reached 3.5 mm in the damping probe and 11.8 mm in the failed clearance probe. All 24,242 prior hashes were independently rechecked intact.

Persistent tmux low_seat_load_0/1 now run load_feedback_original_02 and load_feedback_clearance_02, started 2026-10-17 02:37:42 UTC. These new candidates replace time-scheduled static unloading with bounded torque/encoder/IMU-estimated load transfer. The second also tests a less extreme clearance posture. They reuse the preserved same-GPU baseline recordings, explicitly documented in baseline_reuse.json; no repeated baseline simulation is claimed. Each candidate runs all 24 cases for 20 seconds, with no optimizer updates. Original controllers, failed trials, thresholds and materials are unchanged.

Next: independently audit candidate startup/prefix identity and each physical outcome. Test any viable repair with frozen current policies and the complete existing-success set before PPO. The initial probe uses the repaired resting parent to isolate the physical-controller change; its unchanged mild-slope failures are not evidence that the current early policy has regressed. A separate frozen-current-policy harness is prepared for this comparison.

No repair success, new effort gain or anywhere-sitting claim is established. No policy has been promoted.

Automation finish-resting-ppo-comparison is ACTIVE for this new work, every 30 minutes, quiet during ordinary progress. It must not reuse the completed study's old training instructions. PPO remains conditional on removing the targeted failures and retaining the 60 current successful conditions; the fresh holdout is closed.


Previous study history follows unchanged.

Delivery complete: verified comparison available in validated_low_positive_transfer_01/final_results/review.html; automation finish-resting-ppo-comparison is PAUSED. No policy promoted.

## Completed and independently reviewed — 2026-10-17 UTC

The low_positive_transfer_01 comparison is complete. Both seeds62111/62117 trained16updates in uninterrupted optimizer processes,448000active+10500indexedwarmup physics ticks each,917000total;1400initial settling ticks separate. All training used10–20s continuous bounded feedback. The417-input actor selected3 symmetric hip/knee/ankle residuals every0.5s, applied every0.1s with0.004rad absolute and0.004rad/s slew bounds. The reference and frozen59117 resting policy stayed active. This remains reference-assisted sitting, not unassisted motion discovery.

All three complete verifications (verified_01/02 and independent verified_03) passed and are identical.224frozen source/input hashes and21,135protected prior/preflight files intact. Exact main/smoke initialization, frozen parent, actual417inputs, CPU optimizer replay, complete reward/budget accounting, final/selected loads, physical/sensor startups, locked prefixes, fixed fingers, bounded commands, full20s horizons and holdout timing verified. Accepted actor Adam steps127/128; critic128each. No physics/reward/criteria/budget changes after freeze. All rejected manual repairs and audit recoveries remain preserved.

Fresh holdout24trials (12geometries repeatedtwice;36.5/38.5/40.5/42.5cm ×+7.5/+12.5/+17.5deg): every controller remained supported24/24 with no unwanted contacts or other physical failures, but sliding persists. Clean counts repaired reference/parent18/24; previous61111/61117each20/24; full finals62111=20/24,62117=18/24. Strict-margin-clean finals4/24and2/24. Lowest-seat slips at36.5cm/+12.5,+17.5remain in both placements. Final62111 resolves reference slips38.5cm/+7.5, which previous policies already resolved; final62117does not. Known36cm/+20housing contacts remain in development.

Paired common-clean torque-squared changes for fullfinal62111/62117 versus repairedreference:whole20s−0.69668%/−0.79896%,last10s−1.88285%/−2.15929%,last5s−4.76166%/−5.09194%. Against frozenparent they instead INCREASE whole20s+0.12857%/+0.02545%,last5s+0.69194%/+0.34274%. Against previous61111/61117, whole20s increases0.012–0.128% andlast5s0.285–0.656%. No incremental final-policy effort benefit established.

Development-selected policies were62111update16 and62117update8, both prospective8–20s evaluations; NO training at8s. Selected62111 resolves40cm/+20development slips but adds one42.5cm/+12.5holdout slip (19/24clean). Selected62117 has20/24clean holdout and tiny effort savings versus previous policies (~0.062–0.073%whole20s,0.613–0.648%last5s), but adds two39cm/+11.25development-transfer slips (18/24clean versus20baseline). Fullfinal62117also loses both44cm/+15development successes (16/24clean versus18baseline). No post-holdout reselection or policy promotion.

All2,448geometry observations per fresh-holdout final/selected were available; maxplane offset error2.256mm,normal~0.310deg. Failure/contact union, height/slope breakdowns, strict-policy subsets, squared/absolute/meaninstantaneousRMSNm andleg/upper contributions reviewed. Effort comparisons use only common full-duration clean pairs; no failed-fragment savings. Torque is an effort proxy,not electrical energy/global optimum;two seeds/repeated rigid surfaces are limited evidence.

Remote final_export_01 copied to Mac validated_low_positive_transfer_01/final_results. All338original files SHA256-verified; manifest SHA5509a8462688c03c4cf02719ddf752f1ddc3e0489df83116e18b44b36040283a. comparison.png and originalfindings.html visually inspected; additive review.html/review_audit.json clarify regressions and incremental gains without changing original results. Localreport http://127.0.0.1:8806/review.html. Independent03copy in validated_low_positive_transfer_01/independent_verified_03. Temporary approval-review usage-limit failure delayed copying; the same transfer succeeded on retry, no approval bypass. Report delivery closes this comparison; pause finish-resting-ppo-comparison. No extra training authorized or launched.

Earlier entries below are historical.

---

## All comparisons complete; final audits running — 2026-10-16 22:38 UTC

Both workers completed all training, checkpoint selection and matched evaluations on all three banks (old development, previous holdout now development transfer, and fresh holdout). Original training validators passed actual417 inputs, full rewards, bounds/slew, frozen parent and independent CPU optimizer replay. Both selected8s prospective evaluations passed their development gates:62111 update16 and62117 update8. All training was10–20s;8–20s was tested only after training. Finals and selections preceded the holdout barrier. No policy promoted.

Automatic low_positive_finalize is verifying twice into verified_01/02. Required independent verifier has been launched once in persistent low_positive_independent_03, output ppo_01/verified_03, log independent_verification_03.log. Do not launch another copy or overwrite that directory. Final report/export and complete verification are still pending.

Initial inspection shows mixed results, not a general solution: fresh holdout clean counts reference/parent18/24, previous61111/61117 each20/24, full finals62111=20/24 and62117=18/24; selected8s variants19/24 and20/24. All fresh-holdout trials remain stable with zero unwanted contacts/other physical failures; sliding remains. Final policies use slightly more torque-squared than current policies on common clean pairs despite less than the repaired reference. Selected62117 has small savings but loses two clean cases on development transfer. These are preliminary raw-result observations pending final audits; no deployment/promotion or overall success claim.

---

## Training complete; validation and comparison pending — 2026-10-16 20:38 UTC

Both seeds finished all16updates and exited training successfully. ppo_01/training_completion_snapshot_01.json independently confirms448000active+10500indexedwarmup=458500ticks/seed,917000total;8960decisions/seed,224000teacher+224000feedback-window ticks,700initial settling/seed excluded. Final weights are finite, changed and exactly equal model_16. Accepted actor Adam steps127/128 for62111/62117; critic128each. All training remained10–20s.

The original workers are now executing the complete action/reward/CPU optimizer validator before checkpoint selection. Their last status label still says training; this is stale orchestration labeling, not unfinished PPO updates. Both processes are active and consuming CPU; no failure was reported. Do not restart or add training. Full verification has not yet passed. No development selection or fresh holdout has begun at this snapshot. New-policy physical/effort conclusions remain pending.

---

## Main training launched and independently checked — 2026-10-16 19:10 UTC

Both main runs launched at18:58:59UTC after both recovered preflights and independent freeze passed. Persistent jobs: low_positive_ppo_62111 and low_positive_ppo_62117 on physicalGPU0/GPU1; low_positive_finalize waits for completion. Do not restart or add training. Main has completed2/16updates perseed at this snapshot. Both actor/critic weights are finite and changed. Frozen budget917000physics ticks total; all16updates train10–20s continuous bounded reference-assisted feedback. Any8s start remains prospective evaluation only.

Independent ppo_01/launch_validation_01.json PASSED:224frozen source/input hashes and21,135protected prior/preflight files intact; exact main/smoke fresh actor/critic/Adam/normalization initialization,updates0 andemptyAdam; frozen59117parent exact; same-GPU physical startup arrays exactly match calibration; actual input dimension417. Both preflight optimizer replays passed16actor/16criticsteps. Deterministic smoke evaluations retain26/28stable,18clean,10slip,2housing-contactcases with no lost baseline clean cases or added physical-failure cases. Seed62117 stochastic smoke had one additional sliding trial; it remains recorded. No physical success claim from software admission.

Latest stochastic main episode:62111=26stable/18clean/10slip/2physical;62117=26stable/17clean/11slip/2physical. These are exploration outcomes, not final deterministic or unseen comparisons. Slipping has not been demonstrated fixed and effort benefit is not established. Fresh holdout has not started; both finals/selections must precede it. Old preflight/audit failures and previous orchestration below are historical.

Read current ppo_01/status_62111.json/status_62117.json and seed_<seed>/train/progress.json, plus finalization_status.json. Final independent verifier remains scripts/verify_low_positive_study.py with validatorv3. Preserve frozen definitions and all original outputs. Automation stays active and quiet during normal training; final report/pause only after verified comparison delivery.

---

Current execution note: both seeds have completed their first real PPO smoke update (8 actor/8 critic Adam steps each); second updates continue. Main still gated. Use preflight_recovery_status_62111/62117.json, low_positive_preflight_v2_62111/62117 and low_positive_launch_v2; old GPU1 audit failure is preserved historical evidence. Validatorv3 checks actual recorded memory with unchanged tolerances. Repair diagnosis copied/SHA-verified/visually inspected at validated_low_positive_transfer_01/diagnosis_results.

## Active: continuous transfer PPO preflight — 2026-10-16

Read rl_support_work/LOW_POSITIVE_TRANSFER_STATUS.md and LOW_POSITIVE_TRANSFER_PROTOCOL.md. New data low_positive_transfer_01. Six manual variants failed to remove targeted slipping and are rejected; no repair success claim. New zero interface PASSED exact same-GPU global/warmup/indexed traces and414-channel parent actions after correcting a mismatched audit target. Both GPU preflights now calibrate/explore/two-update smoke/evaluate the417-input continuous-feedback PPO on28 trials including known failures; baselines26stable/18clean/10slip/2housing contacts.

low_positive_launch waits for both complete preflights and independent freeze, then starts62111/62117 with ONE uninterrupted16-update process each. All training10–20s,917000total physics ticks incl indexed warmup;1400initial settling excluded. Any8s start is prospective evaluation only, not8s training. Checkpoint selection must preserve baseline clean cases and physical failure set, with explicit parent fallback. Fresh24trial holdout closed through selection. Existing automation finish-resting-ppo-comparison ACTIVE for this new pipeline; main not claimed until launch_status confirms. No additional training/promotion. Earlier statuses below are historical.

---

## Active continuation — low positive-slope transfer, 2026-10-16

Read `rl_support_work/LOW_POSITIVE_TRANSFER_STATUS.md` and the corresponding remote docs. New study `/data/humanoid-sitting-perception/low_positive_transfer_01`; completed positive_slope_01 remains unchanged. Fresh24-trial holdout reserved and closed. Phase-separated and timing-only physical repairs have not fixed the targeted slips. Delayed predicted support is under physical test onGPU1; new repeated-feedback PPO zero/global/indexed regression runs onGPU0. No new training or holdout launched. Longer-control PPO software built and unit-tested; physical admission still pending. Current sources/status are independent new files;20,120prior files protected. Previous automation remains paused until a new persistent pipeline is ready.

Earlier entries below describe completed studies.

---

## Delivery complete — monitor paused

The final comparison and58.4-second video have been opened in the Mac app and browser. Automation finish-resting-ppo-comparison is PAUSED. Both main runs and all evaluations are complete; no additional training or policy promotion.

## Completed and independently reviewed — 2026-10-16

The positive-slope study is complete. Both seeds61111/61117 trained all12updates,3at8seconds and9at6seconds after passing the current/prospective development gates. Main budget:216000active+3600indexed warmup perseed=439200total physics transitions;3600initial settling ticks are separate. The learned action is only a one-second±.001rad symmetric approach pulse at6–7seconds. The reference resumes until the frozen59117 resting policy acts11/15seconds. This is limited earlier intervention, not continuous PPO takeover or unassisted sitting.

Both full-budget final policies retain18/18clean admitted development trials. Across all24development trials they have22stable,18clean,6slipping and2physical hip-pitch-housing contacts. The six excluded stress trials remain explicit failures:36cm/+15,+20degrees and40cm/+20degrees; housing contacts remain at36cm/+20degrees. No thresholds, frozen definitions or budgets were altered after launch.

On24unseen trials (12geometries repeated twice,37/39/41/43cm×+6.25/+11.25/+16.25degrees), both finals and the repaired reference/parent have24stable,20clean,4slipping,zero unwanted contacts or other physical failures. Slipping remains on37cm/+11.25and+16.25degrees, both placements. Previous reference/current59111/59117/60117 each have18stable/clean,6slipping and4unwanted contacts. The repair improves reliability; the new PPO adds no further clean successes. Conservative margin warnings remain: finals61111/61117 have8/24and4/24strict-margin-clean holdout trials. The legacy force ratio is not actual static friction capacity.

Effort uses only common full-duration clean successes, never failed fragments. On20clean unseen pairs versus repaired reference, final61111/61117 reduce total37-joint torque-squared by1.03277%/1.01799% whole20seconds,3.27835%/3.22883% final10seconds and6.97862%/6.92878% final5seconds. Final5s leg torque-squared changes−12.2356%/−12.1484%; upper-body changes−.0343%/−.0337%. Whole-trial absolute torque decreases.59284%/.58620%; mean instantaneous joint RMS decreases.98255%/.97181%. Almost all savings already come from frozen59117. Against that policy on the same repaired motion, incremental whole20changes are−.00237%/+.01256% and final5changes−.00858%/+.04500%: tiny, inconsistent, no demonstrated new effort benefit. On16common clean cases versus current59117, whole20changes−.00157%/+.00508%; versus currentearly60117, final5effort increases.01554%/.04712%. Grouped height/slope and strict-policy subsets are preserved in verification/review_audit JSON.

Development selection froze before holdout:61111block1 and61117block2, both6seconds. Full-budget block3 finals remain separate. Selected61111's later24-environment development replay slipped once at10.50seconds on44cm/+15degrees case21, despite its18-environment selection evaluation passing. Exact checkpoint load verified; do not attribute this conclusively to batch-size/numerical effects without further evidence. Report17/18admitted clean and17/24overall for this selected checkpoint. Selected61117 retains18/18admitted clean. Both selected policies have20/24clean holdout and no meaningful incremental effort gains. No policy promoted; no post-holdout reselection.

Independent `ppo_01/verified_03/verification.json` passed and exactly equals verified02:188frozen source/input hashes,17,849protected prior/preflight files, finite changed actor/critic weights, CPU optimizer replay, exact Adam continuity, smoke/main initial parity, all active/warmup/pulse/gap/reference/parent budgets, exact final/selected loads,414sensor-channel starts, locked prefixes, bounded commands/fixed fingers, full20second horizons and holdout barrier. Actual accepted actor steps91/93, critic96each; actor rollback respected. CPU replay weight atol5e-5/rtol5e-4, action inference atol/rtol1e-5; physical startup/prefix arrays exact. Actual pulse-window ticks10800perseed; inherited eligibility-after-start counts are not continuous new-actor control.

Geometry was available at all2448audited observations per unseen final through the physical-evaluation failure boundary; maximum plane-offset error2.2403mm and normal error.3232degrees. The observed failures do not show lost geometry observations as their cause. Torque remains an effort proxy, not electrical energy/global optimum; two seeds and repeated nearby rigid slabs do not establish arbitrary-surface generalization.

Final export `/data/humanoid-sitting-perception/positive_slope_01/final_export_01` copied to Mac `validated_positive_slope_01/final_results`; all236original files SHA256-verified. Additive review.html/review_audit.json clarify failure/contact unions and incremental gains; original report and frozen sources unchanged. Independent local verifier copy `validated_positive_slope_01/independent_verified_03`. Video independently rechecked against every trace:480episodes,355SUCCESS/125FAILURE,48conditions,58.4seconds,H264,1920×1080,1752frames,allfirst-event frames included. SHA256 fbe83f6eea2725660d0e98f2c346693e78cede6386d6feec4095cb646823e42b. Explicit offline schematic from recorded transforms/modelCOM. comparison.png,curriculum.png,readable report and representative clean/slip/contact/selected-regression/event frames visually inspected. Local report http://127.0.0.1:8805/review.html; video final_results/video/summary.mp4. No additional simulations or training launched in final review.

All earlier status entries below are historical. Deliver the final comparison and video, then pause automation `finish-resting-ppo-comparison`. Useful next development targets are the remaining low-seat positive-slope transfer slips and sustained feedback control; these are not newly launched experiments.

---

## Main training complete; matched final evaluations running — 2026-10-16 11:33 UTC

Both seeds finished all12updates and the scheduled219600physics transitions each (439200total). Both trained3updates with the8–9s pulse and9updates with the6–7s pulse,after passing both current/prospective18-case gates. Final development gates retained18/18clean outcomes. These are development results,not yet an unseen-surface conclusion. Selection froze before holdout:61111 selected block1;61117 selected block2. Full-budget block3 finals remain separately evaluated. No policy promoted.

Both final checkpoints and selections were verified to precede the holdout barrier;scheduled budgets and exact checkpoint hashes passed the preliminary training-completion audit atppo_01/training_completion_snapshot_01.json. Full independent optimizer/source/action/final-evaluation checks stillawait worker completion. Both workers nowrun matched reference/parent/final/selected evaluations including the24 reserved holdout trials;positive_slope_finalize waits. Rootstatus nowreflects holdoutstarted. Finalcomparison andupdated<60s video remainpending;normal progress requires no intervention.

## Main PPO launched and independently checked — 2026-10-16 10:33 UTC

Both seeds (61111/61117) began the admitted main study at10:29UTC on separate GPUs. Both have completed their first update and remain in block0 at the8-second intervention stage. Their fresh initial actor/critic/optimizer checkpoints exactly match the admitted8-second smoke initializations, with zero inherited optimizer updates; the frozen59117 parent is exact. Independent launch audit:ppo_01/launch_validation_01.json. All188 frozen source/input hashes and17,849 protected prior/preflight files passed.

Both new8-second zero/global/indexed replays matched exactly. Each seed passed36/36 exploration trials,36/36 two-update smoke trials and18/18 matched deterministic smoke evaluations without slipping, unwanted contact or physical failure. CPU optimizer replay passed. Conservative margin warnings remain; this is not a strict-margin or static-capacity certificate.

Frozen main budget:four3-update blocks per seed,18 complete20-second trajectories per update;216000active+3600indexed warmup=219600physics transitions per seed,439200total.3600initial settling ticks excluded. New PPO controls a bounded one-second approach pulse at8–9s;the reference resumes before the frozen resting parent acts11/15s. Candidate6s requires current/prospective stability and effort gates. This is not continuous PPO takeover or unassisted sitting. No reward,control,geometry,criteria or budget changes after freeze.

Persistent sessions:positive_slope_ppo_61111,positive_slope_ppo_61117,positive_slope_finalize. Current status:ppo_01/status_61111.json,status_61117.json andfinalization_status.json. Main is launched;earlier statements that main has not started are historical. Holdout remains untouched until both final checkpoints and selections are fixed. Runtime/validationv4;common/worker/verifier/finalizer/reportv5. Runverify_positive_slope_study_v5.py into freshverified_03 aftercompletion. Finalcomparison andunder-one-minute video stillpending;interimrepair video isnot the newstudy result. Automation staysactive andquietduringnormalprogress.

## Verified at2026-10-16T10:12Z:earlier pulse physical probes passed

Both8–9s pulse probes PASSED:61111=36/36 and61117=36/36 full20s stable/no-slip/no-unwanted-contact/no-physical-failure trials. Independent active-tick,startup/prefix andactual414-channel parent/bounded-action audit passed. Conservative margin warnings remain;no strict-margin or static-capacity claim. Evidence:pulse_timing_probe_05/independent_validation.json. Bothpreflight workers began fresh8s zero/global/indexed replay at10:12:01Z,then2update smokes andmatched evaluations ifexactreplay passes. positive_launch_v6 remainsguarded;main/holdout notstarted. No budget/gate changes.

## Current continuation: pulse timing probe at8s,October16

The9s pulse improved random exploration to71/72clean but61117 stillslippedonce on36cm/+5deg at10.58s;all72stable,nocontact/otherphysicalfailure.61111 additionallypassed36/36smoke,18/18evaluation,CPUreplay anddevelopmentgate;incrementaleffort savings tiny(~0.005%whole20/~0.012%final5),notmain/unseenresults.2smokeupdates only,preserved;main/holdout neverstarted. pulse9_archive_05 preserves350outputs andsources.

Nowtestingsameone-second±.001radpulse at8–9s onbothseeds toallow more settling. Runtime/reference/parent/reward/geometry/physicalthresholds unchanged. Probe:pulse_timing_probe_05/status_61111/61117.json. Bothprobes mustpass beforeconditionalpositive_pulse8_preflight_gpu0/1 performnewzero/indexedexact replay,2updatesmokes,matched20sevaluations. Live preflightpreflight_pulse8_05_status_gpu0/1.json;positive_launch_v6 waitsguarded freeze. Currentorchestration/verifier/finalizer/reportv5,runtime/validationv4. Maininitial8then6onlyifnewadmission/current/prospective gatespass;439200budget unchanged. No9sgatebypass orcontinuousPPOtakeoverclaim. Bothfinals/selections stillrequired beforeholdout. No policy promotion.

## Current continuation: smooth early-release preflight, October16

Smaller persistent.001rad exploration FAILED:61111=35/36clean,61117=33/36clean;72stable,4true pelvis-seat touchdown slips,no foot slips/unwanted contacts/other physical failures. No optimizer/main/holdout started. Preserved in early_span_001_persistent_archive_04 with114 original output hashes;diagnosis exploration_failure_diagnosis_02.json.

New documented pre-main repair uses a one-second smooth±.001rad approach pulse,zero by10s from9s start;reference controls touchdown andgap until unchangedparent59117at11/15s. This is earlier bounded intervention,not continuous PPO takeover. Same reward,physical criteria,geometry and439200 mainbudget. Software envelope checks passed;physical validation pending. Persistent positive_pulse_preflight_gpu0/1 runnewzero replay/exploration/smokes/evaluation underpreflight_pulse_04;live statuspreflight_pulse_04_status_gpu0/1.json. positive_launch_v5 guardsfreezeandmain;never bypassfailure. Currentruntime/worker/freeze/verifier/finalizer/report arev4. Useverify_positive_slope_study_v4.py forindependent finalchecks. Separateactual pulse/reference-gap/parentticks.

Holdout staysclosed untilbothfinals/selections;allprior studiesunchanged. Finalcomparison/video remainpending;interim58.4s repairvideo isnotnewPPO.

## Current continuation: bounded positive-slope preflight, October16

The .004rad exploration FAILED zero-slip admission:61111=34/36clean,61117=32/36clean;72/72stable,no unwanted contacts or physical failures. Six real pelvis-seat touchdown slips remain failures. No optimizer/main/holdout ran. Diagnosed all-tick evidence and original sources/results preserved in early_span_004_archive_03 and exploration_failure_diagnosis_01.json.

Documented pre-main revision reduces only early bound to.001rad;9s start,1s ramp,parent11/15s,reward,geometry,no-slip definition,and439200main budget unchanged. Exact source-transform audit and both saved same-GPU full-trace/414-channel zero replays revalidated. Two persistent positive_bounded_preflight_gpu0/1 workers repeat exploration,2update smokes andmatched20s evaluations under preflight_bounded_03;positive_launch_v4 waits for both checks before independent admission/freeze/main. Inspect preflight_bounded_03_status_gpu0/1.json andlaunch_v2_status.json. Do not call v2 failed statuses active progress. Current scripts common/runner/validation/worker/freeze/verifier/finalizer are v3;launcher v4. Never bypass failure. Main not started at this update.

Final verifier verify_positive_slope_study_v3.py;required report/video andholdout embargo unchanged. Existing58.4s repairvideo remains interim. Automation remains active until new comparison/video delivered.


## Verified continuation at2026-10-16T08:30:55Z

Both same-GPU zero/indexed-reset replays PASSED exactly:every trace array and actual414-channel parent inputs/actions. This confirms the preserved cross-GPU numerical-inference explanation. zero_validation_61111/61117.json identify the paired sources. Both GPUworkers now run explore_61111/61117, then2update smokes andmatchedsmokeevals. positive_launch_v3 remainsguarded;main/holdout notstarted yet. No physical threshold or numerical tolerance relaxed.

Updated168episode repairvideo COMPLETE:58.4sH264,105SUCCESS/63FAILURE,allfirsteventframes andcoverageverified. Copiedtolocalvalidated_positive_slope_01/repair_validation_results,55originalfilesSHAverified. review.html/review_audit.json clarify contactunion;physicalhousingcontact failurescount2. comparison.png andsuccess/failureframes visuallyinspected;browserreportlocalhost8804/review.html,localserverexecsession34520. Video/repairresults delivered asinterim;newPPO/unseencomparisonpending. AutomationACTIVE until finalcomparison/video delivered.

---
# Current continuation — positive-slope repair and earlier PPO (2026-10-16)

Read `rl_support_work/POSITIVE_SLOPE_STATUS.md` and remote `docs/POSITIVE_SLOPE_PROTOCOL.md` first. Work directly by SSH kfuUbuntu; never launch another coding agent. Data `/data/humanoid-sitting-perception/positive_slope_01`.

Repair06 independently validated: reference and existing59117 with conditional11/15s handoff each18/24 clean,22stable; exact first10s/startups. Current original reference/policies14clean. Six low/steep stress trials remain, including two physical housing contacts at36cm/+20deg. Train only9clean geometries×2 after explicit pre-main revision, retain all stress cases in comparisons. Current policy preserved exactly on14previous clean cases; repaired59117 saves1.0346%whole20/6.5922%final5torque² vs repairedreference on18commonclean; these are not newPPO gains.

`positive_launch_v2` gates independent `positive_preflight_v2_gpu0/1`, then freezes and launches seeds61111/61117 only if passed. Inspect launch_v2_status.json/preflight_v2_status_gpu*.json before claiming main started. Proposed12updates perseed in4blocks,219600physics/seed,439200total;earlyintervention9→8→6s only through clean/effort gates. Conditional frozenparent11/15s,414sensorinputs,3symmetric boundedactions. Actual materialaudit disproved treating legacy0.6force ratio as staticcapacity;0.90warnings andactualslip/contact definitions remain unchanged. All failedpreflights/probes preserved. Holdout24trials is stillclosed untilbothfinals/selections.

Main/freeze/independentverifier/report/finalizer built with separate v2 scripts. Full final comparison includes oldreference/current59111/59117/60117, repairedreference and repairedparent. Complete under60s outcomevideo;nohiddenfailures. Agent must independently rerunverified_03,copy/hashcheck/visuallyinspectfinal_export_01,updatehandoffs,anddeliveractualMP4beforepausingautomation.

`positive_repair_export_02` creates updated interim repair_validation_export_02 (168developmentepisodes,old5controllers+repair06reference/parent). Copy to localvalidated_positive_slope_01/repair_validation_results whencomplete; do not call it finalPPO/unseen results. Olderdiagnosis_results alreadycopiedandvisuallyinspected usesrepair01/02. Automationfinish-resting-ppo-comparisonACTIVE with updatedprompt. No policy promoted.

---

# Positive-slope continuation — October16,2026

Active user request: repair steep **positive** slopes (surface descends toward robot feet), validate zero slipping/unwanted contacts, train earlier PPO, compare with current reference/policies, update complete under-one-minute outcome video. Negative slopes are outside this continuation. Work only Mac SSH→kfuUbuntu; no second agent.

Root `/data/humanoid-sitting-perception/positive_slope_01`. Original15596 files protected; initial assessment independent verification passed with exact physical/414-channel startups and full20s horizons. Previous earlier_control_01 remains complete and unchanged. No new PPO training or unseen-surface simulation has started. The main proposal in POSITIVE_SLOPE_PROTOCOL.md is **not admitted or frozen**.

Development12 geometries×2 placements:36/40/44cm ×+5/+10/+15/+20deg. Reserved before simulation:37/39/41/43cm ×+6.25/+11.25/+16.25deg,12 geometries×2. Holdout remains closed.

Assessment complete: current traction reference,59111,59117,60117 each16/24 stable,14/24 clean,6physical forbidden-contact failures,10slipping cases. Rejected60111 has16stable but13clean and11slipping cases. Contact failures concentrate36cm/+15,+20 and40cm/+20;44cm/+20 fails continuous support;40cm/+15 has brief seat sliding. No old negative-slope outcome relabeled.

Diagnosis: original endpoint search falls back to0.66rad despite failed model clearance on these low/steep surfaces, moving support toward the seat edge. A geometry screen in true seat coordinates confirms reduced housing clearance. Failures occur after10s; first10s approach is physically reliable. Two earlier +10deg60111 slip events are at the seat/body patch, not feet. New offline screens and all rejected candidates are preserved. target_screen_02 aborted on an empty contact candidate; robust target_screen_03 records infeasible candidates explicitly and preserves02 unchanged.

Physical repairs:

- `repair_01`: slower transfer to rootx−.18m,pitch.25rad,gap+.001m, land_fraction.6.22/24 stable,18/24 clean; removes active seat-patch legality violations but36cm/+20 still has two physical forbidden-contact failures; six trials slip. Not admitted.
- `repair_02`: deeper rootx−.24m,pitch.55rad,gap+.003m, land_fraction.8,travel_fraction.6.20/24stable,12clean,4impact failures and12slipping cases. Rejected. Not a PPO result.
- `repair_03` persistent tmux positive_repair_03 GPU1:rootx−.18m,pitch.40rad,gap+.006m,both fractions1.0. Uses corrected diagnostic close wrapper_v2. Check full20s results before further decisions.
- `repair_04` persistent tmux positive_repair_04 GPU0: same path as03 plus sensor-only load-feedforward correction. Interpolates no-seat/full-seat static motor biases using existing torque/encoder/IMU foot-wrench estimate; clamps correction±.08rad. No simulator contact/truth enters control. Physical validation still pending.

Original repair01/02 close wrapper accessed `scene` after inherited close had deleted it. Physics and every-tick/results arrays were saved before the exception. `repair_close_recovery_01` preserves original logs/sources/hashes and independently confirms24×1000active ticks,20s,zero indexed warmup. No simulation rerun or original output rewritten. Explicit `closure_recovery.json` records missing original body-name/episode-contract serialization; do not infer sensor body order from robot body order. New wrapper_v2 captures names before close. Legacy failed wrapper remains unchanged.

`positive_diagnosis_export_02` runs export_positive_diagnosis_v2.py, verifies starts/first10s/fullhorizons, and writes diagnosis_export_01 with comparison.png,findings.html,all episodes and video. It compares five original controllers plus repair01/02 (168 episodes,24conditions) in <60s. This is **development physical validation, not new PPO results**. Original export01 failed on the close exception and its log/source are preserved. Copy report to local validated_positive_slope_01/diagnosis_results, verifySHA256,inspectplots/video, but do not call it the final PPO comparison.

New run_positive_slope_ppo.py and positive_slope_validation.py are provisional earlier-control code, unexecuted. Candidate fresh seeds61111/61117 choose3 symmetric±.004rad offsets with1s smoothstep initially9s; frozen59117 resting policy from11s. Four3-update blocks proposed;585600main physics ticks total including indexed warmup,4800initialsettling separate. Before training, integrate only a physically validated controller, validate zero-action reproduction and indexed resets, random probes,2update optimizer smokes, deterministic smokes and CPU replay, then freeze exact protocol/sources/banks. Do not launch on the currently failed admission or silently discard steep cases. If a bounded scope revision is scientifically necessary, record it explicitly before freeze and keep every failed geometry reported as stress—not success. No progression/selection on holdout.

New-study proposal explicitly keeps original .90 conservative warnings and penalties but changes advancement from no-added-warning to physical no-slip/no-contact and grip utilization<1, plus effort retention. This is a new research choice, not a change to completed studies or physical thresholds. Current proposed gate references assessment59117/reference, which contain failures; before freeze it must use validated, matched repaired baselines and avoid effort comparisons from failed fragments. Main training runners/finalizer are not yet built. Do not claim training launched.

Next: inspect03/04 outputs and exact body contact diagnostics; finish physical repair on development without touching holdout, then finalize preflight/training pipeline. Preserve all previous studies. Final report must separate repair gains from incremental PPO, full-budget final from any selection/fallback, use common full-duration clean trials for total37-joint effort, and include every final comparison episode with success/failure labels. No policy promotion; torque is not measured energy.

---
Previous completed studies follow; new user continuation above is current.

# Complete: earlier-control sitting PPO — October 15, 2026 PDT

Study `/data/humanoid-sitting-perception/earlier_control_01` is complete and independently reviewed. Work remains directly through Mac SSH to kfuUbuntu; never another coding agent. Both seeds60111/60117 completed24 updates and all final20s tests. No further training, definition changes or policy promotion. Read `docs/EARLIER_CONTROL_PROTOCOL.md`, frozen `ppo_01/protocol.json`, and the new independent review.

**Outcome: one second earlier for one seed, no new effort savings.** All24 updates of both seeds trained early control at10s; neither reached or tested9/8/6s.60111 failed all four development gates due to added per-case conservative margin ticks; selection explicitly falls back to current59111 at11s.60117 passed only the last gate at10s (18/18 strict-clean), after the last training block, and its full-budget final is also the development-selected candidate. The reference remains active throughout and the frozen resting parent acts from11s. This is bounded reference-assisted sitting, not unassisted motion discovery or a new leaning result.

Both new finals, reference and current parents retain18/18 development and16/16 retention stable/no-slip/no-contact outcomes. On32 fresh wider tests, every controller has22/32 stable support and the same10 physical forbidden-contact failures. However,60111 adds one20ms slipping sample in two otherwise stable trials:36cm/+10deg case2 at10.62s and38cm/+10deg case7 at10.58s. Therefore full-trial stable with zero slip/contact is20/32 for60111 versus22/32 for60117/reference/parents. The original stable-support plot shows22/32 for all; the independent review and video explicitly retain the two added slip failures. Do not report stable-support count alone as a clean success result.

Shared physical failures: both placements at38/42/44cm with−10deg and39/43cm with−7.5deg. No falls, impact-limit, torque-limit or joint-limit failures. Five shared failed trials slip. All-tick legal-contact flags and the separate wrong-contact-force>10N diagnostic have different definitions; preserve both. Conservative strict successes are13/18 and18/18 development,10/16 and11/16 retention,13/32 and14/32 wide for new60111/60117; baselines16/18,8/16,12/32 respectively.60117 has fewer aggregate warnings, but new per-case warnings still occur on held-out surfaces; it is not a universally margin-clean policy.

Wider-bank paired torque² versus repaired reference, whole20s/final10s/final5s:60111 −0.858%/−2.609%/−4.367% on20 common full-duration clean cases;60117 −1.678%/−5.108%/−7.854% on22. Versus their respective current parents:60111 +0.058%/+0.179%/+0.240%;60117 +0.014%/+0.043%/−0.001%. The reference-relative savings were already present in the frozen resting policies.60117 versus the other current parent59111 is −0.852% whole and−3.792% resting;60111 versus59117 is +0.741% whole and+3.163% resting on its20-case subset. The two subsets differ and must not be compared as matched raw averages.

60117 resting absolute torque−2.174%, mean instantaneous37-joint RMS2.522Nm versus2.637Nm reference, leg torque²−10.724%, upper-body+0.004% on22 common clean wider successes. Strict14-case resting savings versus reference9.274%, but+0.020% versus own parent.60111 strict13-case savings versus reference5.749%, but+0.095% versus own parent. Height/slope groups, both current-policy comparisons, absolute/RMS/leg/upper metrics and all three time windows are in verification.json and independent_review.html. No savings from failed fragments. Torque is an effort proxy, not measured energy or a global optimum.

Geometry before physical failure: wide118–120 fresh-observation gaps out of2782 audited observations/controller; maximum plane offset7.30mm, normal error<0.390deg. Stored measured geometry plus encoder/IMU tracking remain sensor-only. Retention0/1632 missing; development58/1836 late gaps. Six shared physical failures precede10s, four occur later. Availability alone does not establish the failure cause.

Independent verified_03 exactly matches verified_01/02. All116 frozen inputs and14208 protected prior/preflight hashes unchanged. Finite changed weights, exact frozen parents, smoke/main initialization parity, CPU optimizer replay (atol5e−5/rtol5e−4), optimizer continuity, loaded checkpoints, exact physical/414-channel startups, locked10s prefixes, full20s horizons, fixed fingers and bounded offsets passed. Actor Adam steps142/148; critic192 each.882000total physics ticks =864000active+18000indexedwarmup;3600initialsettle separate. Perseed216000teacher and216000early/frozen-parent active control ticks. All864 training episodes stable without slip/contact/physical failure. Both finals and development selections precede all held-out tests. Pre-main adaptive exploration admission revision remains disclosed; no frozen main criteria were relaxed.

Final export copied to `/Users/kadir/humanoid-sitting-control/validated_earlier_control_01/final_results`. All109 original SHA256 entries verified. Independent audit and readable review are supplemental; originals unchanged. `independent_review.html` explains clean versus stable counts and is served at http://127.0.0.1:8803/final_results/independent_review.html. comparison.png, curriculum.png, findings.html, independent review and representative success/added-slip/contact-failure frames visually inspected.

**Final video:** `final_results/video/summary.mp4`,59.000s,H264,1770frames. All330 episodes across66conditions,278successes/52failures; required first-event frames and complete coverage independently reverified. SHA256 f10a55adaa7430f95a97df7c59d6e8d5d0bf12a8a29b329c9836e3f6a3f71d9c. Every outcome labeled, conservative warnings amber. Recorded-state schematic at24x nominal speed, not invented simulation. Earlier archived renderer validation and six-update interim videos remain preserved and must not be confused with this final result.

The completion automation finish-resting-ppo-comparison is PAUSED as part of this final delivery. No policy is promoted. Two seeds and repeated rigid surfaces are limited evidence; no arbitrary-surface claim. Further physical repair or training requires a new authorized continuation.

---

Historical records follow; completion above supersedes active-run entries.

# Main PPO active — October15,2026,6:00PM PDT

Both main seeds60111/60117 launched at2026-10-16T01:00:00Z; physical GPU0/GPU1 processes and saved main initial checkpoints verified. Persistent sessions earlier_control_ppo_60111/60117; supervisor earlier_control_finalize_02; completion heartbeat ACTIVE. Study sources/banks/calibration frozen:116 hashes;14208 protected prior/preflight files. No further reward/control/budget/criteria changes or extra training.

Each revised exploration and two-update training run passed36/36 full20s stable/no-slip/no-contact/no-physical-failure trials. Independent CPU replay passed15/16 actor Adam steps and16critic steps. Final deterministic admission policies each18/18 stable, zero slipping/unwanted contacts/physical failures. Conservative warning counts remain15/18 and14/18 strict-clean; each has two added warning cases10/11 versus its parent. Peak grip utilization0.98003/0.92633, all below capacity1. These smoke policies FAIL the stricter advancement/selection gate due to added warnings, which remains unchanged. Main starts10s; no earlier advancement or policy promotion yet. Smoke effort remains within0.015% of parent resting torque², but these are development checks, not main or unseen results.

Explicit pre-main admission revision is archived in admission_revision_02: exploration/smoke allow at most one20ms conservative warning tick pertrial below capacity1, while requiring stable/no-slip/no-contact/physical checks. Original0.90 reporting, penalties and stricter no-added-per-case-margin progression/selection remain. This is a research screening revision, not an execution-only repair; original rejected outputs preserved. Current reference and59111/59117 remain unchanged.

Main24updates perseed, four6-update blocks,441000physics ticks/seed including9000indexedwarmup,882000total;3600initialsettle separate. Earlier controls10→9→8→6s only after current and earlier full20s development gates pass; main fullbudget finals and selected/fallback reported separately. New wider32trials36–44cm at±7.5/±10deg plus16previouslytested retention remain closed until both final checkpoints/selections. Compare repaired reference,currentparents,newfinals anddistinctselected. No arbitrary-surface or unassisted-motion claim.

Video pipeline verified using90 archived episodes (88success,2failure) in58.6s, complete coverage and first-event frames,H264, SHA-checked locally and failure frame visually inspected. This is renderer validation only. Actual final video will include all330–462 new final-comparison episodes across66conditions, with SUCCESS/FAILURE labels and separate margin warnings, in under60s. All files for archived test are under validated_earlier_control_01/renderer_validation; final export destination remains validated_earlier_control_01/final_results.

Read frozen protocol.json and docs/EARLIER_CONTROL_PROTOCOL.md. On completion rerun independent verifier into fresh verified_03, compare01/02/03, verify final SHA/video coverage and inspect plots/video, update both handoffs and deliver honest outcomes plusMP4, then pause finish-resting-ppo-comparison. No policy promotion.

---

Preflight history follows.

# Active: earlier PPO sitting control — October 15, 2026 PDT

Read `rl_support_work/EARLIER_CONTROL_STATUS.md` and `EARLIER_CONTROL_PROTOCOL.md`, or Ubuntu docs counterparts. Work directly through SSH to kfuUbuntu; no other coding agent. New study `/data/humanoid-sitting-perception/earlier_control_01`.

Prior persistent-posture PPO is complete and modestly reduces effort without lost stability. This authorized continuation adds a bounded early actor, initially10s, while freezing59111/59117 for resting from11s. Development-gated candidate times10/9/8/6s. Main training remains blocked until physical exploration, short-training and replay checks pass. Zero early action exactly matches each parent across both reset episodes. A reset-comparison harness error was diagnosed and narrowly fixed with all originals preserved; no physics criteria changed.

Fresh32-trial wider bank36–44cm, slopes±7.5/±10deg reserved before simulations, plus16previously tested retention cases. Main budget24updates/seed60111/60117,882000physics ticks including warmup; no extra training or failed-gate bypass. Final comparison includes repaired reference/current policies and an under-one-minute video covering every final success/failure episode, with independent coverage audit. Renderer failure/success smoke validated and visually checked. Read live preflight/launch/finalization status; no new training savings claim yet.

---

Historical records follow; the active entry above supersedes previous completion-only status.

# Complete: persistent-posture PPO effort comparison — October 15, 2026 PDT

Work directly through Mac SSH to kfuUbuntu; never another coding agent. Study `/data/humanoid-sitting-perception/posture_credit_01`. Both seeds59111/59117 completed10 updates and all matched20s development/holdout evaluations. No additional training or policy promotion. Read frozen `docs/POSTURE_CREDIT_PROTOCOL.md` and `ppo_01/protocol.json` for definitions. The completion monitor finish-resting-ppo-comparison is PAUSED after the verified final comparison; no training remains active.

**Result: modest learned effort savings with retained stability.** Repaired reference, previous58311/58317 and both new policies all pass18/18 development and16/16 holdout, with zero all-tick slipping, unwanted contacts or physical failures. Strict clean remains16/18 development and8/16 holdout for every controller. The eight holdout flags are each one20ms conservative friction-margin sample at10.46–10.54s, before PPO begins11s; maximum utilization0.918454, above0.90 reporting margin but below modeled capacity1.0. Every controller has exactly the same flags. No criterion was relaxed.

On16 common holdout successes, new59111/59117 torque-squared changes versus repaired reference are −0.888%/−0.871% whole20s, −3.155%/−3.094% final10s, and −5.204%/−4.569% final5s. Versus previous58311, final5s gains are5.450%/4.817%; versus58317,4.763%/4.125%. Both new policies beat both previous ones on average. These are incremental PPO gains with the physical reference unchanged, unlike earlier controller-repair gains. Development final5s gains are5.256%/4.622% versus reference.

Resting absolute-torque sums fall1.696%/1.016%. Mean instantaneous37-joint RMS falls from2.10623Nm to2.05199/2.05022Nm. Leg torque-squared falls9.751%/8.546%; upper-body changes only−0.069%/−0.079%. Vertical seatload rises1.870%/2.100% relative to reference, not percentage points. Strict-clean8-case final5s gains persist at5.643%/3.858%. No failed fragments were used for savings.

59111 improves all16 individual holdout placements.59117 improves14/16, but increases resting effort0.381%/0.314% on the two41.75cm/−2deg placements (paired mean+0.347%). Gains therefore are not uniform across seeds/surfaces. Holdout has eight reserved geometries at38.25/39.25/40.75/41.75cm and slopes±2.5/±3.5/±2.5/±2deg, repeatedtwice; both final checkpoints preceded its evaluation and no holdout selection occurred.

Independent verified_03 exactly matches verified_01/02. All94 frozen source/input hashes and12898 protected prior/evidence hashes pass. Finite changed weights, exact main/smoke initialization parity, final checkpoint loads, matched physical and414-channel sensor starts and actual11s actor inputs, locked first11s, full20s horizons, bounded symmetric offsets and fixed fingers passed. CPU replay reproduces every optimizer update with documented tolerances. Actor Adam steps59/49; critic80 each. Budget368100physics transitions =360000active +8100indexed warmup;900initialsettle separate, no pending warmup/unfinished ticks. Each seed180/180 training episodes stable, with no slip/contact/physical failures or post11s margin flags. All preflight/probe experience is separately accounted.

Geometry: all1632 holdout observations available, max plane error2.183mm/0.3681deg. Development retains58 late fresh-depth gaps among1836 audited observations; measured-plane memory and encoder/IMU tracking remain available. Simulator truth is reset/reward/evaluation only; actor/critic use414 sensor/reference channels. This policy selects3 persistent posture offsets at11s behind the unchanged feedback reference; it is not unassisted standing-to-sitting discovery, a new leaning result, or arbitrary-surface generalization. Two seeds, nearby rigid slabs and repeated placements are limited evidence. Torque is a proxy, not electrical energy or proof of minimum possible effort. The bundled learning changes do not establish a single causal diagnosis.

Final report: `/Users/kadir/humanoid-sitting-control/validated_posture_credit_01/final_results/review.html`, served at http://127.0.0.1:8802/final_results/review.html. Original128 SHA256 entries verified. Supplemental REVIEW_SHA256.json covers134 files including independent_verified_03, readable conclusions and per-surface exceptions. comparison.png, findings.html, review.html and replay at41.75cm/−2deg rest plus39.25cm/−3.5deg transfer/rest were visually inspected. Remote final_export_01 receives the same supplement. All previous study outputs and frozen definitions remain unchanged.

---

Historical records follow. Completion above supersedes older active entries.

# Active: persistent-posture learning repair — October 15, 2026 PDT

Read `rl_support_work/POSTURE_CREDIT_STATUS.md` and `POSTURE_CREDIT_PROTOCOL.md`, or their Ubuntu docs counterparts. Work directly through SSH to kfuUbuntu; never another agent. Root `/data/humanoid-sitting-perception/posture_credit_01`.

Verified development probes: sustained positive knee/ankle offsets reduce resting torque²8.64%/6.75%,18/18 stable, zero slipping/unwanted contacts/physical failures. Previous PPO moved only~0.0004rad on average;26/64 rollouts had no eligible learning data. Reward clipping was not active. New PPO uses3 persistent bounded posture choices with complete-outcome effort credit, retaining the repaired reference and first11s. Zero-action regression is exact.

All preflight gates passed: each seed54/54 exploration,36/36 short-training and18/18 final-policy trials stable, without slipping/unwanted contacts/physical failures or added conservative margin flags. CPU optimizer replay passed. Both main workers launched at2026-10-15T22:36:54Z (3:36PM PDT). Main budget10 full-episode updates/180 completed trajectories perseed59111/59117,368100 total physics transitions including warmup, matching prior active experience. New16-trial unseen bank remains reserved until both final checkpoints. Persistent finalizer and the updated completion heartbeat are active. Read live status; no final PPO savings claim yet. Do not bypass admission, extend budgets, alter frozen definitions or promote policies. Earlier studies remain complete and unchanged. Diagnostic report: localhost8802/diagnosis_results/findings.html.

---

Historical records follow; the active entry above supersedes prior completion-only status.

# Complete: traction-safe sitting PPO — October 15, 2026 PDT

Both seeds58311/58317 completed their64 scheduled rollouts and all matched20s evaluations. Independent verified_03 exactly matches verified_01/02;76 frozen inputs and11342 protected prior/preflight hashes remain unchanged. Work directly through SSH to kfuUbuntu; never another coding agent. Root: /data/humanoid-sitting-perception/traction_transfer_01. No additional training or policy promotion.

Unseen seats: repaired traction reference and both new policies16/16 stable, zero all-tick slipping, unwanted contacts or physical failures. Strict clean11/16 for all three. Five cases retain one20ms conservative traction-margin flag each at10.48–10.54s, before learned control starts11s:39.5cm at±4deg (both placements) and40.5cm/−1deg (second placement). Maximum utilization0.94848, above0.90 margin but below modeled capacity1.0. Development18/18 stable,16/18 strict; two20ms flags at42cm/−5deg,10.46s,max0.91402. No labels were erased or thresholds changed.

Incremental PPO on16 common holdout successes versus repaired traction reference: seed58311 torque² whole20s/final10s/final5s +0.116%/+0.406%/+0.245%;58317 −0.087%/−0.304%/−0.551%. No consistent two-seed effort improvement. Strict11-case final5s changes +0.198%/−0.560%. Final5s absolute torque −0.130%/−0.053%, mean instantaneous37-joint RMS2.147/2.138Nm versus2.144Nm. Leg torque² +0.329%/−1.117%; upper-body +0.138%/+0.175%. Final5s seatload about52%BW, essentially unchanged.

Against older corrected reference: new policies whole effort −33.71%/−33.84%, final5s −78.45%/−78.62%,15 common successes (older reference15/16). Against56211/56217: about31–33% whole and76–78% final5s lower,16 common successes. These large savings come from physical controller repair, not PPO. Previous bridge reference/58111/58117 only10/11/12 of16 stable and slip on41.5cm/−3deg. New policies improve reliability but consume about23–25% MORE resting effort than58111/58117 on respective11/12 common successes, and29–30% more than bridge reference on10. No savings from failed fragments.

Budget368640physics transitions =360000active +8640indexed warmup;145800learned-control +214200reference-control ticks. Each seed544actor/608critic Adam steps, actor unchanged during critic-only warmup,180pending warmup ticks,0unfinished active ticks.900initial settle and46080smoke ticks separate. Exact final checkpoint loads, matched physical/414-channel sensor starts, first11s locked reference prefixes, full20s horizons, fixed fingers/bounded-control implementation and sampled slew passed. All resets standing; this remains reference-assisted, not unassisted standing-to-sitting discovery. Both finals preceded reserved evaluation. Inherited base contract labels are superseded by traction_contract.json and frozen traction sources for actual414 channels/all9 eligible surfaces.

Geometry: holdout all1632audited observations available per controller, worst plane error2.18mm/0.374deg. Development retains58 fresh-depth gaps on two high-seat cases; stored measured plane and encoders/IMU maintain sensor-only tracking. Two seeds/eight nearby reserved rigid geometries repeated twice do not establish arbitrary-surface generalization. Torque is effort proxy, not electrical energy/global optimum. Planner fallback is physically tested, not certified clearance.

Final report: /Users/kadir/humanoid-sitting-control/validated_traction_transfer_01/final_results/review.html (http://127.0.0.1:8801/final_results/review.html). Original159 SHA256 entries verified; supplemental REVIEW_SHA256.json includes independent audit and readable conclusions. comparison.png, findings.html, supplemental review and replay at unseen39.5cm/−4deg rest and41.5cm/−3deg transfer were visually inspected. Remote final_export_01 includes the same supplements. Local/remote handoffs updated; completion monitor is PAUSED after the verified comparison. No training pending. Next research decision: obtain meaningful learned effort reduction while retaining the successful physical approach; no new study authorized or launched by this completion check.

---

Historical status below; completion above supersedes prior active entries.

# Active: traction-safe sitting PPO — October 15, 2026

Read `rl_support_work/TRACTION_TRANSFER_STATUS.md` and `TRACTION_TRANSFER_PROTOCOL.md` locally, and the matching files under `/home/kadir/humanoid-sitting/docs` on Ubuntu. Work directly through SSH to `kfuUbuntu`; never start another agent. New root `/data/humanoid-sitting-perception/traction_transfer_01`.

Reference repair now18/18 stable20s across all nine development heights/slopes, zero slipping/unwanted contacts/physical failures. Strict clean16/18 retains two20ms0.914 grip-utilization samples. The first10s approach and the complete probe-to-training-environment replay are exact; reset isolation passed. Whole-motion/final5s torque² falls32.58%/74.79% versus current corrected reference on18 common successes; these are reference-repair gains, not PPO. High-seat forward-leaning rest loses58 fresh-depth samples late; stored measured plane and kinematics remain sensor-only.

Both optimizer smokes passed18/18 stable and16/18 strict with zero slip/physical failures. The launch gate froze76 inputs and11342 prior/preflight files; both main runs are active: seeds58311/58317,64 scheduled rollouts each,368640 total physics transitions, first11s reference-controlled. All9 development surfaces eligible. Frozen414-channel sensor/reference observations,23 bounded residuals,14 fixed fingers. Fresh16-trial holdout stays closed until both final checkpoints. Read launch_status.json, ppo_01/status_58311.json/status_58317.json and finalization_status.json. No automatic policy promotion.

The existing completion heartbeat is ACTIVE and updated for this study. Preflight report is SHA-verified and visually checked at `validated_traction_transfer_01/validation_results`, localhost8801. Prior studies and10326 original protected files remain unchanged. Full current instructions, admission distinction, budgets and final verification/export steps are in the status/protocol documents.

---

# Complete: crouch-to-rest bridge and bounded PPO — October 15, 2026

Read `rl_support_work/CROUCH_REST_BRIDGE_STATUS.md` locally and `docs/CROUCH_REST_BRIDGE_STATUS.md` on Ubuntu, plus frozen `docs/CROUCH_REST_SAFE_PPO_PROTOCOL.md`. Work directly through Mac SSH to `kfuUbuntu`; never another agent. Study root `/data/humanoid-sitting-perception/crouch_rest_bridge_01`.

Both64-rollout PPO seeds58111/58117 finished. Neither progressed beyond control starting15s; all approaches remain reference-controlled. Development admitted clean11/12 and12/12; six stress trials remain0/6 with two forbidden-contact failures.58117passedits48-update current-window gate but failed the separate13s-control gate11/12strict. No gate bypass.

Holdout: older current reference/currentPPO56211/56217 each16/16stable; repaired reference and both new policies each8/16stable,5/16clean,0physical failures. Remaining misses are traction-margin/slip problems; no unwanted holdout contacts. PPO does not help effort: final5s torque² rises1.93%/1.09% versus repaired reference, wholemotion+0.10%/+0.06%, on8common successes. Large savings versus older baselines (~37%whole/~87%rest) come from the reference repair and do not offset lost reliability. No policy promotion.

Verification01/02/03 pass identically;49frozen/8705prior hashes unchanged. Main368640physics ticks include only42000policy-enabled ticks; actor272/critic320Adam steps perseed. Exact startup/checkpoint/prefix matches and full20s horizons. Geometry available; physical transfer is the remaining problem. Final report/export SHA-verified and visually inspected at `validated_crouch_rest_bridge_01/final_results`, localhost8800/final_results/findings.html. Named completion automation paused after final reporting; no training pending. Earlier results below remain unchanged.

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
