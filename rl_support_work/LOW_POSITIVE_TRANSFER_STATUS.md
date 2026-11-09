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

## Audit-only recovery,18:42UTC

GPU1 exploration hit a memory-channel audit difference1.192e-7 from independently accumulated NumPy/CUDA tanh rounding. All414 sensor channels were exact. Corrected validatorv3 reads the actual previous applied command for memory identity while retaining the separate independent slew/action replay and every original tolerance. Both seeds' existing calibration/exploration now pass; both exploratory outcomes26stable/18clean/10slip/2physical. Evidence and original failure sources/status are preserved in input_audit_recovery_02. No simulation/reward/actor/criteria change.

Current persistent recovery jobs: low_positive_preflight_v2_62111/62117; states preflight_recovery_status_62111/62117.json. GPU0 waits for its existing smoke worker to finish; GPU1 starts only its unrun smoke. Existing outputs are reused and independently checked, never overwritten. low_positive_launch_v2 waits both recovery checks, then freezes/launches. Runtime remains run_low_positive_feedback_v2.py; actual audit is validate_low_positive_feedback_v3.py. Main/holdout not started at this update. Automation is updated to these paths.

# Continuous sitting feedback — active, 2026-10-16

The manual repair phase did not solve the remaining slipping. All six variants (phase_only, phase_deeper, timing_mild, timing_strong, load_delay, rolling_mild) are rejected, with366 outputs preserved. Do not claim a completed physical repair or effort improvement.

New data: /data/humanoid-sitting-perception/low_positive_transfer_01. Work directly over SSH kfuUbuntu; never another coding agent. Original20,120 prior files remain protected. Fresh24-trial holdout remains closed:36.5/38.5/40.5/42.5cm ×+7.5/+12.5/+17.5degrees, each twice. Prior study holdout is now development data. Negative slopes excluded.

The new repeated-feedback interface passed EXACT same-GPU fresh-start, warmup and indexed-reset traces and actual414-channel parent inputs/actions. The first comparator mistakenly compared a fresh-start episode with an indexed-reset episode; its failure and sources are preserved in zero_validation_recovery_01. Corrected matching two-episode sequences passed feedback_zero_v2_status.json. No tolerance or control change.

Both preflight calibration baselines:26/28 stable,18/28 clean,10 slipping,2 housing-contact cases,8 strict-margin clean. Training includes14 known geometries (old12 plus37cm/+11.25,+16.25), each twice. Known failures participate in optimization, rather than being omitted. Stable is separate from clean: clean requires full-duration supported success plus zero all-tick slip/unwanted contact/physical failure.

Persistent low_positive_preflight_62111/62117 run physicalGPU0/1:1 calibration,1 random exploration,2-update smoke, matched deterministic smoke evaluation, full action/reward/budget and CPU-optimizer verification. A passed preflight certifies the software checks; it does not claim physically clean exploration. All smoke regressions remain reported. low_positive_launch waits for BOTH preflights and an independent freeze before main. Inspect launch_status.json and preflight_status_62111/62117.json; no main is claimed before successful launch.

Final pre-main design revision uses ONE uninterrupted16-update optimizer process per seed, preserving RNG/Adam/weights. All training uses10–20s continuous residual feedback. Each seed:448000active+10500indexed warmup=458500physics ticks;917000total. Initial settling1400ticks total excluded. Earlier four-block proposal is preserved; learning_revision_02.json and frozen ppo_01/protocol.json supersede its accounting. Best development checkpoint may get an8–20s prospective evaluation; no8s training is claimed.

Actor/critic417 inputs =414 sensor/reference channels +3 prior applied residuals. Three symmetric hip/knee/ankle residuals, .004rad bounds and .004rad/s slew; updates every.5s, applies every.1s. Other17 body residuals zero;14 fingers fixed. The reference and frozen59117 resting policy at11/15s remain active. This is bounded reference-assisted feedback, not unassisted sitting. Reward uses complete undiscounted reward-to-go over all37 physical joint torque-squared plus dense loaded sliding/forbidden contact and binary physical/support/slip/contact/margin penalties. No truth enters networks; no short bootstrap.

Checkpoints4/8/12/16 get development evaluation at10s. Gate preserves every matched-parent clean case, adds no physical failure condition, and allows at most+1% whole20 torque²/+2% final5 torque² on common clean successes. Rank by clean count, physical failures, then effort; explicit parent fallback if none pass. Only the best passing checkpoint gets a prospective8s test; choose it only if that separate gate passes. Both finals/selections must precede fresh holdout. Compare on old24 development, prior24 holdout (now development transfer) and fresh24 holdout against repaired reference, parent59117 and current61111/61117 on each GPU. Report finals and selections separately; no promotion.

Finalizer verifies twice and exports final_export_01. Independently rerun verify_low_positive_study.py into fresh verified_03 after completion, copy/hash-check final export to Mac validated_low_positive_transfer_01/final_results, inspect plots/report, update handoffs and report actual stability/slip/contact/effort outcomes. Unwanted-contact count is the union of forbidden-body and illegal-patch flags. Effort only on common full-duration clean successes. Torque is a proxy, not energy; two seeds and repeated rigid surfaces are limited evidence. Holdout is withheld from this pilot, not necessarily never used anywhere in the project.

Automation finish-resting-ppo-comparison is ACTIVE for this new study and stays quiet during normal progress; pause only after final comparison delivery. No extra training or post-freeze research-definition changes.
