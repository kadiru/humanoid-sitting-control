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

# Independent sitting and leaning PPO running — October 14, 2026 PDT

The user authorized separate sitting/leaning training, contact-safe weight transfer, progression toward standing, and comparison against reference motion. Work directly from this Mac via SSH to kfuUbuntu; never start another coding agent. Preserve all prior studies.

## Validated before launch

The passive probe exactly reproduces the old reference. Sitting fell backward before support; leaning contact approached the friction limit. Sitting uses IMU ankle pitch feedback (kp0.3,kd0.03,cap0.2rad). Leaning uses coordinated knee+0.06/hip−0.03/ankle−0.03rad for approach starts0–4 only. Initial all-start leaning bend passed reference14/14 but learned smoke12/14; that failure is preserved. The revised approach-only controller and learned smoke both pass14/14. No physical criteria were relaxed.

Both final preflights: reference14/14 and two-update PPO14/14, full20seconds, no physical failures, exact physical/sensor starts, reset isolation, finite changed weights and exact loaded checkpoints. Matched old reference: sitting10/14 (4physical failures), leaning6/14 (0physical failures). Standing improved0/2→2/2 for each corrected reference. Whole20s squared-torque reduction for the controller repair vs old reference on common successes:11.10% sitting(n10),10.23% leaning(n6). These are controller gains, not trained PPO gains. Geometry/command/horizon audits passed. Development smoke6144 transitions is separate.

## Running study

Root `/data/humanoid-sitting-perception/split_support_curriculum_02`; live `ppo_01/sitting/status.json` and `ppo_01/leaning/status.json` are authoritative. Independent tmux `split_sitting_ppo_02` onGPU0 runs `scripts/run_split_curriculum_study.py`; `split_leaning_ppo_02` onGPU1 runs `scripts/run_split_leaning_study.py`. No shared curriculum gates. Seeds55211/55217 sitting;55311/55317 leaning. Each seed128updates,16envs,64steps,131072transitions including warmup; total524288. Four32-update blocks start at stages[4,2,0,0], so final64updates explicitly train standing. Last observed: sitting55211 completed128updates and is in final evaluation; leaning55311 finishing stage2. Second seeds and baselines are queued. This snapshot is not a final policy result.

Read frozen `docs/SPLIT_SUPPORT_PROTOCOL.md` and leaning appendix `docs/SPLIT_LEANING_REPAIR.md`, plus each surface's protocol.json. Do not change frozen sources/rewards/contracts/budgets or add training. Sitting verifier is `scripts/verify_split_support.py`; leaning verifier `scripts/verify_split_leaning.py`. Necessary execution/report repairs must preserve originals and document recovery. Prior148distinct hashes preserved; sitting/leaning frozen definitions independent.

Each worker evaluates both final policies and old reference/corrected reference/fixed posture on28 full20s conditions:7known starts repeated at4simulator placements. Verify exact matched physical/sensor starts and full budgets/checkpoints. Physical failures and standing success first; effort on common full-duration successes only. Two seeds and known rigid surfaces do not establish arbitrary-surface generalization. Reference-assisted residual PPO is not unassisted motion discovery. Torque is a proxy, not energy/global optimum. No promotion.

Persistent tmux `split_support_export_02` runs `scripts/finalize_split_reports.py`; after both verifiers pass, it writes `final_export_01` with report, plots, replay, traces, checks and SHA256.json. Copy to Mac `validated_split_support_02/final_results`, verify hashes, visually inspect plots/replay, update handoffs and report the final comparison. Completion follow-up `finish-resting-ppo-comparison` is ACTIVE every30min, quiet during normal progress; pause it only after final comparison is delivered.

Mac validated preflight report: `validated_split_support_02/preflight_report_01/report.html`; replay beside it. Plots and both standing endpoint replays visually inspected. Local preview `http://127.0.0.1:8795/preflight_report_01/motion_replay.html` (server session7814); Codex browser tab2 is a deliverable. This is the preflight evidence, not final PPO results.

---

Historical preflight snapshot:

# Independent sitting / leaning — October 14, 2026 PDT

User authorized separate policies, contact/weight-transfer repair, progression toward standing, and comparison against reference motion. New root: `/data/humanoid-sitting-perception/split_support_curriculum_02`. Prior studies unchanged; 148 distinct protected artifact hashes verified.

Passive instrumentation exactly reproduces the previous reference startup, all trace arrays, and episode results. Diagnosis: sitting loses backward balance before contact; leaning reaches support but some pelvis contacts violate the 0.90 friction reserve. Full original physical criteria retained.

Selected sitting IMU ankle feedback and leaning coordinated +0.06 rad knee flexion pass14/14 full20-second trials each (7 known starts repeated twice). Corresponding old reference: sitting10/14 with4 physical failures, leaning6/14 with0 physical failures. Standing success improves from0/2 to2/2 in each surface. These are reference-controller improvements, not learned PPO results. Independent checker confirms durations, failure accounting, command envelopes/slew, reference offsets, finite traces and geometry audit structure.

Independent preflight workers run matched fixed baselines, indexed reset isolation, and two-update PPO smoke with full20-second evaluation. No full study PPO run has started as of this snapshot. Read separate `sitting_preflight_status.json` / `leaning_preflight_status.json` for live status. When preflight is independently verified, follow `SPLIT_SUPPORT_PROTOCOL.md`: seeds55211/55217 sitting onGPU0;55311/55317 leaning onGPU1; each128 updates and131072 transitions, starts[4,2,0,0] in32-update blocks, last64 updates fromstanding. No shared curriculum gates.

Final28 conditions per surface repeat7 known states four times at simulator placements. Compare old reference, corrected reference, fixed posture and final policies with exact matched physical/sensor starts, full durations, failures first, effort on common successes only. Two known rigid surfaces and repeated starts do not establish arbitrary-surface generalization. Torque is an effort proxy, not energy. No policy promotion.
