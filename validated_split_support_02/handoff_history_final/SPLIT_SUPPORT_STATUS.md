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
