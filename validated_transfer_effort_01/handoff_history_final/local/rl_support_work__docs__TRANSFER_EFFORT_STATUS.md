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
