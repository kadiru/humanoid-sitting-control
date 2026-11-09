# Leaning approach correction revision — October 14, 2026 PDT

This appendix applies to the separate leaning worker only. The sitting study was already frozen and started; its sources, budget and behavior remain unchanged.

The first leaning reference passed14/14, but the two-update PPO smoke passed12/14. All14 trials stayed free of physical failures and both standing starts succeeded. The two failed trials were already-supported starts at stages5/6; the added knee bend pushed pelvis contact to friction utilization about1.01. They remain failures under the unchanged0.90 criterion, and all outputs/checkpoints are preserved.

Repair: use the same0.06rad coordinated knee/hip/ankle correction for approach starts0–4 only. Starts5/6 use the original reference without the extra bend. This avoids unnecessary reconfiguration of already-supported postures. The start-stage variable is an explicit offline motion-library request, not live simulator pose/contact information. Reference/phase inputs already expose the actual requested motion.

New source `scripts/run_split_leaning.py` is a preserved fork of `run_split_support.py`; only the bend activation condition and its profile parameter differ. Profile `leaning_selected_02.json` adds `knee_start_max:4`. No reward, physical success/failure definition, PPO budget, residual bound, geometry or observation dimensions change. Separate preflight and verifier files account for this known reference difference. The original two-update smoke remains an unsuccessful development result, not silently replaced.

Repeat full14-start reference and optimizer-smoke evaluation, plus indexed reset isolation. Reuse the already matched original-reference and constant fixed-posture baselines, whose behavior is unaffected by this correction. Require exact physical/sensor startup matches against those baselines. Strict14/14 corrected-reference and14/14 learned-smoke preflight criteria remain unchanged.

Development optimizer accounting is now three2-update runs: sitting2048, original leaning2048, repaired leaning2048 =6144 transitions. These are separate from the four study policies' total524288 transitions. There is no extra study training. Seeds, stages[4,2,0,0], budgets and final comparisons remain as specified in SPLIT_SUPPORT_PROTOCOL.md. The two surfaces' training and verification workers are independent.
