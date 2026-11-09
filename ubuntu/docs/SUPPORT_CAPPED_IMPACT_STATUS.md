# Capped-impact-only PPO study running

October 11, 2026, 22:16 PDT (October 12 UTC). The user authorized fixing the timer,
four matched32-update PPO jobs (old versus capped impact, two training seeds),
and comparison against the47/64 starting policy. They also requested a fresh-start
RL design; that explanation is saved in SITTING_RL_FRESH_APPROACH.md.

The new training/evaluation path uses int64 supported-hold ticks. Six tests pass,
including CPU/GPU timer boundary/reset checks and cap/reward-isolation tests.
The full timer regression changes only the known case13 from failure to success
at exactly150 ticks: the prior new-reward policy scores10/16 instead of9/16 on
its novel bank. Controller inputs/actions, all startup sensors, poses, kinematics
and physical outcomes match the archived run exactly. Force labels differ by one
float32 unit in three50Hz samples (one10Hz sample), max0.00003052N. The original
strict parity stop, its source, and a narrow documented recovery are preserved.
This exception affects force-label parity only, not success criteria or control.

Training seed52813/old reward is running now. The persistent queue will execute
both old/capped arms for seeds52813 and52817,32 updates each, from the preserved
151-input47/64 checkpoint. The capped arm changes ONLY the impact term; total
additional peak cost cannot exceed5 per episode. Other reward terms, perception,
architecture, motion scaffold and motor limits remain unchanged.

On194 archived admitted training attempts, the proposed peak cost totals706.76
instead of23272.99, with maximum5. This verifies its scale; it is not evidence
of improved learned behavior. Matched policy results are still pending.

Each of four final policies will get80 full20-second tests: the four known
regression banks plus fresh holdout52841/52843. All attempts are retained. The
starting policy is evaluated on the new holdout and archived baselines are
transparently rescored using integer ticks. No checkpoint or seed selection.

- Session: support_capped_impact_01, running on Ubuntu through Mac SSH.
- Study: /data/humanoid-sitting-perception/support_capped_impact_01.
- Live status: study/status.json; learning: study/train_<seed>_<arm>/progress.json.
- Protocol: SUPPORT_CAPPED_IMPACT_PROTOCOL.md.
- Recovery: scripts/resume_support_capped_impact_01.py.
- Parity amendment: SUPPORT_CAPPED_IMPACT_PARITY_AMENDMENT.md.
- Validation: study/timer_validation.json and archived_cost_audit.json.
- Expected report: study/results/REPORT.md.
- Completion follow-up: same-chat automation finish-capped-impact-ppo-comparison,
  every30 minutes, quiet during normal progress, final comparison then pause.
- Prior result/diagnosis: SUPPORT_REWARD_STUDY_COMPLETED_STATUS.md and
  /data/humanoid-sitting-perception/support_reward_study_01/diagnosis_01/REPORT.md.

Expected total duration is about2–2.5 hours. Long processes persist in tmux;
no additional Codex agent runs on Ubuntu. No packages installed or deployment
defaults changed. Original experiment code and checkpoints remain preserved.
