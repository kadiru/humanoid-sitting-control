# Retention study status

Complete. Both matched 32,768-transition PPO continuations pass 15/16 known easier full trials, versus the preserved parent's 16/16. Both fail the retention gate; neither is promoted. No new wider or fresh holdout was used. No jobs need to continue running.

Four new unit tests, exact parent trace reproduction, transition accounting, frozen-normalizer and preservation checks pass. The legacy root-disk reserve warning remains documented separately. See `SUPPORT_RL_LATEST.md` for results and the concrete next task: audit landing-time visibility/rejection reasons and robustness before more training.
