# RL reward study is running

October 11, 2026, approximately 17:16 PDT. This is a launch-time snapshot;
read the remote status file for current progress. The user explicitly authorized
continuing RL. Earlier pause instructions are historical and superseded.

The persistent Ubuntu tmux session `support_reward_study_01` is training two
PPO policies, sequentially on GPU0. Each gets128 updates /131,072 transitions
from the previous151-input PPO checkpoint, using identical training seeds and
fresh optimizers. The first arm strengthens peak-impact and supported leg-effort
rewards; the second retains the old reward. Perception, runtime inputs, motor
limits, and physical success criteria are frozen.

Verified so far: six reward tests passed; first three PPO updates completed;
3,072 new transitions; initial actor/critic values exactly match the checkpoint;
first saved optimizer update changed actor parameters by up to0.000360675;
parameters are finite and normalization is unchanged. No outcome improvement
has been established yet. No checkpoint or default was replaced.

The coordinator automatically runs both final policies on four matched16-case,
20-second evaluation banks and evaluates the starting PPO on the new bank.
It reuses the starting policy's archived results on the other three banks,
checks exact startup/geometry matching, accounts for every training transition,
and writes a physical-metric comparison report. Final checkpoints only; no
selection of favorable intermediate checkpoints. All failures remain included.

- Protocol: [SUPPORT_REWARD_STUDY_PROTOCOL.md](SUPPORT_REWARD_STUDY_PROTOCOL.md).
- Coordinator: scripts/run_support_reward_study_01.py.
- Session: support_reward_study_01.
- Study directory: /data/humanoid-sitting-perception/support_reward_study_01.
- Current job/status: study directory/status.json.
- Learning progress: study directory/train_gentle/progress.json, then train_legacy/progress.json.
- First-update evidence: study directory/first_update_verification.json.
- Expected report after completion: study directory/results/REPORT.md.
- Launcher log: /home/kadir/humanoid-sitting/logs/support_reward_study_01.log.

Estimated total duration at launch: roughly1.5–2 hours, subject to simulation
speed. The remote queue survives SSH disconnection. On any error it records
failure and stops, preserving all outputs. No second agent or scheduler was
started. No software was installed.

Previous completed study: [SUPPORT_SPLIT_PPO_STATUS.md](SUPPORT_SPLIT_PPO_STATUS.md).
Its interface validation and35/48→37/48 pilot results remain valid; the pilot
checkpoint was not promoted because some impacts worsened. This is a reason
to measure the next experiment carefully, not to postpone RL.
