# Longer PPO reward comparison — October 11, 2026

User explicitly authorized continuing RL. Perception work is frozen. This study
compares more training against a changed reward; it does not require perfect
perception or change deployment defaults.

Both arms start from support_split_ppo_train_01/final.pt (151 inputs), SHA256
ece102b139323af18bd1c7aad6d8fb06e478f38946f07e52406a31bd4acc7dda.
Actor, critic and normalizer values are copied exactly. Both optimizers are fresh.
All existing files and checkpoints remain intact. No package installation.

- Arms: legacy reward; gentle reward. Gentle runs first; GPU jobs are sequential.
- Each: 128 updates × 64 steps × 16 robots = 131,072 new transitions.
- Learning rate 3e-5, 3 epochs, 4 minibatches, zero entropy coefficient, zero
  parent KL penalty, frozen normalization; all inherited physical action limits.
- Same mixed 16-seat training bank, geometry seed51621, training seed52713.
- Original nominal scaffold, 151 sensor-only inputs, validated depth/kinematic
  tracker and camera timing remain unchanged. This is residual PPO, not learning
  a complete motion from scratch.
- Training stops each attempt on its first failure; full evaluations run20s.
- Use final checkpoints only, without choosing favorable intermediate results.

The gentle reward replaces two old terms. Its episode impact cost is
`-12 * max_t(max(force/bodyweight - 1, 0)^2)`, applied incrementally whenever the
peak grows. Thus a brief spike incurs its full cost, independent of how many
control samples it occupies. Peak history is reset per robot and used only for
the reward. Stable, upright, correctly supported sitting earns leg economy
`4*exp(-(leg_RMS_Nm/10)^2)` per second. This replaces the old weakly varying
normalized-effort term. All other rewards, physical failures, three-second hold
requirement, and admission criteria remain unchanged. This compares the two
reward changes together; it does not isolate their separate contributions.

Evaluate both final policies on known51403/51407, fresh52641/52643, mixed-bank
new starts52623, and a new broad bank52741/52743:64 attempts per policy. Run the
starting151-input policy on the new bank; reuse its archived deterministic paired
evaluations on the other three banks. Check startup depth/proprioception and
geometry equality. Retain refused and failed attempts. Existing original
controller scores35/48 and starting PPO37/48 are historical benchmarks on the
first48 cases. The new bank has not been used for training or reward selection.

Report success counts, gained/lost cases, force maxima/p95, stable hold, common
successful-case leg torque and foot load, and all failure types. Reward values
across training arms are not directly comparable; evaluate both with the legacy
reward and identical physical metrics. Log individual reward contributions,
finite parameters, normalization invariance, transition accounting, and source
hashes. Single training seed and small evaluation banks are exploratory evidence.
Tracking availability remains a diagnostic, not a reason to postpone this study.

Persistent tmux session: support_reward_study_01. Coordinator script:
scripts/run_support_reward_study_01.py. Outputs:
/data/humanoid-sitting-perception/support_reward_study_01.
Each training/evaluation has its own new subdirectory, command record and log.
Coordinator stops on error and preserves artifacts. More than30GiB free is
required by the existing entrypoint. No automatic checkpoint promotion occurs.
