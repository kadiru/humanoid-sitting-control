# Active independent-distance PPO task

User explicitly authorized fixing the interface, validating it, running a small
PPO pilot, and comparing it against the current controller. This supersedes the
previous blanket RL pause. Work is executed directly from Mac via SSH; no other
agent, reinstall, checkpoint overwrite, gain tuning or default change.

Protocol: SUPPORT_SPLIT_PPO_PROTOCOL.md. All new output directories live under
/data/humanoid-sitting-perception on Ubuntu.

Completed before training:
- Three new unit tests, including exact CPU/GPU actor/critic migration and
  gradient flow to new weights: pass.
- support_split_replay_01: fresh archived sensor replay; old patch features
  bitwise identical; zero accepted scalar estimates lost;271 readings recovered
  while patch unavailable;99.7546% before-failure availability,max4.043mm error.
- support_split_regression_parent_01:16/16; all12 archived rollout arrays,
  all144 original inputs and actions bitwise identical to continuous_control_dev.
  Actual policy landing-distance availability99.6711%,versus82.7303% patch;
  all-command availability99.8063%,max1.738mm error,225 extra valid distance frames.

Runtime: original144 inputs plus seven independent distance/normal/validity/age/
budget inputs. New first-layer columns zero initially; SplitInputLinear retains
old GEMM shape. All normalizers fixed. Original runtime tracker and patch
transport unchanged. New entry point train_support_split_observation.py handles
both full-horizon evaluation and independent-reset training.

Persistent validation: tmux support_split_validation_01; launcher
scripts/run_support_split_validation_01.sh; exitfile logs/support_split_validation_01.exit.
Validation finished with exit0. Indexed-reset probe passed; all unaffected robot
state deltas were exactly zero. Subsequent support_split_online_replay_01 also
passed: every consumed post-warmup packet (15984) matched, including all seven
new channels. No accepted distance estimate was lost.

scripts/run_support_split_ppo_01.sh is now running in tmux support_split_ppo_01.
Training has passed its midpoint (17/32 updates at this note); check current logs.
It requires validation exit0, replays the actual new-interface archive from
sensors, freezes source hashes, then trains32x64x16=32768 transitions from the
preserved parent. It evaluates the final checkpoint and frozen parent on three
matched banks. Seeds, exclusions, promotion criteria and no-sweep rule are in
the protocol. Report builder: scripts/report_support_split_ppo.py.

Read actual logs/status before continuing; this document is a recovery pointer,
not evidence that queued training or comparisons have completed.
