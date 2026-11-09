# Both PPO runs completed; matched evaluations running

Verified October11,2026 at18:30 PDT (October12 at01:30 UTC). This is a status
snapshot, not a final evaluation report. Read remote status.json for live status.

Both old-reward and new-reward arms completed128 updates /131,072 transitions
each, taking about37.6 minutes each. Final checkpoints exist; actor/critic
parameters are finite, normalization remains unchanged, starting weights match,
and completed plus unfinished attempts account for every training transition.
Frozen source hashes still match. No model was promoted.

Preliminary statistics across exploratory training attempts:

| Metric | Old reward | New reward |
|---|---:|---:|
| Successful training attempts |44/205 (21.5%)|37/212 (17.5%)|
| Mean peak force |989.7N|982.4N|
| Median peak force |784.5N|864.4N|
| 95th percentile peak force |2003.5N|2125.0N|
| Mean seated leg RMS on successful attempts |4.32Nm|4.68Nm|

These numbers do not show a clear benefit from the new reward. They are not
matched final-policy evaluations: PPO explores, changes throughout training,
and takes different reset/failure paths. Do not treat the training success
percentages as the final policy's sitting success rate. Compare final policies
on the same initial conditions before drawing a conclusion.

The persistent coordinator finished both training jobs and started novel_initial,
the starting-policy reference on a new bank. It will then evaluate both trained
policies on four matched16-case banks:64 full20-second attempts per policy.
The report will retain failures and compare force, success, and leg effort among
common successful cases, using identical physical criteria.

- Protocol: [SUPPORT_REWARD_STUDY_PROTOCOL.md](SUPPORT_REWARD_STUDY_PROTOCOL.md).
- Session: support_reward_study_01 on Ubuntu, controlled directly through SSH.
- Remote outputs: /data/humanoid-sitting-perception/support_reward_study_01.
- Live status: remote output directory/status.json.
- Saved evidence: remote output directory/training_snapshot.json.
- Expected final comparison: remote output directory/results/REPORT.md.
- Mac evidence: /Users/kadir/humanoid-sitting-control/validated_support_reward_study_01/training_snapshot.json.
- Prior pilot: [SUPPORT_SPLIT_PPO_STATUS.md](SUPPORT_SPLIT_PPO_STATUS.md).

Perception, runtime sensor inputs, motor limits and original checkpoints remain
unchanged. No additional experiment was launched during this status check.
