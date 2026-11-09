# Independent distance interface and PPO pilot: completed

October11,2026. The user authorized fixing the observation interface, validating
it, running one small PPO experiment, and comparing it with the current
controller. All work is complete. Both tmux jobs exited0; no jobs remain.
Continue directly from the Mac over SSH to kfuUbuntu; no other agent was started.

## Outcome

Reliable clearance now has independent validity, age and error-budget channels.
The original144-channel input remains intact; seven fields are appended:
144 clearance_m;145–147 normal_P;148 distance_valid;149 measurement_age_s;
150 conditional_budget_m. Original channel140 still means complete-patch valid.
Unknown/expired distance is explicitly [0,0,0,0,0,1,.02], never silently held.

The migrated parent starts with zero new input weights and retains the old
first-layer GEMM shape. Known-bank actions, old inputs and all12 trajectory
arrays matched the preserved continuous-tracking controller bitwise:16/16.
Known landing availability improved82.73%→99.67%,max accepted error1.74mm.
Three unit tests passed, including CPU/GPU migration and new-weight gradients.
Archived sensor replay lost zero accepted distance readings and recovered271
readings hidden by missing patch edges. Actual new-interface sensor-only replay
matched15,984 consumed post-startup packets exactly. Indexed resets preserved
other robots' states exactly and cleared the reset robot's tracker.

## One fixed PPO run and96 evaluation attempts

32 updates×64 steps×16 robots=32,768 transitions;3e-5 learning rate,3 epochs,
4 minibatches, fresh optimizer, no parent-KL penalty, frozen normalizers. Existing
reward, nominal scaffold,23 bounded residual actions, gains and physical criteria
unchanged.44 completed training attempts plus partial attempts account for every
transition. Final checkpoint evaluated without intermediate checkpoint selection.

| Bank | Parent | PPO | Worst force parent→PPO |
|---|---:|---:|---:|
| Known51403/51407 |16/16|16/16|520.50→616.46N|
| Fresh52641/52643 |12/16|13/16|1078.07→1176.08N|
| Mixed training bank/new starts52623 |7/16|8/16|6447.91→6447.91N|
| Total |35/48|37/48| |

No lost successes. Gained fresh case13 and new-start case2. All96 evaluations run
the full20s, retaining failures/refusals. Startup depth, K, encoders, IMU gravity
and other startup archive fields match bitwise across each paired bank.

**Do not promote this checkpoint:** worst impact increased on two banks. Among
common successful cases, seated leg RMS increased0.112/0.199/0.132Nm in the three
banks. Fresh case13 improved hold2.68→5.14s and mean foot load66.0%→45.4%, but its
torque rises again late in the trial. Sustained low-effort sitting is not proven.

New-start case15 is refused at startup in both policies because the landing
patch is unobservable; it accounts for the identical6448N maximum. It is retained
in the main score, with admitted-only diagnostics clearly secondary.

Distance accepted errors stay below7.65mm across all96 evaluations. Landing
availability96.31–99.92%. Hard-bank before-failure coverage remains98.88% parent /
98.94% PPO, below the99% diagnostic threshold; these failures remain recorded.
The interface cannot recover a measurement that the tracker/startup detector
itself rejects. No invalid estimate is forced valid to pass a score.

The new input weights are nonzero, but clearance-only sensitivity remains weak:
a1cm change produces at most3.0e-7rad bounded correction change on recorded
training inputs. This pilot does not isolate the benefit of extra channels from
PPO using existing inputs; do not attribute the two gains specifically to them.

## Sources, outputs and reproduction

- Pure contract/migration: sitting_task/src/humanoid_sitting/support_split_observation.py.
- Runtime/training: scripts/train_support_split_observation.py.
- Tests: scripts/test_support_split_observation.py.
- Protocol: docs/SUPPORT_SPLIT_PPO_PROTOCOL.md.
- Validation launcher: scripts/run_support_split_validation_01.sh (finished).
- PPO/comparison launcher: scripts/run_support_split_ppo_01.sh (finished).
- Runtime/source freeze: /data/humanoid-sitting-perception/support_split_ppo_frozen_01.
- Report: /data/humanoid-sitting-perception/support_split_ppo_results_01/REPORT.md.
- Mac report: /Users/kadir/humanoid-sitting-control/validated_support_split_ppo/results/REPORT.md.
- Raw evaluations: /data/humanoid-sitting-perception/support_split_{regression,fresh,starts}_{parent,trained}_01.
- New checkpoint: /data/humanoid-sitting-perception/support_split_ppo_train_01/final.pt.
  SHA256 ece102b139323af18bd1c7aad6d8fb06e478f38946f07e52406a31bd4acc7dda.
  Requires the151-channel entry point and split first-layer computation.
- Preserved parent: /data/humanoid-sitting-perception/support_rl_independent_train_04/final.pt.
  SHA256 2c0f6907196be68bce9a1ee2e6b6ef6e4c82bf0ecdff622dc55b81fe8ef58d13.

No checkpoint was overwritten; no software installed; no runtime defaults changed.
All28 frozen source/model hashes verified unchanged; IsaacLab remains clean.
Plots and originals, all paired results, reset/replay checks, training accounting,
post-hoc functional audit and source snapshots accompany the report.

Scope remains initially aligned rigid seats in simulation. No new camera-noise,
hardware, navigation or cushion validation. Tracker inputs are clean simulated
depth/encoder/IMU equivalents; the original109-channel packet retains its noise
and delay. Geometry confidence is a conditional engineering budget.

The interface fix is complete; the remaining learning issues are strong use of
clearance, impact/effort tradeoffs and robustness. Retain the current controller
while those are investigated. Earlier camera-timing work is preserved in
SUPPORT_CONTROL_PRE_SPLIT_ARCHIVE.md and SUPPORT_CONTROL_TIMING_STATUS.md.
