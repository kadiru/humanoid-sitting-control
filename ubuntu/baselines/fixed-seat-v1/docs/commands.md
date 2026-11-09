# Working commands — existing installation

Run from `/home/kadir/humanoid-sitting`. `scripts/run.sh` changes into IsaacLab,
uses the existing environment with `uv run --no-sync`, and selects the 4090 by UUID.
Do not run setup again. Use new output directories and new log filenames.

## Verified scripted seated hold

```bash
bash scripts/run.sh python ../scripts/scripted_sitting.py \
  --mode hold --pose ../runs/scripted_sweep_01/floor_supported_pose.json \
  --seconds 30 --video --output ../runs/new_scripted_hold --device cuda:0 \
  --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'
bash scripts/run.sh python ../scripts/plot_scripted.py ../runs/new_scripted_hold
```

Video recording automatically enables Kit camera/Replicator startup. No extra
package installation is needed. Root state is authored once at reset; subsequent
motion is ordinary gravity-driven PhysX and actuator control.

## Verified standing balance

```bash
bash scripts/run.sh python ../scripts/scripted_sitting.py --mode standing \
  --poses ../runs/standing_feedback_01/selected_poses.json --seconds 12 \
  --video --output ../runs/new_standing_balance --device cuda:0 \
  --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'
```

## Exact executed training chain

These commands were executed, each adding 20 updates in a separate timestamped
run. Repeating them creates new runs rather than modifying the named checkpoints.

```bash
bash scripts/run.sh isaaclab train --rl_library rsl_rl \
  --task Humanoid-Sit-Curriculum-0-G1 --num_envs 64 --max_iterations 20 \
  --device cuda:0 --seed 42 --experiment_name g1_seated_curriculum --run_name seated_stability

bash scripts/run.sh isaaclab train --rl_library rsl_rl \
  --task Humanoid-Sit-Curriculum-1-G1 --num_envs 64 --max_iterations 20 \
  --device cuda:0 --seed 42 --experiment_name g1_seated_curriculum --run_name near_seat \
  --checkpoint logs/rsl_rl/g1_seated_curriculum/2026-10-09_00-06-27_seated_stability/model_19.pt

bash scripts/run.sh isaaclab train --rl_library rsl_rl \
  --task Humanoid-Sit-Curriculum-2-G1 --num_envs 64 --max_iterations 20 \
  --device cuda:0 --seed 42 --experiment_name g1_seated_curriculum --run_name partial_descent \
  --checkpoint logs/rsl_rl/g1_seated_curriculum/2026-10-09_00-10-58_near_seat/model_38.pt
```

The stage 0/1 runs preceded the final thumb-limit/clean-contact tightening; both
checkpoints were evaluated against the subsequent configuration. Source snapshots
are in `logs/source-snapshots/`; runs also contain serialized environment/agent configs.

## Checkpoint evaluation and visualization

```bash
bash scripts/run.sh python ../scripts/evaluate_curriculum.py --stage 2 \
  --num_envs 32 --seconds 12 --device cuda:0 \
  --checkpoint logs/rsl_rl/g1_seated_curriculum/2026-10-09_00-20-27_partial_descent/model_57.pt \
  --output ../runs/new_stage2_evaluation
bash scripts/run.sh python ../scripts/plot_scripted.py ../runs/new_stage2_evaluation

# Representative rendered evaluation: use a separate output directory.
bash scripts/run.sh python ../scripts/evaluate_curriculum.py --stage 2 \
  --num_envs 1 --seconds 12 --device cuda:0 --video \
  --checkpoint logs/rsl_rl/g1_seated_curriculum/2026-10-09_00-20-27_partial_descent/model_57.pt \
  --output ../runs/new_stage2_video \
  --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'
bash scripts/run.sh python ../scripts/plot_scripted.py ../runs/new_stage2_video

bash scripts/run.sh python ../scripts/verify_curriculum_checkpoint.py \
  logs/rsl_rl/g1_seated_curriculum/2026-10-09_00-20-27_partial_descent \
  --reference logs/rsl_rl/g1_seated_curriculum/2026-10-09_00-10-58_near_seat/model_38.pt
```

For stages 0/1 substitute the stage and corresponding checkpoint. Zero residual
baseline evaluation omits `--checkpoint`. No autoreseated success is credited:
completed-episode metrics use measurements before resets; unfinished episodes are
not included in the denominator.

## Promotion-checked bounded training

Prefer this entrypoint for subsequent stage advancement. The prior evaluation
must pass its promotion gate; stage 3 currently rejects the stage 2 result.
To continue refining stage 2, its prior stage 1 result is already passing:

```bash
bash scripts/run.sh python ../scripts/train_curriculum.py --stage 2 \
  --iterations 20 --num-envs 64 \
  --checkpoint logs/rsl_rl/g1_seated_curriculum/2026-10-09_00-20-27_partial_descent/model_57.pt \
  --previous-evaluation ../runs/curriculum_1_policy_eval_01/results.json
```

Re-evaluate the new checkpoint and inspect contacts/video before advancing.
There is no automatic full-budget or furniture expansion.

## Validated complete transition (2026-10-09)

No installation required. Run from `/home/kadir/humanoid-sitting`.

```bash
# Whole-trial scripted reference; 54 perturbed starts
bash scripts/run.sh python ../scripts/evaluate_full_transition.py --device cuda:0 \
  --output ../runs/new_scripted_full_evaluation

# Each of the executed three training seeds: 7, 42, 123
bash scripts/run.sh isaaclab train --rl_library rsl_rl \
  --task Humanoid-Sit-Full-Transition-G1 --num_envs 64 --max_iterations 40 \
  --device cuda:0 --seed 7 --experiment_name g1_full_transition --run_name seed_7 \
  --checkpoint logs/rsl_rl/g1_seated_curriculum/2026-10-09_00-20-27_partial_descent/model_57.pt

# Selected checkpoint, fresh initialization seeds
bash scripts/run.sh python ../scripts/evaluate_full_transition.py --device cuda:0 \
  --checkpoint /home/kadir/humanoid-sitting/IsaacLab/logs/rsl_rl/g1_full_transition/2026-10-09_10-08-57_seed_7/model_96.pt \
  --seeds 2024 2025 2026 --output ../runs/new_full_holdout

# Complete nominal 20-second video, explicitly renderer GPU 0
bash scripts/run.sh python ../scripts/evaluate_full_transition.py --device cuda:0 \
  --checkpoint /home/kadir/humanoid-sitting/IsaacLab/logs/rsl_rl/g1_full_transition/2026-10-09_10-08-57_seed_7/model_96.pt \
  --video --output ../runs/new_full_video \
  --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'

# Preserved seated policy; recorded settled initialization, original gains
bash scripts/run.sh python ../scripts/evaluate_full_transition.py --device cuda:0 \
  --seated-baseline --seated-initialization settled \
  --checkpoint /home/kadir/humanoid-sitting/IsaacLab/logs/rsl_rl/g1_seated_curriculum/2026-10-09_00-06-27_seated_stability/model_19.pt \
  --output ../runs/new_seated_comparison

# State/contact plots (also extracts video frames when available)
bash scripts/run.sh python ../scripts/plot_scripted.py ../runs/new_full_video
```

Commands above reproduce the tested task and bounds. Use new output/run names to
preserve historical evidence. Previous stage-3 failures above are historical;
the new full-transition task has a separate validated reference and reset.
