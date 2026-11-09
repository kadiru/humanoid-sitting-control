# Working commands

Run from `/home/kadir/humanoid-sitting` with GPU and HTTPS access.
The wrapper uses the checkout's locked extras, project caches, `HEADLESS=1`,
and the 4090 UUID. Keep renderer arguments: CUDA masking alone does not prove
Vulkan GPU selection. Driver and existing environments are unchanged.

## Installation and minimal simulation

```bash
bash scripts/setup.sh
bash scripts/run.sh python ../scripts/minimal_sim.py --device cuda:0 --info \
  --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'
```

## G1 infrastructure

```bash
bash scripts/run.sh isaaclab train --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-G1 --num_envs 128 --max_iterations 5 \
  --device cuda:0 --seed 42 --experiment_name g1_infrastructure --run_name smoke --info \
  --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false' \
  physics=isaacsim_physx

bash scripts/run.sh python ../scripts/verify_checkpoint.py

bash scripts/run.sh isaaclab play --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-G1 --num_envs 4 --device cuda:0 --seed 42 \
  --checkpoint /home/kadir/humanoid-sitting/IsaacLab/logs/rsl_rl/g1_infrastructure/2026-10-08_23-13-49_smoke/model_4.pt \
  --video --video_length 120 --info \
  --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false' \
  physics=isaacsim_physx
```

The verifier intentionally compares the recorded run's model 0 and model 4.
New training runs get new timestamped directories.

## Sitting pilot

Already executed; investigate failure metrics before repeating training.

```bash
bash scripts/run.sh isaaclab train --rl_library rsl_rl \
  --task Humanoid-Sit-Fixed-G1 --num_envs 128 --max_iterations 100 \
  --device cuda:0 --seed 42 --experiment_name g1_sitting --run_name pilot \
  --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'

bash scripts/run.sh python ../scripts/evaluate_sitting.py --device cuda:0 \
  --checkpoint /home/kadir/humanoid-sitting/IsaacLab/logs/rsl_rl/g1_sitting/2026-10-08_23-41-44_pilot/model_99.pt \
  --steps 600 --output sitting-pilot-evaluation

bash scripts/run.sh isaaclab play --rl_library rsl_rl \
  --task Humanoid-Sit-Fixed-G1 --num_envs 1 --device cuda:0 \
  --checkpoint /home/kadir/humanoid-sitting/IsaacLab/logs/rsl_rl/g1_sitting/2026-10-08_23-41-44_pilot/model_99.pt \
  --viz kit --video --video_length 400 \
  --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'

# Diagnostic authored seated start, not a training curriculum.
bash scripts/run.sh python ../scripts/evaluate_sitting.py --probe_seated \
  --steps 400 --output sitting-contact-probe-final --device cuda:0
```

The custom task directly configures Isaac Sim PhysX and needs no preset override.
The editable package registers its task through the `isaaclab.tasks` entry point.
Evaluation outputs are in project `logs/`; PPO runs/videos are in
`IsaacLab/logs/rsl_rl/`. Setup and run scripts retain the custom editable package.
