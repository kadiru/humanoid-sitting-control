# Humanoid sitting

Installation and G1 infrastructure validation **passed** on 2026-10-08.
The custom sitting task executes, but its first bounded pilot **did not learn
to sit**. Driver 580.178.04 and existing environments were preserved.

## Installed stack

- Isaac Lab `release/3.0.0`, commit `be9551260b269d4d8a964b71d5849810a1237b33`.
- Project-local Python 3.12.15; Isaac Sim 6.1.0.0 with full/extscache integration.
- PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, RSL-RL 5.5.1.
- Checkout indexes, overrides and unchanged `uv.lock`; separate `video` extra.
- Exact versions: [resolved-packages.txt](logs/resolved-packages.txt).
  Commit/hashes: `logs/isaaclab-commit.txt`, `logs/dependency-hashes.txt`.
  Installation and periodic disk-space logs are in `logs/`.

GPU/network access works with approved execution outside Codex's sandbox.
PhysX selected CUDA device 0; Vulkan playback marked the RTX 4090 active and
RTX 4070 inactive. Both worked with the existing driver. The run wrapper selects
GPU 0 by UUID and configures project caches. No system components were changed.

## Verified execution

1. PhysX cube/floor check: 120 steps; final cube center height 0.0999996 m.
2. Registered `Isaac-Velocity-Flat-G1`: 128 environments, five PPO updates,
   15,360 environment transitions. Actor/critic weights changed and remained
   finite. Final checkpoint `model_4.pt`: 2,027,031 bytes.
3. Checkpoint loaded for bounded inference and 120-frame rendered playback.
   Inspected frames show falling/resetting, consistent with a smoke test.

Infrastructure checkpoint:
`IsaacLab/logs/rsl_rl/g1_infrastructure/2026-10-08_23-13-49_smoke/model_4.pt`.
Evidence: `logs/minimal-sim-result.json`, `logs/g1-train.log`,
`logs/checkpoint-verification.json`, `logs/g1-evaluation.json`, `logs/g1-play.log`.
Video is under that run's `videos/play/`; extracted frames are
`logs/g1-play-frames.jpg`.

## Fixed-seat task and pilot

Separate package: [sitting_task](sitting_task/pyproject.toml), registered task
`Humanoid-Sit-Fixed-G1`. Full G1 collision asset, floating base, known aligned
standing reset, joint targets through configured actuators. Kinematic anchored
seat: 0.50 × 0.90 × 0.10 m, center (-0.40, 0, 0.25), top 0.30 m.
Arms initially clear the seat. Observations contain proprioception, relative
seat position and numerical dimensions; no policy camera inputs.

The source USD has 38 collision meshes. The minimal locomotion asset has only
foot/torso colliders and is unsuitable for sitting contact rewards. Full-asset
pelvis convex hull extends 0.152 m below its body origin; thigh geometry is on
`left_hip_yaw_link` and `right_hip_yaw_link`. Source geometry is recorded in
`logs/g1-full-collision-geometry.json`. Each supporting body has a separate
seat-filtered contact sensor, matching PhysX's single-body filtering requirement.

Success requires one continuous second of pelvis/thigh seat support >25% of
robot weight, pelvis over the seat at seated height, upright posture, low
linear/angular velocity, and descent never faster than 0.8 m/s. Seat contact
and lower pelvis height are allowed; falls and pelvis-floor contact terminate.
Rewards include target pose, support, hold, descent, orientation, smoothness,
joint limits and torque costs. This reward design is experimental.

Contact validation: four authored seated starts ran eight seconds without a
fall or pelvis-floor termination. Seat load was measured, but longest qualifying
hold was only 0.02 s. Earlier failed probes are retained for diagnosis.
This validates contact sensing; it is not a learned sitting transition.

Pilot: 128 environments × 100 PPO iterations (307,200 transitions), 50.92 s
training. Evaluation: 16 environments × 600 steps; **0/736 completed episodes
successful**, 736 fall terminations, no qualifying hold and no measured
pelvis/thigh seat support. Sixteen unfinished episodes are censored.
The checkpoint loads, but the policy fails before reaching the seat.

Pilot checkpoint:
`IsaacLab/logs/rsl_rl/g1_sitting/2026-10-08_23-41-44_pilot/model_99.pt`.
[Evaluation metrics](logs/sitting-pilot-evaluation.json), synchronized pre-action
state/action/contact records: `logs/sitting-pilot-evaluation-rollouts.npz`.
Proxy-array quaternions use x,y,z,w order. Pilot video is `videos/play/clip_model_99_0001.mp4` under the run;
representative frames are `logs/sitting-pilot-frames.jpg`.
No further training budget or furniture variation has been started.

## Resume

```bash
cd /home/kadir/humanoid-sitting
bash scripts/setup.sh
```

Use a terminal with GPU and outbound HTTPS access. Setup installs project-local
uv/Python/environment, preserves the lock, installs the custom editable package
without replacing Lab dependencies, and records versions. No activation needed.
[Exact working commands](docs/commands.md) cover simulation, training and playback.

About 58 GiB remains free on root; cache files share hardlinks with the
environment. `/data` was not used. Only experiment-referenced assets were
fetched. Cosmos, full asset packs and large datasets were not installed.
First rendered launch compiled shaders for about four minutes; later launches
reused the cache. Startup/shutdown warnings remain in the logs, but the verified
runs exited successfully.

Next: diagnose short falling episodes and improve balance/transition shaping
before another pilot. Approach, varied furniture, perception, couch compliance,
Cosmos and real-robot deployment remain later stages. No successful sitting
policy is claimed.

Official references:
- https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/setup/installation/index.html
- https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html
