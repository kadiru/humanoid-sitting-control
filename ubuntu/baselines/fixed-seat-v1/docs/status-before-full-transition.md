# Humanoid sitting

**A physically stable scripted seated hold is now verified.** G1 held the fixed
seat in a 30-second simulation, including a continuous 28-second post-settle
hold under gravity, using its normal joint actuators and a free
floating base. No root constraint, external support force, or gravity override
was used. The installation and NVIDIA driver were preserved; nothing was reinstalled.

The pelvis carried **76.2% of body weight** and the feet carried the remainder
on the floor. Measured total upward contact force was 316.34 N versus 316.26 N
weight. No hand support, knee wedging, or foot-seat contact was measured.
After two seconds of settling, speed stayed below 0.001 m/s, angular speed
below 0.0091 rad/s, and upright cosine above 0.9805. Peak descent was 0.3145 m/s.

[Physics/results](runs/scripted_hold_clean_30s/results.json) ·
[Contact validation](runs/scripted_hold_clean_30s/contact_validation.json) ·
[Video](runs/scripted_hold_clean_30s/scripted_0000.mp4) ·
[Frames](runs/scripted_hold_clean_30s/frames.jpg) ·
[Physics plot](runs/scripted_hold_clean_30s/physics.png)

## Diagnosis and fixes

The original standing reset fell even with constant joint targets. A scripted
ankle controller using center-of-mass position and forward velocity now holds
standing for 12 seconds with both feet supporting the weight. Earlier seated
poses were too far forward; other seemingly stable candidates wedged their
knees against the seat underside. Those candidates were rejected after inspecting
correctly ordered body contact forces, filtered seat sensors, and rendered frames.

Validated seated targets: hip pitch −1.25, knee 1.40, ankle pitch −0.15 rad on
both legs; authored root (−0.34, 0, 0.48) m, settling to approximately
(−0.3769, 0, 0.4493) m. Seat top is 0.30 m. The full G1 collider asset and
anchored rigid seat remain in use. Thumb defaults/targets respect their actual
asymmetric limits. All previous checkpoints, logs, probes and source snapshots
were retained.

## Executed staged PPO pilots

New tasks `Humanoid-Sit-Curriculum-{0,1,2,3}-G1` share observations and 37 joint
actions. PPO learns bounded ±0.08 rad residuals around a scripted joint reference.
Observations include numerical seat information, proprioception, goal joint error
and reference phase. Later stages load the preceding checkpoint. Reset variation
is deliberately small: ±3 mm horizontal position and ±0.01 rad joint offsets.

| Stage | Initial condition | Pilot | Evaluation | Promotion |
|---|---|---|---|---|
| 0 | Seated stability | 64 envs × 20 PPO updates | 64/64 successful | Passed |
| 1 | Near-seat descent | 20 additional updates | 64/64, including ≥3 s hold each | Passed |
| 2 | Partial descent from 0.52 m pelvis height | 20 additional updates | 63/64 with ≥3 s hold; one knee-seat collision | **Held** |
| 3 | Standing-to-sitting | Scripted prototype; no PPO training | 0/12 scripted diagnostic episodes successful | Gated |

These are residual policies around already useful scripted references. Stages 0
and 1 also passed with zero residuals; this is **not evidence that PPO learned a
complete standing-to-sitting skill**. Stage 2's scripted baseline succeeded in
62/65 episodes; its checkpoint succeeded in 63/64 in a separate bounded test.
This is a small single-seed comparison, not a statistically established improvement.

Promotion requires ≥32 completed episodes, ≥90% with a continuous three-second
hold, and **zero** fall, pelvis-floor or invalid seat-contact terminations.
The historical `foot_seat` termination name now includes knee-seat and arm
support checks. Fast descent (>0.8 m/s) disqualifies success. Full standing
trajectory sweeps reached seats but exceeded that descent limit and contacted
the seat with knees, so none is accepted as a successful transition.

Checkpoints (actor/critic updates verified finite; all loaded for evaluation):

- `IsaacLab/logs/rsl_rl/g1_seated_curriculum/2026-10-09_00-06-27_seated_stability/model_19.pt`
- `IsaacLab/logs/rsl_rl/g1_seated_curriculum/2026-10-09_00-10-58_near_seat/model_38.pt`
- `IsaacLab/logs/rsl_rl/g1_seated_curriculum/2026-10-09_00-20-27_partial_descent/model_57.pt`

Each pilot added 30,720 environment transitions. Evaluation state/action/contact
arrays, metrics, videos and plots are in `runs/curriculum_*`; training logs are
in project `logs/`. Episode terminal contact snapshots identify collisions before
automatic resets. [Stage 2 metrics](runs/curriculum_2_policy_eval_01/results.json).

## Resume without installation

```bash
cd /home/kadir/humanoid-sitting
bash scripts/run.sh python ../scripts/evaluate_curriculum.py \
  --stage 2 --num_envs 32 --seconds 12 --device cuda:0 \
  --checkpoint logs/rsl_rl/g1_seated_curriculum/2026-10-09_00-20-27_partial_descent/model_57.pt \
  --output ../runs/my_stage2_evaluation
```

Output directories must be new; existing evidence is never overwritten. The run
wrapper uses `uv run --no-sync`, the existing project Python environment, and the
4090 UUID. [Working commands](docs/commands.md) include scripted holds, videos,
training and the promotion-checked training entrypoint. Do not rerun setup.

Next development stays at stage 2: improve knee clearance and controlled descent
under reset variation before promoting. Stage 3 includes the tested standing
balance controller, but its full transition is experimental. No furniture
variation or longer training budget has been started.

## Preserved infrastructure

Isaac Lab `release/3.0.0` commit
`be9551260b269d4d8a964b71d5849810a1237b33`; Python 3.12.15; Isaac Sim 6.1.0.0;
PyTorch 2.11.0+cu130; RSL-RL 5.5.1. PhysX/training use RTX 4090 GPU 0; rendered
runs explicitly select Vulkan GPU 0. RTX 4070 remains the desktop GPU. About
57 GiB remains free on root. No Cosmos, full asset packs or datasets were added.

[Earlier infrastructure and failed standing pilot](docs/status-before-scripted-hold.md) ·
[Historical commands](docs/commands-before-curriculum.md) ·
[Resolved packages](logs/resolved-packages.txt)
