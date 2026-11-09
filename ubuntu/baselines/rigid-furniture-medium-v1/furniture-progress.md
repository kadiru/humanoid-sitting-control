# Rigid furniture curriculum — 2026-10-09

The fixed-seat baseline is copied to `baselines/fixed-seat-v1`, including source,
checkpoints, demonstrations, evaluation recordings and a SHA-256 manifest. The
working fixed-seat task and its successful checkpoints are unchanged.

## Executed progression

- Small rigid-slab variation: 108/108 PPO trials and 108/108 zero-residual trials.
- Medium slab variation initially failed the two-body-weight force criterion:
  preserved PPO 99/108, scripted reference 98/108. No promotion occurred.
- A 30-update, 64-environment PPO warm-start pilot (46,080 transitions) reduced
  peak force from 735 to 703 N but achieved 98/108, so it was not selected.
- Doubling upright recovery from 2 to 4 seconds resolved the medium slab stage.
  Untouched geometry seed 9311 and initialization seeds 2024–2026: 108/108
  scripted and 108/108 preserved PPO; peaks 564/573 N.
- Rigid furniture initially exposed couch hand/armrest and backrest contacts.
  All failed variants remain in `runs/`, with source snapshots in `logs/`.
  A contact-triggered lift was too late, and simply raising the initial arms
  prevented seating. Geometry-guided arm targets now clear both armrests and
  backrest: shoulder pitch 0.8, shoulder roll ±1.6, elbow pitch 2.2 rad.
  All configured actuator effort limits remain unchanged.
- The final family controller passed 144/144 on untouched geometry seed 9611
  and initialization seeds 2040–2042, with zero falls, unintended contacts,
  excessive impacts or measured joint-torque violations. Peak seat force was
  539.0 N, peak downward speed 0.202 m/s, shortest continuous hold 10.44 s.

These are procedural rigid shapes: chairs have a back and legs, benches have
legs, boxes have a solid base, and couch-shaped seats have a back, arms and legs.
No compliant cushions or downloaded furniture asset library are used.

## Ranges and safety

| Geometry stage | Seat top | Depth | Width | Position x/y | Seat yaw |
|---|---|---|---|---|---|
| Small | 0.300 ±0.005 m | 0.500 ±0.025 m | 0.900 ±0.025 m | ±0.005 m | ±1° |
| Medium | 0.300 ±0.015 m | 0.500 ±0.075 m | 0.900 ±0.100 m | ±0.015 m | ±3° |
| Wider probe | 0.300 ±0.025 m | 0.500 ±0.100 m | 0.900 ±0.150 m | ±0.025 m | ±5° |

Centre x is -0.4 m, y is 0. Every geometry is crossed with three initialization
seeds and three standing poses, plus seeded joint/pose perturbations. Actual USD
seat dimensions, position and yaw are audited. Evaluations last 20 seconds,
never reset after a failure and latch all violations. Force peaks cover every
5 ms physics sample. The original support/posture/hold, fall, pelvis-floor,
10 N unintended-contact, 0.8 m/s descent and 632.5 N force requirements remain.
Joint targets are clipped to joint limits and original actuator limits bound
effort; observed effort-limit violations additionally disqualify evaluation.

The thumb collider exporter needed correction: cube bounds now use the authored
cube size instead of stale extent values. The actual thumb box is approximately
20×30×20 mm. This changed geometry analysis, not simulation colliders or safety.
A conservative hand-collider analysis found at least 24 mm separation from
couch parts over the final development trajectories.

## What PPO contributes

The preserved policy needs the reference: frozen-standing reference yielded
0/108 sitting successes; a one-second reference delay still passed 108/108.
Replacing furniture observations by fixed-seat values also passed 108/108.
Zero residuals already pass both slab stages. Thus these experiments show
robustness of the scripted physical controller, not learned furniture adaptation.
The reference-removal ablation is structurally constrained by ±0.08 rad residual
actions, which cannot reproduce a full hip/knee excursion. Typical residual RMS
is about 0.009 rad. The new pilot learned nonzero weights for appended seat-yaw
observations, but did not demonstrate better whole-trial success.

Numeric observations include actual dimensions, seat-local relative position
and yaw. The previous actor/critic were transferred with two zero-initialized
input columns; original checkpoints were never rewritten. The pilot used a
fresh optimizer and verified finite actor/critic updates.

[Complete metrics](../runs/furniture_progress_01/report.md),
[CSV](../runs/furniture_progress_01/metrics.csv),
[per-family results](../runs/furniture_progress_01/families.json) and
[representative videos](../runs/furniture_progress_01/families.mp4) retain the
successful and unsuccessful experiments. Promotion requires ≥90% success and
zero safety violations; a high aggregate success rate cannot mask couch failures.

## Resume

```bash
cd /home/kadir/humanoid-sitting
bash scripts/run.sh python ../scripts/evaluate_furniture.py --device cuda:0 \
  --geometry-stage 1 --geometry-seed 9611 --geometry-count 16 \
  --seeds 2040 2041 2042 --families --arm-clearance \
  --reference-mode gentle_recovery \
  --checkpoint /home/kadir/humanoid-sitting/IsaacLab/logs/rsl_rl/g1_full_transition/2026-10-09_10-08-57_seed_7/model_96.pt \
  --output ../runs/new_furniture_evaluation
```

Add `--video --video-geometry-index 3` and the explicit renderer GPU-0 Kit
arguments from `docs/commands.md` for a nominal couch rollout. Omit checkpoint
for zero residuals. Output directories must be new. No installation required.

Larger shapes, substantial seat rotations, approach/alignment, perception and
compliance remain later work. Do not describe these aligned 30 cm seating tests
as a general-purpose furniture policy or evidence of real-robot readiness.
