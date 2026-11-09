# Geometry-conditioned sitting — verified progress

The common `contact-v12` scripted controller passed **432/432** rigid-furniture transitions: 144 development cases and **288 held-out cases** across two untouched geometry banks. Each bank crosses 16 geometries with three initialization seeds and three standing poses. Chairs, benches, boxes and rigid couch shapes each passed 108/108 across all three banks. A separate 144-case replay also verified the registered `Humanoid-Sit-Geometry-Conditioned-G1` task; it repeats held-out conditions and is not counted as new evidence.

Validated sampling ranges: seat height 0.275–0.325 m, depth 0.40–0.60 m, width 0.75–1.05 m, center offsets ±0.025 m around the fixed-seat position, and yaw ±5°. These are procedural rigid furniture families with unseen dimensions, not arbitrary CAD furniture, cushions or a learned approach behavior.

| Scripted bank | Cases | Success | Falls / unintended / excessive force / torque violations | Peak seat force | Maximum descent speed | Shortest seated hold |
|---|---:|---:|---:|---:|---:|---:|
| Development 9612 | 144 | 100% | 0 / 0 / 0 / 0 | 496.9 N | 0.179 m/s | 6.54 s |
| Held out 10514 | 144 | 100% | 0 / 0 / 0 / 0 | 469.2 N | 0.189 m/s | 7.56 s |
| Held out 10515 | 144 | 100% | 0 / 0 / 0 / 0 | 616.1 N | 0.181 m/s | 7.40 s |

The generator uses exported G1 joint frames for height-conditioned inverse kinematics. It retains the calibrated fore–aft descent geometry, applies a bounded seated correction from seat depth, yaw, front-edge position and initial alignment, and uses a common height-dependent torso/arm clearance rule. Recovery waits for upward load-bearing pelvis/thigh contact. A bounded slow hip search handles contact that arrives after the eight-second descent. Existing COM/ankle feedback remains active before seat support. Motion has no branches keyed by furniture identity.

Safety is unchanged: verified initial standing; no fall or pelvis-floor contact; unintended contacts ≤10 N; seat force ≤632.53 N (two body weights); downward speed ≤0.8 m/s; original actuator effort limits; and at least three seconds of continuous load-bearing, upright, low-velocity seated contact. Any past safety failure permanently vetoes success. Evaluations disable automatic resets. Forces are checked from four 5 ms samples per 20 ms control step. Actual collider audits found at least 79.9 mm of shin front-edge clearance on bank 10514 and 31.1 mm of conservative hand/back separation on bank 10515; pose audits sample at 20 ms, so force histories remain authoritative.

PPO has **not demonstrated a consistent geometry-adaptive improvement**. On bank 10514, scripted, preserved PPO, the new pilot, and geometry-blind pilot all passed 144/144; replacing the reference geometry with canonical geometry reduced the pilot to 102/144. Thus reference conditioning provides the main adaptation. The preserved PPO lowered mean trial peak force by 12.2 N on bank 10515 (geometry-block bootstrap 95% interval: −22.1 to −2.6 N), but increased it by 3.0 N on bank 10514. On development cases it introduced one force violation (143/144, 697 N); the new pilot introduced three (141/144, 649 N). Neither residual policy is promoted over the scripted controller.

The pilot performed 80 actual PPO updates, 122,880 environment transitions, on training geometry bank 41012 with the **v11** teacher. Its finite actor and critic weights changed and `model_79.pt` was loaded for all subsequent evaluations. Testing that checkpoint with v12 is transfer evaluation, not v12 training. Further training was stopped because it did not establish a consistent improvement. All earlier checkpoints and experiments remain intact.

Paired comparisons assert bitwise-identical initial root/joint states and identical geometry banks. [Primary comparison](../runs/geometry_reference_v12_comparison_report/report.md), [confirmation comparison](../runs/geometry_reference_v12_confirmation_report/report.md), [metrics](../runs/geometry_reference_progress_01/metrics.csv), [validation](../runs/geometry_reference_progress_01/scripted_validation.json), and [video](../runs/geometry_reference_progress_01/transitions.mp4) are saved. The video includes the preserved low-chair failure and three successful new transitions. The matched low-chair example uses unchanged learned weights and the same nominal starting configuration: peak force decreases from 810 N to 325 N.

The old fixed-seat and medium-furniture archives passed all 177 checksum checks. The new source, three demonstration banks, results and comparison policies are preserved in `baselines/geometry-conditioned-rigid-v1`. Simulation and rendering were verified on RTX 4090 / GPU 0. The driver and installed environments were preserved. Approximately 39 GiB remains free after archiving; all raw experiment logs and rollouts were retained.

Resume evaluation from the project root, choosing a fresh output name:

```bash
bash scripts/run.sh python ../scripts/evaluate_geometry_reference.py \
  --v12-reference --production-task --geometry-stage 2 \
  --geometry-seed 10514 --geometry-count 16 --families --arm-clearance \
  --seeds 3107 3108 3109 --output ../runs/geometry_eval_resume_01 --device cuda:0
```

To compare a residual policy, add `--checkpoint ../baselines/rigid-furniture-medium-v1/policy_129_inputs.pt` or `--checkpoint ../runs/geometry_reference_v11_ppo_pilot_01/model_79.pt`. Add `--blind-geometry` to blind policy observations, or `--reference-mode canonical_geometry` to ablate reference conditioning. Keep the geometry bank, seeds, initial poses and duration identical. Existing output directories are rejected rather than overwritten.

Optional future v12 training uses `scripts/train_geometry_reference.py --reference-generator contact-v12`; no v12 training run is claimed here. The next neural-policy milestone should remove the demonstrated dependence on the hand-authored reference before adding perception or Cosmos. Compliant cushions remain a separate physics-modeling stage.
