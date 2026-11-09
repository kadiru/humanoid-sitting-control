# Deployment-observation sitting progress

All execution reported here is **Isaac Sim/PhysX on the RTX 4090**, not a physical G1 rollout. Existing installations, NVIDIA driver, teacher/student checkpoints and earlier results remain intact. Safety limits and whole-trial failure latching were not weakened.

## Preserved baseline and independence audit

The frozen DAgger checkpoint now passes **432/432** across the original two banks and fresh bank 24001. Its SHA-256 remains `4e187d9924c8b0357c9a0bde06a626df784af9c7005f67ae7dfaeecb5525e2e3`. Preservation and supplemental validation: `baselines/dagger-student-v1`.

That policy generates actions without executing the teacher trajectory or IK, but it retains exact numerical geometry, reset time/contact memory, simulator support load, COM/velocity feedback and a fitted classical balance primitive. It is not a deployment-realistic policy. On paired 144-case ablations, canonical geometry passes 98, with 25 force failures; removing timing passes zero; removing balance passes zero with 72 falls and 144 unintended-contact/force failures. The old PPO residual around a fixed standing offset also passes zero. See [complete observation audit](deployment-observation-audit.md) and [original controller audit](controller-information-audit.md).

## Strict student implementation

The new pure-tensor contract packs 109 channels: 23 encoder positions, 23 encoder velocities, 23 estimated motor torques, 3 gyro values, 3 IMU gravity values, 23 previous commanded joint targets, 10 estimated geometry/status values, and one sit command. There are **zero foot-feedback channels**. The inspected G1 `unitree_hg::LowState` supplies motor/IMU fields but does not establish foot-force feedback. SDK headers, pin and tentative wire-slot mapping are saved for review.

RGB-D geometry comes from depth, camera intrinsics, fixed body extrinsics and IMU gravity. Floor height is inferred from depth. No simulator root pose, camera world pose, furniture family, authored geometry or contact forces enter that estimator. Captures include robot self-occlusion. A proposed rear-facing camera mount is used; it is not verified hardware. The initial estimate is latched for the already-aligned, anchored-seat transition. Recorded camera observations are matched to benchmark starts; this is not a live physical camera integration, approach or moving-seat experiment.

The selected neural architecture learns a geometry/command-conditioned recurrent body trajectory and a neural ankle-feedback head. No teacher trajectory, IK, scripted COM correction, exact contact channel or explicit clock is evaluated for its actions. **It still has an implicit timing schedule:** the body head depends on inference count/recurrent memory and has no proprioceptive feedback. The ankle head uses physical sensor equivalents and previous commands. This is not yet a wholly reactive sitting controller.

A 0.05-rad/control-step target guard, original joint limits, original effort caps and the unchanged evaluation criteria apply. Fourteen simulated hand targets remain parked. Exact simulation/contact state is used only by supervision, diagnostics and evaluation. The 12-trajectory adaptation dataset used 95% teacher action assistance and is explicitly not an independent policy benchmark.

## Measured outcomes

The machine-readable table in [metrics.csv](../runs/deployment_interface_01/metrics.csv) contains every failed trial, descent speed, hold duration, torque/descent violations and successful-trial stability measurements. Counts below use unchanged whole-trial criteria. Force is peak **resultant seat normal force**, excluding friction, with a 632.53 N limit.

| Frozen recurrent student 04 experiment | Success | Falls | Unintended contact | Excessive force | Peak normal force |
|---|---:|---:|---:|---:|---:|
| Training-geometry pilot | 9/12 | 0 | 0 | 3 | 718 N |
| Fresh geometry bank 24001 | 86/144 | 3 | 3 | 58 | 2612 N |
| First 12 fresh starts, clean | 7/12 | 1 | 1 | 5 | 2495 N |
| Same starts, 5 mm depth noise + 10% dropout | 0/12 | 8 | 10 | 12 | 5358 N |
| Same starts, 20 mm position / 10 mm height error (Gaussian std) | 5/12 | 1 | 1 | 7 | 2237 N |
| Same starts, 60 ms sensor delay | 3/12 | 5 | 7 | 9 | 5956 N |
| Same starts, 100 ms sensor delay | 0/12 | 6 | 12 | 12 | 7404 N |
| Same starts, recurrent memory reset every step | 0/12 | 0 | 0 | 0 | 0 N |
| Same starts, frozen initial encoder/torque/IMU feedback | 0/12 | 6 | 12 | 12 | 7307 N |
| Same starts, training-mean geometry | 2/12 | 0 | 0 | 10 | 2074 N |
| New elliptical/chamfered rigid seat silhouettes | 7/12 | 0 | 0 | 5 | 886 N |

The scripted teacher and preserved DAgger each pass the paired 144-case fresh bank. Exact root/joint initial conditions and geometry equality are verified in `paired_conditions.json`. The new shapes also pass 12/12 with the teacher. These are bounded procedural families at heights 0.275–0.325 m and small orientation/alignment variation, not arbitrary household furniture.

Successful strict fresh trials hold at least 6.74 seconds; the all-trial minimum is zero. Maximum fresh descent is 0.449 m/s. Typical clean force failures occur during seat loading/recovery around 9.6–12.3 seconds. A stable final appearance does not erase a past excessive impact. Sensor-delay and noisy-estimate failures are substantially more severe. Freezing physical feedback and replacing geometry both reduce performance, demonstrating learned dependence on these inputs; memory reset also confirms the remaining temporal dependency.

## Confidence failure and preprocessing correction

A causal offline diagnostic isolated an untrained constant feature: confidence was 1.0 throughout clean training, but noisy depth reduced it to roughly 0.11–0.24. Changing only confidence shifted generated body targets by up to **1.81 rad**. This is a motion-generation defect, not poor seat-height accuracy. The diagnosis does not itself claim safe execution.

`runs/deployment_status_mask_05/student.pt` preserves all learned weights and masks normalized confidence/validity in the motion networks. **109 channels are packed; 107 are effective neural inputs.** Validity still admits/refuses the command upstream; confidence remains an uncalibrated diagnostic. No geometric visibility gate, force criterion, torque limit or success threshold was relaxed. Clean inference is bitwise identical on 24,000 replayed packets; this is offline equivalence, not another physics benchmark.

The corrected model passes **10/12** on the exact noisy-depth cases that previously passed zero, with zero falls/unintended contacts and two force failures (peak 1152 N). It remains unpromoted. On a new geometry bank 25001 generated after freezing the correction, the teacher passes 36/36. The corrected student passes **15/36 clean** and **21/36 noisy** starts. Both have zero falls, unintended contacts, torque violations or descent violations, but respectively **21 and 15 excessive-force failures**, peaking at 1445 and 1421 N. This independently confirms the strict-input milestone while rejecting robust promotion. The clean all-trial minimum hold is zero; noisy minimum hold is 7.84 seconds, but that does not erase earlier impact failures.

## Perception admission and contact validation

On 144 fresh captured starts, clean seat-height error averages 0.589 mm (max 0.768 mm); noisy error averages 1.16 mm (max 3.95 mm). These idealized simulation measurements do not establish real-sensor accuracy. Visible seating area is smaller than full furniture bounds by an average 0.139 m²; hidden extents are not recovered.

The initial estimator falsely accepted backrest tops under central occlusion, with errors up to 35 cm. The corrected gate requires sufficient visible depth/width/area and at least 75% observed coverage in a 20 × 24 cm aligned landing patch. It retains all 144 clean/noisy estimates and rejects all 144 starts in each 20%/40% central and 50% lateral occlusion test. Those **refusals are not sitting successes**, and no autonomous standing fallback has been validated. Clutter, moving seats and mid-transition perception changes are untested.

The rigid-body audit binds all 44 robot links and the seat, checks every sensor body path, and finds zero warnings on actual rigid-body prims. The 162 unique warning paths come from the broad `create_rigid_body_view("/**")` enumeration of non-rigid scene nodes. During a valid hold, seat/filtered-robot reaction mismatch is zero; total vertical robot normal force is 316.278 N versus 316.264 N bodyweight. An intentionally deep knee/seat intersection is correctly rejected by the unintended-contact sensor. Its 117.9 kN diagnostic force is **not a policy rollout metric**. Audit: [contact_binding_validation_02](../runs/contact_binding_validation_02/audit.json). The diagnostic also exposed and fixed a full-reset slice-handling bug; the previous source is preserved.

## Videos and next work

[Strict student complete transition](../runs/deployment_student_video_04/transition_0000.mp4): 8.36 s hold, 276 N peak, zero violations. [Matched preserved DAgger](../runs/deployment_dagger_comparison_video_01/transition_0000.mp4): 7.80 s hold, 305 N peak. [Force-failing transition](../runs/deployment_student_failure_video_04/transition_0000.mp4) is retained so successful-looking seating cannot hide a failed safety benchmark. Physics plots/contact sheets are saved alongside each video; [failure/success trajectories](../runs/deployment_interface_01/deployment_trajectories.png) and [camera occlusion views](../runs/deployment_interface_01/perception_occlusion.png) are also available.

The next useful training work is sensor-synchronized demonstrations including startup torque histories, delayed-feedback state estimation, and contact/recovery inference from physical channels. Body control should eventually react to estimated load/stability rather than recurrent timing alone. Extensive RL is not justified by the current force/delay failures.

No hardware commands were issued. Actual G1 variant, joint/frame signs, motor mode, onboard IMU/torque timing, camera mount/extrinsics, mass/inertia and actuator envelope are unverified. The simulated 300 Nm ceilings for many joints are not a verified physical G1 limit; camera mounting mass is not modeled. These commissioning questions and failed robustness tests block a hardware-ready claim.

## Resume

Use the existing environment; do not reinstall. Preserve output directories (the evaluators refuse overwriting them). Full commands are saved in each run's `command.json`.

```bash
cd /home/kadir/humanoid-sitting
bash scripts/run.sh python ../scripts/audit_deployment_contract.py \
  --checkpoint ../runs/deployment_status_mask_05/student.pt \
  --output ../runs/deployment_interface_01
bash scripts/run.sh python ../scripts/evaluate_geometry_reference.py \
  --deployment-checkpoint ../runs/deployment_status_mask_05/student.pt \
  --deployment-perception ../runs/deployment_perception_stress_fresh_01/clean \
  --deployment-action-rate .05 --geometry-stage 2 --geometry-seed 24001 \
  --geometry-count 4 --seeds 3201 --compact-recording --summary-only \
  --output ../runs/deployment_resume_NEW --device cuda:0
```

`evaluate_deployment_robustness.sh` and `evaluate_deployment_ablations.sh` reproduce the original candidate-04 protocols with new output names. `capture_deployment_geometry.py` renders sensor starts; `stress_deployment_perception.py` applies measured-pipeline noise/occlusion without ground-truth fallbacks. `train_deployment_student.py` supports measured RGB-D estimates and safe-prefix deployment DAgger packets. Protected banks 9612, 10514, 10515, 24001 and 25001 cannot train either student script. See `runs/deployment_interface_01` for source hashes, SDK audit, paired-condition checks and storage accounting.
