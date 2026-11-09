# Contact and effort diagnosis — October 14, 2026

This records the evidence behind the current repair. Final study conclusions belong in CONTACT_EFFORT_REPAIR_STATUS.md and the independently verified reports. All original studies, rejected probes and checkpoints remain preserved under `/data/humanoid-sitting-perception`.

## Sitting contact failures

The original reference replay reproduced the saved physical and sensor traces exactly. It succeeds in8/18 development trials. Geometry observations remain available before the contact failures: missing seat-distance input is not the explanation for these failures.

The motion transfers the pelvis rearward while lowering before it has established a safe seat contact. The planned reference moves from approximately x=−0.039m at stage4 to−0.260m at stage5, then returns to−0.163m. The actual motor commands are rate-limited and lag the changing reference. Contact diagnostics identify hip housings and disallowed/edge contacts, alongside impacts above the unchanged2-bodyweight threshold. Contact-sensor body order differs from robot body order; diagnosis uses the recorded sensor names, not an assumed index mapping.

The accepted repair constructs a new approach/contact path from the measured plane and joint/IMU kinematics, preserving the initial foot locations. It transfers above the seat before lowering, targets pelvis x=−0.15m in the explicit offline reference coordinate system, and adds bounded ankle feedback from a foot-anchored pelvis-x estimate. Measured-plane calibration, command slew and path parameters were fixed before training. This remains reference-assisted control on static slabs; horizontal seat placement is not autonomously discovered.

The repaired reference and repaired parent55211 pass18/18 development trials without physical failure. Repaired parent55217 passes16/18: its two38cm/+5degree trials exceed the friction safety margin without falling or slipping. Both new sitting seeds therefore use the same admitted55211 parent. The rejected initial both-parent admission is preserved;55217 remains a comparison baseline. No criteria were relaxed.

The repair has a substantial effort cost. On the8 trials where the old reference also succeeds, repaired-reference squared torque rises53.68% over the whole20s and225.88% over the final5s. Across all18 repaired-reference trials, the seat carries only12.44% of body weight in the final5s. The robot is safely supported but still largely crouching on its legs. A stable-support pass is not proof of relaxed sitting.

## Why leaning effort rose

The previous transfer study retained16/16 holdout stability for both leaning finals, but whole-motion squared torque rose1.02% and6.99% against the corrected reference. Leg torque increases outweighed small upper-body savings. These were physically worse posture adjustments, not an artifact of counting failed trajectory fragments. The old selection rule could choose a trained candidate even when its unchanged parent was better; the current selection explicitly requires beating both parent and reference.

The original learner changed23 joint residuals independently every20ms, while filtering and command slew delayed their physical effect. Its short credit-assignment horizon was a plausible obstacle to discovering coordinated resting changes. This is an interpretation, not a single-cause ablation. Controlled persistent perturbations provide direct evidence that better motions exist: coordinated knee flexion with hip/ankle compensation retained9/9 stability and lowered whole-motion torque4.11%, final5s12.56%, against parent55311. An opposite knee adjustment lost two successes. These probes are prescribed motions, not learned results.

The new learner retains a frozen parent for approach and learns two coordinated adjustments held200ms after12s, with a longer learning horizon and critic warmup. One final seed56111 succeeds16/16 on reserved surfaces, lowering whole20s torque3.89% versus reference and2.46% versus its parent. Final5s savings are8.02% and7.50%. The second seed56117 succeeds only2/16 and is rejected in favor of its parent. It straightens the knees and increases ankle commands; mean final5s friction utilization rises from approximately0.362 to0.921, beyond the0.9 safety margin. Contact remains and no physical-failure flag is triggered, but stable supported rest is lost. The two seeds have different frozen parents; this is not a controlled test of random seed alone.

Several learning choices changed together. The new result demonstrates a feasible learned improvement in one run, not a repeatable two-seed solution or proof that one particular change caused it.

## Additional optimizer diagnostic, not a change to the studies

Masking actor advantages outside the resting phase sets actor gradients to zero but does not necessarily freeze actor weights once Adam has momentum. A disposable-copy diagnostic using saved checkpoint parameters and optimizer state demonstrates nonzero parameter movement on a zero-gradient optimizer step. Original checkpoints and training budgets were not modified. The installed PPO also adapts its shared learning rate using all sampled distribution KL values, including approach observations. Saved final leaning learning rates differ substantially:0.01 for56111 and0.0000759375 for56117. The high-rate run is the successful one, so these observations do not establish learning rate or momentum as the sole cause of regression.

A future learner should distinguish actor-active samples in its surrogate/KL accounting and skip actor optimizer steps when no actor learning is intended, with a regression check that parameters and actor optimizer state remain unchanged. This has not been silently applied to the frozen comparison. The current pilot must be interpreted as the implementation it actually ran.

## Separate exploratory unloading continuation

A reference-only continuation after12s brings the torso more upright and moves weight onto the seat. One candidate passes8/9 development geometries; on those8 common successes it lowers whole20s torque3.02%, final5s9.64%, and raises final5s seat load from11.08% to24.76% of body weight. The remaining geometry causes a forbidden contact. A second candidate passes only4/9 and increases effort. Neither is validated for use, neither was used for PPO training or selection, and neither touched the reserved holdout. Small parallel-placement startup/trajectory differences are explicitly recorded in its review. The original plot and a corrected layout are both preserved.

The remaining physical objective is to transfer substantially more body weight to a legal support patch without hitting the hip housings or exceeding friction/impact limits. Further training around a high-effort crouch alone is unlikely to establish low-effort sitting on arbitrary furniture.
