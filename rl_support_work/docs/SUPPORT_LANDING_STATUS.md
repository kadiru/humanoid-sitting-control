# Humanoid sitting: current verified state

Status (October 11, 2026): the landing-distance diagnosis, implementation and 48-trial validation are complete. Accuracy tests pass, but the predeclared all-trial availability gate fails. No RL was run, no new policy was promoted, and all simulation/replay/render jobs have finished. The new tracker is passive beside the frozen policy; it is not yet its observation source. Keep the preserved pilot-3 parent and existing deployment defaults.

Continue as the sole coding agent through Mac SSH to `kfuUbuntu`, project `/home/kadir/humanoid-sitting`. Never start another agent on Ubuntu. Use tmux for long jobs and `/data/humanoid-sitting-perception` for outputs. Preserve old code, datasets and checkpoints. Nothing was installed.

## What was fixed

1. Camera timing: the eager camera updates were offset from control reads. At a read, the image was commonly 95 ms old. Installed `force_recompute=True` does not mark a sensor due. New `support_camera.acquire_current_camera` temporarily sets its period to zero, requests a capture, then restores it, without stepping physics. All 603 validation capture checks have zero sensor timestamp age and unchanged joint/physics state. The image/IMU timing audit independently changes from 100-ms best lag to zero; development maximum normal disagreement is 0.063 degrees.
2. Floor/seat association: the old estimator discards a visible seat if a broad floor patch is absent in the same image. `LandingGeometryTracker` carries the measured floor reference using encoder FK and IMU gravity, associates fresh seat depth, and preserves recent surface identity when the feet unload.
3. Prediction honesty: the first prototype wrongly trusted two feet moving together and produced ten errors above 2 cm, maximum 4.40 cm. It is rejected and preserved. The current bridge lasts at most 100 ms, charges the full inferred displacement to its uncertainty budget, and withholds uncertain outputs. A currently observed horizontal floor can carry the previously measured rigid seat height for up to 0.6 s if the seat is briefly hidden. This has an explicit engineering error budget and a horizontal-floor assumption; it is not a statistical confidence guarantee.

Inputs are only depth, joint positions, IMU gravity/angular rate and static robot/camera calibration. Simulator truth is in separate archives and loaded by scorers only. No runtime seat pose, contact force or root world pose is used by the tracker. Camera rendering naturally uses the simulator scene, as any simulated sensor does.

## Validation, including failures

All 48 attempts run 20 s and are scored at 50 Hz, including time after physical failures. Each bank has 16 attempts.

| Bank | All-tick availability | Landing-window availability | Worst accepted error | Estimate at first sampled seat support |
|---|---:|---:|---:|---:|
| Mixed training shapes, new starts 51823 | 93.58% | 98.68% | 6.02 mm | 15/15 events |
| Known easier regression 51403/51407 | 99.91% | 99.75% | 1.83 mm | 16/16 |
| Fresh broader shapes 51841/51843 | 85.94% | 80.67% | 6.92 mm | 16/16 |

No accepted validation estimate exceeds 2 cm. The new-start run includes one startup refusal (case15: aligned sitting patch occluded/unobservable); the tracker is never initialized for that case. Do not exclude it from the overall score.

Fresh cases 2,3,4,5 have long gaps after actual falls. Their longest gaps begin with torso tilt approximately 106,97,64,69 degrees, after the first physical failure. Foot-supported kinematics and current seat visibility no longer provide a trustworthy reference. Before first physical failure, fresh availability is 99.74%; on the eight successful full trials it is 99.95%. These are post-hoc conditional diagnostics, not replacements for the failed all-frame gate. Do not claim continuous tracking during arbitrary falls.

Online/offline fresh replay matches all seven saved arrays bit for bit. Eight geometry tests and two acquisition tests pass. Added sensor noise produces no accepted error above 2 cm (maximum 6.15 mm), with fresh landing coverage 80.35%, including failed motions. During the 11.8–12.4 s camera blackout, no stale estimate remains valid after the 100-ms bridge limit. All 12 non-fallen cases recover at the first new frame, 12.5 s; the four already-fallen cases stay unavailable. Median online per-robot processing is around 0.7 ms, p95 around 19 ms; this is not a hard real-time guarantee.

## Controller caveat and next work

Synchronous camera acquisition changes the old actor's camera samples even though the actor weights and 144-channel contract stay unchanged. The old actor completes 15/16 known trials with corrected timing versus 16/16 historically. Case14 loses success to impact: 455.21 N previously, 729.20 N now. The passive tracker cannot be credited or blamed as a new control input because its output was not fed to the actor. Better estimates do not automatically make a controller trained with the old timing behave better.

The full release gate fails, so do not restart broad PPO or promote this observation pipeline by silently ignoring startup refusals or fallen intervals. Next isolate controller compatibility with corrected observations using matched motion recordings, and address startup reacquisition without bypassing the visibility guard. Any new tracker changes need a new frozen validation protocol and fresh geometry/start seeds; 51841/51843 are now consumed. Preserve explicit unavailable states during falls or genuine loss of references. Do not tune soft-landing gains to disguise the perception/control mismatch.

## Files, reproduction and preservation

- Tracker: `sitting_task/src/humanoid_sitting/support_landing_geometry.py`, frozen SHA256 `80a05f6cb26e07e54acc539faa4c699205fd68a07e29a623639906ec7a46ffd0`.
- Acquisition helper: `sitting_task/src/humanoid_sitting/support_camera.py`.
- Online entry point: `scripts/capture_support_landing_shadow.py`; base raw recorder and synchronization wrapper: `capture_support_landing.py`, `capture_support_landing_sync.py`.
- Launch: `scripts/run_support_landing_validation_01.sh` (completed, exit0; do not rerun into existing output names). Protocol/source freeze: `docs/SUPPORT_LANDING_VALIDATION_PROTOCOL.md`, `/data/humanoid-sitting-perception/support_landing_frozen_01`.
- Validation captures: `support_landing_{validation,regression,fresh}_capture_01`, each with `shadow_score/`, raw depth/proprioception, separate labels and source hashes.
- Fresh replays: `support_landing_fresh_replay_{none,noise,blackout}_01`.
- Final report: `/data/humanoid-sitting-perception/support_landing_validation_results_02`. It includes results, failure audit, source/protocol, preservation checks and two 3D videos: successful fresh case6 and failed fresh case2. Earlier report/visualization revisions remain preserved.
- Mac delivery: `/Users/kadir/humanoid-sitting-control/validated_support_landing/results`.
- Development: original capture `support_landing_development_capture_01`; camera render/pump attempts `_sync_capture_01` and `_02` were ineffective and retained. Explicit fresh acquisition is `_sync_capture_03`; final development replay is `_sync_track_05`. Rejected unsafe prototype `_development_track_02` and intervening runs remain available.
- Parent checkpoint: `/data/humanoid-sitting-perception/support_rl_independent_train_04/final.pt`, SHA256 `2c0f6907196be68bce9a1ee2e6b6ef6e4c82bf0ecdff622dc55b81fe8ef58d13`.

Parent, pace and both retention checkpoints are hash-verified unchanged. Original shared-frame source remains `daec7a1c769c68579cebfafe79d2063cb19660ca6e09ced705f460218f9fcf5c`. IsaacLab is clean at `be9551260b269d4d8a964b71d5849810a1237b33`. The legacy preservation checker verifies preservation and then fails only its pre-existing root-space reserve (29.53 GiB <30); the actual /data volume has 80.36 GiB free and passes its reserve check. Nothing was deleted to clear the warning.

The base capture's `original_controller_inputs_unchanged` metadata flag refers to returning its original observer output unchanged, not historical image parity. The synchronized subclass changes camera samples; the final report explicitly documents this distinction.

Prior completed RL results are preserved in `docs/SUPPORT_RL_RETENTION_ARCHIVE.md`. Scope remains aligned rigid horizontal seats in simulation, not navigation, deformable cushions, moving hidden seats or hardware deployment.
