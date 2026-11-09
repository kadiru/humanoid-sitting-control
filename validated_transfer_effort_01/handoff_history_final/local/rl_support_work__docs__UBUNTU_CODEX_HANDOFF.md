> **Bounded resting PPO complete — October 12, 2026:** Both trained policies and fixed posture passed56/56 full20-second conditions. No physical failures, but normalized whole-body effort increased0.28%/2.59% by seed, including worse effort on all tested heights. Optional hold never activated and had no effect. Budgets, checkpoints, startup matching and all46 frozen hashes independently verified. No policy promoted or extra training started. Read the latest status and validated_resting_bounded_02/final_results/review.html. Completion monitor paused; all previous outputs preserved. Earlier snapshots below are history.

---

> **Bounded resting PPO pilot active — October 12, 2026:** Old-policy contact failures traced to hip-pitch housings on the seat and the right hip-roll joint limit. New bounded reference offsets retain fixed posture exactly in40/40 full20-second trials plus16/16 with the optional hold rule enabled; reset isolation and repeated resets pass. A separate two-seed64-update study compares fixed posture, PPO and PPO with measured-effort hold over56 conditions, including16 fresh states. Read the latest status and docs/RESTING_BOUNDED_PROTOCOL.md; persistent tmux resting_bounded_ppo_02 uses GPU1. The existing resting completion monitor is active for this new study. Prior studies and all checkpoints are preserved. Earlier snapshots below are history.

---

> **Resting PPO comparison complete — October 12, 2026:** Fixed posture40/40; all four learned policies0/40. Effort reward lengthened qualifying rest but produced no successful policy or validated effort saving. All budgets, checkpoint/startup matching and source preservation verified. See the latest status and validated_resting_posture_01/final_results/review.html. Both resting and capped-impact studies are complete, their monitors paused, and no policy promoted. Earlier snapshots below are preserved history.

---

> **Capped-impact comparison complete — October 12, 2026:** All four32-update runs and matched evaluations verified. Starting policy53/80; old→capped35→36/80 and54→55/80 by seed. Successful-trial peak forces increased in both seeds; the pooled effort benefit did not generalize to the new holdout. No policy promoted. Read the latest status and verified report. The seated-resting study remains separate under its own completion monitor. Earlier status snapshots below are preserved history.

---

> **Active seated-resting PPO study — October11,2026:** Seated resets and fixed-posture baselines validated. PPO is running on physical GPU1 across25/30/35/40cm seats, with two seeds and matched stability/effort arms. Read the latest status. The separate capped-impact comparison continues on GPU0; both checkpoints and histories are preserved.

---

Active capped-impact PPO study (October11,2026): integer-timer fix validated; four32-update runs from47/64 policy and matched evaluations are running in tmux support_capped_impact_01. Read docs/SUPPORT_RL_LATEST.md and SUPPORT_CAPPED_IMPACT_PARITY_AMENDMENT.md. Completion follow-up stays in the same Mac chat. Old states below are history.

---

Completed reward study (October11,2026): matched results47/64 starting PPO,44/64 old reward,42/64 new reward (43 after timer correction). New reward did not help overall; diagnosis identifies overscaled terminal impact cost. All jobs finished. Read docs/SUPPORT_RL_LATEST.md and SUPPORT_REWARD_NEXT_PROPOSAL.md. Older active/pause instructions below are history.

---

Active RL study (October11,2026): two matched128-update PPO runs and automatic evaluations are running in tmux support_reward_study_01. First optimizer updates verified. Read docs/SUPPORT_RL_LATEST.md and remote study status.json. Earlier pause instructions below are historical; the user explicitly authorized RL.

---

Current status (October11,2026): read docs/SUPPORT_RL_LATEST.md. Independent distance observations validated; one PPO pilot completed35/48→37/48 with no lost successes but worse impacts. No promotion or default change. All jobs finished. Earlier handoff below is preserved history.

---

> **Current verified state — October 11, 2026:** Read [SUPPORT_RL_LATEST.md](SUPPORT_RL_LATEST.md). The known timing regression is repaired experimentally; broader controller/availability gates fail. All jobs finished; no RL or promotion. The original October 9 handoff below is preserved history.

# Ubuntu Codex technical handoff

Prepared 2026-10-09, America/Los_Angeles, on `kfuHomeUbuntu` for a new agent on the Mac mini controlling Ubuntu through SSH. Project root: **`/home/kadir/humanoid-sitting`**; Ubuntu user: **`kadir`**.

## 1. Read this first

**New priority: visual servoing using a shared 3D representation of seat, floor and robot pelvis. Pause further touchdown-controller tuning and additional training until that representation is validated.** The proposal in section 9 is not implemented or validated yet.

Installation, G1 simulation, PPO checkpoint loading, scripted sitting, furniture teachers, privileged students and sensor-based experiments already execute on Ubuntu. **Do not reinstall, run `scripts/setup.sh`, change the NVIDIA driver, delete caches, overwrite checkpoints or launch hardware commands.** Preserve failed trials as well as successes. No physical G1 deployment has occurred.

The best broad rigid-furniture controller is the **privileged scripted contact-v12 teacher**, 432/432 transitions. The best preserved privileged DAgger student also passed 432/432. The current **sensor-based controller** is checkpoint 05 plus the original contact estimator 02 and authored ankle governor: **155/168** on four matched rectangular banks, **166/180** including novel silhouettes. It still violates the 633 N impact ceiling. These are different capabilities; neither teacher nor privileged DAgger proves deployment-realistic control. No latest soft-landing candidate was promoted.

This handoff was prepared using read-only inspection and CPU checkpoint metadata loading. Commands below were checked against source parsers, recorded command manifests, environment metadata and existing paths; simulations/training were **not rerun during handoff**. Exact historical commands/configurations remain in the linked records. Fresh-output examples reproduce those protocols with new destinations, not new execution claims.

## 2. Workstation, SSH and environment

- Ubuntu 22.04.5; system Python remains separate from project Python. RTX 4090 GPU 0, 24 GB, for simulation/training; RTX 4070 GPU 1, 12 GB, desktop. NVIDIA driver **580.178.04**, preserved.
- Project environment: `IsaacLab/.venv`, Python **3.12.15**. Local uv executable: `tools/uv`; managed Python and package/runtime caches: `cache/python`, `cache/uv`, `cache/runtime`.
- **Isaac Sim 6.1.0.0**, **Isaac Lab release/3.0.0 stabilization checkout**, commit **`be9551260b269d4d8a964b71d5849810a1237b33`**. Lab 3.0 was pre-release when selected. Do not infer compatibility from old 4.5 metadata in the user/system environment.
- Current installed distribution metadata: `isaaclab==27.0.2`, `isaaclab-assets==2.0.0`, `isaaclab-rl==1.3.4`, `isaaclab-tasks==23.0.1`; those component metadata numbers are **not** the release branch name. Source commit is authoritative.
- torch **2.11.0+cu130**, NumPy **2.5.1**, RSL-RL **5.5.1** (`rsl-rl-lib`), SciPy **1.17.0**, custom package `humanoid-sitting-task==0.1.0`.
- Dependency authority: `IsaacLab/pyproject.toml` and `IsaacLab/uv.lock`, including NVIDIA/PyTorch indexes and checkout overrides. Installed using source-checkout uv workflow with `isaacsim`, `rsl-rl`, `video` extras. Custom package installed editable without replacing checkout dependencies. [Resolved packages](../logs/resolved-packages.txt); [execution/GPU metadata](../runs/deployment_interface_01/stack_and_gpu.json).
- PhysX backend; ordinary free-base gravity simulation. Physics 5 ms, control 20 ms in sitting tasks. Recorded infrastructure startup selected physics and renderer GPU 0; [minimal validation](../logs/minimal-sim-result.json) and `logs/g1-train.log` / `logs/g1-play.log`.

**Always launch from the project root through `bash scripts/run.sh`.** It clears inherited Python/Conda variables, sets project caches and `OMNI_KIT_ACCEPT_EULA=yes`, selects the 4090 by UUID `GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4`, sets `OMP_NUM_THREADS=8`, changes cwd to `IsaacLab`, and executes `uv run --no-sync`. Consequently `../scripts`, `../runs`, `../datasets` refer to the project; `logs/rsl_rl` without `../` refers to `IsaacLab/logs/rsl_rl`. Do not accidentally run the system Python for Isaac scripts. Direct `.venv/bin/python` is suitable for CPU metadata inspection; use the wrapper for actual simulation.

Mac-side example, **not tested from this session**:

```bash
ssh kadir@kfuHomeUbuntu 'cd /home/kadir/humanoid-sitting && hostname && df -h . && nvidia-smi'
ssh kadir@kfuHomeUbuntu 'cat /home/kadir/humanoid-sitting/docs/UBUNTU_CODEX_HANDOFF.md'
```

Use the Mac's configured SSH alias/IP if hostname resolution differs. Passwordless SSH is supplied by the user; this session did not verify Mac connectivity or modify SSH. Run simulation on Ubuntu, not on Apple Silicon. For lengthy authorized work, use a persistent Ubuntu terminal/session and explicit log redirection; do not assume a remote foreground process survives disconnect. Serialize GPU simulations; camera runs need the renderer flags shown below, even headless.

## 3. Project map and implementation

| Location | Purpose |
|---|---|
| `IsaacLab/` | Official pinned source checkout, `.venv`, uv lock, RL logs; its `AGENTS.md` applies to edits inside that repository |
| `sitting_task/pyproject.toml` | Editable task package, Python >=3.12,<3.13, `isaaclab.tasks` entry point |
| `sitting_task/src/humanoid_sitting/tasks/__init__.py` | Gym registrations: Fixed, Curriculum-0..3, Full-Transition, Rigid-Furniture, Geometry-Conditioned, Learned G1 tasks |
| `tasks/config.py`, `tasks/mdp.py`, `tasks/env.py` | Scene/PhysX/actuators, observations/rewards/contact criteria, custom environment and reset handling |
| `tasks/curriculum.py`, `tasks/full_transition.py` | Seated/near-seat/partial-descent stages and fixed-seat full transition; original validated ankle gains 160/4, ankle effort cap 20 Nm |
| `tasks/furniture.py`, `tasks/geometry_curriculum.py` | Procedural rigid furniture and geometry training configurations |
| `tasks/geometry_reference.py`, `tasks/geometry_contact_v12.py` | Exported-G1 leg kinematics and selected common geometry-conditioned contact-aware scripted teacher |
| `tasks/geometry_feedback.py`, `balanced_reference.py`, `geometry_guarded*.py`, `geometry_contact_v11.py` | Earlier preserved controller revisions; do not select merely because of recent filename/time |
| `tasks/student.py` | Full absolute-target student action term and privileged 99-channel student / deployment environment configurations |
| `student_network.py` | Initial MLP and structured privileged student; fitted classical balance buffers |
| `deployment.py` | Pure torch SensorPacket contract, joint order, tentative SDK slots, geometry-recurrent sensor actor; no simulator imports |
| `sim_deployment_sensor.py` | Physical-equivalent simulated encoders/IMU/estimated torque, noise and causal delay; geometry supplied by depth estimator |
| `soft_landing.py` | Pure torch contact estimator and authored phase/action governor; optional rejected `smooth` experiment and staged ankle head |
| `scripts/evaluate_geometry_reference.py` | Primary whole-trial teacher/residual/privileged/sensor comparison, recording and video evaluator |
| `scripts/seat_depth_estimator.py` | Depth unprojection, IMU leveling, floor/support plane and visible sitting-area estimation |
| `scripts/capture_deployment_geometry.py`, `stress_deployment_perception.py` | Proposed pelvis-mounted RGB-D matched-start captures; noise/dropout/occlusion evaluations |
| `scripts/export_g1_model.py`, `export_g1_com_model.py`, `analyze_transition_geometry.py` | Actual USD joint/collision export and offline FK/clearance validation; useful for next milestone |
| `scripts/train_sitting_student.py`, `refine_sitting_student.py` | Privileged BC/DAgger and anchored PPO refinement |
| `scripts/train_deployment_student.py` | 109-input sequence BC/adaptation, optional ankle-only freezing, protected evaluation-bank checks |
| `scripts/train_contact_estimator.py`, `train_window_contact_estimator.py` | Privileged contact labels supervise sensor-only observers |
| `scripts/refine_soft_landing_rates.py` | One-update episode REINFORCE experiment on two governor rates, **not actor PPO** |
| `runs/`, `datasets/`, `logs/`, `baselines/`, `docs/` | Experiment outputs, synchronized traces, launch logs, immutable archives and technical reports |

Paths in the table beginning `tasks/` or a module filename are relative to `sitting_task/src/humanoid_sitting/`. `runs/g1_kinematics_geometry_03/model.json` and `runs/g1_kinematics_geometry_04/model.json` contain actual loaded USD frames/collision data. `geometry_reference.py` resolves the latter from the project layout, so do not relocate a single script/package without the project assets. Mesh seats use normalized geometry and convex hulls; geometry audits are saved per evaluation.

## 4. Learning system and information boundary

### Scripted teacher and residual PPO

Contact-v12 uses exact authored seat top height, depth, width, position/yaw and initial robot alignment. It computes common geometry-conditioned IK targets, a shared descent, contact-triggered recovery, bounded contact search, and COM/velocity ankle balance. Width is cached but does not change targets. Contact/load, COM, masses, foot positions, root velocity and elapsed/contact clocks are privileged. Root is free; control is through joint actuators only. [Exact controller audit](controller-information-audit.md).

Residual PPO receives 129 channels (body velocity/gravity, 37 relative joint positions/velocities, previous residuals, seat-relative center/dimensions, phase/yaw). Action path adds `0.08 * clipped_actor_action` to the teacher trajectory, with joint clipping. Removing reference support destroys the skill. RSL-RL runner/configuration is inherited in `tasks/full_transition.py`; recorded checkpoints/configs, not the latest package defaults, define historical network settings. Old 127-input residual weights were padded with zero yaw columns to 129. General-purpose geometry adaptation is primarily scripted, not learned by PPO.

### Privileged DAgger student

Initial 99-input three-layer 256-wide SiLU MLP had low supervised error but failed closed-loop (0/144). Selected `StructuredStudent` uses a **10-feature geometry/time/contact head, 3 x 256 SiLU layers, 37 absolute joint targets**, plus fitted COM/velocity ankle gains near 8 and 1.5, bounded at ±0.35 rad and disabled after seat loading. `minimal_geometry=True` masks direct depth/width head features. Normalization/target scaling live in checkpoint buffers; head standard deviations floor at 0.01 / targets at 0.02 during initial fitting.

The 99-vector contains simulator body linear/angular velocity/gravity, q/dq, numerical seat geometry, root height, explicit elapsed/contact clocks, initial clearance, COM error and privileged support. Teacher trajectory/IK is absent at inference, but privileges and fitted classical balance remain. DAgger labels at learner states are in `datasets/student_dagger_01`; selected training arguments are `runs/student_dagger_01/training.json`. PPO refinement has actual updates and exported absolute-target network, but not a demonstrated generalization gain over the frozen DAgger baseline. [Observation audit](deployment-observation-audit.md), [controller audit](controller-information-audit.md).

### Sensor-based actor: exact observation/action interface

| Zero-based slice | Channels |
|---|---|
| 0:23 | 23 body joint encoder positions, radians |
| 23:46 | Encoder velocities, rad/s |
| 46:69 | Estimated motor torques, Nm |
| 69:72 / 72:75 | IMU gyro / projected gravity |
| 75:98 | Previous joint-position commands, radians |
| 98:101 | Estimated seat center XYZ in **initial gravity-aligned body frame** |
| 101 | Estimated seat top height above depth-estimated floor |
| 102 / 103 | Visible unobstructed depth / width |
| 104:106 | sin/cos estimated heading |
| 106:108 | Confidence / validity, masked to zero after normalization in actor 05 |
| 108 | Sit command |

109 channels packed, **107 effective actor channels**. Validity rejects a sit upstream; confidence is uncalibrated diagnostics. No verified G1 foot-feedback channels: **zero foot-force input channels**. `deployment.py` is authoritative for BODY_JOINT_NAMES/SDK_WIRE_SLOTS; never substitute the 37-joint simulator order for the 23-body contract.

Selected actor: normalize with saved mean/std, clamp ±20, geometry+command slice -> 128 SiLU -> GRU(hidden 128) -> linear 23; separate ankle head concatenates normalized 109-vector and recurrent latent (237), then 128 SiLU ->64 SiLU ->2, replacing ankle pitch outputs 15/16. Outputs are scaled to **absolute q targets**, not residuals/torque. Fourteen simulated hand targets are parked. Main body head has **no live proprioceptive feedback**; its recurrent inference count is an implicit trajectory clock. Ankle head is feedback-driven. Resetting hidden state every step removes sitting; no external explicit timer/teacher IK is used by this actor.

Sensor emulator reads only q/dq/applied torque/body gyro/projected gravity. Torque EMA is `0.8 previous + 0.2 current`; nominal causal delay 1 step (20 ms). Noise std: q .001 rad, dq .01 rad/s, torque .25 Nm, gyro/gravity .002. Own previous commands are immediate. Geometry estimate is latched at start. Simulation applied torque is only a proxy for hardware `tau_est`. Guard is .05 rad/control step (2.5 rad/s), then original joint-position limits; effort caps remain in actuators. Startup torque can reflect pre-teleport/reset dynamics; no silent zeroing/warmup shortcut is validated. The evaluator stores the previous guarded proposal before the action term's final joint-limit clipping; if future targets reach limits, verify that the sensor history reflects the actual issued command rather than assuming they are identical.

Working soft-landing estimator: 109 normalization clamp ±10, masked status, encoder 64 SiLU, GRU64, sigmoid contact probability / clipped load fraction. Full recurrent model 02; finite-window models use 32 samples (640 ms). Teacher label is **previous post-step support at packet time**, >.05 bodyweight (~15.8 N), not any infinitesimal touchdown. Labels are privileged; actor never receives them. Actions governor limits late ankle steps to `.004-.002*p` rad when mean knee q>1.0; estimated contact p>.65 limits hip/knee steps to .0025 rad. `--disable-contact-feedback` retains a fixed limiter. This authored control provides most safety improvement, not new learned motion. [Measured soft-landing report](soft-landing-progress.md).

Datasets retain synchronized 20 ms observations/actions/q/dq/torque/reference/geometry/contact/outcome records; force histories cover 5 ms physics. Pre-action SensorPacket is at t, post-state at t+.02; respect this when labeling. `soft_contact_teacher_02` has 96 unique successful teacher cases; `_01` repeats 24 of these with a different batch realization. `soft_landing_dagger_01` has 24 successful guarded learner trajectories with teacher targets as supervision only. Protected evaluation banks in the latest sensor trainer include 9612,10514,10515,24001,25001,26001. Do not train on them or concatenate teacher packets and post-state labels without checking alignment.

## 5. Checkpoints: selection and configuration

All paths below are relative to project root; file existence and selected metadata/hashes verified during handoff.

| Role/status | Exact path | Matching configuration/evidence |
|---|---|---|
| G1 infrastructure smoke only | `IsaacLab/logs/rsl_rl/g1_infrastructure/2026-10-08_23-13-49_smoke/model_4.pt` | Isaac-Velocity-Flat-G1, 128 envs, five updates; not sitting |
| Seated-hold PPO baseline | `IsaacLab/logs/rsl_rl/g1_seated_curriculum/2026-10-09_00-06-27_seated_stability/model_19.pt` | Evaluate **settled** initial pose and original gains, not authored penetrating reset |
| Partial-descent warm start | `IsaacLab/logs/rsl_rl/g1_seated_curriculum/2026-10-09_00-20-27_partial_descent/model_57.pt` | Historical curriculum stage 2 |
| Selected fixed-seat residual PPO | `IsaacLab/logs/rsl_rl/g1_full_transition/2026-10-09_10-08-57_seed_7/model_96.pt` | FullTransition task and scripted reference; seed 7 selected for lower measured impact |
| Other successful fixed-seat seeds | `IsaacLab/logs/rsl_rl/g1_full_transition/2026-10-09_10-09-25_seed_42/model_96.pt`; `IsaacLab/logs/rsl_rl/g1_full_transition/2026-10-09_10-09-52_seed_123/model_96.pt` | Same task; preserved, not defaults for sensor actor |
| Best broad geometry teacher | **No neural checkpoint**; `tasks/geometry_contact_v12.py` | Zero residual; archive `baselines/geometry-conditioned-rigid-v1`, 432/432 |
| Preserved furniture residual | `baselines/rigid-furniture-medium-v1/policy_129_inputs.pt` | Geometry-conditioned evaluation; no consistent advantage over scripted |
| Geometry PPO pilot, unpromoted | `runs/geometry_reference_v11_ppo_pilot_01/model_79.pt` | 80 updates with **v11** teacher; v12 evaluations are transfer, not v12 training |
| **Best preserved privileged student** | `baselines/dagger-student-v1/best.pt` (same selected model `runs/student_dagger_01/best.pt`) | Structured99/37, minimal_geometry=True; 432/432 after supplemental fresh evaluation |
| Privileged student PPO, experimental | `runs/student_dagger_rl_pilot_01/student_final.pt` | 60 anchored updates, seed113, bank22007; recorded runner `model_59.pt` is a different serialization than exported student |
| Original sensor actor 04 | `runs/deployment_geometry_recurrent_04/student.pt` | GRU geometry128, 109/23; confidence sensitivity, not preferred |
| **Current sensor actor 05** | `runs/deployment_status_mask_05/student.pt` | Same learned weights as04, masks normalized confidence/validity; **not hardware-ready** |
| **Working soft-landing observer** | `runs/contact_estimator_02/contact.pt` | Teacher96, 1200 fitting steps, full recurrent; pair with actor05, mode `actions` |
| Explicit current-controller archive | `baselines/sensor-soft-landing-155-of-168/{student.pt,contact.pt,soft_landing.py,evaluate_geometry_reference.py}` | `manifest.json`, `sha256.json` and original benchmark references; experimental baseline |
| Latest observer, **not promoted** | `runs/contact_estimator_conservative_05/contact.pt` | Window32, negative loss weight3; improved classification, not safe-tail generalization |

Key verified SHA-256:

- DAgger: `4e187d9924c8b0357c9a0bde06a626df784af9c7005f67ae7dfaeecb5525e2e3`.
- Sensor05: `bdd8ee4a45955e7d27f37956047f2c2ecc68b709baebfb49e32780d276b8a2b1`.
- Sensor04: `6d06843bec0fd610b12f6a412e36fe1545161ffd15fbf15765e4c1e04c96ce24`.

Other immutable archives: `baselines/fixed-seat-v1`, `baselines/rigid-furniture-medium-v1`, `baselines/deployment-student-experimental-v1`, `baselines/deployment-status-mask-experimental-v1`, `baselines/soft-landing-experimental-v1`. Their snapshots are restoration/reference artifacts, not independently installed packages; restore only deliberately and preserve current source first. The DAgger archive's original manifest says 288 trials; supplemental `runs/student_dagger_fresh_01` raises total to 432. Do not edit historical manifests to erase chronology.

## 6. Results, safety and lessons

**Unchanged whole-trial criteria:** 20 s without automatic reset; standing verified before descent; free root/gravity. Fall root Z<.28 m or projected gravity Z>-.65. Pelvis-floor force>30 N invalid. Knee/foot seat or arm/hand unintended/downward support>10 N invalid at any physics sample. Descent faster than .8 m/s invalid. Original joint effort caps invalid if exceeded. Valid pelvis/thigh seat support>.25 bodyweight, bounded seat-relative pelvis position, root speed<.2 m/s, angular speed<.35 rad/s, gravity Z<-.9, and >=3 s sustained seated hold; any latched past failure vetoes success. Force ceiling **632.527832 N** (~2 x316.263916 N bodyweight). Peak is resultant seat **normal** force, not friction or each local mesh force. Existing caps include 300 Nm on many simulated joints and 20 Nm ankles, **not verified hardware limits**. Details in `evaluate_geometry_reference.py` and each `results.json`.

| Validated experiment | Outcome / interpretation |
|---|---|
| Fixed-seat scripted + three PPO seeds | 216/216 complete transitions; zero-residual teacher already works. Detailed older report in [README](../README.md) and [commands](commands.md) |
| Small/medium slabs and rigid furniture | 108+108+144=360/360 selected historical stage trials; geometry expansion exposed wider-range failures; [furniture report](furniture-progress.md) |
| Geometry-conditioned contact-v12 teacher | 432/432, zero violations, worst616 N, min hold6.54s; [geometry report](geometry-conditioned-progress.md) |
| Privileged DAgger | 432/432 across three144 banks; timing/balance ablations fail, canonical geometry reduces success; [deployment report](deployment-progress.md) |
| Sensor05 alone, bank25001 | 15/36 clean, 21/36 noisy; no falls/wrong contacts but peaks1445/1421 N |
| Working soft landing (actor05 + observer02 + governor) | 33/36 clean,32/36 noisy; fresh44/48 clean,46/48 noisy; novel11/12. Four rectangular banks155/168, including novel166/180. Zero falls/wrong contacts in these principal banks; remaining failures are impacts |
| Fixed limiter without contact feedback | 165/180 vs contact166/180; average force benefit of contact feedback ~21 N, but success gain not statistically established |
| Latest conservative observer | Same155/168 rectangular aggregate; fresh noisy45/48 versus46/48 baseline. Improved classification, not consistently safer |

Latest force-tail comparison (all trials, not successes only):

| Bank | Existing working controller success; peak P95/P99/max N | New observer success; peak P95/P99/max N |
|---|---|---|
| Clean36 | 33/36; 690/761/773 | 34/36; 664/892/919 |
| Noisy36 | 32/36; 746/1125/1224 | 32/36; 754/1140/1227 |
| Fresh clean48 | 44/48; 725/1091/1168 | 44/48; 828/991/1114 |
| Fresh noisy48 | 46/48; 596/687/715 | 45/48; 623/704/728 |

Latest clean observer precision/recall 99.41%/99.92% vs98.51%/99.75%; >200 ms early episodes2 vs6. Post-onset P95 latency20 ms, but occasional startup predictions >11 s early remain. Long seated holds dominate pooled sample metrics; this is not a reliable actual-touch detector. [Complete safety report](soft-landing-safety-investigation.md), [quantitative JSON](../runs/soft_landing_02/metrics.json), [all14 baseline failures](../runs/soft_landing_02/baseline_diagnostics.json), [failing examples](../runs/soft_landing_02/failing_examples.png).

**Failures worth retaining and not repeating without new evidence:**

- Original shin cylinder (`knee_link`, ~150 mm) hit the seat front/underside during backward lean. Forward torso lean and correct loaded-G1 geometry resolved this without disabling collision; [geometry validation](../runs/transition_geometry_validated_01/analysis.json).
- Low BC target RMSE did not predict closed-loop stability. Teacher-action imitation and ankle-only DAgger updates broke balance; `runs/soft_contact_bc_01`, `_02`, `soft_contact_ankle_bc_03`, `soft_contact_ankle_dagger_04` are **rejected checkpoints**. Last ankle-only DAgger0/36,18falls,36wrong/force failures; posture-gated fallback still0/36. Do not use these as defaults.
- Body-GRU slowing produced new failures; episode REINFORCE rate pilot18/36 and1490 N tail, not improvement. No extensive sensor-actor PPO was justified.
- `smooth` extra ankle slew limiting lowered clean median418->357 N but worsened max773->923 N. Ankle damping4->8 destabilized (20/36,4falls,5wrong,2577 N max). Both optional experiments rejected.
- All14 principal180 baseline force violations are near **first contact**, not late stable-hold spikes. Predicted contact often precedes any seat force; target steps near impacts are already ~.002 rad. Simple slower commands or greater damping are not sufficient.
- Height-channel-only Gaussian std10 mm test35/36 but1104 N max. Increasing sensor delay20->60 ms gives12/36,11falls,24wrong,7405 N max. Separate20 ms command delay gives31/36,2wrong,1719 N max. Preserve these diagnostic trials; no criteria changed.

Do not pool repeated replays as independent evidence. Bank25001 is12geometries x1seed x3starts;26001 is8 x2 x3. Novel bank24001 uses ellipse/chamfered surfaces. Training banks22001 and22007 are separate. Video single-env sensor random realizations differ from full batches: paired videos match each other, not exactly the corresponding full-batch force value. Contact-aware video512 N vs matched baseline973 N is a selected example, not the full distribution.

## 7. Reproducibility commands

**Documentation only; none executed during this handoff.** From project root. Use genuinely new output names; many shell orchestrators use fixed historical destinations and will refuse overwrite. Do not run those orchestrators unchanged. The new agent should begin with section9's representation work, not the training examples below.

```bash
cd /home/kadir/humanoid-sitting
# Minimal simulator; writes the historical fixed result pathname, so preserve it first if rerunning.
bash scripts/run.sh python ../scripts/minimal_sim.py --device cuda:0 --info \
  --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'

# Infrastructure PPO protocol: existing stack, five iterations, NEW run label.
bash scripts/run.sh isaaclab train --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-G1 --num_envs 128 --max_iterations 5 \
  --device cuda:0 --seed 42 --experiment_name g1_infrastructure --run_name handoff_NEW --info \
  --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false' physics=isaacsim_physx

# Existing locomotion checkpoint loading/G1 inspection; output filenames are fixed in this script.
bash scripts/run.sh python ../scripts/evaluate_g1.py --full_asset --device cuda:0 \
  --checkpoint logs/rsl_rl/g1_infrastructure/2026-10-08_23-13-49_smoke/model_4.pt

# Validated geometry teacher, zero residual, registered task, held-out protocol.
bash scripts/run.sh python ../scripts/evaluate_geometry_reference.py \
  --v12-reference --production-task --geometry-stage 2 --geometry-seed 10514 \
  --geometry-count 16 --families --arm-clearance --seeds 3107 3108 3109 \
  --output ../runs/handoff_teacher_NEW --device cuda:0

# Privileged DAgger protocol; does NOT use deployment-realistic observations.
bash scripts/run.sh python ../scripts/evaluate_geometry_reference.py \
  --student-checkpoint ../baselines/dagger-student-v1/best.pt \
  --geometry-stage 2 --geometry-seed 10514 --geometry-count 16 --seeds 3107 3108 3109 \
  --compact-recording --summary-only --output ../runs/handoff_dagger_NEW --device cuda:0

# Current sensor-based controller, clean matched bank. Do not omit .05 action guard or actions mode.
bash scripts/run.sh python ../scripts/evaluate_geometry_reference.py \
  --deployment-checkpoint ../runs/deployment_status_mask_05/student.pt \
  --contact-estimator ../runs/contact_estimator_02/contact.pt --soft-landing-mode actions \
  --deployment-perception ../datasets/deploy_rgbd_confirmation_05 --deployment-action-rate .05 \
  --geometry-stage 2 --geometry-seed 25001 --geometry-count 12 --seeds 3401 \
  --compact-recording --output ../runs/handoff_sensor_clean_NEW --device cuda:0
# Noisy matched bank: replace perception with
# ../runs/deployment_perception_stress_confirmation_05/depth_noise_5mm_dropout_10pct
# Fresh48 bank: seed26001,count8,seeds3501 3502, perception ../datasets/soft_landing_rgbd_fresh_01
# Fresh noisy perception: ../runs/soft_landing_perception_fresh_01/depth_noise_5mm_dropout_10pct
# Pure actor ablation: omit observer/mode. Fixed limiter: add --disable-contact-feedback.

# Recorded paired-video protocol (nominal N=1, same bank/config/seed, not full-batch noise realization).
bash scripts/run.sh python ../scripts/evaluate_geometry_reference.py \
  --deployment-checkpoint ../runs/deployment_status_mask_05/student.pt \
  --contact-estimator ../runs/contact_estimator_02/contact.pt --soft-landing-mode actions \
  --deployment-perception ../datasets/deploy_rgbd_confirmation_05 --deployment-action-rate .05 \
  --geometry-seed 25001 --geometry-count 12 --seeds 3401 --compact-recording \
  --video --video-geometry-index 4 --video-benchmark-config 0 \
  --output ../runs/handoff_sensor_video_NEW --device cuda:0 \
  --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'
bash scripts/run.sh python ../scripts/plot_scripted.py ../runs/handoff_sensor_video_NEW

# Existing matched-start RGB-D capture protocol; input fixture has root/q initial arrays.
bash scripts/run.sh python ../scripts/capture_deployment_geometry.py \
  --fixture ../runs/deployment_fresh_teacher_fixture_05 --output ../datasets/handoff_rgbd_NEW \
  --device cuda:0 --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'
bash scripts/run.sh python ../scripts/stress_deployment_perception.py \
  --capture ../datasets/handoff_rgbd_NEW --output ../runs/handoff_perception_stress_NEW
```

**Training examples, recorded for reproducibility only; do not launch as the next task:**

```bash
# Selected privileged DAgger fitting protocol.
bash scripts/run.sh python ../scripts/train_sitting_student.py \
  --train ../datasets/teacher_train_01 ../datasets/teacher_train_expanded_01 ../datasets/student_dagger_01 \
  --validation ../datasets/teacher_validation_01 --checkpoint ../runs/student_bc_expanded_01/best.pt \
  --structured --minimal-geometry --steps 8000 --seed 101 --device cuda:0 \
  --output ../runs/handoff_dagger_train_NEW

# Recorded sensor04 learning protocol; this reproduces pre-status-mask model, not automatically actor05.
bash scripts/run.sh python ../scripts/train_deployment_student.py \
  --fixtures ../datasets/teacher_train_01 --perception ../datasets/deploy_rgbd_train_02 \
  --dagger ../datasets/deployment_adaptation_02 --variant geometry_recurrent \
  --steps 4000 --seed 401 --device cuda:0 --output ../runs/handoff_sensor_train_NEW

# Working observer02 fitting protocol.
bash scripts/run.sh python ../scripts/train_contact_estimator.py \
  --dataset ../datasets/soft_contact_teacher_02 --steps 1200 --output ../runs/handoff_contact_train_NEW

# Existing privileged PPO refinement protocol, not sensor-policy refinement.
bash scripts/run.sh python ../scripts/refine_sitting_student.py \
  --checkpoint ../runs/student_dagger_01/best.pt --iterations 60 --seed 113 --geometry-seed 22007 \
  --actor-learning-rate .000002 --critic-learning-rate .0001 \
  --anchor-dataset ../datasets/teacher_train_01 ../datasets/teacher_train_expanded_01 \
  --output ../runs/handoff_privileged_rl_NEW --device cuda:0
```

For fixed-seat PPO commands/configurations see [commands.md](commands.md); infrastructure originals [commands-before-curriculum.md](commands-before-curriculum.md). Latest exact experiment matrix: `scripts/run_soft_landing_causal_checks.sh`, `run_soft_landing_estimator_holdouts.sh`, `run_soft_landing_videos.sh` and each output's `command.json`. `scripts/report_soft_landing_safety_02.py` verifies matching initial arrays/geometry/ceiling and checkpoint hashes, but writes its historical report paths; do not rerun it casually over a fixed evidence set.

## 8. Limitations, warnings and development state

**Perception/frame limitations:** current geometry is captured only at initial standing pose and latched. Camera is simulated pelvis-mounted rear/down facing, 320x240, fixed offset `(0.05,0.18,0.55)` m, ROS optical convention. It is proposed equipment, not a verified G1 camera. Depth-only estimator uses known intrinsics/extrinsics plus IMU gravity, infers floor, finds dominant horizontal support, minimum-area oriented rectangle, and backrest evidence for front/back disambiguation. It assumes width>depth in validated furniture; heading can remain modulo pi without backrest. Hidden extents are not recovered. Clutter, arbitrary shapes, curved/tilted surfaces, compliant cushions, moving furniture, live mid-transition vision and approach/alignment are not solved.

`surface_center` is expressed in a **floor-shifted gravity-aligned body frame**, while `position_initial_body_level` is its equivalent in initial level-body coordinates. XY matches, Z differs by `floor_height_body_m`; these are not simulator world coordinates or the current pelvis frame. The 10-vector contains both body-relative center Z and floor-relative seat height. `estimate()` is an older generic routine and some earlier `capture_seat_rgbd.py` / `evaluate_seat_perception.py` experiments use simulator camera-world extrinsics. Those early tests are not the strict body-frame deployment pipeline. Use `estimate_body_frame()` and `capture_deployment_geometry.py` as the starting point.

Visibility gate requires depth>=.20 m,width>=.35 m,area>=.08 m² and >=75% observed coverage of an aligned20x24cm patch; no fallback to simulator geometry. Corrected central/lateral occlusion tests refuse all starts; refusals are not successful sitting or validated standing fallback. Confidence is point-count-based, not calibrated covariance. Ideal simulated sub-mm/millimetre errors do not establish real camera accuracy.

**Contacts/physics:** `runs/contact_binding_validation_02/audit.json` confirms all44 robot bodies +seat rigid-body bindings, filtered reaction consistency and static total load316.278 N vs316.264 N weight. 162 warning paths originate from broad `create_rigid_body_view("/**")` scanning non-rigid prims; not missing actual sensor links. Warnings were investigated, not suppressed. The deliberately interpenetrating knee positive-control generated117.9 kN and was correctly rejected; it is a diagnostic, not a policy rollout metric. Contact history covers4x5ms and eager sensor updates. Keep these checks when adding visualization or changing camera scene. RTX startup warns that CUDA_VISIBLE_DEVICES and Omniverse enumerations differ; inspect actual GPU startup selection, not env string alone.

**Hardware:** no verified foot-force channels, no publisher/physical rollout; HG LowState q,dq,tau_est+IMU fields and tentative slots are audited in `runs/deployment_interface_01/sdk_audit`. Actual G1 variant, joint signs/frames, motor mode/torque calibration, effort envelope, camera mount/calibration/mass, IMU convention and timing are unverified. Root translation cannot be read from an IMU; robot kinematics alone gives relative link poses, not absolute floating-base position. This is central to the next milestone. Do not feed simulator root translation or forces into a proposed deployed observer.

**Git/versioning:** project root is **not a Git repository**. `git -C IsaacLab status --short` is empty; branch `release/3.0.0`, commit above. No handoff commit created. Root custom-code history is source snapshots, hashes and immutable baseline archives, not Git diffs. Preserve them. Compared with `runs/soft_landing_01/source_final_sha256.json`, latest changes are:

- `sitting_task/src/humanoid_sitting/soft_landing.py`: optional rejected smooth mode.
- `scripts/evaluate_geometry_reference.py`: experimental damping and command-delay arguments; defaults retain working controller.
- `scripts/train_window_contact_estimator.py`: optional negative-class weight.
- `scripts/report_soft_landing.py`: P99 and precision/recall reporting.
- New diagnostics/report/orchestrator scripts: `diagnose_soft_landing.py`, `report_soft_landing_safety_02.py`, `write_soft_landing_report_02.py`, `run_soft_landing_causal_checks.sh`, `run_soft_landing_estimator_holdouts.sh`.
- README, latest safety documentation, new outputs/observer checkpoint and explicit baseline archive. Current five-file final snapshot hashes in `runs/soft_landing_02/final_source_sha256.json` match current files. Previous sources/README are preserved in `runs/soft_landing_02/source_before`.

**Processes:** handoff inspection found no project simulation, training, evaluator, uv-run or storage-monitor jobs. 4090 idle17MiB/0%;4070 desktop~203MiB/0%. SSH listener/session and Ubuntu background Python processes remain; do not kill them. All nine latest simulation evaluations completed, total348 trials; no interrupted/resumable training experiment remains. The shared-frame/visual-servo milestone is unimplemented, not an interrupted job.

**Storage:** roughly33–34GiB free; filesystem93% used. Latest safety stage used181MiB including experiment logs against250MiB budget, minimum30GiB free; [integrity record](../runs/soft_landing_02/integrity.json). README includes historical free-space figures; use `df -h .` for current state. Keep new next-stage data bounded (set a fresh budget before captures), use compressed arrays/selected frames, monitor before rendering. No Cosmos, foundation-model training, huge datasets or compliant-couch simulation yet.

## 9. Next milestone: shared 3D representation and visual servoing (PROPOSED)

User explicitly reprioritized this over touchdown tuning. Preserve the current controllers as comparison baselines; do not start with more actor training.

1. **Define and test explicit frames.** Document sensor optical frame C, pelvis/body P, gravity-aligned frame L, local floor/map frame F, seat frame S and their transform directions, metres/radians, handedness and quaternion ordering. Existing USD/FK exports use quaternion XYZW, camera APIs may differ; verify boundaries. Define the floor plane/normal and seat plane/polygon/front edge in one frame instead of mixing latched body XYZ and floor-relative height.
2. **Build synchronized observation records.** Start from existing compressed RGB-D captures, known K/fixed camera mount, IMU orientation and measured q/dq. Reuse actual loaded G1 joint frames/FK, not assumed limb lengths. Represent seat surface/visible support polygon, floor plane and pelvis pose together, with timestamps, validity, uncertainty and occluded regions. Privileged simulator geometry/body pose may validate transforms only, not supply deployed estimates.
3. **Resolve floating-base observability explicitly.** Using pelvis as a current frame avoids pretending IMU supplies translation; for tracking a local floor/map across motion, use depth registration/visual odometry and/or justified stationary-foot kinematic constraints with IMU. No verified foot-force input is available. Detect/represent possible slip/contact ambiguity rather than silently using true simulator foot contact. Kinematics estimates link/pelvis-relative geometry; it does not independently determine world root position.
4. **Validate an overlay before control.** Save RGB/depth/point-cloud views with estimated floor plane, seat polygon/front edge and pelvis/support-body geometry in the same frame. Compare against simulator truth offline. Quantify transform/plane/position/orientation/clearance errors, delayed-frame drift and occlusion; check robot self-mask/visible regions. Existing `G1Model.fk` and geometry model03/04 offer a validated simulation FK starting point but currently accept privileged root pose; replace that input with an explicit estimated/relative frame when used for deployment.
5. **Introduce live estimates and visual-servo errors.** Track seat-to-pelvis displacement, estimated vertical separation, front-edge clearance and sitting-area occupancy during motion instead of only at startup. Separate geometry estimation from motion policy. Define bounded, uncertainty-aware target updates and stale/invalid-observation refusal/hold behavior; validate them under gravity and latency before any controller integration. Keep approach/alignment separate until aligned-seat servoing is demonstrated.
6. **Evaluate fairly.** Keep the original matched banks, initialization seeds,633N force ceiling, effort/contact/fall/descent criteria and failure latching. First milestone is a coherent, accurate live3D representation across the transition; a 3D visualization alone is not closed-loop visual-servo success. Future servo improvement requires measured clearance/trajectory/safety gains on held-out/noisy cases, with successful and failing overlays/rollouts retained. Launch imitation/RL only after representation and observability are established.

No live point-cloud registration, shared-map estimator, synchronized moving-camera dataset, perception feedback actor or hardware visual servo has been validated. This plan is the next agent's work, not a completed capability.

## 10. Primary evidence to read rather than duplicate

- [Controller information audit](controller-information-audit.md): teacher/residual/privileged student information and inference boundaries.
- [Deployment observation audit](deployment-observation-audit.md) and [deployment progress](deployment-progress.md): strict interface, SDK assumptions, perception admission and ablations.
- [Geometry-conditioned progress](geometry-conditioned-progress.md): 432teacher transitions and unpromoted residual PPO.
- [Soft-landing progress](soft-landing-progress.md): datasets, failed imitation/RL, current governor and matched180 trials.
- [Latest safety investigation](soft-landing-safety-investigation.md): all latest candidates, causal interventions, precision/recall, force P95/P99, joint effort and stability.
- [Latest metrics](../runs/soft_landing_02/metrics.json), [failure case table](../runs/soft_landing_02/baseline_diagnostics.json), [failing examples](../runs/soft_landing_02/failing_examples.json), source hashes and integrity records alongside them.
- [Paired transition video](../runs/soft_landing_01/paired_transition.mp4), [paired physics](../runs/soft_landing_01/paired_physics.png), [impact distributions](../runs/soft_landing_01/impact_distributions.png). Representative success is not certification.
- [Contact bindings audit](../runs/contact_binding_validation_02/audit.json); `runs/deployment_interface_01` for paired-condition assertions, SDK audit and older ablations.
- Exact run-local `command.json`, `results.json`, `initial_conditions.npz`, `rollouts.npz`, model/estimator source snapshots and training configuration are the authority when prose differs from a historical run.

End of handoff. No training, policy changes or installations were performed to prepare this document.
