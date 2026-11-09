# Humanoid Sitting — ChatGPT → Mac Codex handoff

**Prepared:** 2026-10-09 (America/Los_Angeles)  
**Purpose:** Transfer the high-signal technical context and decisions from an extensive ChatGPT conversation to a **new Codex session running locally on the Mac mini**, which will control the Ubuntu GPU workstation over **passwordless SSH**.  
**Status:** Historical account based on screenshots/transcripts supplied in ChatGPT. **The files, code, checkpoints, metrics and active processes on Ubuntu are the source of truth. Verify them before any further modifications.** This is a detailed *handoff summary*, not a verbatim export of every ChatGPT message.

## 0. IMPORTANT — latest user decision supersedes all previous next-step requests

**STOP iterating on the soft-landing contact-force limiter, estimator thresholds, damping or PPO simply to improve the simulated seat-impact metric.** The user has explicitly decided to stop because the research has been circling the same issue without solving spatial perception and relative control.

**NEW PRIMARY TASK:** Build and validate a **shared 3D representation of the seat/floor and the robot's own body**, using deployment-realistic 3D sensing and proprioception. The robot needs to know, in a consistent coordinate frame, **where its pelvis/hips are relative to a candidate sitting surface**, so it can align and lower itself using closed-loop visual/geometric servoing. This geometric state-estimation capability must work before optimizing additional sitting-policy training. Preserve all previous successful controllers and research artifacts.

User's desired functional sequence:

1. Perceive candidate sitting support surfaces (chair, couch, bench, box, table where physically feasible), floor, edges and obstacles with onboard depth/point-cloud sensing (and RGB if available).
2. Fit usable surface geometry and determine what is sittable; do not rely simply on furniture category labels.
3. Register the observed furniture and robot pelvis/feet/kinematics **in the same reference frame** with calibration, robot forward kinematics and IMU/odometry.
4. Estimate and track pelvis-to-seat **height/clearance**, longitudinal/lateral offset, orientation, support region, and uncertainty, including when the surface leaves the camera view as the robot turns.
5. Visual-/geometric-servo the robot to an appropriate approach/alignment pose; then control its descent relative to the estimated seat surface, not to a hard-coded joint-angle schedule.
6. Use robot-available proprioceptive feedback for contact/weight-transfer inference and stabilization; don't require a real chair to report contact forces.
7. Train a student sitting policy only once its inputs are meaningfully grounded in actual onboard sensors and a validated shared 3D frame; maintain privileged simulator state **for teachers, debugging and evaluation only**.

### Exact initial next milestone (do not expand scope yet)

**In Isaac Sim, using simulated depth camera and/or LiDAR measurements plus robot state/kinematics, reconstruct the usable seat and floor, compute pelvis pose in the SAME 3D frame, display/record the result, and demonstrate accurate pelvis-to-seat relative tracking across robot/sensor motion.** Use Isaac Sim ground truth exclusively as a metric for evaluating the estimator; do not feed it into the estimator or deployed actor.

For example, with a gravity-aligned local odometry frame `O`, estimated seat frame `S`, and pelvis/hip frame `H`:

```
T_O_S = estimate_from_depth_point_cloud_and_odometry(...)
T_O_H = T_O_base * FK(base_to_pelvis; joint_encoders)
T_S_H = inverse(T_O_S) * T_O_H
```

Conventions for transform direction must be documented and tested (the symbols above are illustrative); distinguish pelvis/root frame from physical hip contact surface. Useful outputs include signed pelvis-contact-proxy clearance to the seat plane, closest approach to a usable support polygon, yaw error, lateral/forward offset, and confidence/covariance. Correctly model sensor extrinsics and the camera's field of view, occlusions, delayed frames and floor-relative body pose. Avoid using visual ground-truth poses, simulator contact forces, or global reset metadata in runtime estimates.

**Exit criteria:** for multiple chairs/benches/couches/boxes (including geometries excluded from fitting), report seat/floor reconstruction errors, pelvis-to-surface clearance errors in centimeters, lateral/yaw pose errors, frame timestamp consistency, occlusion robustness, and live 3D visualizations showing sensor cloud + fitted planes/surfaces + robot meshes/joint skeleton + pelvis/seat frames. Produce a short video/recording where robot moves and the estimated relative transform updates correctly. Agree a reasonable metric tolerance based on depth-sensor noise and the target joint/contact geometry rather than choosing arbitrary thresholds.

Do not start expensive RL, training new policies, Cosmos, or hardware deployment as part of this first milestone.

---

## 1. Execution environments and intended workflow

### Mac mini — coordinating Codex

- Local user account: `kadir`.
- Mac's local Ethernet IP previously observed: `192.168.68.53` (`en0`, wired 1 Gbps). Wi-Fi/other interface also had `192.168.68.56`, but route to Ubuntu selected `en0`.
- Passwordless SSH is **verified working** from the Mac using the alias:

  ```bash
  ssh kfuUbuntu
  ```

  The SSH alias resolves to `kadir@192.168.68.60`. `scp kfuUbuntu:... ~/Downloads/` should work without prompting for the Ubuntu account password.
- New desired workflow: **Start one Codex agent locally on the Mac**, with this handoff file in its working directory. **That agent should execute work on Ubuntu through SSH**. There is *no need to start a second Codex agent on Ubuntu*. Avoid independent simultaneous agents editing the same repo.
- The Mac Codex agent may execute non-interactive commands such as:

  ```bash
  ssh kfuUbuntu 'hostname; nvidia-smi; df -h /'
  ssh kfuUbuntu 'cd ~/humanoid-sitting && git status --short'
  ssh kfuUbuntu 'cd ~/humanoid-sitting && cat README.md'
  ```

  Verify the correct repository location before assuming Git root. For long-running GPU jobs, arrange **remote `tmux` sessions or another supervised persistent process** and monitor remote logs/exit statuses. A persistent tmux *terminal* is not by itself a continuously autonomous agent; Codex still needs bounded tasks and clear continuation/stop criteria.
- The user previously expressed a preference not to approve every command. This requires deliberate permission configuration for the **Mac Codex session**; `--yolo` removes sandbox/approval safeguards and grants broad access to the Mac user account. Use only with explicit understanding of risks, avoid `sudo` or unrelated host edits, and prefer bounded command scope. Do not bypass safety-critical testing or alter existing system configs simply because command approvals are disabled.

### Ubuntu GPU workstation — execution target

- Hostname: `kfuHomeUbuntu`.
- User: `kadir`; IP: `192.168.68.60`.
- OS: Ubuntu 22.04.5 LTS; GLIBC 2.35; kernel seen as 6.8.0-138-generic.
- CPU: Intel Core i9-14900F, 32 logical CPUs.
- RAM: ~62 GiB.
- GPU 0: **GeForce RTX 4090 (24 GiB)**, intended for simulation/training.
- GPU 1: **GeForce RTX 4070 (12 GiB)**, serving the desktop.
- NVIDIA driver reported: **580.178.04** (CUDA supported by driver 13.0). No driver change was needed for validated runs. Treat future compatibility complaints as diagnosable, not a reason for blind upgrade.
- Early root free space ~86 GiB, falling over many experiments to **~33–35 GiB free by latest reports**. Recheck before downloads/data capture. `/data` was previously reported with ~85 GiB free but mount permissions/status should be verified afresh; do not assume writable. Many datasets, videos and caches are in the project.
- An earlier globally visible Isaac Sim 4.5 Python 3.10 installation predated the new project. Preserve old environments and drivers.
- Current project set up Python **3.12.15**, **Isaac Sim 6.1**, **Isaac Lab `release/3.0.0`** (a pre-release line at project start), **RSL-RL**; exact installed versions, revision, package locks and command syntax must be checked in the local checkout, logs, and `uv.lock`.
- Project root: **`/home/kadir/humanoid-sitting`**.
- Likely structure includes `IsaacLab/`, `sitting_task/`, `scripts/`, `docs/`, `runs/`, `logs/`, `datasets/`, `baselines/`, `cache/`, `tools/`.
- Most existing scripts used `bash scripts/run.sh ...` as the wrapper for the local pinned Isaac Lab runtime. Review `README.md`, `docs/commands.md` if present, existing source and their latest saved runtime commands, rather than inventing CLI flags for current Isaac Lab.
- Existing Ubuntu Codex CLI session was completed and can be closed using `/exit`; no need to preserve the old interactive agent session after handoff, but preserve files, logs, checkpoints, and active remote processes if any.

### Previous access blocker (solved)

The original Ubuntu Codex CLI sandbox hid `/dev/nvidia*` and DNS, despite the host working. It was fixed by permitting execution outside its sandbox: an elevated `nvidia-smi` worked and `git ls-remote https://github.com/isaac-sim/IsaacLab.git HEAD` worked. This was **not** a host network or NVIDIA driver failure. Mac-to-Ubuntu SSH has likewise been tested and succeeded. If Mac Codex reports network failure, check its own sandbox and SSH execution environment before changing the Ubuntu workstation.

---

## 2. Scientific goal and longer roadmap

**Ultimate goal:** a humanoid robot should be able to visually perceive a realistic sitting surface (chair, couch, bench, box, or feasible table), decide whether/where it can safely sit, approach and align itself, and lower its hips onto the surface with controlled contact and stabilization. First in NVIDIA Isaac Sim/Isaac Lab, later robust enough for physical humanoids. Cosmos may later help with image/appearance/data diversity; it cannot replace physical contact modeling or reliable 3D registration.

- Initial robot: **Unitree G1 simulated model**, currently apparently a 23-body-joint interface (verify exact model/joint mappings). Actual real G1 configuration **not yet chosen**.
- Physical sensor assumptions: RGB/depth cameras and/or LiDAR if fitted; joint encoders and joint velocity estimation; IMU; motor torque estimates if supported by selected hardware; previous action history. **Do not assume skin pressure sensors, seat-integrated force sensors, or foot pressure sensors exist.** Verify actual hardware variant and sensor topics before real deployment.
- A real seat/couch does **not** report seat contact forces. Simulated force sensors may be used as privileged training labels and safety evaluation, *never* as mandatory runtime inputs. Proprioception-based contact inference is uncertain and must have failsafes.
- Surface geometry is the transferable abstraction; chair/couch labels are secondary. Cushion deformation/compliance and dynamic load capacity require separate modeling and real-world measurement.
- Aligning to furniture in the room is **not demonstrated yet**: prior success tests started near pre-aligned seating surfaces with small randomized initial poses. Start with shared 3D frames/relative pose tracking, then alignment, then closed-loop descent, then real-device sensor adapter and transfer.
- Do not treat apparent PPO reward or a scripted successful trajectory as proof of visual understanding or learned self-positioning.

---

## 3. Chronology of experiments and major verified/reported outcomes

Below are important numbers **as reported by Codex in screenshots and snippets**. They are not independently reprocessed here. Verify in existing JSON files and logs and retain exact definitions, scenario banks, matched seeds, contact/force thresholds and checkpoints. Different test-bank percentages are often not comparable.

### A. Infrastructure and initial sitting

1. Isolated Isaac Sim 6.1/Isaac Lab 3.0 environment installed on Ubuntu. G1 physics and PPO verified on RTX 4090; Python 3.12.15. Custom `sitting_task` package installed in editable mode. Project script wrapper created.
2. A first fixed-seat PPO run completed 100 iterations and produced `model_99.pt`, but **0/736 successful sits** in initial evaluation; G1 fell before reaching seat. Collision/body contact and pose issues were diagnosed. Initial video playback/rendering occasionally slow or required explicitly selecting `--viz kit`.
3. Verified a physically supported scripted seated pose for **~28 seconds of stable hold in a 30-second simulation**; seat supported ~76% of weight in that test, feet remainder (reported, not direct hardware measurement).
4. Short PPO curricula reached seated hold **64/64**, near-seat descent **64/64**, partial descent **63/64**. Full standing-to-sitting was initially failing because shin/lower-leg collider swept underneath seat.
5. Scripted reference + residual PPO managed complete full transitions after forward lean/hip-knee timing/stronger ankle tracking; fixed-seat reference tests **216/216** with zero falls/unintended contacts and minimum hold >10 sec. Scripted demonstration was not independent learned sitting.

### B. Geometry-conditioned scripted control

6. Controller succeeded within smaller/medium furniture ranges: small slabs **108/108**, medium slabs **108/108**, medium rigid families (chairs/benches/boxes/couch shapes) **144/144**, but wider perturbations **127/144** with unwanted contacts and high peak forces. Several failures involved hands/thumbs hitting couch or shin-seat geometry.
7. A stronger **geometry-conditioned scripted baseline** achieved **432/432** transitions, including **208 held-out scenarios**, zero reported falls/unintended contacts/force or torque violations within its test bank, peak seat force ~616 N with then-633 N comparison threshold, max descent speed ~0.189 m/s, minimum seated hold ~6.54 s. Preservation regression bank **177/177**. These are constrained, simulated, mostly aligned starts; this does not prove unseen physical surface adaptation.
8. Geometry-adaptive PPO residual experiments sometimes regressed. Earlier same-bank comparison: preserved reference **144/144**, geometry-scripted **95/144**, geometry+PPO **105/144**, hidden-geometry PPO **100/144**. Later 80 PPO updates failed to produce reliable residual improvement and caused force violations. The scripted controller was retained as robust teacher/baseline.

### C. Imitation learning / DAgger and sensor-based student

9. Teacher generated demonstrations (robot states/actions, reference trajectories, seat/furniture geometry, simulated contacts, outcomes); supervised behavior cloning and **DAgger (dataset aggregation)** trained a student predicting **full joint target actions** rather than mere residual on a hard-coded teacher action. A learned student reported **288/288**, later **432/432** in tested simulation banks (no teacher trajectory or IK in action path); but ablations revealed important recurrent-timing and shared-balance dependencies. The frozen DAgger student is a useful comparison, not proof of hardware-ready autonomy.
10. Built a simulated depth/point-cloud surface estimator and deployment-oriented actor observation interface. Early prototype reportedly ~**109 numeric channels**, but that is historical and **may not be the current active input dimension**. Feeds derived geometry, joint encoder positions/velocities, estimated torques, IMU, previous actions, sit command and temporal context. **Not a raw-RGB end-to-end vision-to-motor network.** It is essential to inspect the actual code for exact vector layout, input normalization, recurrent architecture, and indirect ground-truth leakage.
11. A depth-based seat/floor estimation pipeline reported valid surface estimates in **96/96** simulated starts, including some robot occlusion, but unverified against real RGB-D/LiDAR or hardware mounting/calibration. Simulated contact forces/COM/root velocity were removed from deployment actor inputs; retained for privileged label generation and evaluation. Foot load sensor was not assumed.
12. First strict deployment student: **86/144** (59.7%) on fresh furniture, with **58 force-limit failures**. Additional independent corrected-student tests were **15/36** clean and **21/36** noisy, zero falls/unintended contacts but impacts ~1.4 kN. Delay/recurrent dependencies remained.

### D. Soft landing — current preserved baseline; now paused

13. Smoothing especially abrupt **ankle joint-target changes** substantially improved successful/safe sitting versus previous student on matched banks:

   | Test | Old student | Soft-landing baseline | Reported max force after |
   |---|---:|---:|---:|
   | Clean | 15/36 | **33/36** | ~773 N |
   | Noisy | 21/36 | **32/36** | ~1224 N |
   | New furniture clean | 37/48 | **44/48** | ~1168 N |
   | New furniture noisy | 36/48 | **46/48** | ~715 N |
   | **Total** | **109/168** | **155/168** | Existing comparison ceiling 633 N |

   No falls/unintended contacts reported on those matched tests. Nonetheless, some contact peaks remained above the 633 N simulated benchmark threshold.
14. **All 14 baseline force-limit violations occurred at first seat contact**; the primary issue was impact before a reactive contact detector could respond. A new learned contact classifier reported ~99.41% precision / 99.92% recall against simulator labels, but **did not consistently improve the overall soft-landing controller**, with test-bank regressions. Other candidate braking/damping strategies lowered medians but worsened upper-tail force or balance. Delayed sensor/motor commands exposed fragility, including very large simulated force spikes under stress tests. No new candidate was promoted; the **155/168 baseline and all checkpoints remain preserved**.
15. Latest final report (screenshot 2026-10-09 ~23:04 PDT) said: preserved **155/168**, all 14 violations at first contact, new estimator improved classification but not safety, worse extreme tails in some cases, and no new controller promoted. Roughly **33–34 GiB** free on root and a larger amount preserved in project files. User explicitly decided to **STOP these iterative soft-landing/force-limit experiments** and refocus on shared 3D pose estimation / visual servoing.

### Limitations to retain when reporting results

- A 100% success rate in a bounded Isaac Sim geometry bank **does not establish** robust perception from hardware cameras, unseen real-world furniture, autonomous approach or safe real-hardware interaction.
- A scripted teacher, reference-only controller and DAgger student are separate systems. Keep identity/ablation labels accurate.
- Force/torque statistics are simulated metrics; the 633 N limit is a **project regression threshold**, not verified safe hardware rating. Ground-truth contacts are **not** a sensor on a real chair.
- Correlation of some geometry-input ablations with better PPO scores was modest; don't imply learned geometric understanding without controlled ablations.
- The latest student is not yet physically deployed and must not be represented as hardware-ready.

---

## 4. Architecture as currently understood; inspect and verify actual code

### Perception

- Isaac Sim RGB-D/depth sensor/point cloud pipeline feeding an estimated seating surface. Past screenshot referenced `scripts/seat_depth_estimator.py` and `scripts/capture_deployment_geometry.py`, as well as `datasets/deploy_rgbd_confirmation_05` (verify paths).
- Historical prototype estimated seat/floor height, support polygon, surface centroid, bounding box/extent, and may have an assumed camera pose. Camera extrinsics, body occlusion and real-hardware field of view are unvalidated. Current user-requested task is a **rigorous common-frame robot/seat representation**.
- Do not feed direct `scene['seat'].data.root_pos_w` or `robot.data.root_pos_w` to runtime; those are allowed for error metrics and privileged labels only. Be especially cautious about preprocessing that silently uses global simulator ground truth or reset geometries.

### Low-level motor observations and actions

- Actor apparently gets encoder positions/velocities, estimated motor torques, IMU, previous-action values, sit command, derived geometric features, and temporal/recurrent state. Verify *exact* current input fields, dimensions, frames, normalization and whether a history window or GRU/LSTM/other recurrent module is active.
- Student predicts full joint target commands. A separate command conditioner/rate limiter and motor control loop implement joint-level constraints, currently particularly smoothing ankle commands.
- Separate privileged ground-truth physics for safety evaluation: seat normal force, contact classification, precise body COM/root, body/world positions, impacts, friction and collisions.
- Timers/balance primitives were removed from later deployment student but recurrent timing still apparently influences performance. Audit for hidden leaks and hard-coded phase progression.

### Training methods already attempted

- RL PPO using Isaac Lab/RSL-RL for initial velocity task, custom fixed-seat and partial-descent curricula, and later residual policy refinements. Most PPO residuals didn't consistently outperform scripted reference on furniture.
- Behavior cloning + DAgger: teacher provides joint targets for states reached by student. Later DAgger student produces full joint targets from observations and history, with no reference teacher in direct action path.
- Supervised contact-state/seat-surface estimators trained or validated against synthetic ground-truth labels. Engineering control changes (e.g. ankle command slew-rate limits) provided major improvements that should not be falsely attributed to neural weight learning.
- Cosmos not yet introduced. Do not begin Cosmos until shared spatial perception and action alignment are validated.

---

## 5. Project resources, evidence and suggested discovery commands

**Ubuntu project root**: `/home/kadir/humanoid-sitting`

Known paths from screenshots (presence must be verified):

- `README.md` — project status and commands.
- `docs/commands.md`, `docs/geometry-conditioned-progress.md`, `docs/deployment-observation-audit.md`, `docs/deployment-progress.md`, `docs/soft-landing-progress.md`, `docs/soft-landing-safety-investigation.md` — some may have different actual names; use `find`.
- `sitting_task/src/humanoid_sitting/tasks/env.py` (historical), `.../tasks/furniture.py`, `.../tasks/geometry_feedback.py`, `.../deployment.py`, `.../soft_landing.py` — inspect actual layout.
- `scripts/run.sh`, `scripts/setup.sh`, evaluation/capture/training scripts; do not rerun setup unless missing environment.
- `baselines/geometry-conditioned-rigid-v1/` (historical screenshot), preserved teacher/student checkpoints; do not overwrite.
- `IsaacLab/logs/rsl_rl/...` — trained PPO artifacts.
- `runs/geometry_reference_progress_01/transitions.mp4` — scripted geometry-adaptive comparison.
- `runs/full_transition_comparison_02/comparison.mp4` — early reference/seat-hold comparison.
- `runs/furniture_progress_01/families.mp4` — furniture family comparison.
- `runs/soft_landing_01/paired_transition.mp4` — paired soft-landing example (~973 N vs ~512 N, not representative of all).
- `runs/soft_landing_02/` — final soft-landing safety findings, metrics, failing examples, reports, hashes.
- `runs/deployment_interface_01/`, `runs/deployment_status_fresh_05/` and `datasets/deploy_rgbd_confirmation_05/` — sensor-based student/camera work; inspect existence.

First use *read-only* commands from Mac Codex:

```bash
ssh kfuUbuntu 'hostname; nvidia-smi; df -h / /data; free -h'
ssh kfuUbuntu 'cd ~/humanoid-sitting && ls -la && ls docs scripts sitting_task runs datasets 2>/dev/null'
ssh kfuUbuntu 'cd ~/humanoid-sitting && find docs -maxdepth 2 -type f | sort | tail -80'
ssh kfuUbuntu 'cd ~/humanoid-sitting && sed -n "1,180p" README.md'
ssh kfuUbuntu 'ps -eo pid,stat,etime,cmd | grep -E "isaac|python|codex|train|tmux" | grep -v grep | head -60'
```

Use existing logs/checkpoint manifests to determine real current baseline. Verify file hashes when provided. Record relevant source revisions and avoid concurrent writes by multiple agents.

---

## 6. Handoff task to execute first on Mac Codex

**Stage 0: Confirm connectivity and protect existing work (read-only).**

- Verify `ssh kfuUbuntu` works without password and execute `nvidia-smi`, `df -h`, inventory project files/processes, check status docs; avoid modifying existing environments, baselines, checkpoints or Ubuntu drivers.
- Determine whether old Codex/Sim jobs are still active before starting new commands. If user already closed Ubuntu Codex, no need to restart it.
- Summarize actual versions, paths, current git status, disk budget and canonical best baseline.

**Stage 1: Audit frames, sensing and geometry pipeline (small, bounded).**

- Inventory precise G1 simulated sensor mounts/extrinsics, available depth image or point cloud, frame IDs, coordinate conventions, floor segmentation logic, actor observations, pelvis/hip frame FK chain, and body-relative seat transforms. Identify use of Isaac Sim ground-truth in any predictor/runtime path; restrict to tests/supervision.
- Define a consistent SE(3) transform tree from RGB-D/LiDAR `camera` → robot base → gravity-aligned local odom → seat frame; compute pelvis frame from robot joint states/kinematics.
- Provide explicit contract for frame direction, timestamps, covariance/uncertainty and transform composition. Use unit tests with synthetic known transforms, yaw rotations, robot base motion and pose resets.

**Stage 2: Implement shared 3D representation and visualize (NO RL training).**

- Obtain actual depth data from simulated onboard-mounted sensor; optionally integrate consecutive scans into robot-centered/gravity-aligned local map using odometry and calibration.
- Segment floor and accessible roughly horizontal seat support regions; compute plane, support polygon, front edge, seat normal, height above floor, uncertainty. Distinguish visually indistinguishable narrow unsafe supports, cushion compliance unknown, furniture backs/armrests.
- Convert robot body/pelvis and seat estimates into same frame and calculate signed clearance from appropriate hip/pelvis contact proxy to the seat top, lateral/fore-aft pose error, yaw error, shin/foot and back/arm clearance estimates.
- Demonstrate relative pose tracking as robot moves and seat becomes partly/fully occluded, using temporal mapping where possible. Compare sensor-only estimates to simulator ground truth *solely offline for quantitative error*.
- Generate a short visualization/video plus numeric report on several held-out furniture models. Use bounded evaluation/time/caches and avoid large datasets.

**Stage 3: Only after Stage 2 validated — closed-loop alignment.**

- Define simple visual/geometric servo control toward a feasible pre-sit pose using the estimated relative transform. Incorporate latency/uncertainty and controlled stop/abort criteria.
- Do not initiate a new sitting-policy/soft-landing PPO retraining loop until this is reliably tracking in the same frame.

**Exit and reporting discipline:** after each bounded milestone, write concise markdown results with commands, source change summary, actual measured errors and videos. Preserve prior baselines. If system is blocked, report exact error and propose limited fix, not silent sweeping changes or expensive parameter sweeps.

---

## 7. Suggested instruction to the newly started Codex agent on Mac

> Read `HUMANOID_SITTING_CODEX_HANDOFF.md` completely. You are running on the **Mac mini**, not on Ubuntu. Your execution target is the Ubuntu host `kfuUbuntu` via already-configured passwordless SSH, project `/home/kadir/humanoid-sitting`. Do **not** run a second Codex instance on Ubuntu; use SSH commands yourself and remote tmux for persistent GPU jobs. First verify SSH/GPU/disk and inspect the real project code, reports and checkpoints, reconciling any mismatch with this historical handoff. The user has STOPPED further seat-impact/PPO iteration. Implement the new priority: consistent **3D robot–seat registration** using simulated onboard depth/point-cloud observations, floor geometry, robot forward kinematics and odometry/IMU. Construct and visualize the seat and pelvis in the SAME coordinate frame and track relative position/orientation/clearance as the robot moves. Privileged simulator poses are allowed only as verification labels. Do not train new sitting policies until this capability is quantified. Preserve all existing checkpoints, baselines, environments, safety constraints and drivers. Work in bounded stages, monitor storage, and report measured results/videos and what to do next.

## 8. Notes on the intended tool architecture

- Starting **Codex locally on Mac** gives it direct access to Mac shell functions and (with appropriate permissions) to `ssh kfuUbuntu`, and from there command execution on Ubuntu. This **does not** make the original web ChatGPT conversation directly control Mac/Ubuntu; user wants to transfer working context and continue in Codex.
- No need for nested Codex on Ubuntu. It would add another approval/context boundary and risk conflicting edits. Reuse the old Ubuntu Codex's *code and artifacts*, not the old agent instance.
- To persist GPU jobs, use remote `tmux` sessions/daemonized logging. For a truly continuing multi-milestone agent, local Mac Codex itself must remain running (or be orchestrated in bounded scheduled invocations). Do not claim that SSH, tmux or a single Codex run automatically means 24/7 task continuation.
- **Latest user instruction is authoritative:** stop unnecessary sitting optimization loops; implement common-reference-frame spatial perception and relative body/seat visual servoing first.
