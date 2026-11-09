> **Direct effort PPO complete — October 12, 2026:** Fixed posture29/29; final PPO54511=11/29 and54517=15/29, with0/12 leaning in each. No falls; one forbidden-contact case per seed. On common successful trials, whole20s total squared torque increased65.46% /55.58%, mostly in the legs. Isolated40cm savings did not generalize. Both full budgets, exact starts/checkpoints,105 frozen and61 prior hashes pass. Steep75° perception errors persist separately. Read EFFORT_REST_STATUS.md and validated_effort_rest_01/final_results/review.html. No policy promoted or extra training; standing transitions remain unsolved. Earlier snapshots below are history.

---

> **Direct effort-minimization PPO running — October 12, 2026:** The user made total joint-effort reduction the primary objective. A separate policy learns bounded posture changes from validated sitting/leaning starts, with the stable motor reference outside the neural network. Fixed posture29/29 and two-update learned smoke13/13 full20s pass. Two fresh256-update seeds54511/54517 now train on GPUs0/1 with matched final torque/stability comparisons queued. Read EFFORT_REST_STATUS.md and EFFORT_REST_PROTOCOL.md. No final effort-saving claim yet; standing transitions remain unsolved. Previous experiments remain complete and preserved. Earlier snapshots are history.

---

> **Standing-to-support PPO complete — October 12, 2026:** Both 224-update seeds scored 0/13 training and 0/16 unseen surfaces; all 58 policy trials physically failed within 0.62–0.80 seconds, although rollouts continued for the full 20 seconds. Fixed standing remained stable in 29/29. Supported endpoint poses are validated, but reaching them is not learned. Command drift appears after the first PPO update; pre-failure tracking remains available but reaches 36 mm error. Budgets, exact startup/checkpoint matching and all 36 new/46 prior hashes pass. Read the current SUPPORTED_SURFACES_STATUS.md and validated_supported_surfaces_01/final_results/review.html. No policy promoted or additional training; completion monitor paused. Earlier snapshots below are history.

---

> **Standing-to-support PPO running — October12,2026:** Supported targets across level seats and slopes through75deg validated; corrected standing reset passed29/29 full20-second tests. Two fresh224-update PPO runs now train reaching support from standing, with effort secondary. Automatic evaluations reserve16 unseen configurations. Read rl_support_work/docs/SUPPORTED_SURFACES_STATUS.md and the frozen SUPPORTED_TRANSITION_PROTOCOL.md. No learned standing-transition or unseen-surface success claimed yet. Existing completion monitor reactivated for this pilot. Earlier snapshots are history.

---

# Bounded resting PPO complete — stability retained, no effort saving

October 12, 2026 (PDT). The persistent study finished at11:39a.m. Both64-update
PPO seeds53111/53117 passed56/56 matched20-second trials. Fixed posture also
passed56/56. Every original, intermediate, new-height and disturbance trial
passed, with zero fall, unwanted-contact, torque, impact or joint-margin flags.
Neither policy is promoted. No further training has been launched.

The primary whole-body normalized squared torque metric increased0.2774% and
2.5872% relative to fixed posture. Whole-body RMS Nm changed−0.0762%/+0.9619%;
leg RMS−0.0847%/+1.0945%; torso/arm/hand RMS−0.0814%/+0.0017%. New holdout
normalized effort increased0.2718%/+2.5429%. Effort worsened at every evaluated
height in both seeds; seed53117 is24.10% worse at26.25cm, so the pooled figure
hides substantial relative increases at low seats. No trial meets10% saving.
All effort comparisons are on common full-duration successes.

The measured-effort hold rule recorded zero entries/releases, and enabling it
produced bitwise-identical trajectories and results to plain PPO. No recorded
10Hz effort sample met its90%-of-initial-reference threshold (minimum ratios
0.91415 and0.99133 by seed). The full50Hz counters confirm no entry. Thus this
study cannot establish benefit or safety after an actual hold entry. The gate's
initial measured reference differs from the trial-average comparison baseline.

Largest joint-command changes were0.3135°/0.4147° from the reference. Policies
remained near the fixed pose; foot load increased from41.89% of bodyweight to
42.01%/42.66%. The bounded controller fixes accumulated command drift and the
trained policies retain stability, but they have not found lower-effort poses.
This evidence does not isolate exploration, short budget, reward sensitivity,
or local posture feasibility as the cause of weak effort learning.

Independent audit passed:131072 total transitions,127340 active plus3732 reset
warmup, completed/unfinished accounting reconciled; finite changed actor/critic
weights; zero initial mean action; all intended final checkpoints matched;
16 learned startup comparisons bitwise exact; equal geometry and disturbances;
constant fixed targets, bounded learned targets, unchanged uncontrolled targets;
full20-second horizons and valid final2s; all46 frozen hashes unchanged. Sensor
acquisition leaves the physics clock unchanged. Four controller tests, indexed
reset isolation, ten settled-reset cycles and40+16 zero-action checks passed.
Original sources, checkpoint files, probe-recovery evidence and studies preserved.

The56 conditions comprise44 unique seated states and12 repeated with impulses;
two seeds reuse these cases. They cover11 rigid level seat heights25–40cm,
including16 fresh states at26.25/31.25/36.25/38.75cm. Torque is an effort proxy,
not energy; reported means use qualifying ticks, and Nm averages instantaneous
all-joint RMS. No claim of soft/tilted furniture or hardware generalization.
Old/new study results differ in contracts, seeds and budgets, not an isolated
ablation of action accumulation.

Mac final report: validated_resting_bounded_02/final_results/review.html
Original results/report and plot: final_results/results.json, report.html,
comparison.png. Independent audit/plot/evidence: final_results/verified_review.
Remote study: /data/humanoid-sitting-perception/resting_bounded_ppo_02
Remote final delivery: study final_delivery/review.html.
Completion monitor finish-resting-ppo-comparison is PAUSED after verified delivery.
Capped-impact and earlier resting studies remain completed and unchanged.

Candidate next experiment, not run: measure whether slow deliberate posture
changes can reduce effort under the same physical success constraints. This
would establish reachable lower-effort targets and distinguish a posture/search
problem from an RL-learning problem before selecting a further PPO change.
Continue only via Mac SSH; do not launch another coding agent.

---

# Bounded resting PPO pilot active — drift diagnosis and retention validated

Live-training verification: all56 fixed-baseline conditions pass the full20s.
Seed53111 reached update10 with10240 recorded transitions;
first saved actor and critic weights changed and are finite, and all46 frozen
hashes still match. Evidence: validated_resting_bounded_02/first_optimizer_validation.json.
GPU1 process placement verified. Final policy quality remains pending.

October 12, 2026 (PDT). The user authorized diagnosing unwanted contacts,
repairing posture drift, validating stable-start retention, and another small
PPO comparison against fixed posture, including an optional low-effort hold rule.
Prior completed studies, code, checkpoints and outputs remain preserved.

Unchanged replays of seed52911 stability and seed52917 effort reproduce every
original result, startup and trajectory array exactly. First failures identify
left/right hip-pitch housings loading the seat and the right hip-roll joint
reaching its physical margin. A separate filtered-contact replay confirms seat
contact in all16 housing-contact trials, with zero housing-to-floor contact.
No physical failure labels were relaxed.

New isolated wrapper scripts/run_resting_bounded.py replaces accumulated target
increments with bounded offsets from the remembered stable starting command:
±0.025rad hip/ankle roll and ±0.05rad other controlled joints, smoothing0.2,
slew0.002rad/tick and0.03rad physical command margin. The actor has393 sensor
channels (previous324 plus remembered23-joint reference in each history frame).
The original reward, simulator and limits remain unchanged.

Validation passed:40/40 full20-second zero-action trials across seven heights
and disturbances, plus16/16 repeated original-height trials with the hold rule
enabled. Body, joint, torque and command traces match fixed posture exactly.
Four focused controller tests, indexed reset isolation and ten settled reset
cycles pass. A repeated-reset probe inference-mode issue was narrowly repaired;
original failed source/output/log and recovery manifest are preserved. The hold
rule never activated at unchanged posture, as no10% effort saving occurred.

New study: /data/humanoid-sitting-perception/resting_bounded_ppo_02
Persistent tmux: resting_bounded_ppo_02, physical GPU1 RTX4070.
Protocol: docs/RESTING_BOUNDED_PROTOCOL.md and study protocol.json.
Two fresh seeds53111/53117,64 updates each,65536 transitions/run,131072 total.
Whole-body-effort reward unchanged. Each final checkpoint is evaluated with and
without the measured-effort hold rule against fixed posture.56 conditions use
44 unique states plus12 impulse repeats.16 fresh states at26.25/31.25/36.25/
38.75cm have passed capture validity. Coordinator requires all matched fixed
baselines to pass before training. Inspect status.json for the current stage.
All46 source/data/protocol files are frozen; no extra training or research
changes during the run. No automatic promotion.

Hold rule: once measured normalized squared torque is<=90% of its initial
settled measurement for2s, with low joint/IMU motion and fresh near-seat depth,
hold PD motor targets. Resume when motion/geometry/attitude/effort deteriorates.
This is an evaluation ablation on the identical policy, not another trained arm.
Independently audit physical validity during hold. The sensor's initial effort
reference differs from the trial-average fixed baseline used for the study's
10% saving target. Torque remains an effort proxy, not electrical energy.

Completion monitor finish-resting-ppo-comparison is ACTIVE for this new study,
quiet during normal progress, with final matched reporting and automatic pause
after delivery. The capped-impact monitor remains PAUSED. Never launch another
coding agent; continue direct Mac SSH only. Prior studies remain unchanged.

Mac diagnosis: validated_resting_bounded_02/diagnosis/review.html
Verified plot and audit: diagnosis/diagnosis_and_retention.png, validation.json.
Remote diagnosis: resting_diagnosis_02/verified_report.
Validation and recovery: resting_bounded_validation_02.
Final destination: validated_resting_bounded_02/final_results.
No learned-policy stability or effort-saving conclusion yet.

---

# Resting PPO comparison complete — no learned policy retained stable rest

October 12, 2026 (PDT). Both objectives and both seeds completed their prescribed
128 updates. Fixed posture passed 40/40 matched 20-second conditions. Stability
PPO and whole-body-effort PPO each passed 0/40 in both seeds (52911, 52917).
Every policy failed all four trials at each tested height, including the three
unseen intermediate heights; the disturbance bank also had zero PPO successes.
No policy is promoted and no additional training was launched.

The effort reward improved a secondary diagnostic: mean total qualifying rest
in a 20-second trial increased 3.18→10.44 seconds for seed52911 and 1.94→6.94
seconds for seed52917. This is not necessarily continuous rest and did not meet
the full success criteria. With zero common successful trials, whole-body, leg
and torso/arm effort savings are undefined at every height and disturbance bank.

Failure flags: seed52911 stability had15 forbidden-contact /31 joint-margin
failures, effort12 /28; seed52917 had40 forbidden-contact failures in each arm.
Categories overlap. No falls, excessive impact or torque-limit violations were
recorded. Corrections stop after the first latched failure, while simulation
continues to20seconds. Recorded commands accumulated substantial posture drift
(median largest drift at first failure0.29–0.48rad by policy). The exact bodies
responsible for forbidden contacts and any reset-contact transient cause remain
unresolved; do not claim a proven simulator or detector bug.

Independent checks passed: four times131072 =524288 transitions;436661 active
and87627 warmup. Completed/unfinished accounting reconciles; finite changed
weights; identical arm initial actor/critic tensors within seed; every evaluation
loaded its intended final weights. All12 startup comparisons are exact, geometry
and disturbance schedules match, fixed commands are constant, all34 frozen
source/data hashes match, all28 held-out captures are valid, indexed resets are
isolated, and12 objective tests passed.

Evaluation uses28 unique captured states, plus12 repeated with a small base
velocity impulse, reused across seeds. Only rigid level surfaces are tested.
Torque is an effort proxy, not measured energy. The Nm diagnostic averages
instantaneous all-joint RMS. Fixed-posture whole-body effort rises from about
2.94Nm at25cm to6.32Nm at40cm; the highest posture leaves92% of weight on feet.

Mac delivery: validated_resting_posture_01/final_results/review.html
Original report/data: final_results/report.html and results.json
Verification: final_results/verified_review/validation.json
Plots: final_results/comparison.png and verified_review/diagnostics.png
Remote: /data/humanoid-sitting-perception/resting_ppo_01/final_delivery/review.html
The completion monitor finish-resting-ppo-comparison is PAUSED after delivery.

Candidate follow-up, not implemented in this check: identify triggering contact
bodies/joint margins, audit selected-row reset contact histories, and evaluate
bounded corrections around a stable reference instead of accumulating increments.
Prove baseline retention before another PPO continuation; do not relax evaluation
criteria to reclassify the current policies. Original outputs are preserved.

The capped-impact study was completed and its own monitor paused earlier. Its
result and all preceding snapshots are preserved below as dated history. Both
bounded studies are now complete, with no automatic promotion.

---

# Capped-impact comparison complete and independently verified

October 12, 2026 (PDT). Four 32-update runs and all matched 20-second evaluations
finished. The preserved starting policy scores 53/80 (47 earlier cases plus 6 new
holdout cases). Old → capped scores: 35→36/80 for seed52813; 54→55/80 for seed52817.
No policy is promoted. Sources, checkpoints and earlier reports are preserved.

Capped impact did not make successful sits gentler. On common successes, peak
seat force rose1.18%/3.44%, while leg RMS fell3.12%/13.61%. New holdout success
rose6→7 and7→8/16, but peak force rose4.07%/7.29% and pooled leg RMS rose0.32%.
Pooled falls decreased24→13; wrong contacts49→34; impact failures58→52 (the
second seed's impact failures increased21→23). Seed sensitivity is large; the
first seed regresses heavily from the parent. Two seeds repeat the same80
placements and do not support a general improvement claim.

Independent verification:131072 total transitions, exact parent tensor parity,
frozen normalizers, finite changed weights, full evaluation horizons, bitwise
startup matching,40 frozen files and2 recovery files unchanged. Integer timer
regression changes onlycase13 (9→10/16). Motion/command/startup arrays are exact;
force labels have the preserved one-ULP exception (max0.0000305176N).

Remote study: /data/humanoid-sitting-perception/support_capped_impact_01
Remote review: VERIFIED_REPORT.md and verified_review/validation.json
Mac report: validated_support_capped_impact_01/VERIFIED_REPORT.md
Mac plots: results/comparison.png and verified_review/impact_effort.png
The capped-impact completion monitor is PAUSED after verified report delivery.

The separate seated-resting study resting_ppo_01 and its own monitor are unchanged.
The first stability-only learner previously scored0/40 versus fixed posture40/40;
that is an interim observation, not its final two-seed comparison. Consult its
live status.json and final report when available. Earlier dated snapshots below
are preserved history and do not supersede this completed capped-impact result.

---

# Seated-resting PPO is now running alongside the capped-impact comparison

October 11, 2026 (PDT). The user authorized building and validating seated resets,
then PPO across seat heights, compared against fixed posture. This is a separate
experiment; the existing capped-impact study continues on physical GPU0.

Validated on physical GPU1 (RTX4070):
- Sixteen full-state seated captures at25/30/35/40cm passed sustained support.
- All16 replayed fixed-command trials passed the full20-second hold.
- Initial depth/body clearance error <=1.049mm; filtered foot forces matched net
  foot forces, and total support balanced weight within0.066% at reset.
- An indexed reset leaves every other robot's state, command, observation and
  physics clock exactly unchanged. Selected geometry/history is invalidated.
- All28 independent held-out states are valid (16new starts plus12intermediate
  heights27.5/32.5/37.5cm). Fixed posture passes28/28 undisturbed and12/12 with
  alternating +/-0.06m/s base-velocity impulses at5seconds.
- Twelve objective/calibration tests pass. Simulator source checkout is clean.

The persistent tmux session `resting_ppo_01` is running its first actual PPO job,
`train_52911_stability`; seven real optimizer updates were verified. Two seeds
(52911,52917),128updates x64steps x16robots per arm:131072transitions/run,
524288total across stability-only and whole-body-effort arms. Warmup ticks are
counted separately and receive zero reward. Initial mean action holds the fixed
baseline command; thereafter PPO controls23 incremental joint targets, without a
frozen timed movement. Actor and critic use324sensor-only history channels.

Effort cost is bounded and strictly monotonic: E/(E+reference), where E is mean
all-joint squared torque normalized by existing peak limits. Reference is the
mean fixed-baseline E, calibrated before PPO. This is an effort proxy, not measured
energy. Reported Nm values average instantaneous all-joint RMS. The40cm starting
posture leaves about92%of weight on feet and is substantially more costly; do not
mistake contact alone for a fully unloaded seated posture.

Study: /data/humanoid-sitting-perception/resting_ppo_01
Live status: status.json; train_<seed>_<arm>/progress.json
Protocol and frozen checksums: protocol.json and frozen/manifest.json
Expected results: results/report.html, results/results.json, results/comparison.png
Launcher: scripts/run_resting_ppo_01.py; core: scripts/run_resting_posture.py
Mac validation: validated_resting_posture_01/
Same-chat follow-up: finish-resting-ppo-comparison,30-minute heartbeat, quiet until
failure/completion; final comparison then pause. Final learned-policy comparisons
are pending. No policy is promoted. Forty evaluation conditions use28unique reset
states, with12repeated under perturbation, and are repeated across training seeds.
Only level rigid surfaces are in this study; oblique/compliant support and longer
holds remain future work. The floor is an anchored cuboid per environment because
PhysX cannot filter friction against its infinite plane; its top stays atz=0.

The previous capped-impact status below remains a separate active experiment.

---

# Capped-impact-only PPO study running

October 11, 2026, 22:16 PDT (October 12 UTC). The user authorized fixing the timer,
four matched32-update PPO jobs (old versus capped impact, two training seeds),
and comparison against the47/64 starting policy. They also requested a fresh-start
RL design; that explanation is saved in SITTING_RL_FRESH_APPROACH.md.

The new training/evaluation path uses int64 supported-hold ticks. Six tests pass,
including CPU/GPU timer boundary/reset checks and cap/reward-isolation tests.
The full timer regression changes only the known case13 from failure to success
at exactly150 ticks: the prior new-reward policy scores10/16 instead of9/16 on
its novel bank. Controller inputs/actions, all startup sensors, poses, kinematics
and physical outcomes match the archived run exactly. Force labels differ by one
float32 unit in three50Hz samples (one10Hz sample), max0.00003052N. The original
strict parity stop, its source, and a narrow documented recovery are preserved.
This exception affects force-label parity only, not success criteria or control.

Training seed52813/old reward is running now. The persistent queue will execute
both old/capped arms for seeds52813 and52817,32 updates each, from the preserved
151-input47/64 checkpoint. The capped arm changes ONLY the impact term; total
additional peak cost cannot exceed5 per episode. Other reward terms, perception,
architecture, motion scaffold and motor limits remain unchanged.

On194 archived admitted training attempts, the proposed peak cost totals706.76
instead of23272.99, with maximum5. This verifies its scale; it is not evidence
of improved learned behavior. Matched policy results are still pending.

Each of four final policies will get80 full20-second tests: the four known
regression banks plus fresh holdout52841/52843. All attempts are retained. The
starting policy is evaluated on the new holdout and archived baselines are
transparently rescored using integer ticks. No checkpoint or seed selection.

- Session: support_capped_impact_01, running on Ubuntu through Mac SSH.
- Study: /data/humanoid-sitting-perception/support_capped_impact_01.
- Live status: study/status.json; learning: study/train_<seed>_<arm>/progress.json.
- Protocol: SUPPORT_CAPPED_IMPACT_PROTOCOL.md.
- Recovery: scripts/resume_support_capped_impact_01.py.
- Parity amendment: SUPPORT_CAPPED_IMPACT_PARITY_AMENDMENT.md.
- Validation: study/timer_validation.json and archived_cost_audit.json.
- Expected report: study/results/REPORT.md.
- Completion follow-up: same-chat automation finish-capped-impact-ppo-comparison,
  every30 minutes, quiet during normal progress, final comparison then pause.
- Prior result/diagnosis: SUPPORT_REWARD_STUDY_COMPLETED_STATUS.md and
  /data/humanoid-sitting-perception/support_reward_study_01/diagnosis_01/REPORT.md.

Expected total duration is about2–2.5 hours. Long processes persist in tmux;
no additional Codex agent runs on Ubuntu. No packages installed or deployment
defaults changed. Original experiment code and checkpoints remain preserved.
