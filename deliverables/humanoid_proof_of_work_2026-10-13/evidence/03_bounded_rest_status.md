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

