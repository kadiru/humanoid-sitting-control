# Bounded resting PPO pilot 02

October 12, 2026. Authorized follow-up to diagnose unwanted contacts, repair
posture drift, prove starting-posture retention, and compare a small PPO run
against fixed posture. This is a new study; all previous studies are preserved.

## Diagnosis and repair

Two unchanged old-policy replays reproduce their original startup, trajectory,
and result arrays exactly. First-failure instrumentation identifies left/right
hip-pitch housings bearing load and the right hip-roll joint reaching its physical
margin. A separate filtered-contact replay confirms hip-housing contact with the
seat in all 16 cases, with no housing-to-floor contact. The contact labels remain
unchanged; these failures are not reclassified as valid sitting.

The old policy added each correction to the previous motor target. A persistent
small bias therefore accumulated. The new interface maps action directly to a
bounded offset from the remembered stable starting command: at most 0.025 rad
for hip/ankle roll and 0.05 rad for other controlled joints. Targets approach the
requested offset with factor 0.2 and at most 0.002 rad per 20 ms tick, retaining
0.03 rad command margin from physical limits. Zero action holds the starting
command, or returns toward it after a nonzero action. Actual joint motion may
overshoot a target; the unchanged physical failure checks remain decisive.

The actor receives its remembered starting command in addition to the original
depth/proprioceptive history: 393 instead of 324 channels. Reward, physical
geometry, gains, torque limits and success definitions remain unchanged.
Simulator truth is used for reward/reset/evaluation labels, not actor or hold
inputs. This does not test tilted or soft surfaces.

## Required validation before training

Four focused controller tests; isolated indexed resets and ten settled reset
cycles; full 20-second zero-action evaluations on 16 original-height states,
12 intermediate-height states, and those 12 states repeated with a disturbance.
Motor commands must exactly match fixed posture; body and joint trajectories
must match within 1e-6. Another 16-trial zero-action evaluation enables the hold
rule. Every trial must pass. A probe-only inference-mode error was repaired;
failed output, source, log, and a recovery manifest remain preserved.

## Learning and comparison

Two fresh training seeds, 53111 and 53117, each receive 64 PPO updates,
64 steps/update and 16 environments: 65,536 transitions per seed, 131,072 total.
Indexed-reset warmup transitions count in that budget and are reported separately.
Initial mean action is zero. Existing 25/30/35/40 cm seated states are reused.
The prior calibrated whole-body-effort reward is unchanged, with effort
reference 0.005249470561742783. There is no reward tuning during this run.

Evaluate fixed posture, each final PPO policy, and each same policy with the
optional hold rule for 20 seconds. Use 40 known regression conditions plus
16 newly captured states at 26.25/31.25/36.25/38.75 cm (capture seed 53203).
Thus 56 conditions comprise 44 unique states plus 12 repeats with alternating
positive/negative 0.06 m/s velocity impulses at 5 seconds. The fixed baseline
must succeed on every bank before training. Startup state, geometry, disturbance
schedule and loaded checkpoint are matched. There is no checkpoint selection
on evaluation results and no automatic promotion.

Primary question: does PPO retain physical success across heights and reduce
whole-body effort versus fixed posture? Report failures first, then paired
effort on shared successful trials, by seed, height and disturbance. Success
requires no physical failure, at least 95% qualifying ticks and continuous
qualification in the final two seconds. A useful pilot target is at least 10%
lower normalized squared torque, with unchanged success. This threshold is a
declared pilot target, not a calibrated physiological or energy threshold.

Effort is the mean squared applied torque divided by each joint's existing
effort limit, over all actuated joints. Original summary metrics average over
qualifying ticks in successful trials. Also report the time-average of
instantaneous all-joint RMS torque in Nm, leg and torso/arm effort, and load
distribution. These are effort proxies, not measured electrical energy.

## Optional measured-effort hold rule

The rule is evaluated on the same checkpoint, without additional training.
It holds current PD targets once filtered measured effort is at most 90% of
the initial settled measured effort for two seconds, together with low joint
speed, low gyro motion, similar gravity orientation, and fresh nearby depth
geometry. It resumes adjustment if motion/attitude/geometry deteriorates or
effort exceeds 105% of its starting level. It does not turn off the motors.

Entry: max controlled-joint speed <0.08 rad/s; gyro norm <0.06 rad/s;
gravity alignment >0.996; depth valid, age <=0.12 s and absolute estimated
clearance <0.015 m. Exit: joint speed >0.2 rad/s; gyro >0.12 rad/s;
alignment <0.99; invalid depth, age >0.2 s or clearance >0.025 m;
or effort above 1.05 times reference. Effort EMA is 0.96 old +0.04 new.
These sensor conditions do not prove correct physical support; separately audit
hold entries against evaluation labels. Gate reference is an initial sensor
sample, whereas the study's 10% effort target uses the matched trial average.

## Preservation and execution

Persistent tmux `resting_bounded_ppo_02`, physical GPU1 (RTX4070), through the
existing `scripts/run_resting_gpu1.sh`. Study root:
`/data/humanoid-sitting-perception/resting_bounded_ppo_02`.
Validation: `resting_bounded_validation_02`; diagnosis: `resting_diagnosis_02`.
Freeze source/data hashes before execution. No extra training, changed rewards,
relaxed failure labels, or overwritten outputs. Narrow execution/report repairs
require preserved originals and a recovery manifest. Two seeds and repeated
states limit generalization; comparisons with the older study also differ in
action/observation contracts, seeds and budget and are not an isolated ablation.
