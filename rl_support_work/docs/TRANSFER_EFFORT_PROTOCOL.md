# Unseen-surface transfer and total-effort PPO — October 14, 2026

The user authorized testing unseen heights/slopes, then training to reduce total joint effort while retaining stability, compared with the corrected reference. Prior studies remain immutable. New root: `/data/humanoid-sitting-perception/transfer_effort_01`. Sitting and leaning remain independent; run directly over SSH from Mac, with persistent Ubuntu processes and no other coding agent.

## Assessment and reserved evaluation

The existing motion, standing reset, controller correction, friction, width/depth, and physical criteria stay unchanged. Change only surface top-center height and pitch about that center. Sitting anchor: 40 cm, 0 degrees. Leaning anchor: top-center height 65.136973 cm, 45 degrees; that height describes the inclined slab center, not a horizontal seat height.

Development grid per task: height changes −2/0/+2 cm crossed with slope changes −5/0/+5 degrees (9 surfaces, 8 unseen by the existing policy). Each starts standing and appears twice at different simulator placements: 18 conditions. First evaluate fixed standing, corrected reference, and both existing final policies before new training. An identity bank reproduces the old controller's complete startup/trace arrays exactly at the unchanged geometry; indexed reset isolation must pass.

Reserved holdout per task, defined before assessment: height changes ±1 cm crossed with slope changes ±2.5 degrees (4), height changes ±3 cm at anchor slope (2), slope changes ±7.5 degrees at anchor height (2). These 8 surfaces, repeated twice, are never used for calibration, training, checkpoint selection or controller tuning. The assessment development set becomes training data; it must not be called unseen after training. Holdout means withheld from this experiment/policy, not necessarily never considered anywhere in the project.

No new geometry-specific reference poses or intermediate resets are supplied. Only the original standing reset is admitted on changed surfaces. Reference assistance is explicit; this does not demonstrate independent discovery of sitting or furniture recognition.

## Effort objective

Runtime actor/critic remain 405 sensor/reference channels; the 23 bounded absolute body-joint residuals and fixed finger commands are unchanged. No live truth geometry/contact labels enter observations or control. Truth remains reset/reward/offline evaluation only. The measured depth/kinematic geometry and IMU correction are unchanged.

Effort is the sum of squared physical Nm over all 37 joints. Reward calibration is a frozen per-surface corrected-reference final-10-second mean on successful full-duration development trials. Where the reference fails, use the mean over successful development reference conditions for reward scaling only; never treat a failed fragment as an effort-saving baseline. Calibration never enters observations or control.

Per-tick reward, with dt=.02:

    dt * ((4*good + 2*good*clamp(1-total_squared_Nm/calibration,-1,1)
           + .25*tracking + .25*balance + .5*transfer
           - .25*friction_cost - .5*late_bad)*alive - .02*slew_cost)
      - 5*first_physical_failure

The original `good` contact, traction, slip, support, speed and balance gate is unchanged. `late_bad` means phase 6 without valid support. `tracking`, `balance`, `transfer`, and friction/slew definitions remain unchanged. Stable support retains at least 2 reward units/s even with the maximum bounded effort penalty; leaving support earns no effort benefit. Behavioral tests exercise the exact executed reward, including lower-torque preference, loss of support, and failure.

## Training and checkpoint selection

After independently verifying the assessment, each task receives two training seeds (55611/55617 sitting, 55711/55717 leaning), initialized respectively from its previous two final policies. Retain the complete inherited actor, critic and optimizer; inherit optimizer iteration continuity. Each trains four 32-update blocks, all from standing, with 18 environments and 64 steps/update: 128 new updates and 147,456 transitions per seed, including indexed-reset warmup. Total new study budget 589,824 transitions. Separate two-update optimizer smoke per task: 2,304 transitions each, 4,608 total, never part of study checkpoints/budget.

Every 32-update candidate is evaluated on the development grid with matched starts and full 20-second horizons. Report the last checkpoint from the full budget regardless of success. Also select a stability-preserving candidate using development data only: it must preserve every condition successful under either the corrected reference or that seed's starting policy, and add no physical failure condition beyond their union. Rank eligible candidates by success count, then lower mean squared torque on corrected-reference common successes. The parent policy is an explicit fallback if no trained candidate qualifies. Selection is not deployment or proof of holdout safety.

After selection, evaluate the corrected reference, both parents, both final checkpoints, and selected checkpoints when different from final on the reserved holdout. Training does not depend on the other task's gate. Stop a task only for execution/validation failure, not because the other task struggles. No automatic policy promotion or extra training budget.

## Validation and reporting

Every comparison uses the same bank, seed, environment count, starts, geometry and simulator placement. Require exact startup physical/sensor agreement where reproducible; any exception requires explicit measured evidence. Retain full 20 seconds after failure for evaluation, and require the final 5 seconds continuously qualifying support with no latched physical failure. Report failures before effort, by task/seed/height/slope and development versus holdout.

Report whole-20-second, last-10-second and last-5-second squared torque, absolute torque, mean instantaneous joint RMS Nm, and leg/upper-body contributions on common full-duration successes only. Report final and selected policies distinctly, and comparisons against both corrected reference and starting PPO. Audit geometry availability/error before physical failure, fixed/finger commands, residual envelopes/slew, loaded checkpoints, finite changed weights, optimizer continuity, sources/prior hashes, and active/warmup accounting. Torque is an effort proxy, not electrical energy or a global optimum. Repeated placements and two seeds are limited evidence; no arbitrary-surface claim.
