# Independent sitting and leaning curriculum — October 14, 2026 PDT

User authorization: separate sitting and leaning training, repair weight transfer/contact, train toward standing, and compare stability and joint effort with reference motion. Previous studies and checkpoints remain unchanged.

## Scope and diagnosis

Two known rigid surfaces from the previous motion bank: a 40 cm level seat and a 45 degree leaning surface. Seven individually retained starting states per surface. This is residual PPO around an explicit offline reference motion, with sensor feedback; it is not unassisted control or generalization to arbitrary furniture.

The passive contact probe reproduces the previous reference's startup, trace arrays, and episode results exactly. Sitting loses backward balance before support and impacts the seat. Leaning maintains contact but some starts exceed the unchanged 0.90 friction-utilization acceptance threshold. Independent static poses do not imply dynamically connected states.

A bounded four-candidate sitting probe and six-candidate leaning probe selected: (1) bilateral ankle correction 0.3*(IMU pitch - interpolated offline pitch target) + 0.03*IMU pitch rate, capped at +/-0.2 rad; (2) coordinated knee flexion +0.06 rad, hip/ankle pitch -0.03 rad, smoothly introduced from reference phase 2 to 6. These are reference-controller repairs, not learned PPO improvements. Initial five-start probes and subsequent seven-start, twice-repeated validations pass. All unsuccessful candidates remain preserved.

## Contracts and reward

Keep the 20 second horizon, full final 5 seconds of qualifying support, original contact legality, traction, slip, impact, joint-limit and fall criteria. No relabeling or relaxed thresholds. Runtime actor and critic receive 405 channels: three sensor frames (encoders, measured joint torque, IMU, previous commands, depth/kinematic support estimate) plus offline joint reference and clock phase. No live simulator world pose/contact force/seat geometry enters policy or feedback control. Simulator contacts are reward and evaluation labels only.

23 body-joint actions remain absolute bounded residual offsets, tanh spans 0.01 rad for hip/ankle roll, 0.05 rad for shoulder/elbow, 0.025 rad otherwise, around the corrected reference. Low-pass factor 0.1, slew limit 0.002 rad/control tick, 50 Hz. Fourteen finger commands stay constant. Fixed posture ignores the correction and exactly retains reset commands.

Reward per control tick:

    dt * ((3*good + .5*tracking + .25*balance + .5*transfer
           - .25*friction_cost - .05*effort*good)*alive
          - .02*slew_cost) - 5*first_physical_failure

`tracking` and `balance` retain the previous definitions. `transfer` is vertical support share normalized by 0.5 BW for sitting or 0.2 BW for leaning, clipped to [0,1], multiplied by legal-contact, no-slip, and balance factors. Friction cost ramps from zero at utilization 0.8 to one at 1.0, over loaded patches (>0.01 BW). Effort is all 37 physical squared Nm summed, divided by that sum plus 1500 (sitting) or 280 (leaning). Effort is secondary and bounded. The reporting success criterion is unchanged; transfer reward targets are not additional success claims or thresholds.

## Independent training

Separate checkpoints, optimizers, observations, workers and status files for each surface. Sitting uses physical GPU0, seeds55211/55217; leaning uses physical GPU1, seeds55311/55317. Each worker proceeds independently; no curriculum gate depends on the other surface or seed.

Each seed trains four 32-update blocks, starting at stages [4,2,0,0]. Every start is already validated physically, and the corrected reference reaches support from all starts. This is a scheduled curriculum, not an assertion that PPO mastered a stage. The last 64 updates explicitly train standing starts. Retain every block's initial/final checkpoint and all episodes. Each block is 64 steps/update *16 environments *32 updates =32768 transitions. Each seed128 updates/131072 transitions; each surface262144; total524288 including indexed-reset warmup, with active and warmup counts reported separately. The initial half-second global settling is outside this rollout budget. Development smoke:2 updates/2048 transitions per surface, separate from study.

PPO retains 128x128 ELU actor/critic, initial log-std corresponding to0.1, zero actor output layer, learning rate3e-5 with adaptive KL target.003, entropy.0001,3 epochs,4 minibatches,gamma.998,lambda.97. Resume full optimizer state and increment the previous saved completed iteration before each new block. No extra training or automatic policy promotion.

## Validation and comparison

Before study: independent reset isolation; all14 starts full20 seconds for corrected reference and two-update PPO smoke; matched old reference and fixed baselines; exact physical/sensor startup matching; finite changed weights and exact checkpoint loading. The appended reference channels may intentionally differ among controllers.

Final:28 conditions per surface (seven known starts, each repeated at four simulator placements). Baselines: unchanged old reference, corrected reference, and fixed posture. Evaluate both final policy checkpoints with identical physical starts/geometry for the full20 seconds, even after failure. Fixed standing is a retention baseline, not a successful sitting/leaning transition. No claim of unseen surfaces or independent28-state generalization. No perturbations in this bounded study.

Report physical failures and standing-transition success first, then success by stage/seed. Compare whole20 seconds, final10 seconds and final5 seconds of all-joint squared torque, absolute torque, instantaneous joint RMS Nm averaged over time, and leg/upper contributions on common full-duration successes only. Report changes both against the old reference and the corrected reference, separating controller repair from learning. Never infer savings from failed fragments. Measure support/foot load and friction utilization as offline diagnostics. Audit depth/kinematic availability and geometry error before physical failure.

Independent verification checks frozen/current/source-copy hashes, prior studies, finite changed actor/critic, exact budgets, optimizer continuity, actual loaded final checkpoint, exact startup and geometry, horizons, constant fixed/finger commands, command envelopes/slew and reference offsets. Report any validation limitations rather than hiding them. Torque is an effort proxy, not electrical energy or a global optimum. Two seeds and two finite rigid surfaces do not prove generalization.
