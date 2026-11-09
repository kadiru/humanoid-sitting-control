# Two-surface curriculum completed and independently verified

October 13, 2026 (PDT). Work ran directly through Mac SSH to kfuUbuntu, with no
other coding agent. No policy was promoted. All earlier studies are preserved.

## Outcome

- Seven poses each for the existing 40 cm seat and 45° leaning surface passed
  independent 20-second retention: 14/14, 100% qualifying stationary ticks.
  The 114 coarse and 105 refined candidates, including failures, remain saved.
- PPO seeds 54911/54917 completed 128/160 updates: 131,072/163,840 transitions.
  Total 294,912 = 288,000 active + 6,912 indexed-reset warmup. Separate smoke:
  2,048 transitions. The recorded gated stopping rule explains the unequal budgets.
- Seed 54911 stopped after two stage-4 failures; seed 54917 after two stage-3
  failures. Leaning traction margin blocked progression. Sitting passed these
  gates. Neither curriculum reached standing training.
- Final matched 20-second tests on all 14 starts: reference 9/14 support successes,
  last policies 6/14 and 9/14; last-passing checkpoints 6/14 and 8/14.
  Reference had 2 physical failures; every PPO variant had 3, all early sitting
  cases, with contact/impact violations followed by falls. Fixed posture retained
  all 14 without physical failure; 7 starts were already supported.
- Seed 54917's last policy leaned from standing on the one known surface in the
  final test. Neither policy sat from standing. This isolated result is not robust
  or unseen-surface evidence, and standing starts were never trained.

## Effort and diagnosis

On common successful seated starts 5/6, last-policy whole-trial total squared
joint torque was 18.96%/19.11% lower than fixed posture. Compared with the reference
motion over common seated transitions 3–6, savings were only 1.97%/1.58%.
Most of the fixed-posture improvement therefore came from the reference motion.
Leaning effort was 11.69%/28.62% higher than fixed posture on each policy's common
successes; versus reference it changed +1.14%/−1.56%, on different case sets.
All 37 joints count. Torque is an effort proxy, not measured electrical energy.
Whole-trial/final-10s/final-5s, absolute torque, RMS, leg and upper-body metrics are
in the verified report/CSV. No savings are claimed from failed fragments.

Failed leaning gates remained nearly stationary with legal contacts and no
excessive slip, but failed the existing friction-utilization margin. Late learned
corrections used only 18–33% of their available span. There were no lost geometry
samples before physical failures: 100% availability in the recorded audit and
no missing motion samples at 10 Hz; maximum offset error 2.104 mm and normal error
0.365° (rounded upward). Control/contact behavior is the immediate bottleneck.

A material repeatability issue remains: identical seed-54917 checkpoint, stage-3
lean, fails the 2-environment GPU1 gate but passes the 14-environment GPU0 test.
Cross-layout starts differ up to 11.27 µm translation, 17.70 µrad joint position,
and 0.000675 in normalized observations. Cause is not isolated. All final
controller-versus-baseline starts are bitwise identical. Do not erase or relabel
the gate failure or infer broad robustness from the final isolated success.

## Verified evidence

Remote root: /data/humanoid-sitting-perception/two_surface_curriculum_01.
Study: ppo_01. Frozen protocol: docs/TWO_SURFACE_CURRICULUM_PROTOCOL.md.
Independent report: final_verification_01; portable evidence: final_export_01.
Mac: /Users/kadir/humanoid-sitting-control/validated_two_surface_curriculum_01/final_results.
Start with findings.html; report.html has detailed tables; motion_replay.html is
a standalone interactive replay of saved body transforms, not a rendered camera.
The plots and replay controls/playback were visually checked; browser errors: none.
All 387 copied files matched the export SHA256 manifest.

Audit passed 33 frozen source/bank hashes, 137 prior hashes, changed finite
actor/critic, exact optimizer/checkpoint continuation, active/warmup accounting,
correct loaded final weights, full 1,000-tick horizons, exact matched starts,
constant fixed/finger commands and bounded learned commands. Reference-command
replay error was 2.39e-7 rad. No execution repair or frozen-definition change was
needed during the main study. Verifier reruns may use a new output directory
with the existing script; it accepts both pending-verification and verified states.

## Scope and next work

This is residual PPO around an offline reference, with that reference kept outside
trainable layers. Actor and critic receive 405 sensor/reference/clock channels;
23 bounded body-joint offsets. No live ground-truth geometry/root position/contact
label is fed into either network. Truth supplies reset/reward/evaluation labels.
The support gate allows >=0.05 bodyweight normal support load; passing it does
not prove full weight transfer or minimum effort. Two finite known rigid surfaces,
repeated starts, no disturbances, and no unseen-surface validation in this pilot.

Recommended next: separate sitting and leaning curricula so one cannot block the
other; validate contact-safe weight transfer across placements before scaling RL.
Retain physical acceptance gates and the reference-only baseline. Do not extend or
alter this completed frozen study; any next run needs its own recorded protocol.
No extra training or automation was launched after completion.
