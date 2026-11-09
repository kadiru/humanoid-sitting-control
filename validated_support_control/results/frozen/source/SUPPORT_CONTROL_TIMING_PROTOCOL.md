# Camera/controller compatibility investigation

October 11, 2026. Evaluation only: fixed pilot-3 parent, sensor05 nominal policy,
contact02 estimator, original motor bounds and gains. No RL or policy fitting.
The frozen landing tracker and original scripts/checkpoints remain preserved.

## Diagnosis

Run the full 16 x 20 s regression bank (geometry 51403, starts 51407) with
four acquisition conditions: historical timing, startup-only corrected,
live-only corrected, and both corrected. Save every 50 Hz policy input,
actor action, nominal target and commanded target alongside separate sensor
and evaluation-label archives. Reproduce historical traces before attributing
differences to timing. Compare first differing inputs/commands, initial depth
geometry and physical outcomes. This is a consumed development bank.

An initial recorder run failed before completing its first control step due
to treating a Python joint-index list as a tensor. The failed output is retained;
the corrected factorial uses the suffix 02.

## Candidate and validation

Select the repair from the diagnosed observation/control mismatch. Do not
restore mislabeled image timestamps, insert simulator truth into control,
alter force thresholds, or tune soft-landing gains to recover a score.
Keep unknown geometry explicitly unavailable. Log the exact features fed to
the controller so passive monitoring cannot be mistaken for feedback.

Freeze the selected candidate source and settings before new validation.
Use paired unchanged-controller and candidate runs with identical seeds.
Reserve fresh broad geometry 51941 / starts 51943 and mixed training geometries
with fresh starts 51923; inspect neither set before freezing the candidate.
The old 51403/51407 bank remains the regression check.

Report all 20 s of every attempt, including refusals and post-failure intervals.
Controller compatibility requires recovery of all 16 known-bank successes,
no increase in failures on either paired fresh bank, and no new physical
failure categories hidden by conditional metrics. Perception gates remain
accepted error <2 cm and >=95% availability over both full trials and landing
windows in each bank. Report per-case outcomes and worst cases. A failure
does not authorize further RL, threshold changes, or promotion.
