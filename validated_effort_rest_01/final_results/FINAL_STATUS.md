# Direct effort PPO complete — no overall improvement

Verified October 12, 2026 (PDT; October 13 UTC). Both fresh PPO seeds completed
256 updates. The experiment started already sitting or leaning. It did not
test reaching support from standing. No policy is promoted and no extra training
was launched during this completion review.

## Stability and physical failures

| Policy | Training surfaces | Held-out surfaces | Sitting | Leaning | Total |
|---|---:|---:|---:|---:|---:|
| Fixed posture | 13/13 | 16/16 | 17/17 | 12/12 | 29/29 |
| PPO 54511 | 5/13 | 6/16 | 11/17 | 0/12 | 11/29 |
| PPO 54517 | 6/13 | 9/16 | 15/17 | 0/12 | 15/29 |

Every evaluation ran the full 20 seconds. Success requires no physical failure,
at least 95% qualifying ticks and a continuous final two seconds of valid support.
No falls, impact violations, torque-limit violations or joint-limit violations
were recorded. Both policies caused forbidden contact on held-out sitting case1,
a 30° surface, at 0.06s and 0.14s respectively. The other 17/13 unsuccessful trials
failed the unchanged support qualification; they are not falls. Some recovered a
long final hold but exceeded the permitted time outside qualification.

## Effort versus matched fixed posture

Only common full-duration successes are compared: 11 pairs for seed54511 and15
for seed54517. Each seed therefore has a different survivor subset. Percentages
are changes in pooled means, not averages of per-case percentages.

| Metric | 54511 whole20s | 54511 final10s | 54517 whole20s | 54517 final10s |
|---|---:|---:|---:|---:|
| Total squared torque, all37 joints | +65.46% | +66.10% | +55.58% | +56.24% |
| Sum absolute torque | +29.16% | +29.45% | +25.12% | +25.39% |
| Instantaneous joint RMS, time averaged | +28.53% | +28.81% | +25.40% | +25.69% |
| Leg squared torque | +72.58% | +73.30% | +61.25% | +61.98% |
| Upper-body squared torque | +0.54% | +0.52% | +0.65% | +0.60% |

Whole20s total squared torque was807.71→1336.40Nm² and852.19→1325.84Nm².
Whole20s joint RMS was4.329→5.564Nm and4.502→5.646Nm. Surface normal support
load decreased about20% and foot vertical load increased about46% /33% on these
same successful subsets. This is consistent with shifting work back into the
legs, the opposite of the intended relaxation behavior.

Bank-specific final10s total squared torque changes were +50.47% /+102.32%
(training /held-out) for54511 and +60.94% /+51.01% for54517. All successful cases
were sitting; there is no valid successful-leaning effort comparison.

There are isolated improvements: both seeds retain success on the40cm training
seat and reduce final10s squared torque by54.2% /50.6%. Seed54517 also saves13.6%
on held-out case2, a10° seat. These isolated savings do not compensate for lost
stability and increased effort elsewhere. The previously borderline35cm ellipse
remains included: fixed and54517 pass,54511 fails. Per-height, slope, outline,
seed and task results are in supplement.json and the HTML per-surface table.

## What is established, and what remains uncertain

The saved motion shows reduced surface support on most sitting cases and all
45°/60° leaning cases. Torque increased in most of those cases. Training-surface geometry remains
available and accurate (saved10Hz samples: p95 offset below1.7mm, normal below
0.29°), so tracking loss does not explain those failures.

The75° held-out cases have a separate perception limitation: observations are
missing or a different plane is marked valid, with errors around90° and up to
186mm. Related errors occur in the fixed baseline too; fixed commands ignore
them. Per-case availability and errors before physical failure and during
qualifying motion are reported separately. This is a10Hz sample audit, not an
every50Hz guarantee, and a valid flag alone does not establish correct geometry.

The reward decreases with squared torque while support qualifies, but provides
no positive credit outside that qualification. A flat reward in those states is
a plausible optimization difficulty. These results do not establish the exact
optimizer cause. Individual friction/slip/motion gate failures were not saved,
so they cannot be reconstructed exactly from the combined good flag. No reward,
failure definition, geometry, observation or action contract was changed.

## Verification and preservation

- Both256-update budgets:425984 transitions per seed,851968 total.
- Seed54511:409845 active +16139 indexed-reset warmup. Seed54517:408961 +17023.
  Total818806 active +33162 warmup. Global settle and the separate3328-transition
  smoke are excluded from this budget.
- Finite changed actor and critic, final iteration255, actual final actor loaded
  for both banks, bitwise exact root/joint/command/geometry/observation starts.
- Full1000-step evaluations, constant fixed commands, unchanged uncontrolled
  joints, bounded learned commands. Runtime checks all50Hz bounds; saved10Hz
  commands provide the independent offline command checks.
- All105 frozen hashes and61 prior hashes match. Frozen copies and all six run
  source snapshots and bank manifests also match. Previous studies are unchanged.
- Original exploratory forbidden-contact failures and preflight launch-criterion
  revision remain preserved. A reporting-only glob repair is documented under
  report_recovery_01; it changed no experiment data or definitions.
- Mac delivery:364 files verified against Ubuntu SHA256 hashes, three plots
  visually inspected. Archive SHA256:
  a057b027cb0b51de6fd83e55dd270e37f4dcc907ca0708628387272e64628a7f.

Remote report:
`/data/humanoid-sitting-perception/effort_rest_01/ppo/verified_review/review.html`.
Mac report:
`/Users/kadir/humanoid-sitting-control/validated_effort_rest_01/final_results/review.html`.
Primary verifier: scripts/verify_effort_rest_study.py. Read-only supplementary
diagnosis: scripts/summarize_effort_rest_review.py.

Two seeds reuse29 finite, rigid, prealigned configurations. Held-out means
withheld from this policy, not new to the project. Torque is an effort proxy,
not electrical energy or proof of a global minimum. This experiment does not
demonstrate arbitrary-surface sitting, leaning, standing transitions or hardware
readiness. The final report is delivered and the completion monitor is paused.
