# Verified capped-impact PPO comparison — October 12, 2026

Capped impact did not produce gentler successful sits. Compared with the old reward, it added one successful case per seed and reduced leg effort on matched successes across all banks. However, successful-trial peak seat force increased in both seeds, the effort benefit did not hold on the new holdout, and the first seed regressed substantially from the preserved starting policy. This is a mixed exploratory result; no policy is promoted.

## Success by training seed and placement bank

Each entry is successful full-20-second attempts out of 16. Refused starts remain in the denominator. The starting policy scored 47/64 on the earlier banks and 6/16 on the new holdout, totaling 53/80.

| Bank | Starting policy | Old 52813 | Capped 52813 | Old 52817 | Capped 52817 |
|---|---:|---:|---:|---:|---:|
|Known regression|16|14|14|16|15|
|Previously fresh|13|8|7|12|13|
|Training shapes, new starts|8|3|4|9|8|
|Previously novel|10|4|4|10|11|
|New holdout|6|6|7|7|8|
| **Total /80** | **53** | **35** | **36** | **54** | **55** |

Across seeds, old reward passed 89/160 outcomes and capped impact passed 91/160. Each seed gained four successes and lost three. These are the same 80 placements evaluated under two learned policies, not 160 unique placements. The first four banks were previously tested; only the final 16 placements were newly held out. Two seeds are insufficient for a strong generalization claim.

![Success by bank and seed](results/comparison.png)

## Impact and effort on cases successful under both rewards

Comparisons below use identical cases within each pair. Percentage changes compare means, not the mean of per-case percentages. Lower peak force and lower leg RMS are better. This study measures leg torque, not whole-body effort or electrical energy.

| Scope | Seed | Shared successes | Peak force, old → capped | Change | Leg RMS, old → capped | Change |
|---|---:|---:|---:|---:|---:|---:|
|All banks|52813|32|381.15 → 385.64 N|+1.18%|4.985 → 4.829 Nm|-3.12%|
|All banks|52817|51|373.17 → 386.00 N|+3.44%|5.036 → 4.351 Nm|-13.61%|
|All banks|pooled|83|376.24 → 385.86 N|+2.56%|5.016 → 4.535 Nm|-9.59%|
|New holdout|52813|6|382.15 → 397.69 N|+4.07%|4.930 → 5.174 Nm|+4.94%|
|New holdout|52817|6|384.94 → 413.00 N|+7.29%|4.922 → 4.709 Nm|-4.32%|
|New holdout|pooled|12|383.55 → 405.34 N|+5.68%|4.926 → 4.942 Nm|+0.32%|

![Paired impact and effort changes](verified_review/impact_effort.png)

On the new holdout, success improved 6→7 and 7→8 out of 16, but matched peak force increased 4.1% and 7.3%. Leg effort rose 4.9% in the first seed and fell 4.3% in the second: effectively no pooled saving (+0.3%). The broader 9.6% pooled leg-effort reduction therefore does not establish a transferable benefit.

Compared with the starting policy, capped impact lost 18 successes and gained one in seed 52813, versus two lost and four gained in seed 52817. On common successes against that parent, leg RMS changed +27.5% and −4.7%. The training-seed effect is much larger than the one-case reward advantage.

## Physical failures and full-case force statistics

Failure categories overlap; these are counts, not mutually exclusive diagnoses.

| Seed / arm | Falls | Wrong contacts | Impact failures | Fast descent | Perception unavailable | Mean peak N | P95 peak N |
|---|---:|---:|---:|---:|---:|---:|---:|
|52813 / legacy|22|36|37|13|4|1159.1|4469.5|
|52813 / capped|13|27|29|8|4|961.0|2444.7|
|52817 / legacy|2|13|21|2|4|803.3|1935.3|
|52817 / capped|0|7|23|0|4|818.4|1836.0|

Pooled falls fell 24→13 and wrong contacts 49→34. Impact failures fell 58→52 overall, but increased 21→23 in the second seed. Full-case mean peak force improved in the first seed and worsened in the second. Force tails include failed/refused attempts and later motion during the full horizon, so they must not be substituted for gentleness on matched successful sits. Every policy has four refused starts out of 80, including two on the new holdout; these remain primary failures. No torque-limit or pelvis-floor failures occurred.

## Verification and preservation

- Four runs completed exactly 32 updates × 64 steps × 16 environments = 32,768 transitions each; total 131,072. Completed and unfinished attempt steps reconcile exactly.
- All actor and critic initial tensors match the preserved 47/64 checkpoint. Final weights are finite, actors changed, and observation normalizers stayed frozen.
- All 320 new trained-policy evaluation attempts ran 1,000 ticks / 20 seconds. The 16-case new-parent evaluation and 16-case timer regression also ran. The earlier 64 parent cases were reused and transparently rescored, not represented as newly simulated.
- All evaluation startup sensor arrays match the corresponding parent evaluation bitwise, and placement geometry matches.
- The six original timer/cap/isolation tests passed. The timer regression changes only case 13: 150 ticks now correctly passes three seconds (9/16 → 10/16).
- Timer motion, controller input/action and startup arrays match exactly. There is a documented, preserved exception for force labels: one 10 Hz value / three 50 Hz values differ by one float32 ULP, at most 0.0000305176 N. All other episode metrics match. The original strict-parity stop and the subsequent validation amendment remain preserved.
- All 40 frozen source/model hashes and both recovery hashes match. The parent checkpoint hash remains ece102b139323af18bd1c7aad6d8fb06e478f38946f07e52406a31bd4acc7dda.
- Original generated report, results, checkpoints and frozen sources are unchanged. This review and its verifier are additive; no further training or policy promotion was performed.

The report generator’s brief timer-parity sentence should be read with the explicit force-label exception above; the raw timer_validation.json already records it.

[Independent verification](verified_review/validation.json) · [All paired comparisons](verified_review/paired_comparisons.json) · [Original complete results](results/results.json) · [Original generated report](results/REPORT.md)

## Interpretation

The capped cost fixed the intended numerical scale and isolated the reward change, but it did not demonstrate gentler successful sitting. The inconsistent seed behavior and loss of effort benefit on unseen placements argue against more tuning of this penalty on the current banks. The already authorized seated-resting study remains separate and has its own completion monitor. No additional experiment is launched by this report.
