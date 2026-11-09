# Matched PPO reward study complete; new reward did not improve overall

Verified October11,2026,21:35 PDT (October12 UTC). Both128-update runs and all144
new full20-second evaluations completed. No jobs remain. No candidate promoted.
The user asked for matched results, diagnosis, and the next experiment; no new
training was launched during the diagnosis.

| Policy | Original successful sits /64 | Mean peak force, same36 successes | Mean leg RMS, same36 successes |
|---|---:|---:|---:|
| Starting PPO |47|375.06N|4.48Nm|
| Continued PPO, old reward |44|431.70N|4.34Nm|
| Continued PPO, new reward |42|404.45N|4.88Nm|

A reproduced float32 timer defect rejects exactly150 held control ticks as
2.999997854 seconds. Only novel/new-reward case13 changes under integer-tick
scoring: corrected totals47,44,43. Original scores stay intact. The conclusion
is unchanged. Use integer hold counters in future runs; PPO's dense hold reward
was unaffected by this success-reporting defect.

The new reward has slightly lower force than continued old-reward training among
common successes, but greater leg effort and fewer successes. Both continuations
regressed relative to the starting PPO.

Strong diagnostic evidence: new impact cost23283 units versus7431 total positive
units; old impact cost115. Of the reconstructed completed/admitted new peak costs,99.2% came from impact-failure attempts. One cost765 units, versus the ordinary
5-unit failure penalty. The peak term was overscaled. Mean final32 value loss
516.6 versus4.02 also reflects the changed target scale; it is not independent
proof of causation. New training produced420s supported sitting versus471s.

Lost cases include three impacts, two late forbidden contacts after sitting,
one short hold, and the timer edge case. Accepted distance errors at first
contact/failure were <=1.70mm; availability92.2–100% in those windows. Gaps remain,
but these results do not justify another perception rewrite. Independent
clearance-only input response is still weak; other geometry inputs may overlap.

Recommended next: start from the47/64 PPO checkpoint, retain current perception
and action limits, and compare old reward against capped-impact-only reward.
Cap the entire peak cost at5 units; retain old effort reward.32 updates per arm,
two training seeds. Then test effort separately. This is a proposed experiment,
not an implemented or validated fix. Integer hold scoring should be corrected
for all prospective arms. Existing banks are now known regression sets; add a
predeclared fresh bank when running the next study.

- Remote study: /data/humanoid-sitting-perception/support_reward_study_01.
- Original report: study/results/REPORT.md.
- Diagnosis and next proposal: study/diagnosis_01/REPORT.md and NEXT_EXPERIMENT.md.
- Mac diagnosis: /Users/kadir/humanoid-sitting-control/validated_support_reward_study_01/diagnosis_01/REPORT.md.
- Detailed JSON, comparison plot, illustrative lost-case plot, and source accompany it.
- Saved next proposal: docs/SUPPORT_REWARD_NEXT_PROPOSAL.md.
- Prior validated interface/pilot: SUPPORT_SPLIT_PPO_STATUS.md.

Both budgets, initial weights, frozen normalization, startup parity and source
preservation verified. Installed IsaacLab remains clean. All checkpoints remain
available, and deployment defaults are unchanged. Continue RL research; earlier
handoff instructions to pause RL are historical and superseded.
