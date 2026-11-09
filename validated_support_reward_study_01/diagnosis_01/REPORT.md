# New reward: no overall improvement in matched sitting tests

Verified October11,2026 (October12 UTC). Both training runs and all144 new
evaluation attempts completed. Each training run used128 PPO updates and131,072
transitions. Each final policy was tested on64 matched20-second attempts.
Startup sensor archives and geometry matched exactly within each test bank.
All failures and refused attempts remain in the primary results.

| Measurement | Starting PPO | More PPO, old reward | More PPO, new reward |
|---|---:|---:|---:|
| Successful sits, original scoring |47/64|44/64|42/64|
| Mean peak seat force, common successes |375.06N|431.70N|404.45N|
| Mean seated leg RMS torque, common successes |4.48Nm|4.34Nm|4.88Nm|
| Mean longest supported hold, common successes |7.58s|6.87s|6.95s|

The last three rows use the exact same36 cases that succeeded under all three
policies. Conditioning on success does not replace the full64-case success
score. Relative to more training with the old reward, the new reward reduced
mean peak force6.3% on these cases but increased leg torque12.5% and lost more
sits. Relative to the starting policy, force increased7.8% and torque9.0%.
Torque is an effort proxy, not a measurement of electrical energy consumption.

![Matched comparison](matched_comparison.png)

The new policy gained2 and lost7 successes relative to the starting policy.
Three lost cases involve impact, two make prohibited contact late in an otherwise
partly successful sit, one holds for only2.94 seconds, and one is a timer-rounding
edge case. The old-reward continuation also regressed, so more training alone
was not beneficial in this experiment.

## What the evidence says about why

**The impact reward was badly scaled.** My change made the cost of a brief
impact independent of its duration, but left it unbounded and quadratic. The
new arm accumulated23,283 impact-penalty units against7,431 total positive
reward units. The old arm accumulated115 impact-penalty units. This is about
200 times more penalty, not a modest reweighting. Reconstructing the new cost
from completed, admitted attempts shows99.2% came from impact-failure attempts;
one attempt cost approximately765 units, compared with the ordinary5-unit
failure penalty. The penalty therefore concentrated on large terminal impacts.

The reward predictor was also inherited from the old objective. Mean value loss
over the final32 updates was516.6 with the new reward versus4.02 with the old
one. Those losses have different target scales, so this ratio is not a
scale-independent measure of model quality. Together with the outsize penalties,
it identifies reward scaling and critic adaptation as strong suspects. The
experiment did not isolate these causes from each other.

**Fewer useful seated examples were produced.** The new arm accumulated420.34
robot-seconds of qualifying supported sitting, versus470.66 with the old reward.
The effort reward only applies during those qualifying intervals. It did not
translate into lower torque in the final matched tests. Both reward terms were
changed together, so the isolated effect of the effort term remains unknown.

**The objective and evaluation also differ in time.** Training earns reward
while the robot is supported and ends an attempt at the first failure. Evaluation
requires the whole20-second attempt to remain failure-free. Two newly lost cases
made prohibited contact at19.14 and19.68 seconds, after accumulating successful
hold time. PPO inherits gamma0.99 and lambda0.95 at50Hz, with64-step/1.28-second
rollouts. These settings rely heavily on the learned value predictor for delayed
outcomes. That is a plausible contributor, not a proven cause or a reason to
change every hyperparameter at once.

**There is no basis here for another perception rewrite.** In the one-second
windows before first contact/failure of the seven lost cases, the new policy's
distance availability was92.2–100%; accepted errors were at most1.70mm. Gaps
remain and have not been causally ruled out. The new independent clearance
channel is still weakly used: adding1cm changed the bounded correction by at
most1.76e-6rad on fixed recorded inputs. This is only a scalar-channel sensitivity
test; other geometry inputs may carry the same information.

The frozen motion scaffold still advances at its fixed pace; PPO learns bounded
joint corrections. Those corrections were not at their clipping limits in the
inspected landing windows. The logs do not establish that increasing action
bounds, adding a pace action, or changing tracking would solve the regression.

![Lost known case15](lost_case15.png)

This illustrative case was selected after seeing the results. The new policy
lands at850N versus445N for the starting policy. Its pelvis closes the final gap
faster near impact, despite close agreement between the accepted distance
estimate and evaluation truth. This is a behavioral example, not causal proof
of a particular reward or joint contribution.

## One scoring defect, disclosed separately

Adding0.02 seconds150 times in float32 gives2.999997854 seconds. The strict
`hold >= 3` comparison therefore rejects a physically three-second hold. This
exact value occurred in new-policy novel case13, with no physical failure.
Counting held control ticks changes totals to47/64,44/64,43/64. Original scores
and artifacts remain unchanged. The conclusion still holds. The next run
should use an integer hold counter and disclose both old and corrected baseline
scores. This defect affected evaluation success reporting, not the dense
supported-hold reward used for PPO.

## Recommended next experiment

Start from the47/64 starting checkpoint. Keep perception and motor limits fixed.
Run a short matched comparison of the old objective against a capped impact-only
change, leaving leg economy unchanged. Limit the entire additional episode
impact cost to5 units, comparable to the existing failure penalty; keep all
physical failure rules. Use32 updates per run and two training seeds, followed
by matched evaluation. This is a proposed parameterization, not a validated fix.

Then test the effort term separately if the impact comparison is informative.
Before adding further rewards, inspect their contributions on recorded attempts
so another terminal spike cannot overwhelm everything else. Fix integer hold
scoring for all prospective arms. Keep the original checkpoint as the reference;
neither128-update candidate should replace it.

The next protocol is saved as `NEXT_EXPERIMENT.md`. No new training was launched
during this diagnosis. This study used one training seed and does not prove a
general result about PPO or sitting on arbitrary surfaces.

Detailed measurements and provenance: `diagnosis.json`, the archived original
`../results/results.json`, and `diagnose_support_reward_study.py`. Frozen source
and model hashes still match; installed IsaacLab remains clean.
