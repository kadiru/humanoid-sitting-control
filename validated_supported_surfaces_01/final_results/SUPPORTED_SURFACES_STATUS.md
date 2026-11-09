# Standing-to-supported sitting and leaning — completed, no learned transition

Verified October 12, 2026 (PDT; October 13 UTC). Both 224-update PPO runs and all matched 20-second evaluations finished. This pilot did not teach the robot to reach support. No policy was promoted and no extra training was launched. The completion monitor is being paused after delivery. Prior studies and all checkpoints remain preserved.

## Physical outcomes

| Controller | Training surfaces | Unseen surfaces | Physical failures |
| --- | --- | --- | --- |
| Fixed standing | 13/13 remain standing | 16/16 remain standing | 0/29 |
| PPO seed 54211 | 0/13 reach stable support | 0/16 reach stable support | 29/29 |
| PPO seed 54217 | 0/13 reach stable support | 0/16 reach stable support | 29/29 |

Fixed standing does not attempt a sit. Each PPO seed scored 0/7 sitting and 0/6 leaning on training surfaces; 0/10 sitting and 0/6 leaning on unseen surfaces. Every policy evaluation eventually latched fall, unwanted-contact and joint-limit failures. One seed-54217 training trial also latched an impact failure; no torque-limit failures. First physical failures occurred at 0.76–0.80 s and 0.62–0.74 s by seed. All rollouts nevertheless ran the specified full 20 seconds without privileged policy freezing.

The final policies never achieved a qualifying support tick. Training also produced zero full-episode successes, including episodes initialized already supported. Maximum supported-reset qualifying holds were only 0.32/0.18 s. Endpoint validation is not learned transition success.

## Diagnosis

The initial network exactly copied the stable command, but that copy was implemented inside trainable layers. CPU replay of unchanged recorded standing observations shows requested joint-target shifts of 16.7/18.0 degrees after the very first optimizer update; final shifts reach 30.9/37.9 degrees. At final evaluation, 17–18 of 23 joints immediately hit the command-rate bound. In the illustrated 25 cm trial, actual commanded joints move by up to 30.1 degrees within 0.7 s. Requested and actual command shifts are distinct metrics. This identifies failed control learning before any later geometry drift; it does not isolate a unique cause.

All 398 saved 10 Hz pre-failure geometry observations remain marked valid, but availability is not accuracy. Distance-error p95 on train/unseen surfaces is 5.46/6.52 mm for seed 54211 and 23.29/23.93 mm for seed 54217; maximum 36.08 mm. Normal-error p95 stays below 0.39 degrees. Camera-time p95 distances alone are only 2.16–2.56 mm and hide between-frame drift. The kinematic bridge assumes stationary feet. These brief unstable motions do not validate tracking through a successful sit, and not every 50 Hz tick was saved. Ground truth remains confined to resets, rewards and accuracy/failure labels; actor and critic use the same 333 sensor-derived channels.

## Budget and integrity

Each seed completed 224 × 64 × 26 = 372,736 transitions. Seed 54211: 188,417 active + 184,319 warmup; seed 54217: 199,899 active + 172,837 warmup. Total 745,472; warmup shares 49.45%/46.37%. The initial 0.5-second global settle and separate 1,664-transition development smoke test are excluded. Episode records reconcile with pending active/warmup ticks. Both actor and critic are finite, actor weights changed, final files match update 223 (zero-based), and evaluation loaded the exact final actors.

All 36 frozen source/bank hashes and 46 prior-study hashes pass. Per-run saved sources and bank manifests also match. Four policy/baseline startup comparisons are bitwise exact for root, joints, commands, geometry and all observations. No tolerance was relaxed. Evaluation durations are exactly 1,000 control ticks; fixed commands and uncontrolled joints are constant, learned command-rate bounds pass. No external disturbance schedule was used. A reporting-only path-canonicalization repair and a plot correction from Euler pitch to total tilt are preserved with their records; neither changed simulation or training.

## Validated infrastructure and limitations

All 13 training endpoints hold for 20 seconds, including 25/30/35/40 cm level seats and tilted sitting/pelvis leaning through 60 degrees. Separate back-supported endpoints at 75 degrees pass; three are unseen tests. One unseen changed-outline endpoint remains borderline (98.7% qualifying ticks but missing its final continuous 2-second endpoint criterion) and remains reported. Vertical 90-degree support is unvalidated. Five geometry/tracker tests, reset isolation, standing retention and the smoke optimizer check passed before launch.

Unseen tests include intermediate 27.5/32.5/37.5 cm level heights, 10/30-degree slopes, narrower elliptical/chamfered outlines and 75-degree back support. Low seats have only standing and supported reset endpoints, leaving a difficult exploration gap. Two seeds reuse a finite prealigned rigid bank; arbitrary furniture, cushions, approach/alignment and hardware are untested. There are no common successful learned transitions for an effort-saving claim. Torque is an effort proxy, not energy; fixed supported endpoints are not matched transition-energy baselines.

## Next implementation direction, not launched

Protect stable reference behavior outside trainable command-copy layers and validate bounded exploration first. Then build physically feasible short motions connecting standing and supported states and train their progression, with support/stability primary and effort secondary. These are proposed remedies, not demonstrated results. Do not blindly extend the failed runs or change their frozen definitions.

## Evidence

Remote study: `/data/humanoid-sitting-perception/supported_surfaces_01/transition_ppo_01`.
Verified report: `verified_review/review.html`, `validation.json`, `supplementary_audit.json`, `diagnosis/diagnosis.json`.
Mac report: `/Users/kadir/humanoid-sitting-control/validated_supported_surfaces_01/final_results/review.html`.
Setup evidence remains in the Mac parent `validation_evidence_01` folder and original Ubuntu directories. Prior status documents are preserved in `final_results/document_history` and the remote report. The frozen protocol is unchanged.
