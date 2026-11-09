# Complete: seat-load target validation and standing-start PPO pilot — October 14, 2026

The bounded study is complete and independently verified. The low-effort seated target works, but neither new PPO seed preserves the current standing-transition reliability. No policy was promoted. No training is pending. Existing completion automation remains paused. The broader goal of reliable low-effort sitting from standing remains unresolved.

Project: `/home/kadir/humanoid-sitting`. Study: `/data/humanoid-sitting-perception/seat_load_transfer_01`. Frozen protocol: `docs/SEAT_LOAD_TRANSFER_PROTOCOL.md`. Final export: `final_export_02`; Mac: `/Users/kadir/humanoid-sitting-control/validated_seat_load_transfer_01/final_results`. Local report/replay: `http://127.0.0.1:8798/final_results/findings.html`.

## Validated seated target

An upright pelvis target around x=-0.18m with the original leg preload removed passed 18/18 full 20-second seated replays across 38/40/42cm and -5/0/+5 degrees. Every physics tick met the support criteria; no physical failures; motor commands were constant. Final-five-second vertical seat load was 73.24–74.20% of bodyweight; sum of squared physical torque over all 37 joints was 91.07–95.64 Nm², and time-averaged instantaneous all-joint RMS was 1.569–1.608 Nm.

At the central 40cm flat seat the target used about 91.6 Nm² versus 869.4 Nm² for the current reference endpoint, with 74.1% versus 10.9% bodyweight on the seat. This compares resting postures with different starts, not matched whole-motion cost. Use `rest_verified_02` and `rest_replay_02`: the earlier normal-force/vertical-force label mistake was corrected with a preserved source, new explicit-force logging and a fresh replay. The physical target itself retained its posture with no unwanted contact.

## Standing-start comparison

Every final evaluation starts standing, runs 20 seconds and uses the unchanged success rule: no physical failure at any time and continuously qualifying support for the final five seconds.

| Controller | Development stable /18 | Physical failures | Reserved stable /16 | Physical failures |
|---|---:|---:|---:|---:|
| Current repaired reference | 18 | 0 | 16 | 0 |
| Current policy 56211 | 18 | 0 | 16 | 0 |
| Current policy 56217 | 18 | 0 | 16 | 0 |
| New reference toward upright rest | 4 | 2 | 0 | 4 |
| New PPO 56811 | 0 | 2 | 0 | 4 |
| New PPO 56817 | 2 | 2 | 1 | 4 |

Development contains nine geometries repeated twice. Reserved cases are eight geometries repeated twice: 38.5/41.5cm with ±2.5 degrees and 39.5/40.5cm with ±7.5 degrees. Both final checkpoints completed before the holdout barrier; no selection or calibration used reserved outcomes.

All new-policy physical failures were unwanted contacts, with no falls, impact-threshold, torque-limit or joint-limit failures. Development failures were at 42cm/-5 degrees. Reserved failures were at 39.5 and 40.5cm/-7.5 degrees, both placements. Earlier instrumented development probes identified a hip-roll housing hitting the seat; final policy evaluation records the forbidden-contact flag/force, not the contacting link identity.

Most other misses were insufficient conservative friction margin while still seated. Saved 100ms traces show very low late root speeds and no late slip flags in these cases, but friction utilization repeatedly exceeds the unchanged 0.9 threshold. Sampled traces can miss brief violations: seed56817 case0 looks good at every final-five-second sampled frame but has only 4.54 continuous qualifying seconds in the authoritative all-tick result. Do not relax or relabel this after evaluation.

## Effort: one successful reserved condition only

Seed56817 succeeds on case1 (38.5cm/-2.5 degrees). Compared with the matched current reference, whole-20s summed squared torque decreases 30.48%, final10s 58.09%, final5s 83.48%. Against current56211/current56217, whole-motion reductions are 27.02%/28.76%; final5s reductions are 80.71%/82.21%. Final5s torque is 137.86 Nm² and vertical seat load 85.16%BW. Whole-trial absolute torque falls 20.74% and mean RMS Nm falls 22.32% versus the current reference. Final5s leg effort falls 93.28%, while upper-body squared torque rises 75.86% (85.04 versus 48.36 Nm²). This is one repeated-placement condition, not a general improvement. Seed56811 has no common successful reserved trials, so no valid savings estimate. Failed fragments do not support savings claims.

## Diagnosis and next research direction

Current commands preload the bent legs and leave most weight on the feet. The new static target demonstrates that substantially lower effort is mechanically feasible. Getting there shifts the feet and leaves larger tangential forces than the desired rest reset. Before any physical failure, the new reserved evaluations retained all 1,438 audited geometry observations per seed: maximum plane offset error about 2.2mm and normal error below 0.39 degrees. Foot-anchored root-x errors reached 18–22mm, with measured horizontal foot displacement 22–24mm. Common foot translation violates the stationary-feet assumption of that estimate. These measurements identify a limitation; they do not by themselves isolate the cause of every failure.

A next bounded study should first bridge from a physically valid near-rest state to the validated target while reducing tangential foot forces and preserving housing clearance, then move the reset earlier only after successful transitions. Force/contact feedback or an explicit contact-constrained reference is a more targeted next change than simply extending this unchanged all-standing PPO budget. This is a proposed next study, not training already launched. Holdout results must not silently become tuning data; reserve new geometries for a future confirmation.

## Training and verification

Two fresh seeds, 64 updates each: 18 environments ×32 decisions ×5 physics ticks ×64 =184,320 transitions per seed; 368,640 total. First eight updates trained only the critic and left actor weights exactly unchanged; 56 updates changed actor weights. Seed56811:178,925 active+5,395 indexed-reset warmup. Seed56817:178,960 active+5,360 warmup. Initial settling adds 450 explicitly excluded ticks per seed. Separate development smokes used 4,320 transitions, including 1,440 in a preserved failed run with zero optimizer updates.

Runtime actor and critic receive 405 sensor/reference channels plus a clock, with 23 bounded body residuals and 14 fixed fingers. Ground truth is restricted to reset/reward/evaluation. The new active-sample PPO excludes reset warmup from actor/critic losses and prevents disabled-actor Adam drift; targeted regression tests passed. Fixed learning rate 1e-4; no frozen contract, geometry, reward, failure threshold or budget changed after launch.

`ppo_01/verified_01` and independently repeated `verified_02` both passed with byte-identical verification JSON. All 52 frozen and 3,341 prior hashes match; finite changed final weights, optimizer counts, exact checkpoint loads, exact physical/sensor startups, full horizons, fixed fingers and bounded sampled commands passed. All actual startup errors were zero despite the verifier permitting up to1e-5 for cross-GPU comparisons. All-tick safety flags come from frozen runtime code; sampled command/trace audits are additional checks.

The first export failed because CSV arm-label names collided with numeric metric names. `export_recovery_01.json` preserves the old source and failed `final_export_01`; corrected export02 changes column names only. Rest-load labeling and optimizer inference-tensor compatibility recoveries are also preserved. Research results were not changed to repair reporting.

Torque is an effort proxy, not electrical energy or proof of a global optimum. Two seeds, nearby rigid slabs and repeated placements do not establish arbitrary-surface generalization. This is reference-assisted residual PPO, not unassisted discovery. Earlier sitting/leaning studies and checkpoints remain unchanged.
