# Camera timing and controller validation

**The original timing regression is repaired on the known bank.** Corrected camera timing with the old observer gives 15/16 successful sits; the frozen policy with 50 Hz tracked geometry gives 16/16. Case14 peak seat force changes from 729.20 N to 362.51 N, below the unchanged 632.53 N limit. No RL, weight update, motor-gain change or failure-threshold change was used.

**Broader release gate: NOT PASSED.** RL remains paused and deployment defaults are unchanged. Fresh tests are reported below, including all unsuccessful cases and worst impacts.

## Why newer camera images changed the behavior

The frozen controller was trained with an observation pipeline containing timing errors and intermittent missing seat measurements. Correcting the first image changed its initial geometric plan; correcting later images changed its feedback. Either change independently lost the same known-bank sit. The historical control reproduces all 12 recorded trajectory arrays and all archived actor packets exactly. The both-corrected control also reproduces the earlier passive-tracker run exactly, ruling out recording overhead or passive tracking as the cause.

| Camera condition | Successful sits | Case14 peak force |
|---|---:|---:|
| Historical timing | 16/16 | 455.21 N |
| Startup corrected only | 15/16 | 697.77 N |
| Descent corrected only | 15/16 | 714.39 N |
| Both corrected | 15/16 | 729.20 N |
| Both corrected + tracked patch at 10 Hz | 15/16 | 669.33 N |
| Both corrected + tracked patch at 50 Hz | 16/16 | 362.51 N |

In case14 the corrected startup seat-height feature changes by -5.219 mm and body-relative seat height by +1.772 mm. The commanded targets change immediately. The failed landing cannot be attributed to one scalar alone: initial geometry, patch validity, timing and closed-loop motion interact.

## Implemented repair

The depth camera remains synchronized. The same observed seat patch is carried into the current pelvis frame every 20 ms using joint forward kinematics and IMU orientation. New images refresh it every 100 ms. The ten actor features contain current clearance, normal, observed-patch center and spans, validity, and the age of the underlying image. Simulator geometry and forces are used only for separate scoring.

Prediction lasts at most 100 ms. Its engineering error budget charges the full inferred 3D body translation, disagreement between the two feet, and any plane reprojection correction. Unknown or over-budget geometry is withheld. Two feet moving together cannot prove contact; the full-motion charge is a conservative guard, not a statistical guarantee. These are tracked observed points, not inferred hidden furniture bounds.

## Frozen paired validation

| Bank | Corrected baseline | Tracked controller | Cases repaired | Cases regressed | Worst force: baseline → tracked |
|---|---:|---:|---|---|---:|
| regression | 15/16 | 16/16 | [14] | [] | 729 → 521 N |
| fresh | 10/16 | 10/16 | [] | [] | 3486 → 3662 N |
| new_starts | 9/16 | 9/16 | [] | [] | 1753 → 1951 N |

Fresh shapes/starts use 51941/51943; new starts on the unchanged mixed training bank use 51923. Source and settings were frozen before those data were viewed. Each attempt lasts 20 s even after a physical failure. There are 160 completed evaluation attempts across diagnosis, development and validation; none are RL training.

| Bank | Tracker full-trial availability | Tracker landing availability | Actor landing availability | Worst distance error | Worst patch transport error |
|---|---:|---:|---:|---:|---:|
| regression | 99.91% | 99.67% | 82.73% | 1.74 mm | 9.95 mm |
| fresh | 93.66% | 92.60% | 72.45% | 6.90 mm | 6.68 mm |
| new_starts | 96.18% | 93.17% | 73.93% | 7.25 mm | 7.29 mm |

The actor requires a complete patch estimate. Its input is withheld more often than scalar distance, particularly during rapid motion or loss of a reliable foot reference. All unavailable and post-failure frames remain in the denominators. Direct camera-point samples have zero *transport* error by definition; the reported transport maximum and p95 also separately cover predicted samples. Distance error is independently measured against true seat clearance.

Worst impacts increased in already-failing fresh/new-start trials despite unchanged success counts. The candidate remains experimental; the known-bank repair does not establish that every motion is safer.

## Gates and verification

- known_regression_restored: **True**
- paired_fresh_success_not_reduced: **True**
- no_new_failure_category: **True**
- tracker_accuracy_and_coverage: **False**
- patch_transport_under_2cm: **True**
- Additional complete-actor-input accuracy/coverage check: **False**
- Additional worst-impact severity check: **False**
- Seventeen tests pass. All known-bank sensor-only replay actor packets match the online geometry bit for bit (15,984 packets).
- All corrected camera acquisition audits have zero sensor timestamp age; no extra physics step is inserted.
- Frozen runtime sources and parent checkpoint are hash verified. Old checkpoints, datasets, experiments and the unsuccessful 10 Hz adapter are preserved.
- The known-bank geometry update takes 1.84 ms median and 37.17 ms p95 per robot, including image frames. The 50 Hz claim describes simulation/control timestamps; hard real-time hardware throughput is not established.

## Scope and next action

The known regression is fixed, but promotion and further RL require the remaining failed gates to be addressed. Next address the loss of complete geometric feedback during fast motion, then the remaining balance/collision failures and worse impacts. A useful next investigation is requesting a new depth frame when the kinematic error budget is about to expire, and separating distance confidence from patch-shape confidence at the controller boundary. Any modified candidate needs new validation seeds; the banks in this report are now consumed. The current work covers aligned rigid horizontal surfaces in simulation; it does not establish arbitrary furniture, cushions, approach/navigation or hardware performance.

See [all machine-readable results](results.json), [timing comparison](timing_control_comparison.png), [fresh validation plot](validation_summary.png), [the repaired landing video](repaired_landing/landing_tracking.mp4), and [a remaining fresh-bank fall](fresh_failure/landing_tracking.mp4). The repaired video is the originally regressed case14; the failure video is fresh case10, the first remaining fall by case index. Both use depth and joint kinematics for the 3D scene; truth appears only on the accuracy graph.
