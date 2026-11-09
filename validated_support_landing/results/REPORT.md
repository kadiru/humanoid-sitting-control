# Landing distance tracking validation

Validation gate: NOT PASSED; no RL training was run.

| Bank | All-tick availability | Landing availability at 50 Hz | Worst accepted error | At first supported contact |
|---|---:|---:|---:|---:|
| new_starts | 93.58% | 98.68% | 0.602 cm | 15/15 |
| regression | 99.91% | 99.75% | 0.182 cm | 16/16 |
| fresh | 85.94% | 80.67% | 0.692 cm | 16/16 |

Fresh-bank loss diagnosis: the four long gaps start after physical failures, with torso tilt about 64–106 degrees. Before the first physical failure, fresh-bank availability is 99.74%; on its eight successful full trials it is 99.95%. These are post-hoc conditional diagnostics, not substitutes for the failed all-frame gate.

Startup refusals (included in the all-tick denominator): {"new_starts": [{"case": 15, "reason": "Aligned landing patch is occluded or not observable"}], "regression": [], "fresh": []}

Overall release requires at least 95% availability both overall and around landing on every bank, and no accepted error above 2 cm. A failed gate blocks promotion and more RL with this new observation pipeline.

Camera reads previously received images about 95 ms old. Explicit acquisition now captures at the controller timestamp. Sensor-only floor-normal/IMU agreement improved from a 100-ms best lag to zero lag; development maximum current angle is 0.063 degrees.

The tracker follows fresh seat depth, uses joints and IMU to carry a measured floor reference, and bridges camera intervals with a short kinematic prediction. If the seat is briefly hidden, a currently observed floor can carry its previously measured rigid height for at most 0.6 seconds. Predictions expire and uncertainty can make the output unavailable.

Online and sensor-only offline replay match bit for bit. Simulator labels are separate and used only for scoring. The tracker remains passive: it does not yet replace the frozen actor observation contract. Synchronized camera samples do change that original actor’s inputs, so historical motion parity is not claimed.

Control limitation: the unchanged policy completed 15/16 known trials with synchronized camera samples, versus 16/16 historically. Lost cases: [{"case": 14, "historical_peak_force_N": 455.20648193359375, "synchronized_peak_force_N": 729.1986694335938, "failures": {"fall": false, "wrong_contact": false, "pelvis_floor": false, "impact": true, "fast_descent": false, "torque": false, "perception_unavailable": false}}]. Improved estimation does not establish improved control, and no new tracker output was sent to the actor.

All code, datasets and checkpoints were retained. Installed IsaacLab remains clean. The old preservation checker verifies preservation but still fails its pre-existing 30-GiB root-disk reserve rule; the /data output volume is checked separately and has more than 30 GiB available.

These results cover aligned rigid horizontal seats and the recorded motion distribution. Simultaneous body translation and foot motion cannot be inferred from joint encoders alone; moving or deforming hidden seats are not validated.

Eight geometry tests and two acquisition tests pass. Online and offline replay match all seven saved arrays. With injected noise, the worst accepted error is 0.615 cm and fresh landing availability is 80.35%, including failed motions. In the camera blackout, stale estimates expire; all 12 non-fallen cases recover at the first new image (12.5 s), while the four already-fallen cases remain unavailable.

The summary figure compares 10-Hz camera updates; the table reports the complete 50-Hz tracking stream. Videos show successful fresh case6 and failed fresh case2, with documented selection rules.
