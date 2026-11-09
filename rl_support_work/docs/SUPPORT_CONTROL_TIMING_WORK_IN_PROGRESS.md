> **Completed:** all jobs finished. Read [final verified status](SUPPORT_CONTROL_TIMING_STATUS.md). The working notes below are preserved history.

# Active controller timing investigation — October 11, 2026

User asked to diagnose/fix the sitting regression after camera timing correction
and validate the controller with the new tracker before more RL. No training.

Primary agent stays on Mac, executing Ubuntu through SSH. Outputs remain under
`/data/humanoid-sitting-perception`; no existing source/checkpoint deleted.

Protocol: `docs/SUPPORT_CONTROL_TIMING_PROTOCOL.md`.
Current jobs: `control_timing_02` runs four matched 16 x 20 s evaluations;
`tracking_control_dev_01` waits for that exact tmux session to finish, then
runs a new tracker input adapter on the known bank. Both persistent jobs.

New scripts: `capture_support_control_timing.py` records all 144 actor inputs,
23 actions, previous full targets and bounded commands at 50 Hz.
`run_support_control_timing_01.sh` currently writes suffix-02 outputs.
Its completion status is `logs/support_control_timing_01.exit`.
An initial suffix-01 recorder failed on a list/tensor API error at the first
step; it is retained. A concurrent restart incurred ~4 minutes of RTX shader
compilation, now completed. No reinstall or cache deletion.

Verified so far:
- Historical timing reproduces 16/16, all 12 original trace arrays exactly,
  all 200 archived actor packets exactly. Reconstructed commands equal the
  next recorded previous commands exactly.
- Startup-only corrected: 15/16, case14 impact 697.77 N (historical 455.21 N).
- Live-only corrected: 15/16, same case14 impact 714.39 N.
- Both-corrected: 15/16, case14 729.20 N, all 12 traces exactly match the prior
  passive-tracking evaluation. The factorial is completed, exit0.
- Case14 corrected startup seat-height feature shifts -5.219 mm and initial
  body-relative seat z shifts +1.772 mm. Heading sine changes -0.006748.
  In fixed-sensor, open-loop nominal replay, the height channel contributes
  more pre-contact target change than XYZ, heading or extents. This is a
  sensitivity diagnostic, not a replacement for physical causal tests.
- Sensor gate flips around 10.5–10.6 s and 12.0–12.6 s differ by condition.
  Both acquisition changes independently expose policy fragility; do not
  claim startup alone explains everything.

Candidate: frozen `LandingGeometryTracker` runs at 50 Hz. The new pure-sensor
`support_tracking_features.measured_tracking_features` emits the original
10 features only for a current, uniquely identified observed seat patch.
The actor keeps its 10 Hz sample/age contract. It can now see a seat even when
the floor is absent. Distance-only FK/floor-map estimates remain available
in tracking logs but are NOT misrepresented as measured patch extents.
`capture_support_tracking_control.py` replaces actor channels 132:142, logs
exact fed features, and leaves all weights/109 original channels/motor limits
unchanged. It is an active actor input, not a passive shadow.

Twelve tests pass (8 existing tracker + 4 new feature-contract checks), including
bitwise parity with the original detector when its complete measurement is valid.
`logs/support_tracking_features_tests_02.log`.

The 10 Hz tracked-patch adapter completed: 15/16, case14 669.33 N. Better than
729.20 N but still above the unchanged 632.53 N limit; it is not a passing fix.

A second development trial, `continuous_control_dev_01`, is now active. It uses
`support_continuous_features.ContinuousPatchFeatures` to transport the actual
observed patch using foot FK and IMU at 50 Hz. It has a 100 ms expiry and charges
the full 3D translation, bilateral disagreement and plane reprojection to a
2 cm engineering budget. Five new geometry/motion/expiry tests pass, alongside
the earlier twelve tests. No gains, rewards or policy weights changed.
`capture_support_continuous_control.py` records probe points for offline
transport scoring. `score_support_continuous_features.py` compares them with
truth only after execution. `replay_support_continuous_features.py` rebuilds
all geometry from sensor archives and compares to actual actor inputs.

Next inspect this candidate result, diagnose if it fails, then freeze a
selected repair before the reserved paired fresh validation seeds. Do not
restart RL, tune physical failure thresholds, or claim promotion prematurely.

## Update: frozen candidate validation active

The continuous 50 Hz candidate restored 16/16 known-bank successes. Case14
peak force is 362.51 N, hold 6.28 s; no physical failure flags. Transport
validation: 15,760 accepted patch samples, 12,544 predictions, worst error
9.947 mm, prediction p95 0.563 mm, no >2 cm errors.

Frozen sources: `/data/humanoid-sitting-perception/support_control_frozen_01`
(25 source files, manifest). No validation data were seen before freezing.
`control_validation_01` now runs paired baseline/candidate evaluations on fresh
geometry/start 51941/51943, then mixed training shapes/new starts 51923.
`control_replay_01` reconstructs all geometry sensor-only on the development
capture and checks exact equality to actual actor packets. No RL or promotion.

Outputs: `support_control_validation_{fresh,starts}_{baseline,candidate}_01`.
Report/figures/video staging: `support_control_validation_results_01`.
Final results and all gates are pending; do not declare the broader problem solved.
