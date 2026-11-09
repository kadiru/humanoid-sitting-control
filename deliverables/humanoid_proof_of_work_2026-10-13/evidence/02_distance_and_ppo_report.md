# Distance interface fixed; one PPO pilot completed

The policy now receives reliable hip-to-seat clearance even when the visible seat
edges become uncertain. Distance has its own validity, age and error-budget
channels. Missing or expired distance is explicitly marked unavailable.

The expanded controller initially reproduced the old controller exactly:
16/16 known successes, identical actions and all12 recorded motion arrays.
During known-bank landings, distance availability increased from **82.73% to
99.67%**, with worst accepted error **1.74mm**. Sensor-only replay reproduced all
15,984 consumed post-startup geometry packets exactly. CPU/GPU migration,
invalid-data tests and isolated-reset checks passed.

## PPO comparison

One fixed32-update run,32,768 transitions, unchanged rewards/gains/action limits,
fixed normalization and a fresh optimizer. The final checkpoint was evaluated
without selecting among intermediate checkpoints. Each policy ran48 full20s
trials; failures and startup refusals remain in the scores.

| Test bank | Frozen controller | After PPO | Worst seat force: frozen → PPO |
|---|---:|---:|---:|
| Known regression |16/16|16/16|521 →616N|
| Fresh seats and starts |12/16|13/16|1078 →1176N|
| Mixed training seats, new starts |7/16|8/16|6448 →6448N|
| **Total** |**35/48**|**37/48**| |

No previously successful case was lost. Both new successes came from the fresh
tests. However, impact severity worsened on two banks, and mean seated leg torque
increased among cases both controllers completed: +0.112,+0.199,+0.132Nm across
the three banks. **The trained policy is preserved but not promoted.**

![Paired PPO comparison](ppo_comparison.png)

## What improved, and what remains limited

Fresh case13 increased supported hold from2.68 to5.14s. Mean foot load during
qualifying support decreased from66.0% to45.4% of body weight; mean leg torque
decreased from7.00 to6.37Nm over those intervals. Its leg torque nevertheless
rose late in the trial, so sustained low-effort sitting is not established.

Known case14 held longer but increased impact from363 to616N. The examples below
were chosen after scoring to show both outcomes; all48 paired cases remain in
the main results.

![One improvement and one regression](case_comparison.png)

The new-start bank contains the same startup refusal (case15) in both policies:
the landing patch was not observable. Sitting was never enabled, yet subsequent
physical contacts are retained; this case accounts for the6448N whole-bank
maximum. Admitted-only diagnostics are secondary and do not replace that score.

Distance accuracy remained below7.65mm for all accepted inputs across all96
evaluations. Landing availability ranges from96.31% to99.92%. Before-first-failure
coverage on the hard new-start bank remains below the99% diagnostic threshold:
98.88% for the parent and98.94% after PPO. Both coverage failures remain recorded.
The interface forwards accepted distance independently of patch validity; it
does not invent estimates when the tracker or startup detector refuses them.

![Distance and patch validity during landing](distance_interface.png)

New distance weights learned nonzero values, but scalar-clearance sensitivity is
very weak: a1cm perturbation changed the bounded correction by at most3.0e-7rad
on recorded training inputs (2.77e-7rad with patch unavailable). This experiment
compares continued PPO against a frozen controller; it does **not** isolate the
benefit of the extra channels from PPO using the original inputs. See
[functional audit](policy_input_audit.json).

## Evidence and reproduction

- [All paired cases, metrics and failures](results.json)
- [Startup parity, training accounting and secondary diagnostics](extended_verification.json)
- [Frozen protocol](frozen/docs/SUPPORT_SPLIT_PPO_PROTOCOL.md)
- [Original automatically generated report](REPORT_generated.md)
- Frozen runtime/training sources are in `frozen/`; subsequent analysis scripts
  are in `posthoc_sources/`. All original checkpoints and frozen source hashes
  verified unchanged. IsaacLab is clean. No software was installed, and no
  runtime defaults changed. Both persistent jobs finished with exit0.

Ubuntu checkpoint: `/data/humanoid-sitting-perception/support_split_ppo_train_01/final.pt`.
SHA256: `ece102b139323af18bd1c7aad6d8fb06e478f38946f07e52406a31bd4acc7dda`.
Load it through `scripts/train_support_split_observation.py`; it requires the
151-channel layout and split first-layer computation.

Scope remains aligned rigid seats in simulation. The geometry tracker uses
clean simulated depth/encoder/IMU equivalents; the original109-channel packet
retains its existing noise/delay. No new camera-noise or hardware validation was
performed. Simulator pose/forces supply rewards and evaluation labels only.
The geometry budget is conditional, not calibrated statistical confidence.
