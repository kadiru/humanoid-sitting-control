# Independent-reset PPO pilot

Experimental candidate only; the existing default controller is unchanged.

Full 20-second evaluation with latched failures. Training resets independently at failure or horizon.

| Evaluation | Success | Worst impact | Failures |
|---|---:|---:|---|
| fresh_zero | 5/16 | 1902.9 N | fall: 1, wrong_contact: 5, impact: 10 |
| fresh_prior | 5/16 | 1968.3 N | fall: 1, wrong_contact: 5, impact: 11 |
| fresh_candidate | 5/16 | 1948.1 N | fall: 2, wrong_contact: 6, impact: 10 |
| regression_prior | 15/16 | 766.5 N | impact: 1 |
| regression_candidate | 16/16 | 570.1 N | none |

Training added 98,304 transitions in 713.3 seconds. Completed 160 attempts, with 25 successes and 10 startup refusals. Training failures end early; these counts are not a held-out success score. All transitions and unfinished attempts are accounted for.

- fresh_zero → fresh_prior: gained []; lost []; leg torque -8.12% on 5 matched successful attempts.
- fresh_prior → fresh_candidate: gained []; lost []; leg torque +13.97% on 5 matched successful attempts.
- fresh_zero → fresh_candidate: gained []; lost []; leg torque +4.72% on 5 matched successful attempts.
- regression_prior → regression_candidate: gained [15]; lost []; leg torque -2.54% on 15 matched successful attempts.

Indexed reset with a one-control-tick pose hold before camera admission; no five-step global settling interval. All comparisons here use this same new startup protocol.

Reset probe: other robots had exactly zero measured state changes; no physics tick; selected controller matched a cold start. Twelve targeted reward/admission/reset tests pass.

The original no-warmup reset exposed old camera transforms. A paired diagnostic reduced maximum startup height error from 49.55 to 5.39 mm, and increased usable live depth samples from 6/32 to 27/32. The two preliminary `_03` full evaluations are preserved as debug runs and excluded from this comparison. No main PPO training had run before that correction.

- Small pilot with 8 fresh geometries and two starts each, not statistical reliability proof.
- Rigid seats 26–35 cm high; no deformable cushions, navigation or hardware validation.
- Frozen motion scaffold and bounded +/-0.08 rad residual remain. Lowering timing is not a learned action yet.
- Sensor-only actor and critic; true pose/contact used only for reward, termination and offline evaluation.
- Foot force remains an evaluation label, not an assumed hardware input.
- Geometry prototypes are fixed within a run; robot starts vary at each independent reset.
- Unobservable starts are counted as refusals; unfinished training attempts are recorded separately.
- Numerical source, reward and final-checkpoint selection were fixed before the corrected fresh holdout. Earlier camera-reset diagnostics are excluded.

On the fresh candidate rollout, 95% of absolute joint corrections were below 0.202 degrees and the maximum was 0.569 degrees. The fraction reaching the existing bound was 0.000000. This run does not establish that the hard action bound caused the failures. The policy used only a small part of its available range; structured exploration and learning the lowering motion are the next hypotheses to test, not a validated fix.


The earlier reset probe 02 stopped on a comparison between different GPU batch sizes (maximum latent difference 0.00005245). The final probe compares the same batch size at unchanged tight tolerances and passes. Its earlier log and source are preserved.
