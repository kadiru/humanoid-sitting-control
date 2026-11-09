# Geometry-conditioned controller comparison

All rows use identical recorded initial states and the same held-out furniture bank. Safety criteria are unchanged.

| Controller | Success | Falls | Unintended | Excess force | Peak N | Min hold s |
|---|---:|---:|---:|---:|---:|---:|
| geometry_reference_v12_holdout_canonical_pilot_01 | 102/144 | 0.0% | 0.0% | 24.3% | 902.3 | 0.00 |
| geometry_reference_v12_holdout_pilot_01 | 144/144 | 0.0% | 0.0% | 0.0% | 486.3 | 7.48 |
| geometry_reference_v12_holdout_pilot_blind_01 | 144/144 | 0.0% | 0.0% | 0.0% | 486.8 | 7.60 |
| geometry_reference_v12_holdout_ppo_01 | 144/144 | 0.0% | 0.0% | 0.0% | 467.3 | 7.42 |
| geometry_reference_v12_holdout_scripted_01 | 144/144 | 0.0% | 0.0% | 0.0% | 469.2 | 7.56 |

Paired estimates and geometry-block uncertainty are in `comparison.json`. No row with a safety violation is promoted.
