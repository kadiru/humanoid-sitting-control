# Geometry-conditioned controller comparison

All rows use identical recorded initial states and the same held-out furniture bank. Safety criteria are unchanged.

| Controller | Success | Falls | Unintended | Excess force | Peak N | Min hold s |
|---|---:|---:|---:|---:|---:|---:|
| geometry_reference_v12_confirmation_pilot_01 | 144/144 | 0.0% | 0.0% | 0.0% | 487.2 | 7.46 |
| geometry_reference_v12_confirmation_ppo_01 | 144/144 | 0.0% | 0.0% | 0.0% | 482.9 | 6.78 |
| geometry_reference_v12_confirmation_scripted_01 | 144/144 | 0.0% | 0.0% | 0.0% | 616.1 | 7.40 |

Paired estimates and geometry-block uncertainty are in `comparison.json`. No row with a safety violation is promoted.
