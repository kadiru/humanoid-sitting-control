# Landing visibility diagnostic

First sampled seat support >25% body weight, offline simulator label only. One second before through half a second after the event.

| Policy | Current depth unavailable at event |
|---|---:|
| Preserved policy | 11/16 |
| Continued PPO | 13/16 |
| PPO + parent penalty | 12/16 |

No causal intervention. Parent succeeds despite gaps. This audits the live ten-channel depth feature, not all robot perception or its proprioceptive contact estimate. Stored traces omit per-frame rejection reasons; do not infer the exact occlusion cause from validity alone.

Case 15 is lost by both trained candidates. All three have unavailable current depth at the support event. Accepted measurements remain accurate, but accuracy on accepted frames does not establish continuous availability during landing.

Next diagnostic: record raw sensor frames and rejection reasons through the landing phase; evaluate a short-term kinematic/geometry estimate with explicit uncertainty when a fresh plane cannot be fitted. Keep simulator truth as evaluation labels, and test on training geometries before any new policy experiment. Also test parent sensitivity to small command changes, since the preserved policy can succeed with the same validity gaps.

![Availability during landing](visibility.png)
