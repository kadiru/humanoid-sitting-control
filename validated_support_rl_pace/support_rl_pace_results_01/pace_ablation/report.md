# Pace-disabled diagnostic

Diagnostic on one known regression bank; not a new holdout, additional training, or promotion decision. Fixing pace also fixes its correctly observed actuator state.

| Controller | Full-attempt success | Worst impact |
|---|---:|---:|
| Previous PPO | 16/16 | 570.1 N |
| New PPO, pace enabled | 8/16 | 781.2 N |
| Same new PPO, pace disabled | 7/16 | 775.8 N |

Disabling pace regains cases [] and loses cases [4].
Cases still lost relative to the previous PPO: [2, 4, 6, 7, 9, 12, 13, 14, 15].
The candidate and diagnostic have exactly identical actor and critic weights. Both run for the full 20 seconds with latched failures. No checkpoint was retrained or selected using this test.
