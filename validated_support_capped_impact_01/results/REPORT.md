# Capped-impact-only PPO results

All four runs completed32 updates /32,768 transitions each. Final policies evaluated on the same80 cases per seed; every attempt ran20 seconds. No policy promoted.

| Training seed | Starting policy | Old reward | Capped impact |
|---|---:|---:|---:|
|52813|53/80|35/80|36/80|
|52817|53/80|54/80|55/80|

![Matched successes](comparison.png)

The timer regression reproduced all archived motion/sensor/command arrays bitwise and corrected only the known150-tick case. Existing reference results were transparently rescored using ticks. All evaluation startup archives and geometries match.

The first64 placements are known regression sets; the last16 are a fresh holdout. Results are reported by seed, not by selecting the best seed. Pooled outcomes repeat placements across two different learned policies. Two seeds remain exploratory evidence.

## Seed52813

Capped versus old reward: gained4, lost3. Among32 common successes, mean peak-force change4.491694927215576N; leg RMS change-0.1556728482246399Nm.

## Seed52817

Capped versus old reward: gained4, lost3. Among51 common successes, mean peak-force change12.834773344152113N; leg RMS change-0.6853703798032275Nm.

Full failure types, force tails, per-case outcomes, training accounting and source verification are in results.json. Neither actor nor critic gained privileged runtime inputs. Effort reward, perception and motor limits stayed fixed.

