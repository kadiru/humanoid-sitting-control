# Longer PPO reward experiment

Both arms completed128 updates (131,072 transitions each) from the same151-input checkpoint. Perception, motor limits and success criteria stayed fixed. No policy was promoted.

| Bank | Starting PPO | Old reward | New reward |
|---|---:|---:|---:|
| regression | 16/16 | 16/16 | 15/16 |
| fresh | 13/16 | 12/16 | 10/16 |
| starts | 8/16 | 8/16 | 8/16 |
| novel | 10/16 | 8/16 | 9/16 |

Totals out of64: {'initial': 47, 'legacy': 44, 'gentle': 42}. Original controller scored35/48 on the first three banks; the starting PPO scored37/48.

![Comparison](comparison.png)

Full case outcomes, force maxima, failures, gained/lost successes and leg/foot-load changes on common successes are in results.json. All144 new evaluation attempts ran the full20 seconds; startup sensor archives and geometry matched within each bank. Final checkpoints were evaluated without selection.

This is one training seed and initially aligned rigid seats. The two reward changes are tested together. It does not establish general sitting, hardware readiness, or a causal benefit from the distance channels. Tracking imperfections remain part of the experiment. Training returns use different objectives and should not be compared directly.

## regression

Worst forces (N): starting 616.5, old reward 624.1, new reward 850.3.

legacy: gained []; lost []; leg RMS change among 16 common successes: 0.419643297791481 Nm.
gentle: gained []; lost [15]; leg RMS change among 15 common successes: 0.3396116256713867 Nm.

## fresh

Worst forces (N): starting 1176.1, old reward 1542.9, new reward 1478.4.

legacy: gained []; lost [4]; leg RMS change among 12 common successes: -0.49968258539835614 Nm.
gentle: gained []; lost [0, 4, 13]; leg RMS change among 10 common successes: 0.22524232864379884 Nm.

## starts

Worst forces (N): starting 6447.9, old reward 6447.9, new reward 6447.9.

legacy: gained [8, 14]; lost [4, 6]; leg RMS change among 6 common successes: -0.4702550172805786 Nm.
gentle: gained [14]; lost [2]; leg RMS change among 7 common successes: 0.9884443283081055 Nm.

## novel

Worst forces (N): starting 5984.8, old reward 5623.7, new reward 5623.7.

legacy: gained [10]; lost [8, 12, 14]; leg RMS change among 7 common successes: -0.44256326130458284 Nm.
gentle: gained [10]; lost [12, 13]; leg RMS change among 8 common successes: 0.833836704492569 Nm.

