# Pace exploration mismatch

Isolated action mapping with constant zero Gaussian mean and std 0.5; no robot simulation or success score.

With a zero action mean and standard deviation 0.5, the actual pace filter averages 0.8632 under sampled training actions, versus 1.0000 when executing the mean. Upper clipping at nominal speed makes zero-mean exploration systematically slower. Executing the raw Gaussian mean is not equivalent to the average transformed training action.

This establishes an action-distribution mismatch, not that it alone caused the robot regressions. Actual policy means and states change during training.
