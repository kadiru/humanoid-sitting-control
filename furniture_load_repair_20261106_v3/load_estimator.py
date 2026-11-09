"""Qualified load observations. Unknown is never reported as zero or contact.

The frozen regressor's ensemble spread alone is not a validity test. A query
must be inside its demonstrated sensor envelope and have current geometry.
The interval is empirical; it is not a physical force sensor certificate.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LoadReading:
    state: str
    mean_bw: object
    lower_bw: object
    upper_bw: object
    reason: str
    outside_channels: tuple = ()

    @property
    def loaded(self):
        return self.state == 'loaded'

    @property
    def available(self):
        return self.state in ('loaded', 'unloaded')


class QualifiedLoadEstimator:
    """Sensor-only wrapper; callers may never fall back to the raw prediction."""
    def __init__(self, predictor, low, high, error_bound):
        self.predictor = predictor
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.error_bound = np.asarray(error_bound, dtype=np.float64)
        assert self.low.shape == self.high.shape == (333,)
        assert self.error_bound.shape == (3,)
        assert np.isfinite(self.low).all() and np.isfinite(self.high).all()
        assert np.isfinite(self.error_bound).all() and (self.error_bound >= 0).all()
        assert (self.low <= self.high).all()

    def read(self, history):
        x = np.asarray(history, dtype=np.float64).reshape(-1)
        if x.shape != (333,) or not np.isfinite(x).all():
            return LoadReading('unknown', None, None, None, 'invalid_sensor_packet')
        frame = x[-111:]
        if not (frame[109] > .5 and 0 <= frame[110] <= .21
                and .99 <= np.linalg.norm(frame[100:103]) <= 1.01):
            return LoadReading('unknown', None, None, None, 'geometry_unavailable')
        # Only the previously accepted float32 rounding allowance is used.
        outside = np.flatnonzero((x < self.low-2e-7) | (x > self.high+2e-7))
        if outside.size:
            return LoadReading('unknown', None, None, None,
                               'outside_training_sensor_envelope', tuple(outside.tolist()))
        mean, spread = self.predictor(x.astype(np.float32))
        mean = np.asarray(mean, dtype=np.float64)
        spread = np.asarray(spread, dtype=np.float64)
        if (mean.shape != (3,) or spread.shape != (3,)
                or not np.isfinite(mean).all() or not np.isfinite(spread).all()
                or (spread < 0).any()):
            return LoadReading('unknown', None, None, None, 'invalid_prediction')
        radius = self.error_bound + 3*spread
        lo, hi = mean-radius, mean+radius
        if lo[0] >= .05:
            return LoadReading('loaded', mean, lo, hi, 'qualified_lower_bound')
        if hi[0] < .01:
            return LoadReading('unloaded', mean, lo, hi, 'qualified_upper_bound')
        return LoadReading('unknown', mean, lo, hi, 'interval_crosses_contact_threshold')


def load_feedback(reading, previous_fraction, target_share, step=.008):
    """No unloading or extra downward search can be authorized by UNKNOWN.

    Existing bounded geometric approach commands are independent. This return
    governs only load-triggered posture changes, not the physical score.
    """
    if reading.state == 'unknown':
        return float(previous_fraction), False
    if reading.loaded:
        request = np.clip((reading.lower_bw[0]-.02)/max(float(target_share), .05), 0., 1.)
        return float(previous_fraction + np.clip(request-previous_fraction, -step, step)), False
    return float(previous_fraction), True
