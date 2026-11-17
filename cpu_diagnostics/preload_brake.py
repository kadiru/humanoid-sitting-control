"""Slow the demonstrated motion through seat load transfer using sensed clearance."""

import numpy as np

from prepare_sitting_repairs import smoothstep
from seat_plane_approach import stance_axes


FIRST_TICK = 350
BRAKE_GAP_M = .022
FULL_BRAKE_GAP_M = .009
TOUCH_GAP_M = .003
MIN_RATE = .28
SETTLE_TICKS = 40
RELEASE_TICKS = 30
MAX_DELAY = 84.
RECOVERY_START = 685.
RECOVERY_RATE = .20
GAP_INTERVAL = 5


def clearance(approach, row, frame, offset):
    feet = approach.feet(frame[:23])
    normal = stance_axes(feet, frame[69:72]) @ approach.normal[row]
    return np.min(approach.vertices @ normal) - feet[:, :3, 3].mean(0) @ normal + offset


class PreloadBrake:
    def __init__(self, approach):
        self.approach = approach
        rows = approach.active
        self.active = np.asarray([i for i in rows if np.ptp(approach.offsets[i]) <= .010], dtype=int)
        self.offset = np.asarray([np.median(approach.offsets[i]) for i in self.active])
        count = len(self.active)
        self.clock = np.full(count, FIRST_TICK - 1., dtype=np.float64)
        self.touch = np.full(count, -1, dtype=int)
        self.started = np.full(count, -1, dtype=int)
        self.gap = np.full(count, np.inf)
        self.slowdown = np.zeros(count)
        self.trace = []

    def update(self, tick, frames=None):
        if frames is not None:
            self.gap = np.asarray([
                clearance(self.approach, row, frame, offset)
                for row, frame, offset in zip(self.active, frames, self.offset, strict=True)
            ])
            seen = (self.started < 0) & (self.gap < BRAKE_GAP_M)
            self.started[seen] = tick
            touch = (self.touch < 0) & (self.gap <= TOUCH_GAP_M)
            self.touch[touch] = tick
            fraction = smoothstep((BRAKE_GAP_M - self.gap) / (BRAKE_GAP_M - FULL_BRAKE_GAP_M))
            self.slowdown = np.maximum(self.slowdown, fraction)

        release = smoothstep((tick - self.touch - SETTLE_TICKS) / RELEASE_TICKS)
        release[self.touch < 0] = 0.
        rate = 1. - (1. - MIN_RATE) * self.slowdown * (1. - release)
        # Reserve time for the original hold. Ease off the brake as the
        # remaining permitted delay shrinks, including if vision saw no touch.
        remaining = MAX_DELAY - (tick - 1 - self.clock)
        rate = np.maximum(rate, 1. - np.clip(remaining / 15., 0., 1.))
        # Repay the delay during the already settled part of the reference.
        recovery = smoothstep((self.clock - RECOVERY_START) / 35.)
        rate += RECOVERY_RATE * recovery * np.where(self.touch >= 0, release, 0.)
        self.clock = np.minimum(tick, self.clock + rate)
        self.trace.append(np.stack((self.clock.copy(), self.gap.copy()), axis=1))
        return self.clock

    def save(self, path):
        np.savez_compressed(path, active=self.active, first_tick=FIRST_TICK, started=self.started,
                            touch=self.touch, trace=np.asarray(self.trace),
                            columns=['reference_tick', 'estimated_pelvis_gap_m'])
