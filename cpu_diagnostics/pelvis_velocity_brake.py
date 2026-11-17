"""Damp measured rearward pelvis motion before the tilted seat takes load."""

import numpy as np

from prepare_sitting_repairs import smoothstep
from seat_plane_approach import stance_axes


FIRST_TICK = 350
INTERVAL = 5
DT = .02 * INTERVAL
BRAKE_GAP_M = .050
CONTACT_GAP_M = .002
SPEED_TARGET_M_S = .012
MAX_SHIFT_M = .015


class PelvisVelocityBrake:
    def __init__(self, approach):
        self.approach = approach
        self.active = np.asarray([i for i in approach.active if np.ptp(approach.offsets[i]) <= .010], dtype=int)
        self.offset = np.asarray([np.median(approach.offsets[i]) for i in self.active])
        count = len(self.active)
        self.previous = np.full(count, np.nan)
        self.forward = np.zeros(count)
        self.delta = np.zeros((count, 23), dtype=np.float32)
        self.started = np.full(count, -1, dtype=int)
        self.touch = np.full(count, -1, dtype=int)
        self.rejected = np.zeros(count, dtype=int)
        self.trace = []

    def measure(self, row, frame, offset):
        feet = self.approach.feet(frame[:23])
        axes = stance_axes(feet, frame[69:72])
        normal = axes @ self.approach.normal[row]
        tangent = axes[:, 0] - normal * (axes[:, 0] @ normal)
        tangent /= np.linalg.norm(tangent)
        vertices = self.approach.vertices
        distances = vertices @ normal
        nearest = distances.min()
        patch = np.average(vertices, axis=0, weights=np.exp((nearest-distances)/.001))
        center = feet[:, :3, 3].mean(0)
        return (patch-center) @ tangent, nearest - center @ normal + offset

    def update(self, tick, frames, desired):
        for i, row in enumerate(self.active):
            frame = frames[i]
            position, gap = self.measure(row, frame, self.offset[i])
            previous = self.previous[i]
            self.previous[i] = position
            speed = (previous-position)/DT if np.isfinite(previous) else 0.
            if self.touch[i] < 0 and gap <= CONTACT_GAP_M:
                self.touch[i] = tick
            available = 1. if self.touch[i] < 0 else 1. - smoothstep((tick-self.touch[i]-5)/10)
            amount = smoothstep((BRAKE_GAP_M-gap)/.020) * available
            rate = min(.075, max(0., speed-SPEED_TARGET_M_S)) * amount
            self.forward[i] = min(MAX_SHIFT_M, self.forward[i] + DT * rate)
            if self.forward[i] and self.started[i] < 0:
                self.started[i] = tick
            if self.forward[i]:
                translated = self.approach.correct(desired[i], frame, self.approach.normal[row], 0., self.forward[i])
                if translated is None:
                    self.rejected[i] += 1
                    self.delta[i] = 0.
                else:
                    self.delta[i] = translated[0]
            self.trace.append([tick, row, gap, speed, self.forward[i], np.max(np.abs(self.delta[i]))])
        return self.delta

    def save(self, path):
        np.savez_compressed(path, active=self.active, started=self.started, touch=self.touch,
                            rejected=self.rejected, forward_m=self.forward, trace=np.asarray(self.trace),
                            columns=['tick', 'row', 'gap_m', 'backward_m_s', 'forward_bias_m', 'max_delta_rad'])
