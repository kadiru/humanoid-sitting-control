"""Arrest leg descent at first supported contact if a hip shell is nearly tangent."""

import numpy as np

from seat_plane_approach import stance_axes


class LoadedClearanceHold:
    def __init__(self, approach, model, metadata):
        self.approach = approach
        self.model = model
        self.joint_ids = metadata['joint_ids']
        count = len(approach.normals)
        self.first_load = np.full(count, -1)
        self.clearance = np.full(count, np.nan)
        self.enabled = np.zeros(count, dtype=bool)
        self.target = np.zeros((count, len(approach.legs)), dtype=np.float32)

    def observe(self, tick, load, frames, previous, braked):
        for row in np.flatnonzero((load >= .05) & (self.first_load < 0)):
            self.first_load[row] = tick
            if not braked[row]:
                continue
            samples = np.asarray(self.approach.normals[row])
            if len(samples) < 4 or np.ptp(samples, axis=0).max() > .035:
                continue
            frame = frames[row]
            feet = self.approach.feet(frame[:23])
            direction = np.median(samples, axis=0)
            normal = stance_axes(feet, frame[69:72]) @ (direction / np.linalg.norm(direction))
            full = np.zeros(len(self.model.names))
            full[self.joint_ids] = frame[:23]
            poses = self.model.fk(full)
            shell_heights = []
            for side in ('left', 'right'):
                link = side+'_hip_pitch_link'
                pose = poses[link]
                heights = self.model.vertices[link] @ (pose[:3, :3].T @ normal) + pose[:3, 3] @ normal
                shell_heights.append(heights.min())
            self.clearance[row] = min(shell_heights) - np.min(self.approach.vertices @ normal)
            if self.clearance[row] <= .001:
                self.enabled[row] = True
                self.target[row] = previous[row, self.approach.legs]

    def save(self, path):
        np.savez_compressed(path, active=np.flatnonzero(self.enabled), first_load_tick=self.first_load,
                            hip_clearance_m=self.clearance, held_leg_target=self.target)
