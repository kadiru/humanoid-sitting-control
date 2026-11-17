"""Keep both pitch housings above the observed support while the pelvis takes load."""

import numpy as np

from humanoid_sitting.supported_surface import observed_support
from prepare_sitting_repairs import smoothstep
from seat_plane_approach import ROLL_START, ROLL_FULL, SeatPlaneApproach, stance_axes


def seat_top_plane(model, depth, intrinsics, transform, joints, gravity, tracked):
    up = -gravity / np.linalg.norm(gravity)
    if tracked[11] and abs(tracked[2:5] @ up) > .9:
        return tracked
    fk = model.fk(joints)
    ankles = [fk[s+'_ankle_roll_link'][:3, 3] for s in ('left', 'right')]
    v, u = np.indices(depth.shape)
    rays = np.stack(((u-intrinsics[0, 2])/intrinsics[0, 0],
                     (v-intrinsics[1, 2])/intrinsics[1, 1], np.ones_like(u)), axis=-1)
    z = np.where(np.isfinite(depth), depth, 0.)
    points = (rays*z[:, :, None]) @ transform[:3, :3].T + transform[:3, 3]
    heights = (points - np.mean(ankles, axis=0)) @ up
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    normals = np.cross(dx, dy)
    top = np.zeros_like(z, dtype=bool)
    top[1:-1, 1:-1] = np.abs(normals @ up) > .90*np.linalg.norm(normals, axis=-1)
    clipped = np.where(top & (heights > .25) & (heights < .52), z, 0.)
    plane, _ = observed_support(model, intrinsics, transform, clipped, joints, gravity)
    return plane if plane is not None and abs(plane[2:5] @ up) > .90 else tracked


class HousingClearance(SeatPlaneApproach):
    def __init__(self, model, metadata, count):
        super().__init__(model, metadata, count)
        self.model = model
        self.joint_ids = metadata['joint_ids']
        self.shoulders = [metadata['joint_names'].index(s+'_shoulder_pitch_joint') for s in ('left', 'right')]
        self.trim = np.zeros(count)
        self.trace = []

    def hip_trim(self, row, frame):
        feet = self.feet(frame[:23])
        axes = stance_axes(feet, frame[69:72])
        normal = axes @ self.normal[row]
        shell = np.min(self.vertices @ normal)
        gap = shell - feet[:, :3, 3].mean(0) @ normal + np.median(self.offsets[row])
        full = np.zeros(len(self.model.names))
        full[self.joint_ids] = frame[:23]
        poses = self.model.fk(full)
        hull_points = []
        for side in ('left', 'right'):
            name = side+'_hip_pitch_link'
            pose = poses[name]
            points = self.model.vertices[name] @ pose[:3, :3].T + pose[:3, 3]
            heights = points @ normal
            hull_points.append(np.average(points, axis=0, weights=np.exp((heights.min()-heights)/.001)))
        difference = hull_points[0] - hull_points[1]
        lever = np.cross(axes[:, 0], difference) @ normal
        error = difference @ normal
        gain = .45 * smoothstep((.085-gap)/.045)
        if lever > .12:
            self.trim[row] = np.clip(self.trim[row] - np.clip(gain*error/lever, -.010, .010), -.065, .065)
        return gap, error

    def update(self, tick, desired, frames):
        tilt = smoothstep((tick-ROLL_START)/(ROLL_FULL-ROLL_START))
        prepare = smoothstep((tick-ROLL_START)/200.)
        for i, row in enumerate(self.active):
            gap, error = self.hip_trim(row, frames[i])
            self.attempts[row] += 1
            result = self.correct(desired[i], frames[i], self.normal[row], tilt*self.roll[row]+self.trim[row],
                                  forward_m=-.009*prepare, pitch=-.055*prepare)
            if result is not None:
                self.delta[row], residual = result
                self.accepted[row] += 1
                self.max_foot_error[row] = np.maximum(self.max_foot_error[row], residual)
            self.delta[row, self.shoulders] = -.12*prepare
            self.trace.append((tick, row, gap, error, self.trim[row], result is not None))

    def save(self, path):
        super().save(path)
        np.savez_compressed(path.with_name('housing_clearance.npz'), trace=np.asarray(self.trace),
                            columns=['tick', 'row', 'shell_gap_m', 'hip_height_difference_m', 'roll_trim_rad', 'accepted'])
