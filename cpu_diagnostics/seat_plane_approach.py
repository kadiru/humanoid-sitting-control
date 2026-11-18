"""Roll the pelvis toward an observed cross-slope while preserving both foot poses."""

import numpy as np
from scipy.spatial.transform import Rotation

from prepare_sitting_repairs import leg_fk, smoothstep


SENSOR_FRAME = slice(222, 333)
MEASURE_UNTIL = 40
ROLL_START = 220
ROLL_FULL = 510
UPDATE_INTERVAL = 10
ROTATION_WEIGHT = .18
JOINT_MARGIN = .03


def stance_axes(feet, gravity):
    up = -np.asarray(gravity, dtype=float)
    up /= np.linalg.norm(up)
    forward = np.cross(feet[0, :3, 3] - feet[1, :3, 3], up)
    forward /= np.linalg.norm(forward)
    return np.column_stack((forward, np.cross(up, forward), up))


def seat_normal(feet, gravity, plane):
    if not plane[11] or plane[12] != 0:
        return None
    normal = stance_axes(feet, gravity).T @ plane[2:5]
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    return normal


def select_cross_slope(samples):
    if len(samples) < 4:
        return 0.
    pitch = np.rad2deg(np.arctan2(np.linalg.norm(samples[:, :2], axis=1), samples[:, 2]))
    yaw = np.rad2deg(np.arctan2(samples[:, 1], samples[:, 0]))
    if (np.min(pitch) < 12 or np.max(pitch) > 20 or np.ptp(pitch) > 1
            or np.min(np.abs(yaw)) < 5 or np.max(np.abs(yaw)) > 20 or np.ptp(yaw) > 2):
        return 0.
    normal = np.median(samples, axis=0)
    return float(np.arctan2(-normal[1], normal[2]))


def foot_error(desired_positions, desired_rotations, feet):
    rotation_error = desired_rotations @ feet[:, :3, :3].transpose(0, 2, 1)
    return np.column_stack(
        (desired_positions - feet[:, :3, 3], ROTATION_WEIGHT * Rotation.from_matrix(rotation_error).as_rotvec())
    ).ravel()


class SeatPlaneApproach:
    def __init__(self, model, metadata, count):
        self.feet, self.legs = leg_fk(model, metadata['joint_names'])
        self.vertices = model.vertices['pelvis']
        self.lower = np.asarray(metadata['lower'])[self.legs] + JOINT_MARGIN
        self.upper = np.asarray(metadata['upper'])[self.legs] - JOINT_MARGIN
        self.normals = [[] for _ in range(count)]
        self.offsets = [[] for _ in range(count)]
        self.normal = np.zeros((count, 3))
        self.roll = np.zeros(count)
        self.active = np.empty(0, dtype=int)
        self.delta = np.zeros((count, 23), dtype=np.float32)
        self.last_frame = -1
        self.attempts = np.zeros(count, dtype=int)
        self.accepted = np.zeros(count, dtype=int)
        self.max_foot_error = np.zeros((count, 2))

    def observe(self, tick, planes, frames):
        if tick != self.last_frame and tick <= MEASURE_UNTIL:
            self.last_frame = tick
            for row, (plane, frame) in enumerate(zip(planes, frames, strict=True)):
                feet = self.feet(frame[:23])
                normal = seat_normal(feet, frame[69:72], plane)
                if normal is not None and np.all(np.isfinite(normal)):
                    self.normals[row].append(normal)
                    sign = 1. if plane[2:5] @ frame[69:72] < 0 else -1.
                    self.offsets[row].append(sign * (plane[5] + feet[:, :3, 3].mean(0) @ plane[2:5]))

    def freeze(self):
        for row, samples in enumerate(self.normals):
            self.roll[row] = select_cross_slope(np.asarray(samples))
            if self.roll[row]:
                normal = np.median(samples, axis=0)
                self.normal[row] = normal / np.linalg.norm(normal)
        self.active = np.flatnonzero(self.roll)

    def correct(self, desired, frame, normal, roll, forward_m=0., pitch=0., lateral_m=0., normal_m=0.):
        q = desired.astype(float, copy=True)
        q[self.legs] = np.clip(q[self.legs], self.lower, self.upper)
        feet = self.feet(q)
        # Express gravity and the latched plane in the reference pelvis frame.
        # Both reference and measured foot orientations come from encoders/FK.
        measured_feet = self.feet(frame[:23])
        measured_axes = stance_axes(measured_feet, frame[69:72])
        relative = feet[0, :3, :3] @ measured_feet[0, :3, :3].T
        axes = relative @ measured_axes
        normal = axes @ normal
        rotate = Rotation.from_rotvec(roll * axes[:, 0] + pitch * axes[:, 1]).as_matrix()
        shift = normal * (np.min(self.vertices @ normal) - np.min(self.vertices @ rotate.T @ normal))
        if normal_m:
            shift += normal_m * normal
        if forward_m or lateral_m:
            tangent = axes[:, 0] - normal * (axes[:, 0] @ normal)
            shift += forward_m * tangent / np.linalg.norm(tangent)
            if lateral_m:
                shift += lateral_m * np.cross(normal, tangent / np.linalg.norm(tangent))
        positions = (feet[:, :3, 3] - shift) @ rotate
        rotations = rotate.T @ feet[:, :3, :3]

        updated = feet
        for _ in range(2):
            error = foot_error(positions, rotations, updated)
            jacobian = np.empty((12, len(self.legs)))
            for col, joint in enumerate(self.legs):
                perturbed = q.copy()
                perturbed[joint] += .0005
                jacobian[:, col] = (error - foot_error(positions, rotations, self.feet(perturbed))) / .0005
            step = np.linalg.solve(jacobian.T @ jacobian + 1e-7 * np.eye(len(self.legs)), jacobian.T @ error)
            q[self.legs] = np.clip(q[self.legs] + step, self.lower, self.upper)
            updated = self.feet(q)
        error = foot_error(positions, rotations, updated).reshape(2, 6)
        position_error = np.linalg.norm(error[:, :3], axis=1).max()
        rotation_error = np.linalg.norm(error[:, 3:], axis=1).max() / ROTATION_WEIGHT
        delta = q - desired
        if (not np.all(np.isfinite(delta)) or position_error > .002 or rotation_error > .01
                or np.max(np.abs(delta)) > .12):
            return None
        return delta, (position_error, rotation_error)

    def update(self, tick, desired, frames):
        fraction = smoothstep((tick - ROLL_START) / (ROLL_FULL - ROLL_START))
        for i, row in enumerate(self.active):
            self.attempts[row] += 1
            result = self.correct(desired[i], frames[i], self.normal[row], fraction * self.roll[row])
            if result is not None:
                self.delta[row], error = result
                self.max_foot_error[row] = np.maximum(self.max_foot_error[row], error)
                self.accepted[row] += 1

    def save(self, path):
        np.savez_compressed(path, active=self.active, roll_rad=self.roll,
                            attempts=self.attempts, accepted=self.accepted,
                            samples_per_case=[len(samples) for samples in self.normals],
                            max_foot_error=self.max_foot_error)
