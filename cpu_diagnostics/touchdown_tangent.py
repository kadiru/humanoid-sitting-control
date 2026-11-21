"""Damp the backward component of pelvis velocity just before seat loading."""
import numpy as np
import torch


class TouchdownTangent:
    def __init__(self, count):
        self.position = np.full(count, np.nan)
        self.speed = np.zeros(count)
        self.shift = np.zeros(count)
        self.touched = np.zeros(count, dtype=bool)
        self.mapping = [None] * count
        self.history = []

    def apply(self, learner, obs, tick, target):
        from common import DT
        from seat_plane_approach import SENSOR_FRAME, stance_axes

        env = learner.env
        brake = learner.velocity_brake
        frames = obs[:, SENSOR_FRAME].cpu().numpy()
        sensed_load = env.supported_lap.previous_load.cpu().numpy()
        arm_released = env.supported_lap.started.cpu().numpy()
        selected = env.arm_active.cpu().numpy()
        joint_ids = learner.contact_feedback.legs
        kin = learner.contact_feedback
        result = target
        samples = np.full((len(frames), 6), np.nan, dtype=np.float32)
        for index, row in enumerate(brake.active):
            if not selected[row]:
                continue
            frame = frames[row]
            position, gap = brake.measure(row, frame, brake.offset[index])
            previous = self.position[row]
            self.position[row] = position
            if np.isfinite(previous):
                measured_speed = (previous - position) / DT
                self.speed[row] += 0.4 * (measured_speed - self.speed[row])
            self.touched[row] |= sensed_load[row] >= 0.05
            if arm_released[row]:
                self.shift[row] = max(0., self.shift[row] - DT * 0.004)
            elif not self.touched[row]:
                # Ramp braking up over the last 20 mm. Continue the original
                # descent along the support normal and retain both foot poses.
                gain = np.clip((0.020 - gap) / 0.010, 0., 1.)
                rate = np.clip(self.speed[row] - 0.006, 0., 0.025)
                self.shift[row] = min(0.006, self.shift[row] + DT * gain * rate)
            max_delta = 0.
            normal_component = 0.
            if self.shift[row] > 0.:
                q = frame[:23].astype(float)
                feet = kin.feet(q)
                axes = stance_axes(feet, frame[69:72])
                normal = axes @ brake.approach.normal[row]
                tangent = axes[:, 0] - normal * np.dot(axes[:, 0], normal)
                tangent /= np.linalg.norm(tangent)
                shift = self.shift[row] * tangent
                normal_component = np.dot(normal, shift)
                if self.mapping[row] is None or tick % 5 == 0:
                    self.mapping[row] = kin.foot_translation(q, feet)
                delta = self.mapping[row] @ shift
                delta /= max(1., np.abs(delta).max() / 0.05)
                if result is target:
                    result = target.clone()
                result[row, joint_ids] += torch.as_tensor(delta, device=target.device, dtype=target.dtype)
                max_delta = np.abs(delta).max()
            samples[row] = gap, self.speed[row], self.shift[row], sensed_load[row], max_delta, normal_component
        self.history.append(samples)
        if tick == 999:
            np.savez_compressed(env.out/'touchdown_tangent.npz', first_tick=tick+1-len(self.history),
                                samples=np.stack(self.history),
                                fields=['gap_m','backward_m_s','tangent_shift_m','load_bw','max_delta_rad','normal_shift_m'])
        return result
