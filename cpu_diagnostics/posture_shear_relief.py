"""Yield to the pelvis sensor's tangential force after the final sitting motion is selected."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation


class PostureShearRelief:
    def __init__(self, env, settings):
        self.settings = settings
        self.weight = (env.robot.data.body_mass.torch.sum(1) * 9.81).cpu().numpy()
        self.offset = np.zeros((env.num_envs, 3))
        self.filtered = np.zeros_like(self.offset)
        self.confirmation = settings.get('support_confirmation_ticks', 0)
        self.settled_steps = np.zeros(env.num_envs, dtype=int)
        self.enabled = np.zeros(env.num_envs, dtype=bool)
        self.mapping = [None] * env.num_envs
        self.trace = []

    def apply(self, learner, tick, nominal):
        from common import DT

        env = learner.env
        sensor = env.raw.scene['pelvis_seat'].data
        normal_force = sensor.normal_force_matrix_w.torch.sum((1, 2)).cpu().numpy()
        friction = sensor.friction_force_matrix_w.torch.sum((1, 2)).cpu().numpy()
        load = np.linalg.norm(normal_force, axis=1)
        normal = normal_force / np.maximum(load[:, None], 1e-8)
        shear = friction - normal * (friction * normal).sum(1, keepdims=True)
        loaded = load >= self.settings['min_load_bw'] * self.weight
        if self.confirmation:
            # Leave impact and load transfer to the existing controller. A short
            # touch that unloads resets the confirmation clock for that robot.
            supported = env.supported_lap.started.cpu().numpy() & (load >= self.settings['support_load_bw'] * self.weight)
            self.settled_steps = np.where(supported, self.settled_steps + 1, 0)
            self.enabled |= self.settled_steps >= self.confirmation
            loaded &= self.enabled
        ratio = shear / np.maximum(load, self.settings['min_load_bw'] * self.weight)[:, None]
        self.filtered += self.settings['filter_fraction'] * (ratio - self.filtered)
        self.filtered[~loaded] = 0.
        magnitude = np.linalg.norm(self.filtered, axis=1)
        velocity = self.settings['admittance_m_s'] * np.maximum(0., magnitude - self.settings['deadband_ratio'])
        direction = self.filtered / np.maximum(magnitude[:, None], 1e-8)
        step = DT * np.minimum(velocity, self.settings['max_speed_m_s'])[:, None] * direction

        # Move within the observed contact plane. After release return gently
        # to the original leg target; the motor's own limiter still runs last.
        target = self.offset + step
        target[loaded] -= normal[loaded] * (target[loaded] * normal[loaded]).sum(1, keepdims=True)
        target[~loaded] = 0.
        target /= np.maximum(1., np.linalg.norm(target, axis=1, keepdims=True) / self.settings['max_offset_m'])
        step = target - self.offset
        step /= np.maximum(1., np.linalg.norm(step, axis=1, keepdims=True) / (DT * self.settings['max_speed_m_s']))
        self.offset += step

        kin = learner.contact_feedback
        active = np.flatnonzero(np.linalg.norm(self.offset, axis=1) > 1e-8)
        result = nominal
        if len(active):
            q = env.previous[:, env.ids].cpu().numpy()
            quat = env.robot.data.root_quat_w.torch.cpu().numpy()
            local = Rotation.from_quat(quat[active]).inv().apply(self.offset[active])
            delta = []
            for j, i in enumerate(active):
                if self.mapping[i] is None or tick % 8 == 0:
                    self.mapping[i] = kin.foot_translation(q[i], kin.feet(q[i]))
                delta.append(self.mapping[i] @ local[j])
            delta = np.asarray(delta)
            delta /= np.maximum(1., abs(delta).max(axis=1, keepdims=True) / self.settings['max_correction_rad'])
            result = nominal.clone()
            rows = torch.as_tensor(active, device=nominal.device)
            legs = torch.as_tensor(kin.legs, device=nominal.device)
            result[rows[:, None], legs] += torch.as_tensor(delta, device=nominal.device, dtype=nominal.dtype)
        self.trace.append(np.concatenate((normal_force, friction, self.offset),axis=1).astype(np.float32))
        if tick == 999:
            np.savez_compressed(env.out / 'posture_shear.npz', first_tick=tick+1-len(self.trace),
                                readings=np.stack(self.trace), fields=['normal_world_N','friction_world_N','offset_world_m'])
        return result
