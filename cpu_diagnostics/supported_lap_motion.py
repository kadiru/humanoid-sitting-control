"""Move seated arms after the pelvis contact sensor records sustained load."""
import numpy as np
import torch


class SupportedLapMotion:
    def __init__(self, env, settings):
        self.settings = settings
        self.weight = env.robot.data.body_mass.torch.sum(1) * 9.81
        self.previous_load = torch.zeros_like(self.weight)
        self.steady_steps = torch.zeros_like(self.weight, dtype=torch.long)
        self.started = torch.zeros_like(self.weight, dtype=torch.bool)
        self.load_history = []
        self.advance_history = []

    def apply(self, env, tick, nominal):
        from common import FILTER, STEP

        sensor = env.raw.scene['pelvis_seat'].data
        force = sensor.normal_force_matrix_w.torch.sum((1, 2))
        load = torch.linalg.vector_norm(force, dim=1) / self.weight
        supported = ((load >= self.settings['minimum_load_bw']) &
                     ((load - self.previous_load).abs() <= self.settings['maximum_load_change_bw']))
        self.previous_load = load
        self.steady_steps = torch.where(supported, self.steady_steps + 1, 0)
        advance = env.arm_active & (self.steady_steps >= self.settings['confirmation_ticks'])
        self.started |= advance

        previous = env.previous[:, env.ids][:, env.arm_indices]
        # Ramp up after first support or reacquisition. Do not catch up after a pause.
        ramp = ((self.steady_steps - self.settings['confirmation_ticks'] + 1) /
                self.settings['ramp_ticks']).clamp(0., 1.)
        cap = (STEP * ramp * advance)[:, None]
        delta = torch.clamp(env.arm_goal - previous, -cap, cap)
        requested = previous + delta / FILTER
        result = nominal.clone()
        result[:, env.arm_indices] = torch.where(self.started[:, None], requested, nominal[:, env.arm_indices])
        self.load_history.append(load.detach())
        self.advance_history.append(advance.detach())
        if tick == 999:
            np.savez_compressed(env.out / 'arm_support.npz', first_tick=tick+1-len(self.load_history),
                                load_bw=torch.stack(self.load_history).cpu().numpy(),
                                advancing=torch.stack(self.advance_history).cpu().numpy())
        return result
