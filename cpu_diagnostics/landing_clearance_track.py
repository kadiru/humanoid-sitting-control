"""Retarget a demonstrated descent to the observed seat with both feet planted."""
import numpy as np


class LandingClearanceTrack:
    def __init__(self, motion, learner, routes, reference):
        from chair_contact_motion import descriptor

        env = learner.env
        rims, planes, tiles = map(np.asarray, zip(*motion.measurements, strict=True))
        self.first = reference['ticks'][0]
        self.reference = np.column_stack([
            np.interp(np.arange(self.first, 1000), reference['ticks'], reference[key])
            for key in ('gap_m', 'seat_load_bw')
        ])
        self.rows = []
        self.forward = np.zeros(env.num_envs)
        for row in learner.velocity_brake.active:
            selected = motion.selected[row]
            observed = descriptor(rims[:, row], planes[:, row], tiles[:, row])
            rule = routes.match(observed)
            if selected >= 0 and rule is not None and motion.library['ids'][selected] == rule['to_motion']:
                self.rows.append(row)
                self.forward[row] = rule.get('forward_shift_m', 0.)
        self.initial_error = np.zeros(env.num_envs)
        self.integral = np.zeros(env.num_envs)
        self.depth = np.zeros(env.num_envs)
        self.mapping = {}
        self.history = []

    def apply(self, learner, obs, tick, nominal):
        from common import DT
        from seat_plane_approach import SENSOR_FRAME, stance_axes

        env = learner.env
        kin = learner.contact_feedback
        target_gap, target_load = self.reference[tick - self.first]
        readings = env.touchdown_tangent.transfer_history[-1]
        phase = min(1., (tick - self.first) / 140.)
        ease = phase * phase * (3. - 2. * phase)
        result = nominal
        samples = np.zeros((len(readings), 4), dtype=np.float32)
        for row in self.rows:
            gap, load = readings[row, :2]
            if tick == self.first:
                self.initial_error[row] = np.clip(gap - target_gap, -.016, .016)
            if gap < .050:
                # Below 1.5% body weight the clearance error is more informative.
                # Once both loads are established, follow the force reference.
                if load < .015 or target_load < .015:
                    rate = 1.8 * (gap - target_gap)
                else:
                    rate = .060 * (target_load - load)
                step = DT * np.clip(rate, -.008, .008)
                self.integral[row] = np.clip(self.integral[row] + step, -.010, .010)
            desired = np.clip(self.initial_error[row] * ease + self.integral[row], -.022, .022)
            self.depth[row] += np.clip(desired - self.depth[row], -DT * .015, DT * .015)
            if self.depth[row] or self.forward[row]:
                frame = obs[row, SENSOR_FRAME].cpu().numpy()
                joints = frame[:23].astype(float)
                down = frame[69:72].astype(float)
                down /= np.linalg.norm(down)
                if row not in self.mapping or tick % 5 == 0:
                    self.mapping[row] = kin.foot_translation(joints, kin.feet(joints))
                displacement = down * self.depth[row]
                if self.forward[row]:
                    feet = kin.feet(joints)
                    axes = stance_axes(feet, down)
                    normal = axes @ learner.seat_plane.normal[row]
                    tangent = axes[:, 0] - normal * np.dot(axes[:, 0], normal)
                    tangent /= np.linalg.norm(tangent)
                    displacement += tangent * (ease * self.forward[row])
                delta = self.mapping[row] @ displacement
                delta /= max(1., np.abs(delta).max() / .10)
                if result is nominal:
                    result = nominal.clone()
                result[row, kin.legs] += nominal.new_tensor(delta)
            samples[row] = gap - target_gap, load - target_load, self.initial_error[row], self.depth[row]
        self.history.append(samples)
        if tick == 999:
            np.savez_compressed(env.out / 'landing_clearance_track.npz', rows=self.rows, first_tick=self.first,
                                samples=np.stack(self.history),
                                fields=['gap_error_m','load_error_bw','start_gap_difference_m','down_m'])
        return result
