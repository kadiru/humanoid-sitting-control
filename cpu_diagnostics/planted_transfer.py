"""Control sole pitch before loading and lower the pelvis into light seat contact."""
import numpy as np
import torch

from touchdown_tangent import TouchdownTangent


class PlantedTransfer(TouchdownTangent):
    def __init__(self, count):
        super().__init__(count)
        self.reference_pitch = np.full((count, 2), np.nan)
        self.ankle_bias = np.zeros((count, 2))
        self.depth = np.zeros(count)
        self.last_gap = np.full(count, np.nan)
        self.closing = np.zeros(count)
        self.transfer_history = []

    def apply(self, learner, obs, tick, target):
        from common import DT
        from seat_plane_approach import SENSOR_FRAME, stance_axes

        nominal = super().apply(learner, obs, tick, target)
        env = learner.env
        kin = learner.contact_feedback
        if not hasattr(self, 'ankles'):
            names = env.metadata['joint_names']
            self.ankles = [names.index(side+'_ankle_pitch_joint') for side in ('left','right')]
        foot_load = []
        for sensor_name in ('rest_left_floor','rest_right_floor'):
            forces = env.raw.scene[sensor_name].data.normal_force_matrix_w.torch.sum((1, 2))
            foot_load.append(torch.linalg.vector_norm(forces, dim=1) / env.supported_lap.weight)
        foot_load = torch.stack(foot_load,dim=1).cpu().numpy()
        frames = obs[:, SENSOR_FRAME].cpu().numpy()
        result = nominal
        samples = np.zeros((len(frames), 10))
        for row in learner.velocity_brake.active:
            gap, _, _, load, _, _ = self.history[-1][row]
            if not np.isfinite(gap):
                continue
            q = frames[row,:23].astype(float)
            gravity = frames[row,69:72].astype(float)
            gravity /= np.linalg.norm(gravity)
            feet = kin.feet(q)
            pitch = np.arcsin(np.clip(feet[:,:3,0] @ gravity, -1.,1.))
            if np.isnan(self.last_gap[row]):
                self.reference_pitch[row] = pitch
            else:
                closing = (self.last_gap[row]-gap)/DT
                self.closing[row] += .25*(closing-self.closing[row])
            self.last_gap[row] = gap
            near = np.clip((.040-gap)/.015,0.,1.)
            both_loaded = foot_load[row].min() >= .15
            pitch_error = self.reference_pitch[row]-pitch
            # Both ankle-pitch axes increase sole pitch in this model.
            excess = np.sign(pitch_error)*np.maximum(0.,abs(pitch_error)-.0015)
            goal = near*np.clip(1.2*excess,-.035,.035)*both_loaded*np.clip((.12-load)/.07,0.,1.)
            self.ankle_bias[row] += np.clip(goal-self.ankle_bias[row],-.08*DT,.08*DT)
            # Approach slowly along the camera plane normal when the gap is
            # small and closing slowly, or when the pelvis feels initial load.
            # Stop pressing when the seat supports 8% BW; unwind above 12%.
            eligible = both_loaded and gap < .008 and (load >= .008 or abs(self.closing[row]) < .018)
            if eligible and load < .08:
                self.depth[row] = min(.009,self.depth[row]+DT*.008*np.clip((.08-load)/.06,0.,1.))
            elif load >= .12 or gap > .012 or not both_loaded:
                self.depth[row] = max(0., self.depth[row]-.003*DT)
            leg_delta = np.zeros(len(kin.legs))
            if self.depth[row]:
                axes = stance_axes(feet, gravity)
                normal = axes @ learner.seat_plane.normal[row]
                if self.mapping[row] is None or tick%5==0:
                    self.mapping[row] = kin.foot_translation(q, feet)
                leg_delta = self.mapping[row] @ (-self.depth[row]*normal)
                leg_delta /= max(1.,max(abs(leg_delta))/.06)
            if np.any(leg_delta) or np.any(self.ankle_bias[row]):
                if result is nominal:
                    result = nominal.clone()
                result[row,kin.legs] += target.new_tensor(leg_delta)
                result[row,self.ankles] += target.new_tensor(self.ankle_bias[row])
            samples[row] = [gap,load,*foot_load[row],*pitch,*self.ankle_bias[row],self.depth[row],self.closing[row]]
        self.transfer_history.append(samples.astype(np.float32))
        if tick==999:
            np.savez_compressed(env.out/'planted_transfer.npz', first_tick=tick+1-len(self.transfer_history),
                samples=np.stack(self.transfer_history),fields=['gap_m','seat_bw','left_foot_bw','right_foot_bw',
                'left_pitch_rad','right_pitch_rad','left_ankle_bias_rad','right_ankle_bias_rad','seatward_offset_m','closing_m_s'])
        return result
