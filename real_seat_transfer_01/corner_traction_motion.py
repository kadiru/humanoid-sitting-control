"""Unload fore-aft seat friction while retaining the measured normal-load transfer."""
import numpy as np
from loaded_corner_motion import StanceTransfer as Parent
from stance23_control import P

class StanceTransfer(Parent):
    def __init__(self, picked, env, ref):
        super().__init__(picked, env, ref)
        with np.load(P/'corner_traction_fit.npz') as fit:
            self.fore = fit['tangent']
            self.normal = fit['normal']
            self.joints = fit['ids'].tolist()
            self.gain = fit['gain']
            self.compliance = float(fit['compliance'])
        self.shear_offset = {}
        self.shear_trace = []

    def apply(self, env, tick, command, seat_load):
        out = super().apply(env, tick, command, seat_load)
        trace = np.zeros((len(out), 4))
        for j, (index, _, _) in self.cases.items():
            if index != 30 or seat_load[j] <= .04 or tick < 640:
                continue
            sensor = env.raw.scene['pelvis_seat'].data
            force = (sensor.normal_force_matrix_w.torch[j] + sensor.friction_force_matrix_w.torch[j]).sum((0,1)).detach().cpu().numpy()
            normal = max(0., float(force@self.normal))
            along = float(force@self.fore)
            offset = self.shear_offset.get(j, 0.)
            excess = min(0., along + .70 * .60 * normal)
            if normal > 0. and np.min(self.corridor_log[-1][j,3:5]) > .06:
                offset = max(-.002, offset + max(-.00005, .12 * self.compliance * excess))
            self.shear_offset[j] = offset
            out[j,self.joints] += out.new_tensor(self.gain*offset)
            trace[j] = [along, normal, excess, offset]
        self.shear_trace.append(trace)
        if tick == 999:
            np.savez_compressed(env.out/'corner_traction.npz', data=np.stack(self.shear_trace))
        return out

