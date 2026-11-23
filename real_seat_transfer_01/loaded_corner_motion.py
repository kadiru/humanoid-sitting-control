"""Use the measured seat-loading direction only after both feet and the seat bear load."""
import numpy as np
from supported_corner_transfer import StanceTransfer as Transfer
from stance23_control import P

class StanceTransfer(Transfer):
    def __init__(self, picked, env, ref):
        super().__init__(picked, env, ref)
        with np.load(P/'loaded_corner_fit.npz') as fit:
            assert self.seat_ids == fit['joints'].tolist()
            self.load_direction = fit['slope'].copy()
        self.trim = {}
        self.load_history = []

    def apply(self, env, t, before, seat_load):
        out = super().apply(env, t, before, seat_load)
        history = np.zeros((len(out), 4))
        for j, (index, _, _) in self.cases.items():
            if index != 30 or t < 572:
                continue
            load = float(seat_load[j])
            feet = self.corridor_log[-1][j, 3:5]
            correction = self.trim.get(j, 0.)
            if np.min(feet) > .06 and load >= .04:
                # Bound the excursion to the span of the measured loading motion.
                error = .105 - load
                if abs(error) > .003:
                    correction = np.clip(correction + .045*error, 0., .052)
            elif correction:
                correction = max(0., correction-.0015)
            self.trim[j] = correction
            out[j,self.seat_ids] += out.new_tensor(self.load_direction*correction)
            history[j] = (load, correction, *feet)
        self.load_history.append(history)
        if t == 999:
            np.savez_compressed(env.out/'corner_load_control.npz', data=np.stack(self.load_history))
        return out
