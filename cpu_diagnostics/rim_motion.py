"""Select a touchdown motion from directly observed standing rim and support plane."""
import numpy as np
import torch


# Edge, support height, pitch and cross-slope. Independent capture tolerances.
SCALE = np.array([.007, .006, .9, .025])


def features(measurement, plane):
    return [measurement['edge'], measurement['height'], measurement['pitch'], plane[3]/plane[4]]


class RimMotion:
    def __init__(self, library, count, screen):
        self.library = library
        self.samples = [[] for _ in range(count)]
        self.selected = np.full(count,-1,dtype=int)
        self.screen = screen

    def apply(self, learner, obs, tick, nominal):
        env = learner.env
        if tick in (0,10,20,30,40):
            frame, measured = env.live
            assert frame == tick
            for i,m in enumerate(measured):
                if m is not None and m['direct']:
                    self.samples[i].append(features(m,env.geometry[i]))
        if tick == 40:
            for i,history in enumerate(self.samples):
                if len(history)<4 or not len(self.library['features']):
                    continue
                feature = np.median(history,axis=0)
                distances = np.max(abs(self.library['features']-feature)/SCALE,axis=1)
                index = i if self.screen else int(distances.argmin())
                if distances[index]<=1.:
                    self.selected[i] = index
            if self.screen:
                assert np.array_equal(self.selected,np.arange(len(obs)))
            self.rows = np.flatnonzero(self.selected>=0)
            self.paths = torch.as_tensor(self.library['paths'][self.selected[self.rows]],device=obs.device)
        if tick == 999:
            np.savez_compressed(env.out/'rim_motion.npz',selected=self.selected,donors=self.library['ids'])
        if tick<50 or not self.rows.size:
            return nominal
        # Recordings supply the approach. During the hold the live actor supplies its residual.
        blend = np.clip((tick-50)/100.,0.,1.)
        blend = blend**2*(3.-2.*blend)
        target = self.paths[:,tick]
        if tick>=750:
            target = target + learner.actor(learner.normalize(obs))[self.rows]
        out = nominal.clone()
        out[self.rows] = torch.lerp(nominal[self.rows],target,float(blend))
        return out
