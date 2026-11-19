"""Choose the observation stream used by legacy sitting approaches at the standing scan."""
import numpy as np

# Capture range of the existing calibrated rim motions, in metres, degrees and slope.
from rim_motion import SCALE


def signature(rims, planes):
    rims, planes = np.asarray(rims), np.asarray(planes)
    valid = (planes[:,11] > 0) & (planes[:,12] == 0) & (planes[:,4] > .68)
    frontal = valid.sum() >= 4 and np.max(np.abs(planes[valid,3] / planes[valid,4])) < .008
    direct = valid & np.isfinite(rims).all(axis=1)
    if direct.sum() < 4:
        return frontal, None
    samples = np.column_stack((rims[direct], planes[direct,3] / planes[direct,4]))
    return frontal, np.median(samples,axis=0)


def select_view(frontal, feature, library):
    # The nominal-view approach library was validated on sagittal slabs. Its
    # missing-reading fallback is also part of that controller.
    if frontal:
        return True
    if feature is None:
        return False
    def nearest(points):
        return np.max(np.abs((points-feature)/SCALE),axis=1).min(initial=np.inf)
    compatible = nearest(library['nominal'])
    return compatible <= 1. and compatible + .15 < nearest(library['protect'])


class ApproachView:
    def __init__(self, count, library):
        self.library = library
        self.history = {}
        self.feedback = {}
        self.nominal = np.zeros(count,bool)

    def record(self, tick, downward, nominal, planes):
        self.history[tick] = (downward, nominal, planes.copy())

    def resolve(self, learner):
        ticks = (0,10,20,30,40)
        history = [self.history[t] for t in ticks]
        routed = list(history[-1][0])
        for i in range(len(routed)):
            values = []
            for downward, _, _ in history:
                m = downward[i]
                values.append([m['edge'],m['height'],m['pitch']] if m is not None and m['direct']
                              else [np.nan]*3)
            frontal, feature = signature(values, [p[i] for _,_,p in history])
            self.nominal[i] = select_view(frontal,feature,self.library)
            if self.nominal[i] and feature is not None and 'conflict' in self.library:
                close = np.max(abs((self.library['conflict']-feature)/SCALE),axis=1) <= 1.
                if close.any():
                    if self.library['use_feedback'].item():
                        observations = [m for m,(down,_,_) in zip(values,history,strict=True)
                                        if down[i] is not None and down[i]['direct']
                                        and down[i]['samples']>=4 and down[i]['span']<=.012]
                        if len(observations)>=4 and np.all(np.ptp(observations,axis=0)<=[.020,.010,1.]):
                            self.feedback[i] = np.median(observations,axis=0)
                        else:
                            self.nominal[i] = False
                    else:
                        self.nominal[i] = False
            if self.nominal[i]:
                # The first routing decision is at tick 40. Rebuild its entire
                # history in one view, with each original consumer's validity filter.
                prior = [frame[i] for _,frame,_ in history[:-1] if frame[i] is not None]
                def observation(m):
                    return [m['edge'],m['height'],m['pitch']]
                learner.features[i] = [observation(m) for m in prior] if not learner.protected[i] else []
                learner.camera_history[i] = [observation(m) for m in prior if m['samples']>=4 and m['span']<=.012]
                routed[i] = history[-1][1][i]
        return ticks[-1], routed
