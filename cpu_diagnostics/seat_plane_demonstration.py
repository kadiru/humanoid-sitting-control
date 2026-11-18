"""Transfer a clean approach only within the measured seat-plane capture range."""

import numpy as np

from prepare_sitting_repairs import smoothstep
from seat_plane_approach import ROLL_START


def plane_features(planes):
    # Height at the typical approach point 20 cm behind the standing pelvis,
    # plus the two slopes. Each comes directly from the measured camera plane.
    n = planes[:,2:5]
    dz = n[:,2]
    usable = (planes[:,11] > 0) & (planes[:,12] == 0) & (np.abs(dz) > .80)
    div = np.where(usable,dz,1.)
    return np.column_stack(((.20*n[:,0]-planes[:,5])/div,n[:,:2]/div[:,None])), usable


def match(donors,features,usable):
    diff = features[:,None] - donors[None]
    # Worst vertical disagreement over a 20 x 25 cm patch centred on the approach.
    err = np.abs(diff) @ np.array([1.,.100,.125])
    err[features[:,None,2]*donors[None,:,2] <= 0.] = np.inf
    index = err.argmin(axis=1)
    distance = err[np.arange(len(features)),index]
    change = diff[np.arange(len(features)),index]
    # A sub-millimetre match already has its established successful controller.
    enabled = usable & (distance > .0005) & (distance <= .010) & (np.abs(change[:,0]) <= .005)
    return np.where(enabled,index,-1), distance, change


class PlaneDemonstration:
    def __init__(self,approach,library,planes):
        self.approach = approach
        self.library = library
        features, valid = plane_features(planes)
        valid &= np.isin(np.arange(len(valid)),approach.active)
        self.index, self.distance, self.change = match(library['features'],features,valid)
        self.established = valid & (self.distance <= .0005)
        self.rows = np.flatnonzero(self.index >= 0)
        self.delta = np.zeros_like(approach.delta)
        self.rejected = np.zeros(len(valid),int)

    def targets(self,tick,frames,nominal):
        phase = smoothstep((tick-ROLL_START)/200.)
        donor = self.library['desired'][self.index[self.rows],tick]
        if tick % 5 == 0:
            for j,row in enumerate(self.rows):
                h,px,py = self.change[row]
                normal = self.approach.normal[row]
                fixed = self.approach.correct(donor[j],frames[row],normal,-py,pitch=px,normal_m=h*normal[2])
                if fixed is None:
                    self.rejected[row] += 1
                else:
                    self.delta[row] = fixed[0]
        return nominal + phase*(donor + self.delta[self.rows] - nominal)

    def save(self,path):
        np.savez_compressed(path,index=self.index,selected_ids=self.library['ids'][self.index[self.rows]],
                            rows=self.rows,plane_difference=self.change,worst_error_m=self.distance,rejected=self.rejected)
