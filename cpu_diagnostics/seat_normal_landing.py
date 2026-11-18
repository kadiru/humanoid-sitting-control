"""Catch a fast incoming pelvis before load, then lower along the observed normal."""

import numpy as np

from seat_plane_approach import stance_axes


class SeatNormalLanding:
    def __init__(self, approach, model, metadata):
        self.approach = approach
        self.model = model
        self.joint_ids = metadata['joint_ids']
        n = len(approach.roll)
        self.start = np.full(n, -1)
        self.anchor = np.zeros_like(approach.delta)
        self.target = self.anchor.copy()
        self.travel = np.zeros(n)
        self.limit = np.zeros(n)
        self.pitch = np.zeros(n)
        self.pitch_goal = np.zeros(n)
        self.gap = np.zeros(n)
        self.rejected = np.zeros(n, dtype=int)
        self.trace = []

    @staticmethod
    def incoming(samples, before_load):
        tick, row, gap, rearward = samples[:, :4].T
        rows = row.astype(int)
        # At least 5 mm of headroom gives the bounded actuator time to respond.
        return rows[before_load[rows] & (gap > .005) & (gap <= .025) & (rearward > .020)]

    def housing_margin(self, row, joints, frame):
        feet = self.approach.feet(joints)
        measured = self.approach.feet(frame[:23])
        axes = feet[0,:3,:3] @ measured[0,:3,:3].T @ stance_axes(measured, frame[69:72])
        normal = axes @ self.approach.normal[row]
        full = np.zeros(len(self.model.names))
        full[self.joint_ids] = joints
        poses = self.model.fk(full)
        shell = []
        for side in ('left','right'):
            name = side + '_hip_pitch_link'
            pose = poses[name]
            shell.append(np.min(self.model.vertices[name] @ (pose[:3,:3].T @ normal)) + pose[:3,3] @ normal)
        return min(shell) - np.min(self.approach.vertices @ normal)

    def clearance_pitch(self, row, joints, frame):
        margin = self.housing_margin(row, joints, frame)
        if margin >= .003:
            return 0.
        # A finite-difference pitch derivative with both feet fixed.
        probe = self.approach.correct(joints,frame,self.approach.normal[row],0.,pitch=.015)
        if probe is None:
            return 0.
        slope = (self.housing_margin(row,joints+probe[0],frame) - margin) / .015
        if abs(slope) < .005:
            return 0.
        goal = float(np.clip((.003-margin)/slope,-.055,.055))
        check = self.approach.correct(joints,frame,self.approach.normal[row],0.,pitch=goal)
        if check is None or self.housing_margin(row,joints+check[0],frame) <= margin:
            return 0.
        return goal

    def update(self,tick,brake,load,never_loaded,frames,previous):
        if tick % 5 == 0 and brake.active.size:
            samples = np.asarray(brake.trace[-len(brake.active):])
            if len(samples) and np.all(samples[:,0] == tick):
                self.gap[samples[:,1].astype(int)] = samples[:,2]
                ready = (self.start < 0) & never_loaded & (load <= .01)
                for row in self.incoming(samples,ready):
                    self.start[row] = tick
                    self.target[row] = self.anchor[row] = previous[row]
                    self.limit[row] = self.gap[row] + .012
                    self.pitch_goal[row] = self.clearance_pitch(row,previous[row],frames[row])
        if tick % 2:
            return
        for row in np.flatnonzero(self.start >= 0):
            # After first touch keep loading gradually towards the existing 12% BW target.
            step = min(.04*np.clip(.08*(.12-load[row]),0.,.009),self.limit[row]-self.travel[row])
            pitch_step = np.clip(self.pitch_goal[row]-self.pitch[row],-.002,.002)
            if step <= 0. and not pitch_step:
                continue
            result = self.approach.correct(self.target[row],frames[row],self.approach.normal[row],
                                           0.,pitch=pitch_step,normal_m=-step)
            if result is None:
                self.rejected[row] += 1
                continue
            self.target[row] += result[0]
            self.travel[row] += step
            self.pitch[row] += pitch_step
            self.trace.append((tick,row,load[row],self.travel[row],self.pitch[row]))

    def save(self,path):
        np.savez_compressed(path,start=self.start,travel_m=self.travel,limit_m=self.limit,
                            pitch_rad=self.pitch,pitch_goal_rad=self.pitch_goal,rejected=self.rejected,
                            trace=np.asarray(self.trace),columns=['tick','row','load_bw','travel_m','pitch_rad'])
