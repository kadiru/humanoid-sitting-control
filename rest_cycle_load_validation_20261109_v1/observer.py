"""Full spatial leg-wrench observer using encoders, IMU and motor-axis effort.

The six joint equations retain lateral force and all three foot moments.
They assume no contact on the leg above the foot and quasi-static dynamics.
Neither assumption is silently certified by solving a square system. This
module is measurement-only until an independent recorded-data audit admits it.
"""
import numpy as np
from humanoid_sitting.shared_frame import apply, level_rotation


class SpatialLoadObserver:
    def __init__(self, model, inertia):
        self.model = model
        self.mass = dict(zip(inertia['body_names'], inertia['mass']))
        self.com = dict(zip(inertia['body_names'], inertia['local_com']))
        self.weight = sum(self.mass.values()) * 9.81
        self.joints = {j['name']: j for j in model.joints}
        self.sides = []
        self.desc = {}
        for side in ('left', 'right'):
            names = [side+'_'+s+'_joint' for s in
                     ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
            self.sides.append(names)
            for name in names:
                bodies = {self.joints[name]['body1'][0]}
                while True:
                    children = {j['body1'][0] for j in model.joints
                                if j['body0'] and j['body0'][0] in bodies}
                    if children <= bodies:
                        break
                    bodies |= children
                self.desc[name] = bodies

    def equations(self, q, gravity):
        fk = self.model.fk(q)
        rotation = level_rotation(np.asarray(gravity, float).copy())
        rows, grav = [], []
        for side, names in zip(('left', 'right'), self.sides):
            foot = rotation @ fk[side+'_ankle_roll_link'][:3, 3]
            jac, g = [], []
            for name in names:
                joint = self.joints[name]
                frame = fk[joint['body0'][0]] @ self.model.frames[name][0]
                origin = rotation @ frame[:3, 3]
                axis = rotation @ frame[:3, 'XYZ'.index(joint['axis'])]
                jac.append(np.r_[np.cross(axis, foot-origin), axis])
                g.append(sum(self.mass[b]*9.81*np.cross(
                    axis, rotation @ apply(fk[b], self.com[b])-origin)[2]
                    for b in self.desc[name] if b in self.mass))
            rows.append(np.asarray(jac)); grav.append(np.asarray(g))
        return np.asarray(rows), np.asarray(grav)

    def estimate(self, q, torque, gravity):
        q = np.asarray(q, dtype=float)
        torque = np.asarray(torque, dtype=float)
        gravity = np.asarray(gravity, dtype=float)
        # Unknown measurements are not a solved wrench. In particular, numpy
        # can solve a finite matrix with a NaN right-hand side without raising.
        unknown = (None, np.full(2, np.inf), np.full((2, 6, 6), np.nan),
                   np.full((2, 6), np.nan))
        if (q.shape != (len(self.model.names),) or torque.shape != q.shape or
            gravity.shape != (3,) or not np.isfinite(q).all() or
            not np.isfinite(torque).all() or not np.isfinite(gravity).all() or
            not np.isfinite(np.linalg.norm(gravity)) or np.linalg.norm(gravity) == 0):
            return unknown
        jac, grav = self.equations(q, gravity)
        indices = [[self.model.names.index(n) for n in names] for names in self.sides]
        rhs = grav - torque[indices]
        if not np.isfinite(jac).all() or not np.isfinite(rhs).all():
            return None, unknown[1], jac, rhs
        conditions = np.array([np.linalg.cond(a) for a in jac])
        if not np.isfinite(conditions).all() or conditions.max() >= 200:
            return None, conditions, jac, rhs
        foot = np.stack([np.linalg.solve(a, b) for a, b in zip(jac, rhs)])
        rest = np.array([0., 0., self.weight]) - foot[:, :3].sum(0)
        value = np.r_[rest, foot.ravel()]
        return (value if np.isfinite(value).all() else None), conditions, jac, rhs
