"""Camera-derived interior landing, with bilateral feet and balance constraints.

CPU construction only. No simulator imports, scene identity, or saved action path.
Coordinates are gravity aligned at the startup pelvis, with measured feet fixed.
"""
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import apply, level_rotation, transform
from sagittal_static_bias import static_bias


class SeatPlan:
    def __init__(self, model, inertia, shapes, q, gravity, geometry, points, previous):
        self.model, self.inertia = model, inertia
        self.q0 = np.asarray(q, float).copy()
        self.previous = np.asarray(previous, float).copy()
        self.R0 = level_rotation(np.asarray(gravity, float).copy())
        self.euler0 = Rotation.from_matrix(self.R0).as_euler('xyz')
        self.n = self.R0 @ np.asarray(geometry[2:5], float)
        self.n /= np.linalg.norm(self.n)
        self.d = float(geometry[5])
        assert geometry[11] and geometry[12] <= .21 and self.n[2] > .97
        self.cloud = np.asarray(points) @ self.R0.T
        self.e = np.array([1., 0., 0.]) - self.n * self.n[0]
        self.e /= np.linalg.norm(self.e)
        self.b = np.cross(self.n, self.e)
        self.hull = ConvexHull(np.c_[self.cloud @ self.e, self.cloud @ self.b])
        self.mesh = {b: np.concatenate([s['vertices'] for s in shapes if s['body'] == b])
                     for b in {s['body'] for s in shapes}}
        self.legids = [model.names.index(f'{s}_{j}_joint') for s in ('left', 'right')
                       for j in ('hip_pitch', 'knee', 'ankle_pitch')]
        limits = {j['name']: np.deg2rad(j['limits']) for j in model.joints if j['name'] in model.names}
        self.lower = np.array([limits[model.names[j]][0] + .031 for j in self.legids])
        self.upper = np.array([limits[model.names[j]][1] - .031 for j in self.legids])
        fk = model.fk(self.q0)
        self.feet = [transform(self.R0) @ fk[s + '_ankle_roll_link'] for s in ('left', 'right')]
        soles = np.concatenate([apply(t, self.mesh[s + '_ankle_roll_link'])
                                for s, t in zip(('left', 'right'), self.feet)])
        self.floor = float(soles[:, 2].min())
        sole = soles[soles[:, 2] < self.floor + .004]
        self.foot_bounds = [float(sole[:, 0].min()), float(sole[:, 0].max())]
        self.z0 = np.r_[self.q0[self.legids], 0., 0., self.euler0[1]]
        self.deadline = None
        self.arm_reach = 0.
        self.armids = [model.names.index(f'{side}_{joint}_joint') for side in ('left','right')
                       for joint in ('shoulder_pitch','shoulder_roll','elbow_pitch')]
        self.arm_target = np.array([-1.3,.15,.10,-1.3,-.15,.10])
        self.shoulder_ids = [self.armids[0], self.armids[3]]
        self.elbow_ids = [self.armids[2], self.armids[5]]
        self.arm_initial = np.array([self.q0[self.shoulder_ids].mean(), self.q0[self.elbow_ids].mean()])
        self.z0 = np.r_[self.z0, self.arm_initial]
        self.arm_lower = np.array([max(limits[model.names[j]][0] for j in ids)+.031 for ids in (self.shoulder_ids,self.elbow_ids)])
        self.arm_upper = np.array([min(limits[model.names[j]][1] for j in ids)-.031 for ids in (self.shoulder_ids,self.elbow_ids)])

    def state(self, z):
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise TimeoutError('45_SECOND_CONSTRUCTION_CAP')
        q = self.q0.copy(); q[self.legids] = z[:6]
        q[self.armids] += self.arm_reach*(self.arm_target-self.q0[self.armids])
        q[self.shoulder_ids] = self.q0[self.shoulder_ids] + z[9]-self.arm_initial[0]
        q[self.elbow_ids] = self.q0[self.elbow_ids] + z[10]-self.arm_initial[1]
        angles = self.euler0.copy(); angles[1] = z[8]
        root = transform(Rotation.from_euler('xyz', angles).as_matrix(), [z[6], 0., z[7]])
        fk = self.model.fk(q)
        pelvis = apply(root, self.mesh['pelvis'])
        heights = pelvis @ self.n + self.d
        tip = pelvis[heights.argmin()]
        com = np.average([apply(root @ fk[b], c) for b, c in
                          zip(self.inertia['body_names'], self.inertia['local_com'])],
                         axis=0, weights=self.inertia['mass'])
        return q, root, fk, pelvis, heights, tip, com

    def residual(self, z, gap, tangent, balance):
        q, root, fk, pelvis, heights, tip, com = self.state(z)
        error = []
        for s, anchor in zip(('left', 'right'), self.feet):
            t = root @ fk[s + '_ankle_roll_link']
            error.extend((t[:3, 3] - anchor[:3, 3])[[0, 2]])
            error.append(.2 * Rotation.from_matrix(anchor[:3, :3].T @ t[:3, :3]).as_rotvec()[1])
        return np.r_[error, heights.min() - gap, tip @ self.e - tangent, com[0] - balance, .001*(z[9:11]-self.arm_initial)]

    def fit(self, seed, gap, tangent, balance):
        lo = np.r_[self.lower, -.5, -.6, -.15, self.arm_lower]
        hi = np.r_[self.upper, .1, .1, np.deg2rad(40) - .05, self.arm_upper]
        fit = least_squares(self.residual, seed, bounds=(lo, hi),
                            args=(gap, tangent, balance), max_nfev=60,
                            xtol=1e-9, ftol=1e-9, gtol=1e-9)
        if np.max(np.abs(fit.fun[:9])) > .002:
            raise ValueError(dict(reason='BILATERAL_BALANCED_POSE_INFEASIBLE',
                                  residual=fit.fun.tolist(), variables=fit.x.tolist(),
                                  gap=gap, tangent=tangent, balance=balance))
        return fit.x

    def solve(self):
        self.deadline = time.monotonic() + 45.
        q, root, fk, pelvis, h, tip, com = self.state(self.z0)
        # Front edge of the observed convex support patch at the contact lane.
        # 50mm leaves room for the finite patch and the unchanged 10mm scoring inset.
        eq = self.hull.equations
        lane = float(tip @ self.b)
        forward = eq[:, 0] > 1e-5
        edge = np.min((-eq[forward, 2] - eq[forward, 1] * lane) / eq[forward, 0])
        target = float(edge - .05)
        balance = self.foot_bounds[0] + .02
        seed = self.z0.copy(); nodes = []
        initial_gap, initial_tangent, initial_balance = float(h.min()), float(tip @ self.e), float(com[0])
        # Solve fresh measured-geometry constraints; no prior failed action vector.
        for s in np.linspace(0, 1, 33):
            self.arm_reach = s
            gap = (1-s)*initial_gap + s*.012
            tangent = (1-s)*initial_tangent + s*target
            cop = (1-s)*initial_balance + s*balance
            if s:
                seed = self.fit(seed, gap, tangent, cop)
            q, root, fk, pelvis, heights, tip, com = self.state(seed)
            floor_root = root.copy(); floor_root[2, 3] -= self.floor
            pose = np.r_[floor_root[:3, 3], Rotation.from_matrix(root[:3, :3]).as_quat()]
            bias, static = static_bias(self.model, self.inertia, q, pose, float(tip[0]), 0.)
            arm_bias=[]
            for joint in self.armids:
                potentials=[]
                for sign in (-1,1):
                    trial=q.copy();trial[joint]+=sign*1e-5
                    tf=self.model.fk(trial)
                    potentials.append(sum(m*9.81*apply(root@tf[b],c)[2] for b,m,c in
                        zip(self.inertia['body_names'],self.inertia['mass'],self.inertia['local_com'])))
                arm_bias.append((potentials[1]-potentials[0])/(2e-5*40.))
            arm_bias=np.asarray(arm_bias)
            if not nodes:
                arm_preload=self.previous[self.armids]-q[self.armids]-arm_bias
                preload = self.previous[self.legids] - q[self.legids] - bias
            cmd = self.previous.copy(); cmd[self.legids] = q[self.legids] + bias + (1-s)*preload
            cmd[self.armids]=q[self.armids]+arm_bias+(1-s)*arm_preload
            nodes.append(dict(q=q.tolist(), root=root.tolist(), command=cmd.tolist(),
                              phase='approach', gap=float(heights.min()),
                              contact_tangent=float(tip @ self.e), balance=float(com[0]), static=static))
        nodes[0]['command'] = self.previous.tolist()
        # Conservative command arc timing with a half-second speed ramp at each end.
        commands = np.array([x['command'] for x in nodes])
        arc = np.r_[0., np.cumsum(np.max(abs(np.diff(commands, axis=0)), axis=1))]
        approach_seconds = float(arc[-1]/.125 + 1.)
        return dict(nodes=nodes, command_arc=arc.tolist(), approach_seconds=approach_seconds,
                    earliest_contact_seconds=approach_seconds+.5+.012/.006,
                    earliest_settled_seconds=approach_seconds+.5+.012/.006+1.,
                    target_tangent=target, target_balance=balance,
                    sensor_front_edge=float(edge), measured_floor=self.floor,
                    foot_bounds=self.foot_bounds, surface_normal=self.n.tolist(),
                    surface_offset=self.d, independent_audit_required=True,
                    physical_success=False)
