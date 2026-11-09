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
        self.arm_goal = None
        self.progress = 0.
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
        arm_reference = self.arm_initial if self.arm_goal is None else (1-self.progress)*self.arm_initial+self.progress*self.arm_goal
        return np.r_[error, heights.min() - gap, tip @ self.e - tangent, com[0] - balance, .001*(z[9:11]-arm_reference)]

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

    def command_for_state(self,q,root,tip):
        floor_root=root.copy();floor_root[2,3]-=self.floor
        pose=np.r_[floor_root[:3,3],Rotation.from_matrix(root[:3,:3]).as_quat()]
        bias,static=static_bias(self.model,self.inertia,q,pose,float(tip[0]),0.)
        arm_bias=[]
        for joint in self.armids:
            potentials=[]
            for sign in (-1,1):
                trial=q.copy();trial[joint]+=sign*1e-5;tf=self.model.fk(trial)
                potentials.append(sum(m*9.81*apply(root@tf[b],c)[2] for b,m,c in
                    zip(self.inertia['body_names'],self.inertia['mass'],self.inertia['local_com'])))
            arm_bias.append((potentials[1]-potentials[0])/(2e-5*40.))
        cmd=self.previous.copy();cmd[self.legids]=q[self.legids]+bias
        cmd[self.armids]=q[self.armids]+np.asarray(arm_bias)
        return cmd,static

    def solve(self):
        self.deadline=time.monotonic()+45.
        q0,root0,fk0,pelvis0,h0,tip0,com0=self.state(self.z0)
        # Arms stay at the measured standing pose. The knee bend determines
        # height; no straight-line pelvis gap/tangent track is prescribed.
        self.arm_target=self.q0[self.armids].copy()
        balance=float(com0[0])
        lo=np.r_[self.lower,-.5,-.6,-.15]
        hi=np.r_[self.upper,.1,.1,np.deg2rad(40)-.05]
        self.stage_evidence=dict(initialization='measured q0 only',balance_target=balance)
        def state9(z):return self.state(np.r_[z,self.arm_initial])
        def feet(z):
            qi,rt,fk,*_=state9(z);rows=[]
            for side,anchor in zip(('left','right'),self.feet):
                foot=rt@fk[side+'_ankle_roll_link']
                rows.extend((foot[:3,3]-anchor[:3,3])[[0,2]])
                rows.append(.2*Rotation.from_matrix(anchor[:3,:3].T@foot[:3,:3]).as_rotvec()[1])
            return rows
        def endpoint_residual(z):
            qi,rt,fk,pelvis,height,tip,com=state9(z)
            return np.r_[feet(z),height.min()-.012,com[0]-balance,
                .2*((z[1]-self.z0[1])-(z[4]-self.z0[4]))]
        endpoint=least_squares(endpoint_residual,self.z0[:9],bounds=(lo,hi),max_nfev=80,
            xtol=1e-10,ftol=1e-10,gtol=1e-10)
        self.stage_evidence['endpoint']=dict(success=bool(endpoint.success),nfev=int(endpoint.nfev),
            residual=endpoint.fun.tolist(),variables=endpoint.x.tolist(),global_optimality_claim=False)
        if not endpoint.success or np.max(abs(endpoint.fun))>2e-6:
            raise ValueError('FOOT_CURVE_ENDPOINT_REJECTED')
        qe,re,fe,pe,he,te,ce=state9(endpoint.x)
        # Derive the setup from the same22mm-eroded observed hull rule.
        eq=self.hull.equations;lane=float(te@self.b);footprint=[]
        for pitch in np.linspace(self.euler0[1],np.deg2rad(40)-.05,33):
            angles=self.euler0.copy();angles[1]=pitch
            pts=self.mesh['pelvis']@Rotation.from_euler('xyz',angles).as_matrix().T
            height=pts@self.n;bottom=pts[height<height.min()+.002];reference=pts[height.argmin()]
            footprint.extend(np.c_[(bottom-reference)@self.e,(bottom-reference)@self.b])
        padding=np.max(np.asarray(footprint)@eq[:,:2].T,axis=0)
        rhs=-.022-eq[:,2]-eq[:,1]*lane-padding;forward=eq[:,0]>1e-5;back=eq[:,0]<-1e-5
        old_target=float(np.min(rhs[forward]/eq[forward,0]))
        if (back.any() and old_target<np.max(rhs[back]/eq[back,0])) or np.any((abs(eq[:,0])<=1e-5)&(rhs<0)):
            raise ValueError('ERODED_PATCH_EMPTY')
        shift=float(te@self.e)-old_target
        if not 0<=shift<=.25:raise ValueError('SETUP_TRANSLATION_OUTSIDE_FROZEN_RANGE')
        self.setup_translation=shift*self.e;self.cloud+=self.setup_translation
        self.hull=ConvexHull(np.c_[self.cloud@self.e,self.cloud@self.b])
        self.stage_evidence['translation_level_m']=self.setup_translation.tolist()
        # Direct command timing witness precedes all33 full-path nodes.
        command_end=self.command_for_state(qe,re,te)[0]
        from complete_seat_path import arc_clock
        length=float(np.max(abs(command_end-self.previous)));duration=arc_clock(0,length)[1]
        count=int(np.ceil(duration/.02));prev=self.previous.copy();peak=0.;margin=float('inf')
        limits={j['name']:np.deg2rad(j['limits']) for j in self.model.joints if j['name'] in self.model.names}
        for tick in range(count):
            distance,_=arc_clock(min((tick+1)*.02,duration),length)
            command=self.previous+(command_end-self.previous)*distance/length
            desired=prev+(command-prev)/.1
            peak=max(peak,float(abs(command-prev).max()))
            margin=min(margin,min(min(desired[j]-limits[self.model.names[j]][0],limits[self.model.names[j]][1]-desired[j]) for j in range(23)))
            prev=command
        self.stage_evidence['command_time_witness']=dict(controls=count,duration_s=count*.02,
            max_increment_rad=peak,minimum_desired_margin_rad=margin,full_path_or_physics_witness=False)
        if count*.02>10.4 or peak>.003+1e-10 or margin<.03:raise ValueError('DIRECT_COMMAND_TIME_REJECTED')
        nodes=[];seed=self.z0[:9].copy();initial_command=self.command_for_state(q0,root0,tip0)[0]
        preload=self.previous-initial_command
        for index,s in enumerate(np.linspace(0,1,33)):
            knee=(1-s)*self.z0[1]+s*endpoint.x[1]
            pitch=(1-s)*self.z0[8]+s*endpoint.x[8]
            def curve_residual(z):
                qi,rt,fk,pelvis,height,tip,com=state9(z)
                return np.r_[feet(z),.2*(z[1]-knee),.2*(z[8]-pitch),com[0]-balance]
            if index:
                fit=least_squares(curve_residual,seed,bounds=(lo,hi),max_nfev=60,
                    xtol=1e-9,ftol=1e-9,gtol=1e-9)
                self.stage_evidence['last_curve_solve']=dict(node=index,residual=fit.fun.tolist(),nfev=int(fit.nfev))
                if not fit.success or np.max(abs(fit.fun))>2e-6:raise ValueError('FOOT_CURVE_NODE_REJECTED')
                seed=fit.x
            q,rt,fk,pelvis,height,tip,com=state9(seed)
            command,static=self.command_for_state(q,rt,tip);command+=(1-s)*preload
            nodes.append(dict(q=q.tolist(),root=rt.tolist(),command=command.tolist(),phase='approach',
                gap=float(height.min()),contact_tangent=float(tip@self.e),balance=float(com[0]),static=static))
        nodes[0]['command']=self.previous.tolist()
        # Endpoint equality must survive the independently parameterized curve.
        if abs(nodes[-1]['gap']-.012)>.002 or abs(nodes[-1]['contact_tangent']-float(te@self.e))>.002:
            raise ValueError('CURVE_ENDPOINT_DOES_NOT_REACH_PLANNED_CONTACT')
        self.arm_goal=self.arm_initial.copy()
        commands=np.asarray([n['command'] for n in nodes]);arc=np.r_[0.,np.cumsum(abs(np.diff(commands,axis=0)).max(axis=1))]
        approach=float(arc[-1]/.125+1.)
        return dict(nodes=nodes,command_arc=arc.tolist(),approach_seconds=approach,
            earliest_contact_seconds=approach+2.5,earliest_settled_seconds=approach+3.5,
            target_tangent=float(te@self.e),target_balance=balance,target_rule='foot_anchored_knee_curve',
            setup_translation_level=self.setup_translation.tolist(),endpoint_solver=self.stage_evidence['endpoint'],
            command_time_witness=self.stage_evidence['command_time_witness'],arm_endpoint=self.arm_goal.tolist(),
            measured_floor=self.floor,foot_bounds=self.foot_bounds,surface_normal=self.n.tolist(),surface_offset=self.d,
            independent_audit_required=True,physical_success=False)
