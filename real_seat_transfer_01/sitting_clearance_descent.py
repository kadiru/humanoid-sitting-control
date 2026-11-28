"""Plan a slow seat approach with fixed measured soles and a checked contact order."""
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.stats import qmc


BODIES = ('hip_pitch','hip_roll','hip_yaw','elbow_pitch','elbow_roll','palm','knee')
FEET = ('left_ankle_roll_link','right_ankle_roll_link')
KP = np.array([200,200,200,150,150,40,40,150,150,40,40,200,200,40,40,160,160,40,40,160,160,40,40])
START = 40
END = 460
MARGIN = .008
STANCE_MARGIN = .010
ANCHOR_LOAD = .22
H = .004


def smooth(u):
    u = np.clip(u,0.,1.)
    return u**3*(10.+u*(-15.+6.*u))


def roof(seat, points):
    ij = np.rint((points[...,:2]-seat.origin)/H).astype(int)
    valid = ((ij>=0)&(ij<np.array(seat.z.shape))).all(-1)
    i,j = np.moveaxis(np.clip(ij,0,np.array(seat.z.shape)-1),-1,0)
    return np.where(valid, seat.z[i,j], -np.inf), np.where(valid,seat.margin[i,j],0.)


def spaced(vertices, spacing):
    _, index = np.unique(np.rint(vertices/spacing),axis=0,return_index=True)
    return vertices[index]


def batched_sole_ik(chain, angle, target_position, target_rotation, lower, upper):
    count = len(angle)
    for step in range(24):
        pose = np.repeat(np.eye(4)[None],count,axis=0)
        axes, pivots = [], []
        for j,(_,left,right,axis) in enumerate(chain):
            pose = pose@left
            axes.append(pose[:,:3,:3]@axis)
            pivots.append(pose[:,:3,3].copy())
            spin = np.repeat(np.eye(4)[None],count,axis=0)
            spin[:,:3,:3] = Rotation.from_rotvec(angle[:,j,None]*axis).as_matrix()
            pose = pose@spin@right
        angular = np.stack(axes,axis=1)
        linear = np.cross(angular,pose[:,None,:3,3]-np.stack(pivots,axis=1))
        error = np.column_stack((target_position-pose[:,:3,3],Rotation.from_matrix(
            target_rotation@pose[:,:3,:3].transpose(0,2,1)).as_rotvec()))
        if step==23:
            return angle,np.linalg.norm(error[:,:3],axis=1),np.linalg.norm(error[:,3:],axis=1)
        J = np.concatenate((linear,angular),axis=2).transpose(0,2,1)
        JT = J.transpose(0,2,1)
        delta = np.linalg.solve(JT@J+2.e-5*np.eye(6),(JT@error[...,None]))[...,0]
        angle = np.clip(angle+np.clip(delta,-.12,.12),lower,upper)


class Corridor:
    def __init__(self, scene, q, root, R, feet, arms):
        from rearstep_model import Robot
        from stance23_planner import chains, legs, lo, hi, model
        from surface import Seat

        self.seat = Seat(scene)
        self.robot = Robot()
        self.chains, self.legs, self.lo, self.hi = chains, legs, lo, hi
        self.root, self.R, self.feet, self.q0 = root, R, feet, q
        self.foot_rotation = Rotation.from_quat(feet[:,3:]).as_matrix()
        self.heading = np.arctan2(R[1,0],R[0,0])
        self.axes = Rotation.from_euler('z',self.heading).as_matrix()
        self.foot_center = feet[:,:3].mean(0)
        self.foot_vertices = [model.vertices[n]@r.T+p[:3] for n,r,p in zip(FEET,self.foot_rotation,feet,strict=True)]
        self.hull = ConvexHull(np.vstack(self.foot_vertices)[:,:2]).equations
        planes = [self.hull, *[ConvexHull(v[:,:2]).equations for v in self.foot_vertices]]
        self.hull, self.shoe_hulls = planes[0], planes[1:]
        self.pelvis = spaced(model.vertices['pelvis'],.005)
        self.parts = {self.robot.names.index(side+'_'+name+'_link'):spaced(model.vertices[side+'_'+name+'_link'],.008)
                      for side in ('left','right') for name in BODIES}
        self.arm_ids = np.array(arms['left']['joints']+arms['right']['joints'])
        self.arm_angles = np.array(arms['left']['angles']+arms['right']['angles'])

    def soles(self, p, R, q):
        pos, angle = np.zeros(len(q)), np.zeros(len(q))
        inv = R.transpose(0,2,1)
        for side,(chain,ids) in enumerate(zip(self.chains,self.legs,strict=True)):
            goal = (inv@(self.feet[side,:3]-p)[...,None])[...,0]
            ori = inv@self.foot_rotation[side]
            q[:,ids], e, a = batched_sole_ik(chain,q[:,ids],goal,ori,self.lo[ids],self.hi[ids])
            pos = np.maximum(pos,e)
            angle = np.maximum(angle,a)
        return pos,angle

    def fk(self, q, p, R):
        T = np.repeat(np.eye(4)[None,None],len(q),axis=0)
        T = np.repeat(T,len(self.robot.names),axis=1)
        T[:,0,:3,:3], T[:,0,:3,3] = R,p
        for j,(parent,index,axis,left,right) in enumerate(self.robot.tree):
            spin = np.repeat(np.eye(4)[None],len(q),axis=0)
            if index>=0:
                spin[:,:3,:3] = Rotation.from_rotvec(q[:,index,None]*axis).as_matrix()
            T[:,j+1] = T[:,parent]@left@spin@right
        centers = (T[:,:,:3,:3]@self.robot.cc[None,:,:,None])[...,0]+T[:,:,:3,3]
        return T,np.average(centers,axis=1,weights=self.robot.mass)

    def clearance(self, T, parts):
        gap = np.full(len(T),1.)
        for index,points in parts.items():
            v = points[None]@T[:,index,:3,:3].transpose(0,2,1)+T[:,index,None,:3,3]
            top,_ = roof(self.seat,v)
            gap = np.minimum(gap,(v[:,:,2]-top).min(1))
        return gap

    def margin(self, p):
        return -(p@self.hull[:,:2].T+self.hull[:,2]).max(1)

    def loads(self, state, seat_force, cp, acceleration):
        com = np.average(state[1],axis=0,weights=self.robot.mass)
        g = 9.81
        support = (com[:2]-seat_force*cp[:2]-com[2]*acceleration[:2]/g)/(1.-seat_force)
        relative = (support-self.foot_center[:2])@self.axes[:2,:2]
        toes = (self.feet[:,:3]-self.foot_center)@self.axes
        dy = toes[0,1]-toes[1,1]
        part = np.clip((relative[1]-toes[1,1])/dy,.13,.87)
        fractions = np.array([part,1.-part])
        cop = self.feet[:,:3].copy()
        cop[:,:2] += support-fractions@cop[:,:2]
        cop[:,2] = 0.
        # Each foot's pressure point must remain inside its own sole.
        margin = min(float(-(h[:,:2]@p[:2]+h[:,2]).max()) for h,p in zip(self.shoe_hulls,cop,strict=True))
        reaction = fractions[:,None]*self.robot.total*(1.-seat_force)*np.r_[acceleration[:2],g]
        torques = self.robot.forces(state,acceleration,cop,reaction)[:23]
        return torques,margin

    def endpoints(self):
        sample = qmc.Sobol(7,scramble=True,seed=8837).random_base2(13)
        reports = np.zeros(5,dtype=int)
        candidates = []
        for block in np.array_split(sample,32):
            count = len(block)
            a = block
            root = self.root + np.c_[-.018-.225*a[:,0],-.080+.160*a[:,1],np.zeros(count)]@self.axes.T
            root[:,2] = 0.
            euler = np.c_[(-.17+.34*a[:,3]),(-.05+.68*a[:,2]),self.heading-.30+.60*a[:,4]]
            R = Rotation.from_euler('xyz',euler).as_matrix()
            v = self.pelvis[None]@R.transpose(0,2,1)+root[:,None]
            top, margin = roof(self.seat,v)
            required = top-v[:,:,2]
            height = required.max(1)
            contact = np.argmax(required,axis=1)
            contact_margin = margin[np.arange(count),contact]
            keep = np.isfinite(height)&(height<self.root[2]-.008)&(height>.37)&(contact_margin>=.016)
            reports[0] += keep.sum()
            if not keep.any():
                continue
            a,root,R,contact_margin = a[keep],root[keep],R[keep],contact_margin[keep]
            root[:,2] = height[keep]
            cp = v[np.arange(count),contact][keep]+np.c_[np.zeros((len(root),2)),root[:,2]]
            q = np.tile(self.q0,(len(root),1))
            q[:,self.arm_ids] = self.arm_angles
            q[:,[5,6]] -= (.12+.62*a[:,5,None])
            q[:,[17,18]] += (.15+.73*a[:,6,None])
            error, angle = self.soles(root,R,q)
            good = (error<.0012)&(angle<.012)&(q[:,11:13].min(1)>.015)
            reports[1] += good.sum()
            if not good.any():
                continue
            q,root,R,cp,contact_margin = q[good],root[good],R[good],cp[good],contact_margin[good]
            T,com = self.fk(q,root,R)
            clearance = self.clearance(T,self.parts)
            admissible = clearance>=MARGIN
            reports[2] += admissible.sum()
            center = (com[:,:2]-ANCHOR_LOAD*cp[:,:2])/(1.-ANCHOR_LOAD)
            stable = (self.margin(com[:,:2])>=STANCE_MARGIN)&(self.margin(center)>=STANCE_MARGIN)
            reports[3] += (stable&admissible).sum()
            for i in np.flatnonzero(admissible&stable):
                state = self.robot.state(q[i],R[i],root[i])
                unloaded,m0 = self.loads(state,0.,cp[i],np.zeros(3))
                loaded,m1 = self.loads(state,ANCHOR_LOAD,cp[i],np.zeros(3))
                max_ankle = max(np.abs(unloaded[15:17]).max(),np.abs(loaded[15:17]).max())
                if min(m0,m1)<.004 or max_ankle>18.5:
                    continue
                reports[4] += 1
                rank = (min(clearance[i],.040)+.4*min(contact_margin[i],.04)
                        +.3*min(m0,m1)-.025*np.linalg.norm(root[i]-self.root)-.0009*max_ankle)
                candidates.append((float(rank),q[i],root[i],R[i],cp[i],float(contact_margin[i])))
        candidates.sort(reverse=True,key=lambda row:row[0])
        return candidates,reports.tolist()

    def path(self, entry, knots=61):
        for _,q_end,final,R_end,cp,patch in entry[:48]:
            for schedule in ((.84,1.,.92),(.94,.85,1.),(1.,1.,1.)):
                u = np.linspace(0.,1.,knots)
                velocity_limit = (END-START)*.02
                fore, lower, tilt = (smooth(u/v) for v in schedule)
                p = self.root+(final-self.root)*np.column_stack((fore,fore,lower))
                R = Rotation.from_rotvec(tilt[:,None]*Rotation.from_matrix(R_end@self.R.T).as_rotvec()).as_matrix()@self.R
                q = np.tile(self.q0,(knots,1))
                q[:,self.arm_ids] += smooth(u[:,None]/.46)*(q_end[self.arm_ids]-q[0,self.arm_ids])
                error, angle = self.soles(p,R,q)
                if error.max()>.0015 or angle.max()>.013:
                    continue
                T,com = self.fk(q,p,R)
                gap = self.clearance(T,self.parts)
                if gap.min()<MARGIN:
                    continue
                pelvis = self.pelvis[None]@R.transpose(0,2,1)+p[:,None]
                top,interior = roof(self.seat,pelvis)
                height = pelvis[:,:,2]-top
                if np.min(height[:-5])<.0005 or np.min(height[interior<.012],initial=1.)<.002:
                    continue
                v = np.gradient(com,velocity_limit/(knots-1),axis=0)
                # This is a preview, with every point softened by the actual 50 Hz controller.
                # Before contact keep both the projection and its momentum inside the soles.
                projected = com[:,:2]+v[:,:2]*np.sqrt(com[:,2,None]/9.81)
                if self.margin(projected).min()<STANCE_MARGIN or (v[:,:2]@self.axes[:2,0]).max()>.045:
                    continue
                acceleration = np.gradient(v,velocity_limit/(knots-1),axis=0)
                ff, pressure_gap = [], []
                for qa,root,ori,acc in zip(q,p,R,acceleration,strict=True):
                    tau,m = self.loads(self.robot.state(qa,ori,root),0.,cp,acc)
                    ff.append(tau/KP)
                    pressure_gap.append(m)
                ff = np.asarray(ff)
                loaded_tau, _ = self.loads(self.robot.state(q[-1],R[-1],p[-1]),ANCHOR_LOAD,cp,np.zeros(3))
                load_ff = loaded_tau/KP
                command = q[:,:23]+ff
                joint_margin = float(np.minimum(command-self.lo,self.hi-command).min())
                if min(pressure_gap)<.004 or np.abs(ff[:,15:17]*KP[15:17]).max()>18.5 or joint_margin<.003:
                    continue
                # On accepted contact, lower a further fraction of a centimetre
                # without moving the soles. Abort that preload if clearance fails.
                press = p[-1]+[0.,0.,-.005]
                press_q = q[-1:].copy()
                pe,pa = self.soles(press[None],R[-1:],press_q)
                press_T,_ = self.fk(press_q,press[None],R[-1:])
                if pe[0]>.0015 or pa[0]>.013 or self.clearance(press_T,self.parts)[0]<MARGIN:
                    continue
                verify = dict(min_housing_elbow_gap_m=float(gap.min()),max_sole_fit_m=float(error.max()),
                              max_sole_rotation_rad=float(angle.max()),min_capture_margin_m=float(self.margin(projected).min()),
                              max_forward_com_speed_m_s=float((v[:,:2]@self.axes[:2,0]).max()),
                              landing_patch_margin_m=patch,min_joint_target_margin_rad=joint_margin,
                              peak_planned_ankle_pitch_nm=float(np.abs(ff[:,15:17]*KP[15:17]).max()),
                              schedule=schedule,goal_pelvis_m=final.tolist(),contact_m=cp.tolist())
                return dict(q=q[:,:23],ff=ff,press=press_q[0,:23]-q[-1,:23],
                            release=load_ff-ff[-1],com=com-self.foot_center, basis=self.axes,
                            feet=self.feet, start_q=self.q0, proof=verify)
        return None
