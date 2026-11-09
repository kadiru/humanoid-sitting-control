"""Bounded fixed-foot continuation, initialized only from a physical start.

The two feet provide twelve equations. An optional root translation supplies
the thirteenth unknown for a prescribed centre-of-mass x coordinate. Each
accepted state is independently checked with the original forward kinematics.
This is a kinematic construction, not a dynamic or contact-success certificate.
"""
import numpy as np
from scipy.spatial.transform import Rotation


class ClosureFailure(ValueError):
    pass


class FixedFeet:
    def __init__(self, model, inertia, initial_q, initial_root, lower, upper):
        self.model=model
        self.legs=[model.names.index(side+'_'+name+'_joint')
                   for side in ('left','right')
                   for name in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
        self.feet=[side+'_ankle_roll_link' for side in ('left','right')]
        self.initial=np.asarray(initial_q,float).copy()
        self.lower=np.asarray(lower,float)+.031
        self.upper=np.asarray(upper,float)-.031
        self.bodies=inertia['body_names'];self.mass=np.asarray(inertia['mass'],float)
        self.local=np.asarray(inertia['local_com'],float)
        self.joints={j['name']:j for j in model.joints}
        fk=model.fk(self.initial)
        self.anchors=[initial_root@fk[b] for b in self.feet]
        self.desc=[]
        for index in self.legs:
            children={self.joints[model.names[index]]['body1'][0]}
            while True:
                added={j['body1'][0] for j in model.joints if j['body0'] and j['body0'][0] in children}
                if added<=children:break
                children|=added
            self.desc.append(children)
        self.mass_membership=np.asarray([[b in ds for b in self.bodies] for ds in self.desc])
        self.calls=0;self.linear_solves=0

    def evaluate(self,q,root,direction=None):
        fk=self.model.fk(q);poses=[root@fk[b] for b in self.feet]
        frames=[root@fk[self.joints[self.model.names[i]]['body0'][0]]@self.model.frames[self.model.names[i]][0]
                for i in self.legs]
        axes=np.asarray([f[:3,'XYZ'.index(self.joints[self.model.names[i]]['axis'])]
                         for i,f in zip(self.legs,frames)])
        origins=np.asarray([f[:3,3] for f in frames])
        columns=12+(direction is not None);J=np.zeros((columns,columns));error=[]
        for k,(body,pose,anchor) in enumerate(zip(self.feet,poses,self.anchors)):
            error.extend(anchor[:3,3]-pose[:3,3])
            error.extend(Rotation.from_matrix(anchor[:3,:3]@pose[:3,:3].T).as_rotvec())
            belongs=np.asarray([body in ds for ds in self.desc])
            J[6*k:6*k+3,:12]=(np.cross(axes,pose[:3,3]-origins)*belongs[:,None]).T
            J[6*k+3:6*k+6,:12]=(axes*belongs[:,None]).T
            if direction is not None:J[6*k:6*k+3,12]=direction
        centers=np.asarray([(root@fk[b])[:3,:3]@p+(root@fk[b])[:3,3]
                            for b,p in zip(self.bodies,self.local)])
        com=np.average(centers,weights=self.mass,axis=0)
        if direction is not None:
            derivatives=np.cross(axes[:,None,:],centers[None,:,:]-origins[:,None,:])
            J[12,:12]=np.sum(derivatives[:,:,0]*self.mass*self.mass_membership,axis=1)/self.mass.sum()
            J[12,12]=direction[0]
        self.calls+=1
        return np.asarray(error),J,com

    def solve(self,previous_q,root,*,direction=None,target_com_x=None):
        q=np.asarray(previous_q,float).copy();root=np.asarray(root,float).copy()
        if q.shape!=(37,) or root.shape!=(4,4) or not np.isfinite(q).all() or not np.isfinite(root).all():
            raise ClosureFailure('FINITE_CURRENT_STATE_REQUIRED')
        fixed=np.ones(37,bool);fixed[self.legs]=False
        if not np.array_equal(q[fixed],self.initial[fixed]):
            raise ClosureFailure('NONLEG_OR_FINGER_CHANGE')
        if (direction is None)!=(target_com_x is None):raise ClosureFailure('BALANCE_EQUATION_PAIR_REQUIRED')
        if direction is not None:
            direction=np.asarray(direction,float)
            if direction.shape!=(3,) or not np.isfinite(direction).all() or not np.isfinite(target_com_x):
                raise ClosureFailure('FINITE_BALANCE_DIRECTION_REQUIRED')
        for iteration in range(9):
            error,J,com=self.evaluate(q,root,direction)
            residual=np.r_[error,target_com_x-com[0]] if direction is not None else error
            if not np.isfinite(residual).all() or not np.isfinite(J).all():raise ClosureFailure('NONFINITE_CLOSURE')
            if max(abs(residual))<=1e-9:
                if np.any(q[:23]<=self.lower) or np.any(q[:23]>=self.upper):raise ClosureFailure('ORIGINAL_PLANNING_JOINT_BOX')
                return q,root,dict(iterations=iteration,maximum_residual=float(max(abs(residual))))
            if iteration==8:break
            try:step=np.linalg.solve(J,residual)
            except np.linalg.LinAlgError as e:raise ClosureFailure('SINGULAR_CLOSURE') from e
            self.linear_solves+=1
            if not np.isfinite(step).all() or max(abs(step[:12]))>.08:
                raise ClosureFailure('CONTINUATION_NEEDS_A_DIFFERENT_PATH')
            q[self.legs]+=step[:12]
            if direction is not None:
                if abs(step[12])>.02:raise ClosureFailure('BALANCE_TRANSLATION_STEP_TOO_LARGE')
                root[:3,3]+=direction*step[12]
            if np.any(q[:23]<=self.lower) or np.any(q[:23]>=self.upper):raise ClosureFailure('ORIGINAL_PLANNING_JOINT_BOX')
            if not -.5<=root[0,3]<=.15:raise ClosureFailure('ORIGINAL_ROOT_DOMAIN')
        raise ClosureFailure('FINITE_EIGHT_STEP_CLOSURE_FAILED')
