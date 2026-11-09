"""Positive-bend geometric inverse kinematics, audited with the full model.

The planar construction eliminates six pitch variables. Measured small roll/yaw
angles remain unchanged; full 3-D FK, not the planar approximation, decides
admission. The positive bend is supported by an admitted sitting demonstration.
No failed solver vector or new joint/action limit is used.
"""
import importlib.util
from pathlib import Path
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import apply, transform

path=Path(__file__).resolve().parent.parent/'furniture_collision_repair_20261106_v8/legal_support_plan.py'
spec=importlib.util.spec_from_file_location('support_original',path)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def ry(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

class SeatPlan(base.SeatPlan):
    def __init__(self,*args,reference,**kwargs):
        super().__init__(*args,**kwargs)
        self.reference=reference
        self.chain={}
        joints={j['name']:j for j in self.model.joints}
        intervals=[]
        for side,anchor in zip(('left','right'),self.feet):
            js=[joints[f'{side}_{n}_joint'] for n in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
            assert all(np.max(abs(j['p1']))==0 for j in js)
            offsets=[Rotation.from_quat(j['q0']).as_euler('xyz')[1] for j in (js[0],js[3],js[4])]
            upper=sum((np.asarray(j['p0']) for j in js[1:4]),np.zeros(3))
            lower=np.asarray(js[4]['p0'])
            self.chain[side]=(js,offsets,upper,lower)
            v=apply(anchor,self.mesh[side+'_ankle_roll_link']);sole=v[v[:,2]<v[:,2].min()+.004]
            intervals.append([float(sole[:,0].min()),float(sole[:,0].max())])
        self.common_foot_interval=[max(x[0] for x in intervals),min(x[1] for x in intervals)]
        assert self.common_foot_interval[1]-self.common_foot_interval[0]>.02
        self.balance_midpoint=sum(self.common_foot_interval)/2
        self.ik_calls=0

    def pitch_ik(self,root):
        angles=[]
        for side,anchor in zip(('left','right'),self.feet):
            js,offsets,u,v=self.chain[side]
            roll=self.q0[self.model.names.index(side+'_ankle_roll_joint')]
            ankle_rotation=anchor[:3,:3]@Rotation.from_rotvec([-roll,0,0]).as_matrix()
            ankle_position=anchor[:3,3]-ankle_rotation@np.asarray(js[5]['p0'])
            target=root[:3,:3].T@(ankle_position-root[:3,3])-np.asarray(js[0]['p0'])
            l1=np.linalg.norm(u[[0,2]]);l2=np.linalg.norm(v[[0,2]])
            cosine=(np.dot(target[[0,2]],target[[0,2]])-l1*l1-l2*l2)/(2*l1*l2)
            if not -1<=cosine<=1:raise ValueError('GEOMETRIC_LEG_REACH_EXCEEDED')
            bend=np.arccos(cosine)
            a_u=np.arctan2(u[0],-u[2]);a_v=np.arctan2(v[0],-v[2])
            knee=bend-offsets[1]-a_u+a_v
            vector=u+ry(offsets[1]+knee)@v
            hip_total=np.arctan2(vector[0],-vector[2])-np.arctan2(target[0],-target[2])
            hip=hip_total-offsets[0]
            foot_pitch=Rotation.from_matrix(root[:3,:3].T@ankle_rotation).as_euler('xyz')[1]
            ankle=foot_pitch-hip_total-offsets[1]-knee-offsets[2]
            angles.extend([hip,knee,ankle])
        self.ik_calls+=1
        return np.asarray(angles)

    def to_full(self,reduced):
        z=self.z0.copy();z[6:9]=reduced[:3];z[9:11]=reduced[3:]
        angles=self.euler0.copy();angles[1]=z[8]
        root=transform(Rotation.from_euler('xyz',angles).as_matrix(),[z[6],0.,z[7]])
        z[:6]=self.pitch_ik(root)
        return z

    def fit(self,seed,gap,tangent,balance):
        if self.contact_target is None:
            # Reference root displacement and pitch from a clean complete sit;
            # it supplies a branch/posture, never a command for the new object.
            reduced=np.r_[self.reference['root_delta_xz'],self.reference['pitch'],self.reference['arms']]
        else:
            reduced=np.r_[seed[6:9],seed[9:11]]
            if not getattr(self,'landing_active',False):
                rise=min(1.,max(self.progress,self.arm_reach)/.25)
                gap=(1-rise)*self.start_gap+rise*.012
        self.last_reduced=None
        def residual(y):
            if time.monotonic()>self.deadline:raise TimeoutError('OUTER_CONSTRUCTION_DEADLINE')
            z=self.to_full(y)
            r=super(SeatPlan,self).residual(z,gap,tangent,balance)
            # All full-model foot errors and original limits are retained.
            violations=np.r_[np.maximum(self.lower-z[:6],0),np.maximum(z[:6]-self.upper,0)]
            return np.r_[r,violations]
        fit=least_squares(residual,reduced,bounds=(np.r_[-.5,-.6,-.15,self.arm_lower],np.r_[.1,.1,np.deg2rad(40)-.05,self.arm_upper]),max_nfev=60,xtol=1e-9,ftol=1e-9,gtol=1e-9)
        z=self.to_full(fit.x)
        if np.max(abs(fit.fun[:9]))>.002 or np.any(z[:6]<self.lower) or np.any(z[:6]>self.upper):
            raise ValueError(dict(reason='GEOMETRIC_BRANCH_POSE_REJECTED',residual=fit.fun.tolist(),variables=z.tolist(),nfev=fit.nfev))
        if self.contact_target is None:self.contact_target=float(self.state(z)[5]@self.e)
        return z

    def solve(self):
        # The inherited builder uses foot_bounds[0]+20mm. Express the unique
        # common-sole midpoint there only during construction; restore the true
        # foot bounds before the independent audit.
        actual=self.foot_bounds
        self.foot_bounds=[self.balance_midpoint-.02,actual[1]]
        try:result=super().solve()
        finally:self.foot_bounds=actual
        result.update(foot_bounds=actual,target_balance=self.balance_midpoint,common_foot_interval=self.common_foot_interval,ik_calls=self.ik_calls,reference_role='branch_and_posture_only',construction='positive_bend_geometric_leg_elimination',physical_success=False)
        return result
