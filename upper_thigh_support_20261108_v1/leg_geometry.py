"""Named pelvis/back separation seed; not a motion or balance certificate."""
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import apply,transform

def ry(a):return Rotation.from_euler('y',a).as_matrix()

def chains(plan):
    result={}
    for side,anchor in zip(('left','right'),plan.feet):
        js=[plan.joints[side+'_'+n+'_joint'] for n in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
        assert all(np.max(np.abs(np.asarray(j['p1'],float)))==0 for j in js)
        offsets=[Rotation.from_quat(j['q0']).as_euler('xyz')[1] for j in (js[0],js[3],js[4])]
        upper=sum((np.asarray(j['p0']) for j in js[1:4]),np.zeros(3));lower=np.asarray(js[4]['p0'])
        roll=plan.q0[plan.model.names.index(side+'_ankle_roll_joint')]
        ar=anchor[:3,:3]@Rotation.from_rotvec([-roll,0,0]).as_matrix()
        ap=anchor[:3,3]-ar@np.asarray(js[5]['p0'])
        result[side]=(js,offsets,upper,lower,ar,ap)
    return result

def pitch_ik(plan,root,chain):
    q=plan.q0.copy()
    for side,(js,offsets,u,v,ar,ap) in chain.items():
        target=root[:3,:3].T@(ap-root[:3,3])-np.asarray(js[0]['p0'])
        l1=np.linalg.norm(u[[0,2]]);l2=np.linalg.norm(v[[0,2]])
        cosine=(target[[0,2]]@target[[0,2]]-l1*l1-l2*l2)/(2*l1*l2)
        if not -1<=cosine<=1:raise ValueError('SEED_LEG_REACH')
        bend=np.arccos(cosine);au=np.arctan2(u[0],-u[2]);av=np.arctan2(v[0],-v[2])
        knee=bend-offsets[1]-au+av
        vec=u+ry(offsets[1]+knee)@v
        hip_total=np.arctan2(vec[0],-vec[2])-np.arctan2(target[0],-target[2])
        hip=hip_total-offsets[0]
        ankle=Rotation.from_matrix(root[:3,:3].T@ar).as_euler('xyz')[1]-hip_total-offsets[1]-knee-offsets[2]
        q[[plan.model.names.index(side+'_'+n+'_joint') for n in ('hip_pitch','knee','ankle_pitch')]]=[hip,knee,ankle]
    return q

