"""Model-based static feedforward; no simulator state or contact truth inputs."""
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import apply,transform

def static_bias(model,inertial,q,root,seat_x,contact_fraction=1.):
    R=Rotation.from_quat(root[3:]).as_matrix();T=transform(R,root[:3]);fk=model.fk(q)
    com=np.average([apply(T@fk[b],c) for b,c in zip(inertial['body_names'],inertial['local_com'])],axis=0,weights=inertial['mass'])
    weight=sum(inertial['mass'])*9.81
    ankle={s:T@fk[s+'_ankle_roll_link'] for s in ['left','right']};fx=np.mean([v[0,3] for v in ankle.values()])
    share=float(np.clip((fx-com[0])/(fx-seat_x),0,.9))*contact_fraction
    foot_cop=(com[0]-share*seat_x)/(1-share)
    # Explicit feasibility audit. No force or failure criterion is relaxed.
    points={s:np.linalg.inv(v)@np.r_[foot_cop,v[1,3],0.,1.] for s,v in ankle.items()}
    ids=[model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    stiffness=np.array([200.,200.,160.]*2);torque=[];eps=1.e-5
    for j in ids:
        values=[]
        for sign in [-1,1]:
            trial=q.copy();trial[j]+=sign*eps;f=model.fk(trial)
            potential=sum(m*9.81*apply(T@f[b],c)[2] for b,m,c in zip(inertial['body_names'],inertial['mass'],inertial['local_com']))
            work=sum((weight*(1-share)/2)*(T@f[s+'_ankle_roll_link']@points[s])[2] for s in ['left','right'])
            values.append(potential-work)
        torque.append((values[1]-values[0])/(2*eps))
    return np.array(torque)/stiffness,dict(com=com.tolist(),predicted_seat_bw=share,foot_cop_x=float(foot_cop),foot_cop_relative_to_ankle=float(foot_cop-fx),torque=torque,scope='Quasi-static vertical-force allocation, not measured support; dynamics validation required')
