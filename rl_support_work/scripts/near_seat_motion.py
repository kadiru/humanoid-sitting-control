"""Kinematic rolling reference from depth, encoders and IMU; no live truth."""
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import level_rotation
from humanoid_sitting.contact_aware_reference import measured_level_plane
from build_near_seat_candidates import solve

def rolling_plan(model,q,gravity,measurement,reference_origin,command,calibration):
    n,d,origin=measured_level_plane(model,q,gravity,measurement,reference_origin)
    raw_n=n.copy();raw_d=d;rn=np.array(calibration['reference_normal']);rd=calibration['reference_offset']
    height=.4-(-.4*n[0]+d)/n[2]+(-.4*rn[0]+rd)/rn[2]
    slope=np.arctan2(n[0],n[2])-np.arctan2(rn[0],rn[2]);n=Rotation.from_euler('y',slope).as_matrix()[:,2];d=-n@np.array([-.4,0.,height])
    R=level_rotation(gravity);fk=model.fk(q);feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
    pitch=Rotation.from_matrix(R).as_euler('xyz')[1];vertices=model.vertices['pelvis'];inside=(vertices@R.T)[:,0]+origin[0]<-.16;assert inside.any()
    pivot=vertices[np.flatnonzero(inside)[np.argmin((vertices[inside]@R.T)@n)]]
    leg=[j for j,name in enumerate(model.names) if any(s in name for s in ['hip_','knee','ankle'])]
    commands=[command.copy()];qs=[q.copy()];errors=[];roots=[]
    for u in np.linspace(0,1,25)[1:]:
        alpha=u*u*(3-2*u);angle=pitch*(1-alpha);target_x=origin[0]+((R-Rotation.from_euler('y',angle).as_matrix())@pivot)[0]
        qi,root,error=solve(model,q,feet,n,d,target_x,angle,y=origin[1]);assert error<.003,(float(u),error,origin,n,d,pitch)
        c=command.copy();c[leg]=qi[leg];commands.append(c);qs.append(qi);errors.append(error);roots.append(root)
    return np.array(commands),np.array(qs),dict(raw_normal=raw_n.tolist(),raw_offset=raw_d,normal=n.tolist(),offset=float(d),origin=origin.tolist(),pitch=float(pitch),pivot=pivot.tolist(),target_root=roots[-1].tolist(),ik_error=max(errors))
