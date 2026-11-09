"""Continuous encoder/IMU/depth-planned transfer; truth is not an input."""
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import level_rotation
from humanoid_sitting.contact_aware_reference import measured_level_plane
from build_near_seat_candidates import solve

def smooth(x):
    x=np.clip(x,0.,1.)
    return x*x*(3.-2.*x)

def continuous_plan(model,q,gravity,measurement,reference_origin,command,calibration,
                    bias_delay=0.,rear_shift=0.,final_pitch=0.,gap_final=.001):
    n,d,origin=measured_level_plane(model,q,gravity,measurement,reference_origin)
    raw_n=n.copy();raw_d=float(d)
    rn=np.array(calibration['reference_normal']);rd=calibration['reference_offset']
    height=.4-(-.4*n[0]+d)/n[2]+(-.4*rn[0]+rd)/rn[2]
    slope=np.arctan2(n[0],n[2])-np.arctan2(rn[0],rn[2])
    n=Rotation.from_euler('y',slope).as_matrix()[:,2];d=-n@np.array([-.4,0.,height])
    R=level_rotation(gravity);fk=model.fk(q)
    feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
    pitch=Rotation.from_matrix(R).as_euler('xyz')[1]
    vertices=model.vertices['pelvis'];world=vertices@R.T+origin
    inside=(world[:,0]<-.16)&(world[:,0]>-.64);assert inside.any()
    pivot=vertices[np.flatnonzero(inside)[np.argmin(world[inside]@n)]]
    gap_initial=float((world[inside]@n+d).min())
    sagittal=[model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    q0,r0,e0=solve(model,q,feet,n,d,origin[0],pitch,gap=gap_initial,y=origin[1])
    bias=command[sagittal]-q0[sagittal]
    commands=[];qs=[];roots=[];errors=[]
    # Fixed25-knot contract matches the existing environment's interpolation.
    for u in np.linspace(0,1,25):
        alpha=smooth(u);beta=smooth((u-bias_delay)/(1-bias_delay))
        angle=pitch+(final_pitch-pitch)*alpha
        x=origin[0]+((R-Rotation.from_euler('y',angle).as_matrix())@pivot)[0]+rear_shift*alpha
        qi,root,error=solve(model,q,feet,n,d,x,angle,gap=gap_initial+(gap_final-gap_initial)*alpha,y=origin[1])
        assert error<.003,(float(u),error)
        c=command.copy();c[sagittal]=qi[sagittal]+(1-beta)*bias
        commands.append(c);qs.append(qi);roots.append(root);errors.append(error)
    commands=np.array(commands);commands[0]=command
    return commands,np.array(qs),dict(raw_normal=raw_n.tolist(),raw_offset=raw_d,normal=n.tolist(),offset=float(d),origin=origin.tolist(),pitch=float(pitch),pivot=pivot.tolist(),gap_initial=gap_initial,gap_final=gap_final,bias=bias.tolist(),bias_delay=bias_delay,rear_shift=rear_shift,target_root=roots[-1].tolist(),ik_error=max(errors),first_command_exact=bool(np.array_equal(commands[0],command)),first_knot_jump_rad=float(abs(commands[1]-commands[0]).max()))
