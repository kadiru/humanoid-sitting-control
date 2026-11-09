"""Frozen-candidate bridge planner, extracted unchanged from the validated probe.

Uses robot model, encoders, IMU, depth plane and explicit reference placement.
Static force allocation is a prediction, not contact truth.
"""
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from crouch_rest_motion import continuous_plan,smooth
from build_near_seat_candidates import solve
from sagittal_static_bias import static_bias
from humanoid_sitting.shared_frame import level_rotation
static_args=SimpleNamespace(settle_fraction=.1,goal_x=-.18,goal_pitch=.1,contact_gap=-.001,contact_ramp=.15)

def plan(model,q,gravity,measurement,reference_origin,command,calibration,inertial):
    commands,qs,a=continuous_plan(model,q,gravity,measurement,reference_origin,command,calibration)
    R=level_rotation(gravity);fk=model.fk(q);origin=np.array(a['origin']);n=np.array(a['normal']);d=a['offset'];pitch=a['pitch']
    feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
    ids=[model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    root0=np.r_[origin,Rotation.from_matrix(R).as_quat()]
    predicted0,_=static_bias(model,inertial,q,root0,-.2,0.)
    offset=command[ids]-q[ids]-predicted0
    statics=[];roots=[]
    for k,u in enumerate(np.linspace(0,1,25)):
        land=smooth(u/static_args.settle_fraction);roll=smooth((u-static_args.settle_fraction)/(1-static_args.settle_fraction))
        angle=pitch+(static_args.goal_pitch-pitch)*roll
        x=origin[0]+(static_args.goal_x-origin[0])*roll
        gap=a['gap_initial']+(static_args.contact_gap-a['gap_initial'])*land
        qi,root,error=solve(model,q,feet,n,d,x,angle,gap=gap,y=origin[1]);assert error<.003,(u,error)
        world=Rotation.from_quat(root[3:]).apply(model.vertices['pelvis'])+root[:3];inside=(world[:,0]<-.16)&(world[:,0]>-.64)
        vertices=world[inside];minimum=(vertices@n+d).min();patch=vertices[vertices@n+d<minimum+.002];sx=float(patch[:,0].mean())
        fraction=smooth((u-static_args.settle_fraction*.5)/static_args.contact_ramp)
        bias,detail=static_bias(model,inertial,qi,root,sx,fraction)
        commands[k]=command;commands[k,ids]=qi[ids]+bias+(1-smooth(u))*offset
        qs[k]=qi;roots.append(root.tolist());statics.append(detail)
    commands[0]=command
    a.update(static_config=vars(static_args),root_path=roots,static_path=statics,target_root=roots[-1],first_command_exact=bool(np.array_equal(commands[0],command)),first_knot_jump_rad=float(abs(commands[1]-commands[0]).max()))
    return commands,qs,a
