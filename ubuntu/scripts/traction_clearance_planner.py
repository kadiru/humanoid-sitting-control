"""Choose the most upright model-clear rolling endpoint, with feet fixed.

All inputs are static robot geometry, encoders/IMU, measured seat plane and the
explicit reference placement. Simulator contacts and live pose are not inputs.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import level_rotation,apply,transform
from crouch_rest_motion import continuous_plan,smooth
from build_near_seat_candidates import solve
from sagittal_static_bias import static_bias
from plan_seat_load_targets import clipped_vertices

def plan(model,q,gravity,measurement,reference_origin,command,calibration,inertial,clearance=.006,gap=.001):
    commands,qs,a=continuous_plan(model,q,gravity,measurement,reference_origin,command,calibration)
    R=level_rotation(gravity);fk=model.fk(q);origin=np.array(a['origin']);n=np.array(a['normal']);d=a['offset'];pitch=a['pitch'];pivot=np.array(a['pivot'])
    feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
    ids=[model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    def pose(angle,contact_gap):
        x=origin[0]+((R-Rotation.from_euler('y',angle).as_matrix())@pivot)[0]
        qi,root,error=solve(model,q,feet,n,d,x,angle,gap=contact_gap,y=origin[1]);assert error<.003,error
        T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);f=model.fk(qi);values=[]
        for side in ['left','right']:
            for joint in ['hip_pitch','hip_roll']:
                b=side+'_'+joint+'_link';v=clipped_vertices(apply(T@f[b],model.vertices[b]))
                if len(v):values.append(float((v@n+d).min()))
        return qi,root,min(values) if values else 1.
    candidates=[];goal=pitch
    for angle in np.linspace(.1,pitch,17):
        qi,root,cl=pose(angle,gap);candidates.append(dict(pitch=float(angle),clearance=cl))
        if cl>=clearance:goal=float(angle);break
    predicted0,_=static_bias(model,inertial,q,np.r_[origin,Rotation.from_matrix(R).as_quat()],-.2,0.);offset=command[ids]-q[ids]-predicted0
    roots=[];statics=[];clear=[]
    for k,u in enumerate(np.linspace(0,1,25)):
        land=smooth(u/.1);roll=smooth((u-.1)/.9);angle=pitch+(goal-pitch)*roll
        qi,root,cl=pose(angle,a['gap_initial']+(gap-a['gap_initial'])*land)
        world=Rotation.from_quat(root[3:]).apply(model.vertices['pelvis'])+root[:3];v=world[(world[:,0]<-.16)&(world[:,0]>-.64)];minimum=(v@n+d).min();sx=float(v[v@n+d<minimum+.002,0].mean())
        bias,detail=static_bias(model,inertial,qi,root,sx,smooth((u-.05)/.15))
        commands[k]=command;commands[k,ids]=qi[ids]+bias+(1-smooth(u))*offset;qs[k]=qi;roots.append(root.tolist());statics.append(detail);clear.append(cl)
    commands[0]=command
    a.update(root_path=roots,static_path=statics,target_root=roots[-1],first_command_exact=bool(np.array_equal(commands[0],command)),clearance_candidates=candidates,model_clearance_path=clear,chosen_pitch=goal,requested_clearance=clearance,contact_gap=gap,endpoint_screen_passed=candidates[-1]['clearance']>=clearance)
    return commands,qs,a
