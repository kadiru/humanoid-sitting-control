"""Roll the pelvis about its measured-plane contact edge instead of dragging it."""
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import level_rotation,apply,transform
from positive_slope_planner_v2 import plan as previous
from build_near_seat_candidates import solve
from sagittal_static_bias import static_bias
from crouch_rest_motion import smooth
from plan_seat_load_targets import clipped_vertices

def plan(model,q,gravity,measurement,reference_origin,command,calibration,inertial,
         final_pitch=-.2,steep_pitch=-.35,land_fraction=.6,contact_gap=.001):
    commands,qs,a=previous(model,q,gravity,measurement,reference_origin,command,calibration,inertial=inertial,target_x=-.18,target_pitch=.25,contact_gap=.001,land_fraction=.6,travel_fraction=1.)
    if not a.get('positive_candidate_applied',False):return commands,qs,a
    n=np.array(a['normal']);d=a['offset'];origin=np.array(a['origin']);R=level_rotation(gravity);fk=model.fk(q);pitch=a['pitch']
    vertices=model.vertices['pelvis'];world=vertices@R.T+origin;inside=(world[:,0]<-.16)&(world[:,0]>-.64)
    pivot=vertices[np.flatnonzero(inside)[np.argmin(world[inside]@n)]]
    feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
    ids=[model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    b0,_=static_bias(model,inertial,q,np.r_[origin,Rotation.from_matrix(R).as_quat()],-.2,0.);bias0=command[ids]-q[ids]-b0
    height=(n[0]*.4-d)/n[2];slope=np.arctan2(n[0],n[2]);goal=steep_pitch if height<.385 and slope>np.deg2rad(18) else final_pitch
    seatR=Rotation.from_euler('y',slope).as_matrix();center=np.array([-.4,0.,height])
    roots=[];statics=[];errors=[];clear=[];margins=[]
    for k,u in enumerate(np.linspace(0,1,25)):
        angle=pitch+(goal-pitch)*smooth(u);x=origin[0]+((R-Rotation.from_euler('y',angle).as_matrix())@pivot)[0]
        qi,root,error=solve(model,q,feet,n,d,x,angle,gap=a['gap_initial']+(contact_gap-a['gap_initial'])*smooth(u/land_fraction),y=origin[1]);assert error<.003,error
        T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);f=model.fk(qi);pelvis=apply(T,vertices);local=(pelvis-center)@seatR;patch=local[:,2]<=local[:,2].min()+.002
        margin=float(min(.25-abs(local[patch,0]).max(),.45-abs(local[patch,1]).max()));sx=float(pelvis[patch,0].mean())
        bias,detail=static_bias(model,inertial,qi,root,sx,smooth((u-.2)/.45))
        commands[k]=command;commands[k,ids]=qi[ids]+bias+(1-smooth(u))*bias0;qs[k]=qi
        c=[]
        for side in ['left','right']:
            for joint in ['hip_pitch','hip_roll']:
                body=side+'_'+joint+'_link';v=(apply(T@f[body],model.vertices[body])-center)@seatR;v[:,0]-=.4;v=clipped_vertices(v)
                if len(v):c.append(float(v[:,2].min()))
        clear.append(min(c) if c else 1.);margins.append(margin);roots.append(root.tolist());statics.append(detail);errors.append(error)
    commands[0]=command
    a.update(root_path=roots,static_path=statics,target_root=roots[-1],chosen_pitch=goal,model_clearance_path=clear,model_patch_margin_path=margins,endpoint_screen_passed=clear[-1]>=.006 and margins[-1]>=.02,ik_error=max(errors),contact_gap=contact_gap,rolling_contact=dict(pivot=pivot.tolist(),target_pitch=goal,tangential_drag_offset=0.),first_command_exact=bool(np.array_equal(commands[0],command)),certification='Geometric rolling candidate; no physical no-slip claim until validated')
    return commands,qs,a
