"""Separate tangential prepositioning from normal landing using measured geometry."""
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import level_rotation,apply,transform
from positive_slope_planner_v2 import plan as previous
from crouch_rest_motion import smooth
from build_near_seat_candidates import solve
from sagittal_static_bias import static_bias
from plan_seat_load_targets import clipped_vertices

def plan(model,q,gravity,measurement,reference_origin,command,calibration,inertial,
         low_target_x=-.18,low_target_pitch=.25,low_gap=.001,travel_end=.45,
         landing_start=.55,landing_end=1.,load_start=.60,load_end=1.):
    commands,qs,a=previous(model,q,gravity,measurement,reference_origin,command,calibration,inertial=inertial,
        target_x=-.18,target_pitch=.25,contact_gap=.001,land_fraction=.6,travel_fraction=1.)
    if not a.get('positive_candidate_applied',False):return commands,qs,a
    n=np.array(a['normal']);d=a['offset'];origin=np.array(a['origin']);pitch=a['pitch']
    height=(n[0]*.4-d)/n[2]
    xgoal=low_target_x if height<.385 else -.18
    pgoal=low_target_pitch if height<.385 else .25
    gap=low_gap if height<.385 else .001
    R=level_rotation(gravity);fk=model.fk(q)
    feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
    ids=[model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    b0,_=static_bias(model,inertial,q,np.r_[origin,Rotation.from_matrix(R).as_quat()],-.2,0.)
    initial_bias=command[ids]-q[ids]-b0
    seatR=Rotation.from_euler('y',np.arctan2(n[0],n[2])).as_matrix();center=np.array([-.4,0.,height])
    roots=[];statics=[];clear=[];margins=[];errors=[]
    for k,u in enumerate(np.linspace(0,1,25)):
        travel=smooth(u/travel_end);land=smooth((u-landing_start)/(landing_end-landing_start))
        angle=pitch+(pgoal-pitch)*travel;x=origin[0]+(xgoal-origin[0])*travel
        qi,root,error=solve(model,q,feet,n,d,x,angle,gap=a['gap_initial']+(gap-a['gap_initial'])*land,y=origin[1])
        assert error<.003,dict(u=float(u),error=error,height=height)
        T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);f=model.fk(qi);values=[]
        for side in ['left','right']:
            for joint in ['hip_pitch','hip_roll']:
                body=side+'_'+joint+'_link';v=(apply(T@f[body],model.vertices[body])-center)@seatR;v[:,0]-=.4;v=clipped_vertices(v)
                if len(v):values.append(float(v[:,2].min()))
        pelvis=apply(T,model.vertices['pelvis']);local=(pelvis-center)@seatR;patch=local[local[:,2]<=local[:,2].min()+.002]
        margin=float(min(.25-abs(patch[:,0]).max(),.45-abs(patch[:,1]).max()))
        low=local[:,2]<=local[:,2].min()+.002;sx=float(pelvis[low,0].mean())
        bias,detail=static_bias(model,inertial,qi,root,sx,smooth((u-load_start)/(load_end-load_start)))
        commands[k]=command;commands[k,ids]=qi[ids]+bias+(1-smooth(u))*initial_bias;qs[k]=qi
        roots.append(root.tolist());statics.append(detail);clear.append(min(values) if values else 1.);margins.append(margin);errors.append(error)
    commands[0]=command
    a.update(low_positive_phased=True,root_path=roots,static_path=statics,target_root=roots[-1],chosen_pitch=pgoal,contact_gap=gap,model_clearance_path=clear,model_patch_margin_path=margins,endpoint_screen_passed=clear[-1]>=.006 and margins[-1]>=.02,ik_error=max(errors),first_command_exact=bool(np.array_equal(commands[0],command)),phase_contract=dict(travel_end=travel_end,landing_start=landing_start,landing_end=landing_end,load_start=load_start,load_end=load_end),candidate=dict(target_x=xgoal,target_pitch=pgoal,contact_gap=gap),certification='Kinematic screen only; no claimed physical clearance or no-slip guarantee')
    return commands,qs,a
