"""Development candidate: keep the contact patch inside a positive sloping seat.

Sensor/kinematic inputs only. A model screen is diagnostic, not certification.
The inherited mild-slope behavior is left exactly unchanged.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import level_rotation,apply,transform
from traction_clearance_planner_v2 import plan as original
from crouch_rest_motion import smooth
from build_near_seat_candidates import solve
from sagittal_static_bias import static_bias
from plan_seat_load_targets import clipped_vertices

def plan(model,q,gravity,measurement,reference_origin,command,calibration,inertial,
         target_x=-.18,target_pitch=.25,contact_gap=.001,land_fraction=.6,travel_fraction=1.):
    commands,qs,a=original(model,q,gravity,measurement,reference_origin,command,calibration,inertial)
    n=np.array(a['normal']);slope=float(np.arctan2(n[0],n[2]));a['positive_candidate_applied']=False
    if slope<=np.deg2rad(10.5):return commands,qs,a
    R=level_rotation(gravity);fk=model.fk(q);origin=np.array(a['origin']);d=a['offset'];pitch=a['pitch']
    feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
    ids=[model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    pred,_=static_bias(model,inertial,q,np.r_[origin,Rotation.from_matrix(R).as_quat()],-.2,0.)
    offset=command[ids]-q[ids]-pred;roots=[];statics=[];clearances=[];patch_margins=[];bias_none=[];bias_full=[];full_shares=[]
    seatR=Rotation.from_euler('y',slope).as_matrix();center=np.array([-.4,0.,(n[0]*.4-d)/n[2]])
    for k,u in enumerate(np.linspace(0,1,25)):
        travel=smooth(u/travel_fraction);land=smooth(u/land_fraction)
        angle=pitch+(target_pitch-pitch)*travel;x=origin[0]+(target_x-origin[0])*travel
        qi,root,error=solve(model,q,feet,n,d,x,angle,gap=a['gap_initial']+(contact_gap-a['gap_initial'])*land,y=origin[1]);assert error<.003,error
        T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);f=model.fk(qi);values=[]
        for side in ['left','right']:
            for joint in ['hip_pitch','hip_roll']:
                b=side+'_'+joint+'_link';v=(apply(T@f[b],model.vertices[b])-center)@seatR;v[:,0]-=.4;v=clipped_vertices(v)
                if len(v):values.append(float(v[:,2].min()))
        pelvis=apply(T,model.vertices['pelvis']);local=(pelvis-center)@seatR
        low=local[:,2]<=local[:,2].min()+.002;patch=local[low]
        patch_margins.append(float(min(.25-abs(patch[:,0]).max(),.45-abs(patch[:,1]).max())))
        sx=float(pelvis[low,0].mean())
        bias,detail=static_bias(model,inertial,qi,root,sx,smooth((u-.20)/.45))
        bn,_=static_bias(model,inertial,qi,root,sx,0.)
        bf,df=static_bias(model,inertial,qi,root,sx,1.)
        bias_none.append(bn.tolist());bias_full.append(bf.tolist());full_shares.append(df['predicted_seat_bw'])
        commands[k]=command;commands[k,ids]=qi[ids]+bias+(1-smooth(u))*offset;qs[k]=qi
        roots.append(root.tolist());statics.append(detail);clearances.append(min(values) if values else 1.)
    commands[0]=command
    a.update(positive_candidate_applied=True,bias_noseat_path=bias_none,bias_fullseat_path=bias_full,fullseat_share_path=full_shares,root_path=roots,static_path=statics,target_root=roots[-1],
        chosen_pitch=target_pitch,model_clearance_path=clearances,model_patch_margin_path=patch_margins,
        endpoint_screen_passed=clearances[-1]>=.006 and patch_margins[-1]>=.02,
        first_command_exact=bool(np.array_equal(commands[0],command)),contact_gap=contact_gap,
        candidate=dict(target_x=target_x,target_pitch=target_pitch,land_fraction=land_fraction,travel_fraction=travel_fraction),
        certification='None; full physical validation required; no blind claim from geometry screen')
    return commands,qs,a
