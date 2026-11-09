"""Delay presumed seat support while retaining the geometric reference path."""
import numpy as np
from positive_slope_planner_v2 import plan as previous
from positive_slope_planner import plan as candidate
from sagittal_static_bias import static_bias
from crouch_rest_motion import smooth
from humanoid_sitting.shared_frame import apply,transform
from scipy.spatial.transform import Rotation

def plan(model,q,gravity,measurement,reference_origin,command,calibration,inertial,
         load_start=.4,load_end=.85,low_target_x=None,low_target_pitch=.1,low_gap=.01):
    args=(model,q,gravity,measurement,reference_origin,command,calibration)
    commands,qs,a=previous(*args,inertial=inertial,target_x=-.18,target_pitch=.25,contact_gap=.001,land_fraction=.6,travel_fraction=1.)
    if not a.get('positive_candidate_applied',False):return commands,qs,a
    n=np.array(a['normal']);height=(n[0]*.4-a['offset'])/n[2]
    if low_target_x is not None and height<.385:
        commands,qs,a=candidate(*args,inertial=inertial,target_x=low_target_x,target_pitch=low_target_pitch,contact_gap=low_gap,land_fraction=.6,travel_fraction=1.)
    old_statics=a['static_path'];ids=[model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    stiffness=np.array([200.,200.,160.]*2);new_statics=[]
    for k,u in enumerate(np.linspace(0,1,25)):
        root=np.array(a['root_path'][k]);T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);pelvis=apply(T,model.vertices['pelvis'])
        depth=pelvis@n+a['offset'];patch=depth<=depth.min()+.002;seat_x=float(pelvis[patch,0].mean())
        bias,detail=static_bias(model,inertial,qs[k],root,seat_x,smooth((u-load_start)/(load_end-load_start)))
        commands[k,ids]+=bias-np.array(old_statics[k]['torque'])/stiffness;new_statics.append(detail)
    commands[0]=command
    a.update(original_static_path=old_statics,static_path=new_statics,load_timing=dict(start=load_start,end=load_end),first_command_exact=bool(np.array_equal(commands[0],command)))
    return commands,qs,a
