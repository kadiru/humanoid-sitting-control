"""Development-only arm counterbalance on unresolved low steep seats.

The pelvis/foot trajectory and feedback gains stay at guarded normal landing.
Arm posture changes the model COM and corresponding leg static feedforward.
No simulator contact/pose labels enter control. Physical admission is pending.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import apply,transform
from sagittal_static_bias import static_bias
from crouch_rest_motion import smooth
import low_seat_normal_timing_controller as normal
import low_seat_load_controller as load

def plan(*args,inertial,config):
    commands,qs,a=normal.plan(*args,inertial=inertial,config=config)
    n=np.asarray(a['normal']);height=(n[0]*.4-a['offset'])/n[2]
    slope=np.rad2deg(np.arctan2(n[0],n[2]))
    active=bool(a.get('landing_repair_applied') and height<.38 and slope>14.)
    a['counterbalance_probe']=dict(applied=active,arm_offset_rad=-.35,duration_s=3.5,
        scope='Measured low steep development surfaces only; reference shoulder/elbow pitch changes, not learned residuals. No physical success implied.')
    if not active:return commands,qs,a
    model=args[0];initial_command=args[5]
    arm=[model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['shoulder_pitch','elbow_pitch']]
    leg=[model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    a['counterbalance_probe']['reference_arm_before']=commands[:,arm].tolist()
    records=[];zero=[];full=[];maxshare=[];scheduled=[];statics=[]
    for k,u in enumerate(np.linspace(0,1,25)):
        delta=-.35*smooth(u/.7)
        oldq=qs[k].copy();qs[k,arm]+=delta;commands[k,arm]+=delta
        root=np.asarray(a['root_path'][k]);T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3])
        pelvis=apply(T,model.vertices['pelvis']);depth=pelvis@n+a['offset'];sx=float(pelvis[depth<depth.min()+.002,0].mean())
        b0,s0=static_bias(model,inertial,qs[k],root,sx,0.)
        b1,s1=static_bias(model,inertial,qs[k],root,sx,1.)
        _,oldst=static_bias(model,inertial,oldq,root,sx,0.)
        blend=smooth((u-.85)/.15);bias=b0+blend*(b1-b0)
        commands[k,leg]+=bias-np.asarray(a['load_feedback_basis']['scheduled'][k])
        zero.append(b0.tolist());full.append(b1.tolist());maxshare.append(s1['predicted_seat_bw']);scheduled.append(bias.tolist())
        statics.append(dict(s1,predicted_seat_bw=s1['predicted_seat_bw']*blend,
            torque=(bias*np.array([200.,200.,160.]*2)).tolist()))
        records.append(dict(u=float(u),arm_delta_rad=float(delta),before_com_x=oldst['com'][0],after_com_x=s0['com'][0],
            no_seat_cop_before=oldst['foot_cop_relative_to_ankle'],no_seat_cop_after=s0['foot_cop_relative_to_ankle']))
    commands[0]=initial_command
    a['static_path']=statics
    a['load_feedback_basis']=dict(zero=zero,full=full,max_share=maxshare,scheduled=scheduled)
    a['counterbalance_probe'].update(model_path=records,arm_joint_ids=arm,
        maximum_arm_offset_rad=.35,planned_arm_peak_rate_rad_s=.15,
        limitation='Housing clearance is not enlarged by this arm motion; it tests reduced backward tipping. Full physical checks remain mandatory.')
    return commands,qs,a

def install(harness,config):
    load.install(harness,config)
    bridge=harness.parent_harness.traction.bridge
    bridge.progressive.rolling_plan=lambda *a,**kw:plan(*a,**kw,inertial=bridge.inertial,config=config)
