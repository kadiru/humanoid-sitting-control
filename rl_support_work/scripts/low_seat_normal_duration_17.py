"""Approach the measured plane normally after optional in-air orientation.

This separately versioned candidate avoids the previous large inward transfer.
It retains the bounded sensor-based load controller and all physical labels.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import level_rotation,apply,transform
from positive_slope_planner_v2 import plan as previous
from build_near_seat_candidates import solve
from sagittal_static_bias import static_bias
from crouch_rest_motion import smooth
from plan_seat_load_targets import clipped_vertices
from low_seat_contact_controller import selected
import low_seat_load_duration_17 as load

def plan(model,q,gravity,measurement,reference_origin,command,calibration,inertial,config):
    commands,qs,a=previous(model,q,gravity,measurement,reference_origin,command,calibration,inertial=inertial,
        target_x=-.18,target_pitch=.25,contact_gap=.001,land_fraction=.6,travel_fraction=1.)
    a['landing_repair_applied']=selected(a,config)
    if not a['landing_repair_applied']:return commands,qs,a
    R=level_rotation(gravity);fk=model.fk(q);origin=np.asarray(a['origin']);n=np.asarray(a['normal']);d=a['offset']
    feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
    ids=[model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    root0=np.r_[origin,Rotation.from_matrix(R).as_quat()];bias0,_=static_bias(model,inertial,q,root0,-.2,0.)
    offset=command[ids]-q[ids]-bias0
    fraction=config['orientation_fraction'];slope=np.rad2deg(np.arctan2(n[0],n[2]))
    goal=a['pitch'] if fraction==0 else (config['steep_pitch'] if slope>18 else config['pitch'])
    seatR=Rotation.from_euler('y',np.arctan2(n[0],n[2])).as_matrix();center=np.array([-.4,0.,(n[0]*.4-d)/n[2]])
    roots=[];statics=[];zero=[];full=[];shares=[];clear=[];margins=[];errors=[]
    for k,u in enumerate(np.linspace(0,1,25)):
        rotate=smooth(u/fraction) if fraction else 0.
        land=smooth((u-fraction)/(config['landing_fraction']-fraction))
        angle=a['pitch']+(goal-a['pitch'])*rotate
        gap=a['gap_initial']+(config['gap']-a['gap_initial'])*land
        # After orientation, translate along -normal: zero planned tangential speed.
        x=origin[0]-(a['gap_initial']-config['gap'])*n[0]*land
        qi,root,error=solve(model,q,feet,n,d,x,angle,gap=gap,y=origin[1]);assert error<.003,error
        T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);f=model.fk(qi)
        pelvis=apply(T,model.vertices['pelvis']);depth=pelvis@n+d;patch=depth<depth.min()+.002
        sx=float(pelvis[patch,0].mean());local=(pelvis-center)@seatR
        margin=float(min(.25-abs(local[patch,0]).max(),.45-abs(local[patch,1]).max()))
        b0,s0=static_bias(model,inertial,qi,root,sx,0.);b1,s1=static_bias(model,inertial,qi,root,sx,1.)
        scheduled=smooth((u-.85)/.15);bias=b0+scheduled*(b1-b0)
        commands[k]=command;commands[k,ids]=qi[ids]+bias+(1-smooth(u))*offset;qs[k]=qi
        values=[]
        for side in ['left','right']:
            for joint in ['hip_pitch','hip_roll','knee','ankle_pitch']:
                body=side+'_'+joint+'_link'
                if body not in model.vertices:continue
                v=(apply(T@f[body],model.vertices[body])-center)@seatR;v[:,0]-=.4;v=clipped_vertices(v)
                if len(v):values.append(float(v[:,2].min()))
        roots.append(root.tolist());statics.append(dict(s1,predicted_seat_bw=s1['predicted_seat_bw']*scheduled,
            torque=(bias*np.array([200,200,160]*2)).tolist()))
        zero.append(b0.tolist());full.append(b1.tolist());shares.append(s1['predicted_seat_bw'])
        clear.append(min(values) if values else 1.);margins.append(margin);errors.append(error)
    commands[0]=command
    a.update(root_path=roots,static_path=statics,target_root=roots[-1],chosen_pitch=goal,contact_gap=config['gap'],
        model_clearance_path=clear,model_patch_margin_path=margins,ik_error=max(errors),first_command_exact=True,
        landing_config=config,landing_repair_applied=True,normal_approach=True,
        load_feedback_basis=dict(zero=zero,full=full,max_share=shares,
            scheduled=[(np.asarray(s['torque'])/np.array([200,200,160]*2)).tolist() for s in statics]),
        certification='None; model screen only, physical validation required')
    return commands,qs,a

def install(harness,config):
    load.install(harness,config)
    bridge=harness.parent_harness.traction.bridge
    bridge.progressive.rolling_plan=lambda *a,**kw:plan(*a,**kw,inertial=bridge.inertial,config=config)
