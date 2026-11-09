"""Sensor/model-only early approach, hover, then normal landing candidate.

The 3 mm model clearance screen is not a physical safety certificate. The 7 mm
reserve searched in33/34 was not achieved on all hard surfaces. This candidate
tests settling before touchdown, retaining the tested-.35rad arm offset.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import apply, transform, level_rotation
from build_near_seat_candidates import solve
from sagittal_static_bias import static_bias
from plan_seat_load_targets import clipped_vertices
from low_seat_joint_path_controller_17 import measured

DURATION=14.; START=1.; DT=DURATION/24
APPROACH_END=19*DT; LAND_START=20*DT; LAND_END=23*DT

def eligible(plane):return bool(plane[3]<.38 and 14.<plane[4]<21.)

def plan(model,q,gravity,geometry,reference,command,cal,inertial,config):
    q=np.asarray(q,dtype=float);command=np.asarray(command,dtype=float)
    assert geometry[11] and geometry[12]<=.5
    n,d,origin,h,slope,rn,rd=measured(model,q,gravity,geometry,reference,cal)
    assert h<.38 and 14.<slope<21.
    R=level_rotation(gravity);fk=model.fk(q)
    feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
    legs=[model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    arms=[model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['shoulder_pitch','elbow_pitch']]
    seed=q.copy();seed[arms]-=.35
    SR=Rotation.from_euler('y',np.deg2rad(slope)).as_matrix();center=np.array([-.4,0,h])
    def endpoint(x,gap=.001):
        qi,root,err=solve(model,seed,feet,n,d,x,.9,gap=gap,y=origin[1])
        T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);f=model.fk(qi)
        com=np.average([apply(T@f[b],c) for b,c in zip(inertial['body_names'],inertial['local_com'])],axis=0,weights=inertial['mass'])
        feet_vertices=np.concatenate([apply(T@f[s+'_ankle_roll_link'],model.vertices[s+'_ankle_roll_link']) for s in ['left','right']])
        sole=feet_vertices[feet_vertices[:,2]<feet_vertices[:,2].min()+.003]
        return qi,root,err,float(com[0]-sole[:,0].min())
    # A deterministic scalar model solve, not a geometry-index lookup.
    assert endpoint(-.25)[3]<.0055<endpoint(-.1)[3]
    x=float(brentq(lambda v:endpoint(v)[3]-.0055,-.25,-.1,xtol=1e-9))
    end,endroot,enderr,_=endpoint(x)
    hover,hoverroot,hovererr,_=endpoint(x+.019*n[0],.02)
    root0=np.r_[origin,Rotation.from_matrix(R).as_quat()]
    bias0,_=static_bias(model,inertial,q,root0,-.2,0.)
    preload=command[legs]-q[legs]-bias0
    def point(qi,pre,screen=False):
        f=model.fk(qi)
        rotations=[feet[s][1]@f[s+'_ankle_roll_link'][:3,:3].T for s in ['left','right']]
        rr=rotations[0];positions=[feet[s][0]-rr@f[s+'_ankle_roll_link'][:3,3] for s in ['left','right']]
        xyz=np.mean(positions,axis=0);T=transform(rr,xyz);root=np.r_[xyz,Rotation.from_matrix(rr).as_quat()]
        pelvis=apply(T,model.vertices['pelvis']);depth=pelvis@n+d;patch=depth<depth.min()+.002;local=(pelvis-center)@SR;sx=float(pelvis[patch,0].mean())
        b0,s0=static_bias(model,inertial,qi,root,sx,0.)
        c=command.copy();c[legs]=qi[legs]+b0+pre*preload;c[arms]+=qi[arms]-q[arms]
        info={}
        if screen:
            foot=np.concatenate([apply(T@f[s+'_ankle_roll_link'],model.vertices[s+'_ankle_roll_link']) for s in ['left','right']]);sole=foot[foot[:,2]<foot[:,2].min()+.003]
            clear=[]
            for name,verts in model.vertices.items():
                if name in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
                v=(apply(T@f[name],verts)-center)@SR;v[:,0]-=.4;v=clipped_vertices(v)
                if len(v):clear.append((float(v[:,2].min()),name))
            info=dict(gap=float(depth.min()),clearance=min(clear)[0],nearest_body=min(clear)[1],patch=float(min(.25-abs(local[patch,0]).max(),.45-abs(local[patch,1]).max())),cop=float(min(s0['foot_cop_x']-sole[:,0].min(),sole[:,0].max()-s0['foot_cop_x'])),foot_position_error=float(np.linalg.norm(positions[0]-positions[1])),foot_rotation_error=float(np.linalg.norm(Rotation.from_matrix(rotations[0].T@rotations[1]).as_rotvec())))
        return c,qi,root,b0,s0,sx,info
    grid=np.linspace(0,1,97);dense=[point(q+v*(hover-q),1-v,True) for v in grid]
    commands=np.array([v[0] for v in dense]);commands[0]=command
    arc=np.r_[0.,np.cumsum(np.max(abs(np.diff(commands,axis=0)),axis=1))]
    peak=float(arc[-1]/(APPROACH_END-.25))
    landing=[];errs=[enderr,hovererr]
    for v in np.linspace(0,1,49):
        gap=.02-v*.019;qi,r,e,_=endpoint(x+(gap-.001)*n[0],gap);errs.append(e);landing.append(point(qi,0.,True))
    checks=[v[-1] for v in dense+landing];near=[v for v in checks if v['gap']<.025]
    screen=dict(peak_nominal_approach_rate_rad_s=peak,minimum_clearance_m=min(v['clearance'] for v in checks),minimum_gap_m=min(v['gap'] for v in checks),minimum_near_patch_margin_m=min(v['patch'] for v in near),minimum_no_seat_cop_margin_m=min(v['cop'] for v in checks),max_foot_position_error_m=max(v['foot_position_error'] for v in checks),max_foot_rotation_error_rad=max(v['foot_rotation_error'] for v in checks),max_ik_error=max(errs))
    assert peak<=.175 and screen['minimum_clearance_m']>=.003 and screen['minimum_gap_m']>=0 and screen['minimum_near_patch_margin_m']>=.015 and screen['minimum_no_seat_cop_margin_m']>=.005 and screen['max_foot_position_error_m']<.003 and screen['max_foot_rotation_error_rad']<.003 and max(errs)<.003,screen
    cs=[];qs=[];roots=[];zero=[];full=[];shares=[];statics=[];phases=[]
    for k,elapsed in enumerate(np.linspace(0,DURATION,25)):
        if k<=19:
            if elapsed<.25:distance=.5*elapsed**2/.25
            elif elapsed>APPROACH_END-.25:distance=APPROACH_END-.25-.5*(APPROACH_END-elapsed)**2/.25
            else:distance=elapsed-.125
            f=float(np.interp(distance/(APPROACH_END-.25)*arc[-1],arc,grid))
            c,qi,r,b0,s0,sx,_=point(q+f*(hover-q),1-f)
            c=np.array([np.interp(f,grid,commands[:,j]) for j in range(len(q))])
            effective=c[legs]-qi[legs]-(1-f)*preload;phase='approach'
        elif k==20:
            c,qi,r,b0,s0,sx,_=point(hover,0.);effective=b0;phase='hover'
        else:
            f=min(1.,(k-20)/3);f=f*f*(3-2*f);gap=.02-.019*f
            qi,r,e,_=endpoint(x+(gap-.001)*n[0],gap)
            c,qi,r,b0,s0,sx,_=point(qi,0.);effective=b0;phase='normal_descent' if k<=23 else 'settled'
        b1,s1=static_bias(model,inertial,qi,r,sx,1.)
        cs.append(c);qs.append(qi);roots.append(r.tolist());zero.append(effective.tolist());full.append(b1.tolist());shares.append(s1['predicted_seat_bw']);phases.append(phase)
        statics.append(dict(s1,predicted_seat_bw=0.,torque=(effective*np.array([200,200,160]*2)).tolist()))
    cs=np.array(cs);qs=np.array(qs);cs[0]=command;qs[0]=q
    screen['maximum_resampled_command_rate_rad_s']=float(abs(np.diff(cs,axis=0)).max()/DT)
    assert screen['maximum_resampled_command_rate_rad_s']<=.175001
    audit=dict(raw_normal=rn.tolist(),raw_offset=rd,normal=n.tolist(),offset=d,origin=origin.tolist(),pitch=float(Rotation.from_matrix(R).as_euler('xyz')[1]),gap_initial=checks[0]['gap'],root_path=roots,static_path=statics,target_root=roots[-1],chosen_pitch=.9,contact_gap=.001,ik_error=max(errs),first_command_exact=True,landing_config=config,landing_repair_applied=True,positive_candidate_applied=True,
        load_feedback_basis=dict(zero=zero,full=full,max_share=shares,scheduled=zero),
        staged_36=dict(start_s=START,duration_s=DURATION,approach_end_s=START+APPROACH_END,normal_descent_start_s=START+LAND_START,normal_descent_end_s=START+LAND_END,arm_offset_rad=-.35,model_endpoint_x=x,screen=screen,phases=phases,runtime_inputs=dict(q=q.tolist(),gravity=np.asarray(gravity).tolist(),geometry=np.asarray(geometry).tolist(),reference=np.asarray(reference).tolist(),command=command.tolist())),
        certification='None. Model3mm clearance/5mm foot support screens only.7mm research reserve not met everywhere. Full physical validation remains mandatory; no physical criterion change.')
    return cs,qs,audit
