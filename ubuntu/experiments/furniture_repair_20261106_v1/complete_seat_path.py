"""Explicit actuator-rate scheduling and purely normal final landing.

Uses the unchanged .003rad/20ms action contract, including its .1 filter.
Unlike the previous conservative arc estimate, this constructs every command.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from sagittal_static_bias import static_bias

DT=.02; STEP=.003; FILTER=.1; RATE=STEP*.99999/DT

def arc_clock(t, length):
    ramp=1.; duration=length/RATE+ramp
    if length<RATE*ramp:raise ValueError('SHORT_ARC_REQUIRES_SEPARATE_SCHEDULE')
    if t<ramp:distance=.5*RATE*t*t/ramp
    elif t<duration-ramp:distance=.5*RATE*ramp+RATE*(t-ramp)
    else:distance=length-.5*RATE*max(0.,duration-t)**2/ramp
    return float(np.clip(distance,0,length)),duration

def interpolate(values, axis, point):
    k=min(max(np.searchsorted(axis,point,side='right')-1,0),len(axis)-2)
    fraction=np.clip((point-axis[k])/max(axis[k+1]-axis[k],1e-12),0,1)
    return (1-fraction)*values[k]+fraction*values[k+1]

def complete(plan, construction):
    approach=np.array([n['command'] for n in construction['nodes']])
    arc=np.asarray(construction['command_arc']);length=arc[-1]
    duration=arc_clock(0,length)[1]
    commands=[];latent=[];previous=approach[0].copy()
    for tick in range(int(np.ceil(duration/DT))):
        s,_=arc_clock(min((tick+1)*DT,duration),length)
        target=interpolate(approach,arc,s)
        desired=previous+(target-previous)/FILTER
        applied=previous+np.clip(FILTER*(desired-previous),-STEP,STEP)
        if np.max(abs(applied-target))>1e-10:raise ValueError('FILTER_REALIZATION_FAILED')
        commands.append(applied);latent.append(desired);previous=applied
    end=construction['nodes'][-1];root=np.array(end['root']);q=np.array(end['q'])
    plan.arm_reach=1.
    arm=np.array([q[plan.shoulder_ids].mean(),q[plan.elbow_ids].mean()])
    pitch=Rotation.from_matrix(root[:3,:3]).as_euler('xyz')[1]
    seed=q[plan.legids].copy();land=[]
    for travel in np.linspace(0,.015,16):
        xyz=root[:3,3]-travel*plan.n
        def state(values):return plan.state(np.r_[values,xyz[0],xyz[2],pitch,arm])
        def error(values):
            qi,rt,fk,*_=state(values);rows=[]
            for side,anchor in zip(('left','right'),plan.feet):
                foot=rt@fk[side+'_ankle_roll_link'];rows.extend((foot[:3,3]-anchor[:3,3])[[0,2]])
                rows.append(.2*Rotation.from_matrix(anchor[:3,:3].T@foot[:3,:3]).as_rotvec()[1])
            return rows
        fit=least_squares(error,seed,bounds=(plan.lower,plan.upper),max_nfev=60,xtol=1e-9,ftol=1e-9,gtol=1e-9)
        if np.max(abs(fit.fun))>.002:raise ValueError('NORMAL_LANDING_IK_REJECTED')
        seed=fit.x;qi,rt,fk,pelvis,h,tip,com=state(seed)
        floor_root=rt.copy();floor_root[2,3]-=plan.floor
        pose=np.r_[floor_root[:3,3],Rotation.from_matrix(rt[:3,:3]).as_quat()]
        zero,s0=static_bias(plan.model,plan.inertia,qi,pose,float(tip[0]),0.)
        full,s1=static_bias(plan.model,plan.inertia,qi,pose,float(tip[0]),1.)
        cmd=np.array(end['command']);cmd[plan.legids]=qi[plan.legids]+zero
        land.append(dict(travel_m=float(travel),q=qi.tolist(),root=rt.tolist(),command=cmd.tolist(),
                         zero=zero.tolist(),full=full.tolist(),share=s1['predicted_seat_bw'],gap=float(h.min())))
    # This is an executable nominal approach. The real quiet gate may take longer;
    # failure to finish within the original horizon remains a failed demonstration.
    return dict(approach_commands=np.asarray(commands).tolist(),approach_desired=np.asarray(latent).tolist(),
                approach_seconds=len(commands)*DT,landing=land,normal_speed_m_s=.006,
                quiet_current_ticks=25,load_settle_seconds=1.,
                nominal_latest_settled_s=len(commands)*DT+.5+.015/.006+1.,
                original_rate_limit_rad_s=.15,rate_limit_changed=False,physical_success=False)
