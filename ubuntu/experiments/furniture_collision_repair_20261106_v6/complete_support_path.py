"""Same normal stroke/rate/filter, applied to the actual legal support witness."""
import numpy as np
from scipy.spatial.transform import Rotation
from sagittal_static_bias import static_bias
from complete_seat_path import arc_clock,interpolate,DT,STEP,FILTER


def complete(plan,construction):
    approach=np.array([n['command'] for n in construction['nodes']]);arc=np.array(construction['command_arc'])
    duration=arc_clock(0,arc[-1])[1];commands=[];desired=[];previous=approach[0].copy()
    for tick in range(int(np.ceil(duration/DT))):
        s,_=arc_clock(min((tick+1)*DT,duration),arc[-1]);target=interpolate(approach,arc,s)
        latent=previous+(target-previous)/FILTER
        applied=previous+np.clip(FILTER*(latent-previous),-STEP,STEP)
        if abs(applied-target).max()>1e-10:raise ValueError('FILTER_REALIZATION_FAILED')
        commands.append(applied);desired.append(latent);previous=applied
    end=construction['nodes'][-1];q=np.array(end['q']);rt=np.array(end['root'])
    seed=np.r_[q[plan.legids],rt[0,3],rt[2,3],Rotation.from_matrix(rt[:3,:3]).as_euler('xyz')[1],q[plan.shoulder_ids].mean(),q[plan.elbow_ids].mean()]
    plan.arm_reach=plan.progress=1.;plan.landing_active=True;land=[]
    for travel in np.linspace(0,.015,16):
        seed=plan.fit(seed,.012-travel,plan.contact_target,end['balance'])
        qi,root,fk,points,height,tip,com=plan.state(seed)
        floor_root=root.copy();floor_root[2,3]-=plan.floor
        pose=np.r_[floor_root[:3,3],Rotation.from_matrix(root[:3,:3]).as_quat()]
        zero,_=static_bias(plan.model,plan.inertia,qi,pose,float(tip[0]),0.)
        cmd=np.array(end['command']);cmd[plan.legids]=qi[plan.legids]+zero
        # Retain arm gravitational preloads while following the solved small
        # arm changes. The rollout never commands a force-derived unloading.
        cmd[plan.armids]+=qi[plan.armids]-q[plan.armids]
        land.append(dict(travel_m=float(travel),q=qi.tolist(),root=root.tolist(),command=cmd.tolist(),
                         zero=zero.tolist(),gap=float(height.min()),support_tip=tip.tolist(),
                         load_feedback_used=False))
    return dict(approach_commands=np.asarray(commands).tolist(),approach_desired=np.asarray(desired).tolist(),
                approach_seconds=len(commands)*DT,landing=land,normal_speed_m_s=.006,quiet_current_ticks=25,
                load_settle_seconds=1.,nominal_latest_settled_s=len(commands)*DT+.5+.015/.006+1.,
                original_rate_limit_rad_s=.15,rate_limit_changed=False,physical_success=False)
