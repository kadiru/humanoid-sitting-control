"""Named pelvis/back separation seed; not a motion or balance certificate."""
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import apply,transform

def ry(a):return Rotation.from_euler('y',a).as_matrix()

def chains(plan):
    result={}
    for side,anchor in zip(('left','right'),plan.feet):
        js=[plan.joints[side+'_'+n+'_joint'] for n in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
        assert all(np.max(np.abs(np.asarray(j['p1'],float)))==0 for j in js)
        offsets=[Rotation.from_quat(j['q0']).as_euler('xyz')[1] for j in (js[0],js[3],js[4])]
        upper=sum((np.asarray(j['p0']) for j in js[1:4]),np.zeros(3));lower=np.asarray(js[4]['p0'])
        roll=plan.q0[plan.model.names.index(side+'_ankle_roll_joint')]
        ar=anchor[:3,:3]@Rotation.from_rotvec([-roll,0,0]).as_matrix()
        ap=anchor[:3,3]-ar@np.asarray(js[5]['p0'])
        result[side]=(js,offsets,upper,lower,ar,ap)
    return result

def pitch_ik(plan,root,chain):
    q=plan.q0.copy()
    for side,(js,offsets,u,v,ar,ap) in chain.items():
        target=root[:3,:3].T@(ap-root[:3,3])-np.asarray(js[0]['p0'])
        l1=np.linalg.norm(u[[0,2]]);l2=np.linalg.norm(v[[0,2]])
        cosine=(target[[0,2]]@target[[0,2]]-l1*l1-l2*l2)/(2*l1*l2)
        if not -1<=cosine<=1:raise ValueError('SEED_LEG_REACH')
        bend=np.arccos(cosine);au=np.arctan2(u[0],-u[2]);av=np.arctan2(v[0],-v[2])
        knee=bend-offsets[1]-au+av
        vec=u+ry(offsets[1]+knee)@v
        hip_total=np.arctan2(vec[0],-vec[2])-np.arctan2(target[0],-target[2])
        hip=hip_total-offsets[0]
        ankle=Rotation.from_matrix(root[:3,:3].T@ar).as_euler('xyz')[1]-hip_total-offsets[1]-knee-offsets[2]
        q[[plan.model.names.index(side+'_'+n+'_joint') for n in ('hip_pitch','knee','ankle_pitch')]]=[hip,knee,ankle]
    return q

def planar_construct(plan):
    if plan.task!='leaning' or plan.n[0]<.97:raise ValueError('ONLY_NAMED_UPRIGHT_BACK_TARGET')
    fk=plan.model.fk(plan.q0);pelvis=plan.mesh['pelvis']
    torso=np.concatenate([apply(fk[b],plan.mesh[b]) for b in ('torso_link','logo_link')])
    def separation(theta):
        direction=(plan.R0@ry(theta)).T@plan.n
        return float(np.min(torso@direction)-np.min(pelvis@direction))
    reserve=getattr(plan,'separation_reserve',.00601)
    if separation(0.)>=reserve:theta=0.
    else:
        if separation(.65)<reserve:raise ValueError('NO_BRACKETED_TORSO_PELVIS_SEPARATION_SEED')
        theta=float(brentq(lambda t:separation(t)-reserve,0.,.65,xtol=1e-12,maxiter=60))
    R=plan.R0@ry(theta);direction=R.T@plan.n;projection=pelvis@direction
    marker=pelvis[projection<=projection.min()+1e-10].mean(0)
    a=(-plan.d-float(marker@direction))/plan.n[0];b=-plan.n[2]/plan.n[0]
    # t_x=a+b*t_z enforces the observed support plane exactly. Intersect the
    # two positive-knee reach intervals; choose the highest reachable pelvis.
    chain=chains(plan);lo,hi=-.6,.15
    for js,offsets,u,v,ar,ap in chain.values():
        l1=np.linalg.norm(u[[0,2]]);l2=np.linalg.norm(v[[0,2]])
        au=np.arctan2(u[0],-u[2]);av=np.arctan2(v[0],-v[2])
        bend=offsets[1]+.04+au-av
        radius2=l1*l1+l2*l2+2*l1*l2*np.cos(bend)
        d0=(R.T@(ap-np.array([a,0.,0.]))-np.asarray(js[0]['p0']))[[0,2]]
        dz=(R.T@np.array([-b,0.,-1.]))[[0,2]]
        roots=np.roots([dz@dz,2*d0@dz,d0@d0-radius2])
        if np.max(abs(roots.imag))>1e-10:raise ValueError('NO_SEED_REACH_INTERVAL')
        roots=np.sort(roots.real);lo=max(lo,float(roots[0]));hi=min(hi,float(roots[1]))
    z=min(0.,hi)
    if z<lo:raise ValueError('EMPTY_BILATERAL_SEED_REACH_INTERVAL')
    root=transform(R,[a+b*z,0.,z]);q=pitch_ik(plan,root,chain)
    x=np.r_[q[:23],root[:3,3],Rotation.from_matrix(plan.R0.T@R).as_rotvec(),0.,0.]
    s=plan.state(x,0.);foot_error=float(np.max(abs(s['foot_error'])))
    if foot_error>.003:raise ValueError(('SEED_FULL_3D_FOOT_CHECK',foot_error))
    if np.any(q[:23]<plan.lower+.031) or np.any(q[:23]>plan.upper-.031):raise ValueError('SEED_JOINT_MARGIN')
    return x,dict(source='Original current encoders/static body hulls/current observed back plane only',
        pitch_rad=theta,root_position=root[:3,3].tolist(),torso_pelvis_plane_separation_m=separation(theta),
        reach_interval_m=[lo,hi],full_3D_foot_residual=foot_error,
        balance_certified=False,collision_certified=False,physical_success=False)


def construct(plan):
    """Close each original anchored foot in6D after the geometric seed.

    Root pose and contact body stay fixed. This is planning initialization,
    never an executed joint increment or a changed foot/reset location.
    """
    x,report=planar_construct(plan);initial=x.copy();steps=[]
    contact=plan.state(x,0.)['point'].copy()
    for side,offset in [('left',0),('right',6)]:
        ids=[plan.model.names.index(side+'_'+n+'_joint') for n in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
        history=[]
        for iteration in range(6):
            state=plan.state(x,0.);error=state['foot_error'][offset:offset+6]
            norm=float(np.max(abs(error)));history.append(norm)
            if norm<=1e-11:break
            if iteration==5:raise ValueError(('SPATIAL_SEED_CLOSURE_LIMIT',side,history))
            J=plan.jets(x,0.)['feet'][offset:offset+6][:,ids]
            delta=np.linalg.solve(J,-error)
            if not np.isfinite(delta).all():raise ValueError('NONFINITE_SPATIAL_SEED_STEP')
            trial=x.copy();trial[ids]+=delta
            if np.max(abs(trial[ids]-initial[ids]))>.001:raise ValueError('SPATIAL_SEED_INITIALIZATION_BOX')
            if np.any(trial[:23]<plan.lower+.031) or np.any(trial[:23]>plan.upper-.031):raise ValueError('SPATIAL_SEED_JOINT_MARGIN')
            if trial[plan.model.names.index(side+'_knee_joint')]<.04:raise ValueError('SPATIAL_SEED_KNEE_BRANCH')
            new_error=plan.state(trial,0.)['foot_error'][offset:offset+6]
            if np.max(abs(new_error))>=norm:raise ValueError('SPATIAL_SEED_NO_PROGRESS')
            x=trial
        steps.append(dict(side=side,updates=len(history)-1,residual_history=history))
    state=plan.state(x,0.)
    if np.max(abs(state['foot_error']))>1e-11:raise ValueError('SPATIAL_SEED_ORIGINAL_FK_REJECTED')
    if not np.array_equal(x[23:29],initial[23:29]):raise ValueError('SPATIAL_SEED_ROOT_CHANGED')
    if not np.array_equal(state['point'],contact):raise ValueError('SPATIAL_SEED_CONTACT_POINT_CHANGED')
    report.update(spatial_feet=steps,maximum_initialization_joint_correction_rad=float(np.max(abs(x[:23]-initial[:23]))),
        full_3D_foot_residual=float(np.max(abs(state['foot_error']))),root_and_contact_point_bitwise_unchanged=True,
        construction='Original geometric seed followed by bounded bilateral6D foot closure; no terminal point read')
    return x,report
