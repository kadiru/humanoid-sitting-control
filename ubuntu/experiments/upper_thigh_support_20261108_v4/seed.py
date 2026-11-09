"""Original-input proximal-thigh witness: fixed feet, dual contact, clearance."""
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import apply,transform
from paired_contact import BODIES,GUARDS
from leg_geometry import chains,pitch_ik,ry

def construct(plan):
    if plan.task!='leaning' or plan.n[0]<.97:raise ValueError('PAIRED_SEED_TARGET_ORIENTATION')
    fk=plan.model.fk(plan.q0)
    # Thigh contacts move with hip/knee articulation; rigidly tilting the
    # ORIGINAL standing joints is not a feasibility prerequisite. Initialize
    # at measured root orientation, then solve the anchored contact geometry.
    theta=0.
    R=plan.R0@ry(theta);root=transform(R)
    markers=plan.exposed_local(root,fk)
    P=np.array([apply(root@fk[b],m) for b,m in zip(BODIES,markers)])
    a=(-float(np.mean(plan.support_offsets))-float(np.mean(np.einsum('bi,bi->b',P,plan.support_normals))))/plan.n[0]
    b=-plan.n[2]/plan.n[0]
    chain=chains(plan);lo,hi=-.6,.15
    for js,offsets,u,v,ar,ap in chain.values():
        l1=np.linalg.norm(u[[0,2]]);l2=np.linalg.norm(v[[0,2]])
        au=np.arctan2(u[0],-u[2]);av=np.arctan2(v[0],-v[2])
        bend=offsets[1]+.041+au-av;radius2=l1*l1+l2*l2+2*l1*l2*np.cos(bend)
        d0=(R.T@(ap-np.array([a,0.,0.]))-np.asarray(js[0]['p0']))[[0,2]]
        dz=(R.T@np.array([-b,0.,-1.]))[[0,2]]
        roots=np.roots([dz@dz,2*d0@dz,d0@d0-radius2])
        if np.max(abs(roots.imag))>1e-10:raise ValueError('PAIRED_SEED_NO_REACH_INTERVAL')
        roots=np.sort(roots.real);lo=max(lo,float(roots[0]));hi=min(hi,float(roots[1]))
    z=min(0.,hi)
    if z<lo:raise ValueError('PAIRED_SEED_EMPTY_REACH_INTERVAL')
    root=transform(R,[a+b*z,0.,z]);q=pitch_ik(plan,root,chain)
    x=np.r_[q[:23],root[:3,3],Rotation.from_matrix(plan.R0.T@R).as_rotvec(),0.,0.]
    legids=[plan.model.names.index(s+'_'+n+'_joint') for s in ('left','right')
            for n in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
    columns=legids+[23,25,26,27]
    kneeids=[plan.model.names.index(s+'_knee_joint') for s in ('left','right')]
    knee=min(kneeids,key=lambda j:abs(x[j]-.041))
    lower=np.r_[plan.lower+.031,[-.5,-.2,-.6],[-.65,-.65,-.35],0.,0.]
    upper=np.r_[plan.upper-.031,[.15,.2,.15],[.65,.65,.35],np.inf,14.4]
    lower[kneeids]=np.maximum(lower[kneeids],.04)
    def residual(y):
        return np.r_[plan.equal(y,0.),np.min(plan.housing_rows(y))-.00001]
    def jacobian(y):
        guards=plan.housing_rows(y);J=plan.guard_rows_jac(y)
        return np.vstack([plan.equal_jac(y,0.),J[int(np.argmin(guards))]])[:,columns]
    history=[];backtracks=0
    if np.any(x<lower) or np.any(x>upper):raise ValueError('PAIRED_GEOMETRIC_SEED_BOX')
    # This is one fresh geometry-root construction, not a failed terminal state.
    # Foot, contact and buffer equations are evaluated by the original FK.
    for iteration in range(13):
        E=residual(x);norm=float(np.max(abs(E)));history.append(norm)
        if norm<=1e-10:break
        if iteration==12:raise ValueError(dict(reason='PAIRED_ROOT_LIMIT',residuals=E.tolist(),history=history))
        J=jacobian(x);delta,_,rank,_=np.linalg.lstsq(J,-E,rcond=None)
        if rank<len(E) or np.max(abs(J@delta+E))>1e-9:raise ValueError('PAIRED_ROOT_LINEAR_CLOSURE')
        if not np.isfinite(delta).all():raise ValueError('PAIRED_NONFINITE_ROOT_STEP')
        accepted=False
        for power in range(9):
            trial=x.copy();trial[columns]+=delta*(.5**power)
            if np.any(trial<lower) or np.any(trial>upper):continue
            new=residual(trial)
            if np.isfinite(new).all() and float(new@new)<float(E@E):
                x=trial;backtracks+=power;accepted=True;break
        if not accepted:raise ValueError(dict(reason='PAIRED_ROOT_NO_PROGRESS',residuals=E.tolist(),history=history))
    s=plan.state(x,0.);gap=np.einsum('bi,bi->b',s['support_points'],plan.support_normals)+plan.support_offsets
    report=dict(source='Original current encoders, anchored feet, static proximal thigh hulls and observed back plane only',
        construction='15-equation fixed-foot/proximal-dual-contact/expanded-clearance geometry root with16original coordinates; minimum-norm linear correction',
        support_bodies=list(BODIES),proximal_fraction_interval=[0.,.5],initial_rigid_pitch_rad=theta,
        geometry_root_updates=len(history)-1,residual_history=history,line_search_halvings=backtracks,
        full_3D_foot_residual=float(np.max(abs(s['foot_error']))),support_gaps_m=gap.tolist(),
        minimum_expanded_guard_clearance_m=float(np.min(plan.housing_rows(x))),
        root_position=s['root'][:3,3].tolist(),knee_angles_rad=x[kneeids].tolist(),
        knee_lower_bound_rad=.04,knee_endpoint_not_fixed=True,maximum_initialization_joint_change_rad=float(np.max(abs(x[:23]-plan.q0[:23]))),
        no_prior_optimizer_point_read=True,balance_certified=False,physical_success=False)
    return x,report
