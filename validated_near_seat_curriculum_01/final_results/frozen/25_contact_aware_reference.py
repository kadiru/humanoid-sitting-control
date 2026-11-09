"""Adapt an offline motion to a depth-measured plane using calibrated leg FK.

No live simulator state, furniture metadata or privileged contact labels are accepted.
The optional inertial data are static robot model parameters for offline planning.
The floor is inferred from both foot soles and the IMU at the standing start.
The reference library is explicit assistance, not a learned motion discovery.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .shared_frame import apply, level_rotation


def measured_level_plane(model, q, gravity, geometry, initial_reference_root):
    if not geometry[11] or geometry[12] > .5:
        raise ValueError('No current measured plane at standing start')
    R = level_rotation(gravity)
    fk = model.fk(q)
    soles = np.concatenate([apply(fk[s+'_ankle_roll_link'], model.vertices[s+'_ankle_roll_link']) for s in ('left','right')]) @ R.T
    origin = np.array([initial_reference_root[0], initial_reference_root[1], -soles[:,2].min()])
    normal = R @ geometry[2:5]
    normal /= np.linalg.norm(normal)
    offset = float(geometry[5] - normal @ origin)
    if normal[2] < .9:
        raise ValueError('Sitting adaptation is limited to near-horizontal planes')
    return normal, offset, origin


def adapt_motion(model, qs, commands, roots, normal, offset, rear_shift=0., pitch_gain=0., reference_normal=None, reference_offset=None, inertial=None, balance_com=False, balance_pitch=False, prelanding_lift=0.):
    """Preserve each reference foot pose; solve sagittal legs for a shifted pelvis.

    Surface slopes/heights come only from the measured plane. The original
    motion's 0.40m design plane is an explicit offline reference constant.
    """
    n = len(qs)
    adapted = qs.astype(float).copy()
    new_roots = roots.astype(float).copy()
    errors = []
    slope = np.arctan2(normal[0],normal[2])
    reference_slope = np.arctan2(reference_normal[0],reference_normal[2]) if reference_normal is not None else 0.
    reference_height = -(-.4*reference_normal[0]+reference_offset)/reference_normal[2] if reference_normal is not None else .4
    measured_height = -(-.4*normal[0]+offset)/normal[2]
    slope_delta = slope-reference_slope
    height_delta = measured_height-reference_height
    for stage in range(1,n):
        alpha = np.clip((stage-1)/3,0,1)
        alpha = alpha*alpha*(3-2*alpha)
        # Rearward clearance is established before the original landing segment.
        rear_weight = [0.,0.,.3,.7,1.,.3,0.][stage]
        q = qs[stage].astype(float).copy()
        R0 = Rotation.from_quat(roots[stage,3:]).as_matrix()
        xyz = roots[stage,:3].astype(float).copy()
        xyz[0] -= rear_shift*rear_weight
        # Rigidly transform the reference body relative to its design plane;
        # calibration is a recorded depth/encoder/IMU observation, not truth.
        Rdelta=Rotation.from_euler('y',pitch_gain*slope_delta*alpha).as_matrix()
        center=np.array([-.4,0.,.4])
        xyz=center+Rdelta@(xyz-center)+np.array([0.,0.,np.clip(height_delta,-.07,.07)*alpha])
        xyz[2]+=prelanding_lift*[0.,0.,.3,.7,1.,0.,0.][stage]
        R=Rdelta @ R0
        original = model.fk(q)
        for side in ('left','right'):
            name = side+'_ankle_roll_link'
            ids = [model.names.index(side+'_'+j+'_joint') for j in ('hip_pitch','knee','ankle_pitch')]
            target = original[name].copy()
            target[:3,:3] = R.T @ R0 @ target[:3,:3]
            target[:3,3] = R.T @ (R0 @ original[name][:3,3] + roots[stage,:3]-xyz)
            def residual(values):
                trial=q.copy();trial[ids]=values;f=model.fk(trial)[name]
                return np.r_[f[:3,3][[0,2]]-target[:3,3][[0,2]], .2*Rotation.from_matrix(target[:3,:3].T@f[:3,:3]).as_rotvec()[1]]
            solution = least_squares(residual,q[ids],bounds=([-2.4,.05,-.85],[.6,2.5,.5]),xtol=1e-8,ftol=1e-8,gtol=1e-8,max_nfev=35)
            q[ids]=solution.x
            errors.append(float(np.linalg.norm(residual(solution.x))))
        if balance_com:
            masses=np.asarray(inertial['mass']);local=np.asarray(inertial['local_com'])
            def com(fk):
                return np.average(np.array([fk[name][:3,:3]@point+fk[name][:3,3] for name,point in zip(inertial['body_names'],local)]),axis=0,weights=masses)
            original_com=R0@com(original)+roots[stage,:3]
            transformed_com=R@com(original)+xyz
            contact_blend=np.clip((stage-4)/2,0,1)
            desired_com=(1-contact_blend)*original_com+contact_blend*transformed_com
            legids=[model.names.index(side+'_'+j+'_joint') for side in ('left','right') for j in ('hip_pitch','knee','ankle_pitch')]
            initial_xyz=xyz.copy()
            feet={side:(R0@original[side+'_ankle_roll_link'][:3,3]+roots[stage,:3],R0@original[side+'_ankle_roll_link'][:3,:3]) for side in ('left','right')}
            def residual_balance(values):
                trial=q.copy();trial[legids]=values[:6];position=initial_xyz.copy();position[0]=values[6]
                position[2]-=normal[0]/normal[2]*(values[6]-initial_xyz[0])
                fk=model.fk(trial);res=[]
                for side in ('left','right'):
                    f=fk[side+'_ankle_roll_link'];target_pos,target_rot=feet[side]
                    res.extend((R@f[:3,3]+position-target_pos)[[0,2]])
                    res.append(.2*Rotation.from_matrix(target_rot.T@R@f[:3,:3]).as_rotvec()[1])
                res.append((R@com(fk)+position-desired_com)[0])
                return np.asarray(res)
            solution=least_squares(residual_balance,np.r_[q[legids],xyz[0]],bounds=([-2.4,.05,-.85]*2+[roots[stage,0]-.2],[.6,2.5,.5]*2+[roots[stage,0]+.2]),max_nfev=50,xtol=1e-8,gtol=1e-8,ftol=1e-8)
            q[legids]=solution.x[:6];xyz[0]=solution.x[6];xyz[2]-=normal[0]/normal[2]*(xyz[0]-initial_xyz[0])
            errors.append(float(np.linalg.norm(residual_balance(solution.x))))
        if balance_pitch:
            masses=np.asarray(inertial['mass']);local=np.asarray(inertial['local_com'])
            def com(fk):
                return np.average(np.array([fk[name][:3,:3]@point+fk[name][:3,3] for name,point in zip(inertial['body_names'],local)]),axis=0,weights=masses)
            # Stay balanced over the feet until the first seated waypoint.
            source_stage=min(stage,4) if stage<6 else stage
            source_R=Rotation.from_quat(roots[source_stage,3:]).as_matrix()
            desired_com=(source_R@com(model.fk(qs[source_stage]))+roots[source_stage,:3])[0]
            if stage==6:desired_com=(R@com(original)+xyz)[0]
            legids=[model.names.index(side+'_'+j+'_joint') for side in ('left','right') for j in ('hip_pitch','knee','ankle_pitch')]
            feet={side:(R0@original[side+'_ankle_roll_link'][:3,3]+roots[stage,:3],R0@original[side+'_ankle_roll_link'][:3,:3]) for side in ('left','right')}
            euler=Rotation.from_matrix(R).as_euler('xyz')
            def residual_pitch(values):
                trial=q.copy();trial[legids]=values[:6]
                angles=euler.copy();angles[1]=values[6];rot=Rotation.from_euler('xyz',angles).as_matrix()
                fk=model.fk(trial);res=[]
                for side in ('left','right'):
                    f=fk[side+'_ankle_roll_link'];target_pos,target_rot=feet[side]
                    res.extend((rot@f[:3,3]+xyz-target_pos)[[0,2]])
                    res.append(.2*Rotation.from_matrix(target_rot.T@rot@f[:3,:3]).as_rotvec()[1])
                res.append((rot@com(fk)+xyz)[0]-desired_com)
                return np.asarray(res)
            solution=least_squares(residual_pitch,np.r_[q[legids],np.clip(euler[1],-.84,.89)],bounds=([-2.4,.05,-.85]*2+[-.85],[.6,2.5,.5]*2+[.90]),max_nfev=65,xtol=1e-8,gtol=1e-8,ftol=1e-8)
            q[legids]=solution.x[:6];euler[1]=solution.x[6];R=Rotation.from_euler('xyz',euler).as_matrix()
            errors.append(float(np.linalg.norm(residual_pitch(solution.x))))
        adapted[stage]=q
        new_roots[stage]=np.r_[xyz,Rotation.from_matrix(R).as_quat()]
    return adapted, commands+(adapted-qs), new_roots, dict(max_sagittal_residual_m=max(errors),slope_rad=float(slope),normal=normal.tolist(),offset=offset,height_delta=float(height_delta),slope_delta=float(slope_delta))


def front_contact_motion(model,qs,commands,roots,normal,offset,calibration,contact_pitch=.45,contact_x=-.18,contact_gap=-.002,approach_x=-.06,approach_pitch=.43,approach_gap=.045):
    """Replace the old rearward overshoot with a forward-leaning front-seat goal.

    Feet retain their standing transforms. Height/slope are relative to recorded
    sensor calibration. Horizontal placement is explicit reference assistance.
    """
    rn=np.asarray(calibration['reference_normal']);rd=calibration['reference_offset']
    slope=np.arctan2(normal[0],normal[2])-np.arctan2(rn[0],rn[2])
    height=.4-(-.4*normal[0]+offset)/normal[2]+(-.4*rn[0]+rd)/rn[2]
    plane_n=Rotation.from_euler('y',slope).as_matrix()[:,2]
    plane_d=-plane_n@np.array([-.4,0.,height])
    qnew=qs.astype(float).copy();cnew=commands.astype(float).copy();rnew=roots.astype(float).copy()
    Rstart=Rotation.from_quat(roots[0,3:]).as_matrix();start=model.fk(qs[0]);targets={}
    for side in ('left','right'):
        f=start[side+'_ankle_roll_link'];targets[side]=(Rstart@f[:3,3]+roots[0,:3],Rstart@f[:3,:3])
    errors=[]
    for stage,x,gap,pitch in [(4,approach_x,approach_gap,approach_pitch),(5,contact_x,contact_gap+.008,contact_pitch),(6,contact_x,contact_gap,contact_pitch)]:
        angles=Rotation.from_quat(roots[stage,3:]).as_euler('xyz');angles[1]=pitch
        R=Rotation.from_euler('xyz',angles).as_matrix();xyz=np.array([x,roots[0,1],0.])
        vertices=model.vertices['pelvis']@R.T
        selected=vertices[(vertices[:,0]+x<-.16)&(vertices[:,0]+x>-.64)] if stage>=5 else vertices
        assert len(selected),'No intended pelvis patch over the reference seat'
        xyz[2]=(gap-np.min(selected@plane_n)-plane_n[:2]@xyz[:2]-plane_d)/plane_n[2]
        q=qs[stage].astype(float).copy()
        for side in ('left','right'):
            ids=[model.names.index(side+'_'+j+'_joint') for j in ('hip_pitch','knee','ankle_pitch')]
            target_pos,target_rot=targets[side]
            def residual(values):
                v=q.copy();v[ids]=values;f=model.fk(v)[side+'_ankle_roll_link']
                return np.r_[(R@f[:3,3]+xyz-target_pos)[[0,2]],.2*Rotation.from_matrix(target_rot.T@R@f[:3,:3]).as_rotvec()[1]]
            fit=least_squares(residual,q[ids],bounds=([-2.4,.05,-.85],[.6,2.5,.5]),max_nfev=50,xtol=1e-9,gtol=1e-9,ftol=1e-9)
            q[ids]=fit.x;errors.append(float(np.linalg.norm(residual(fit.x))))
        qnew[stage]=q;cnew[stage]=q+(commands[stage]-qs[stage] if stage==4 else commands[6]-qs[6])
        rnew[stage]=np.r_[xyz,Rotation.from_matrix(R).as_quat()]
    return qnew,cnew,rnew,dict(max_sagittal_residual_m=max(errors),measured_relative_height=float(height),measured_relative_slope=float(slope),contact_pitch=contact_pitch,contact_x=contact_x,contact_gap=contact_gap,approach_x=approach_x,approach_pitch=approach_pitch,approach_gap=approach_gap)
