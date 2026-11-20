"""Correct measured touchdown travel of a fixed pelvis point, with both feet anchored."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from prepare_sitting_repairs import encode, smoothstep
from seat_plane_approach import SeatPlaneApproach, stance_axes
from study_live_sitting import read


def construct(stage, row, trial, meta, geometry, limits, model):
    p=stage/'evaluation'
    result=read(p/'results.json')
    with np.load(p/'trajectory.npz') as z:
        pose=z['body'][:,row,0].astype(float)
        pose[:,:3]-=result['origins'][row]
        q=z['q'][:,row]
        obs=z['obs'][:,row,222:333]
        load=np.linalg.norm(z['physical_force_w'][:,row,0],axis=-1)/z['physical_bodyweight_N'][:,row]
        bad=(load>.01)&(z['physical_slip_m_s'][:,row,0]>.02)
        forbidden=np.flatnonzero(z['flags'][:,row,1])
    with np.load(p/'edge_audit.npz') as z: point=z['pelvis_contact_world_local'][:,row]
    legal=np.flatnonzero(bad & np.isfinite(point).all(axis=1))
    if not len(legal) and not len(forbidden):return []
    normals=np.asarray(geometry['normal'])
    rot=Rotation.from_quat(pose[:,3:]).as_matrix()
    pivot=int(legal[0]) if len(legal) else int(forbidden[0])
    # Always use the first contiguous touchdown burst: a later rest contact has
    # a different material point and cannot supply this derivative.
    burst=legal[legal<=pivot+45]
    end=int(burst[-1]) if len(burst) else pivot
    a=max(80,pivot-14)
    b=min(955,end+8)
    t=np.arange(1000)
    tangent=np.zeros((1000,3))
    if len(burst):
        material=rot[pivot].T@(point[pivot]-pose[pivot,:3])
        track=rot@material+pose[:,:3]
        travel=track[np.clip(t,a,b)]-track[a]
        tangent=-(travel-(travel@normals)[:,None]*normals)*smoothstep((t-a)/12.)[:,None]
        tangent*=np.minimum(1.,.018/np.maximum(np.linalg.norm(tangent,axis=1),1e-8))[:,None]
    tool=SeatPlaneApproach(model,meta,1)
    ang=np.zeros(2)
    contact=max(80,int(forbidden[0])-1) if len(forbidden) else pivot
    def basis(tick):
        axes=stance_axes(tool.feet(obs[tick,:23]),obs[tick,69:72])
        return rot[tick]@axes
    if len(forbidden):
        body_frame=basis(contact)
        normal=body_frame.T@normals
        full=q[contact].copy()
        initial=tool.feet(obs[contact,:23])[0,:3,:3]
        def gaps(angles):
            correction=tool.correct(obs[contact,:23],obs[contact],normal,float(angles[0]),pitch=float(angles[1]))
            if correction is None:return None
            changed=full.copy()
            changed[meta['joint_ids']]+=correction[0]
            poses=model.fk(changed)
            orientation=rot[contact]@initial@tool.feet(changed[meta['joint_ids']])[0,:3,:3].T
            axis=orientation.T@normals
            shell=(model.vertices['pelvis']@axis).min()
            out=[]
            for side in ('left','right'):
                name=side+'_hip_pitch_link'
                f=poses[name]
                out.append((model.vertices[name]@(f[:3,:3].T@axis)).min()+f[:3,3]@axis-shell)
            return np.array(out)
        g=gaps([0.,0.])
        xp=gaps([.012,0.])
        yp=gaps([0.,.012])
        if g is not None and xp is not None and yp is not None:
            jac=np.stack(((xp-g)/.012,(yp-g)/.012),axis=1)
            required=max(.004,g.min()+.004)-g
            fit=minimize(lambda x:x@x+2.e4*np.square(np.maximum(required-jac@x,0.)).sum(),
                         [0.,0.],bounds=[(-.085,.085),(-.10,.10)],method='SLSQP')
            if fit.success and (jac@fit.x)[g.argmin()]>.001:ang=fit.x
    if np.linalg.norm(tangent,axis=1).max()<.001 and not ang.any():return []
    knots=np.unique(np.r_[np.arange(0,1000,10),np.arange(a-6,b+9,2),999])
    candidates=[]
    for scale in (1.,.55):
        # Set the clearance orientation before load arrives. It is then retained.
        angles=smoothstep((t-max(100,contact-215))/175.)[:,None]*ang*scale
        cmd=trial['command']
        offsets=[]
        maxerr=0.
        for k in knots:
            at=min(int(k),contact) if len(forbidden) else int(k)
            dw=basis(at)
            direction=dw[:,0]-normals*(normals@dw[:,0])
            direction/=np.linalg.norm(direction)
            lateral=np.cross(normals,direction)
            shift=tangent[k]*scale
            if not angles[k].any() and not np.any(shift):
                offsets.append(np.zeros(23))
                continue
            fit=tool.correct(cmd[k],obs[at],dw.T@normals,float(angles[k,0]),pitch=float(angles[k,1]),
                             forward_m=float(shift@direction),lateral_m=float(shift@lateral))
            if fit is None:break
            offsets.append(fit[0]);maxerr=max(maxerr,float(fit[1][0]))
        if len(offsets)!=len(knots):continue
        delta=np.stack([np.interp(t,knots,col) for col in np.array(offsets).T],axis=1)
        wanted=cmd+delta
        # Allocate enough time for the original actuator step bound through
        # the changed segment rather than clipping independently moving joints.
        step=np.diff(wanted,axis=0,prepend=trial['initial'][None,:])
        clock=np.cumsum(np.maximum(1.,abs(step).max(axis=1)/.00299))-1.
        mapped=np.interp(t,clock,t)
        target=np.stack([np.interp(mapped,t,col) for col in wanted.T],axis=1)
        path,actual=encode(target,trial['initial'],*limits)
        if np.max(abs(actual-target))>.010:continue
        path[750:]-=trial['residual']
        candidates.append((f'arrest_{scale:g}',path,dict(source=str(stage),row=int(row),burst=[pivot,end],
                  tangent_offset_m=np.round(tangent[b]*scale,6).tolist(),clearance_angles_rad=(ang*scale).tolist(),
                  max_foot_fit_m=maxerr,max_command_fit_rad=float(abs(actual-target).max()))))
    return candidates
