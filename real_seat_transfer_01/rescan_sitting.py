"""Register the seat after a step and fit a verified descent to the two planted feet."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from seat_alignment import UNITS, points, register, transform


def shoe_frame(model, joints, gravity):
    from seat_plane_approach import stance_axes
    fk=model.fk(joints)
    feet=np.stack([fk[f'{side}_ankle_roll_link'] for side in ('left','right')])
    return feet,feet[:,:3,3].mean(0),stance_axes(feet,gravity)


def match(reference, observed):
    pose, valid=register(reference,observed)
    if not valid:
        return None
    pose=pose[:3]*UNITS
    tree=cKDTree(reference)
    near=tree.query(reference,k=12)[1]
    edges=reference[near]-reference[near].mean(1,keepdims=True)
    _,axes=np.linalg.eigh(edges.transpose(0,2,1)@edges)
    normals=axes[:,:,0]
    for _ in range(30):
        moved=transform(observed,pose)
        dist,index=tree.query(moved)
        keep=dist<max(.012,np.quantile(dist,.90))
        pp=moved[keep]
        nn=normals[index[keep]]
        delta=reference[index[keep]]-pp
        jac=np.zeros((len(pp),3,3))
        jac[:,0,0]=jac[:,1,1]=1.
        jac[:,0,2]=-pp[:,1]+pose[1]
        jac[:,1,2]=pp[:,0]-pose[0]
        j=np.einsum('nki,nk->ni',jac,nn)
        e=(delta*nn).sum(1)
        fit=np.linalg.lstsq(np.r_[j,.12*jac.reshape(-1,3)],np.r_[e,.12*delta.ravel()],rcond=None)[0]
        pose+=np.clip(fit,[-.015,-.015,-.05],[.015,.015,.05])
        if np.max(abs(fit))<1.e-5:
            break
    final,_=tree.query(transform(observed,pose))
    if np.median(final)>.013 or np.mean(final<.025)<.75 or np.max(np.abs(pose[:2]))>.105:
        return None
    return pose,float(np.median(final))


def smooth(x):
    u=np.clip(x,0.,1.)
    return u*u*(3.-2.*u)


def retarget(demo, model, sample, previous, shift=0., clearance_pitch=0., clearance_roll=0.):
    from stance23_planner import chains,legs,solve
    from rearstep_model import Robot
    from scipy.ndimage import gaussian_filter1d
    robot=Robot()
    depth, intrinsics, q, gravity, mount=sample
    cloud=points(depth,intrinsics,mount,q,gravity,model)
    observed=match(demo['cloud'],cloud)
    if observed is None:
        return None,dict(admitted=False,reason='seat registration failed')
    se2, error=observed
    calibrated=Rotation.from_euler('z',se2[2]).as_matrix()
    axes=demo['frame']@calibrated
    origin=demo['origin']+demo['frame']@np.r_[se2[:2],0.]
    feet, mid, basis=shoe_frame(model,q,gravity)
    current_R=axes@basis.T
    current_p=origin-current_R@mid
    current_feet_R=np.einsum('ij,njk->nik',current_R,feet[:,:3,:3])
    current_feet=feet[:,:3,3]@current_R.T+current_p
    ref_p=demo['body'][:,0,:3]
    ref_R=Rotation.from_quat(demo['body'][:,0,3:]).as_matrix()
    ref_f=demo['body'][:,1:,:3]
    ref_fR=Rotation.from_quat(demo['body'][:,1:,3:].reshape(-1,4)).as_matrix().reshape(1000,2,3,3)
    shift_f=current_feet-ref_f[0]
    turn_f=current_feet_R@ref_fR[0].transpose(0,2,1)
    root_turn=Rotation.from_matrix(current_R@ref_R[0].T).as_rotvec()
    load=int(demo['loaded'])
    ticks=np.unique(np.r_[0,np.arange(0,1000,5),load,999])
    angles=[]
    errors=[]
    balance=[]
    centers=[]
    acc=np.gradient(np.gradient(gaussian_filter1d(ref_p,5,axis=0),.02,axis=0),.02,axis=0).clip(-1.2,1.2)
    force=gaussian_filter1d(demo['force'],6,axis=0)
    # The implicit position drives use the calibrated simulator gains.
    kp=np.array([200,200,200,150,150,40,40,150,150,40,40,200,200,40,40,160,160,40,40,160,160,40,40])
    for t in ticks:
        mix=smooth(t/120.)
        seat_mix=smooth((t-(load-170))/135.)
        p=ref_p[t]+(1-mix)*(current_p-ref_p[0])+mix*(1-seat_mix)*np.mean(shift_f,axis=0)
        # Target the same support patch in the rescanned chair frame.
        p[0]+=seat_mix*shift
        R=Rotation.from_rotvec((1-mix)*root_turn).as_matrix()@ref_R[t]
        # Tip the pelvis forward before approaching the rim. Rotate about
        # the underside of the pelvis so the intended support point stays put.
        lean=smooth((t-(load-240))/155.)*(1-smooth((t-load-30)/140.))
        cleared=Rotation.from_euler('xy',lean*np.array([clearance_roll,clearance_pitch])).as_matrix()@R
        p-=(cleared-R)@np.array([-.0011,0.,-.152])
        R=cleared
        state=demo['q'][t,:23].copy()
        for side,(chain,ids) in enumerate(zip(chains,legs)):
            goal=np.eye(4)
            # Keep lateral and longitudinal offsets at the measured landing.
            # Carry over the reference's small foot pitch/roll changes under load.
            foot=ref_f[t,side]+shift_f[side]
            goal[:3,3]=R.T@(foot-p)
            goal[:3,:3]=R.T@(turn_f[side]@ref_fR[t,side])
            state[ids], err=solve(chain,state[ids],goal,ids)
            errors.append(err)
        angles.append(state)
        old=robot.state(demo['q'][t],ref_R[t],ref_p[t])
        full=demo['q'][t].copy()
        full[:23]=state
        new=robot.state(full,R,p)
        co0=np.average(old[1],axis=0,weights=robot.mass)
        co1=np.average(new[1],axis=0,weights=robot.mass)
        centers.append(co1-origin)
        f=force[t]
        fraction=np.maximum(f[:,2],0.)
        fraction=fraction/max(1e-6,fraction.sum())
        cop_old=ref_f[t].copy()
        cop_old[:,2]=0.
        center=co0[:2]-co0[2]*acc[t,:2]/9.81
        # Stay within the feet; at seat support the reference has additional
        # reactions, so only the change in floor moment is used below.
        cop_old[:,:2]+=np.clip(center-fraction@cop_old[:,:2],[-.062,-.024],[.089,.024])
        cop_new=cop_old.copy()
        needed=np.clip((co1-co0)[:2]-fraction@shift_f[:,:2],[-.038,-.017],[.038,.017])
        cop_new[:,:2]+=shift_f[:,:2]+needed
        torque=robot.forces(new,acc[t],cop_new,f)-robot.forces(old,acc[t],cop_old,f)
        balance.append(np.clip(torque[:23]/kp,-.115,.115))

    fit=PchipInterpolator(ticks,np.stack(angles),axis=0)(np.arange(1000))
    delta=PchipInterpolator(ticks,np.stack(balance),axis=0)(np.arange(1000))
    latent=demo['desired']+fit-demo['q'][:,:23]+delta
    blend=smooth(np.arange(1000)/42.)[:,None]
    latent=(1-blend)*previous+blend*latent
    proof=dict(admitted=max(errors)<.0025,scan_translation_m=se2[:2].tolist(),scan_heading_rad=float(se2[2]),
               registration_median_m=error,estimated_start_m=current_p.tolist(),
               max_drive_feedforward_rad=float(np.max(abs(delta))),
               foot_shift_m=shift_f.tolist(),max_ik_position_error_m=float(max(errors)))
    if not proof['admitted']:
        return None,proof
    center=PchipInterpolator(ticks,np.stack(centers),axis=0)(np.arange(1000))
    return dict(latent=latent,joints=fit,center=center,axes=axes,contact=load),proof


def attach(cfg,launcher,options,original_attach):
    from chair_contact_motion import ChairContactMotion
    import torch
    prior=ChairContactMotion.apply
    from pathlib import Path
    import json
    payload=cfg.get('rescan_library')
    if payload is None:
        return original_attach(launcher,options)
    with np.load(payload,allow_pickle=False) as z:
        raw={k:z[k] for k in z.files}
    names=cfg['templates_by_row']
    def apply(motion,learner,obs,tick,nominal):
        # These project modules import simulator dependencies. Load them after Kit opens.
        from rearstep_kin import MassModel
        from stance23_planner import chains, legs, forward
        from seat_plane_approach import stance_axes
        output=prior(motion,learner,obs,tick,nominal)
        env=learner.env
        if tick==0:
            path=env.out/'single_step.npz'
            with np.load(path,allow_pickle=False) as z:
                valid=(z['options']>0)&~z['slip'].any(axis=(0,2))&~z['flags'].any(axis=(0,2))
                valid&=z['verified_floor'].all(axis=(0,2))
                valid&=(z['floor_load'][-50:,:,:]>.15).all(axis=(0,2))&(z['flat_speed'][-50:,:]<.02).all(0)
            motion.rescan_paths={}
            motion.balance_model=MassModel()
            motion.rescan_feedback={}
            receipts=[]
            for i,name in enumerate(names):
                if not valid[i] or name+'_cloud' not in raw:
                    continue
                t,d,k,q,g,_,_,_,mount=env.standing_samples[-1]
                assert t==0
                demo={k[len(name)+1:]:v for k,v in raw.items() if k.startswith(name+'_')}
                clearance=cfg.get('wire_clearance',{}) if name=='Chair_Wire' else {}
                fit,proof=retarget(
                    demo,env.model,(d[i],k[i],q[i],g[i],mount),env.previous[i,:23].cpu().numpy(),
                    shift=cfg.get('seat_shift_m',0.)+clearance.get('fore_aft_m',0.),
                    clearance_pitch=cfg.get('clearance_pitch',0.) if name in ('Chair_Wire','Anora') else 0.,
                    clearance_roll=clearance.get('roll_rad',0.))
                receipts.append(dict(row=i,asset=name,**proof))
                if fit is not None:
                    motion.rescan_paths[i]=fit
                    if name in cfg.get('balance_feedback_assets',[]):
                        motion.rescan_feedback[i]=dict(error=0.,velocity=0.,filtered=np.zeros(2),clock=0.,enabled=False,
                                                       recenter=bool(name=='Chair_Wire' and cfg.get('wire_hold')),
                                                       anchor=None,seat_trim=0.,last_x=None,x_speed=0.,
                                                       settled=0,hold_target=None,
                                                       log=[])
            (env.out/'rescan_plan.json').write_text(json.dumps(receipts,indent=2)+'\n')
        for i,p in motion.rescan_paths.items():
            data=env.robot.data
            q=data.joint_pos.torch[i,:23]
            state=motion.rescan_feedback.get(i)
            k=tick if state is None else int(state['clock'])
            target=p['latent'][k].copy()
            correction=(cfg.get('tracking_gain',.25)*(q.new_tensor(p['joints'][k])-q)).clamp(-.035,.035)
            correction*=float(smooth(tick/60.))
            if state is not None and tick>=170:
                allq=data.joint_pos.torch[i].cpu().numpy()
                g=data.projected_gravity_b.torch[i].cpu().numpy()
                fk=np.stack([forward(chain,allq[ix])[0] for chain,ix in zip(chains,legs)])
                frame=p['axes']@stance_axes(fk,g).T
                center,feet=motion.balance_model.estimate(allq,frame)
                relative=center-feet.mean(0)
                err=float(relative[0]-p['center'][k,0])
                vel=state['velocity']*.80+.20*(err-state['error'])*50 if state['enabled'] else 0.
                state.update(error=err,velocity=vel,enabled=True)
                bw=float(data.body_mass.torch[i].sum())*9.81
                foot=np.array([env.raw.scene['rest_'+side+'_floor'].data.normal_force_matrix_w.torch[i,:,:,2].sum().item()
                               for side in ('left','right')])
                load=float(env.raw.scene['pelvis_seat'].data.normal_force_matrix_w.torch[i,:,:,2].sum())/bw
                # Small pressure-center changes oppose rearward velocity before
                # first contact. Fade them once the seat carries the body.
                displacement=np.clip(max(.4,relative[2])/9.81*(8.2*err+5.0*vel),-.050,.050)
                gain=cfg['balance_strength']*smooth((tick-170)/50.)*(1-smooth((load-.035)/.10))
                pressure=foot*displacement/160.*gain
                state['filtered']+=np.clip(pressure-state['filtered'],-.002,.002)
                target[[15,16]]+=state['filtered']
                if state['recenter']:
                    xpos=-feet[:,0].mean()
                    prev=xpos if state['last_x'] is None else state['last_x']
                    state['x_speed']=.82*state['x_speed']+.18*(xpos-prev)*50.
                    state['last_x']=xpos
                    if state['anchor'] is None and k>p['contact']-50 and load>.035:
                        state['anchor']=xpos
                    if state['anchor'] is not None:
                        # Keep the newly loaded pelvis inside the rim. A virtual
                        # rearward pelvis displacement leaves both shoe goals fixed.
                        error=xpos-state['anchor']+.008+.10*state['x_speed']
                        desired=np.clip(-1.15*error,-.026,0.)
                        state['seat_trim']+=np.clip(desired-state['seat_trim'],-.00042,.00042)
                        shift=frame.T@np.array([-state['seat_trim'],0.,0.])
                        rhs=np.r_[shift,[0.,0.,0.]]
                        for chain,ids in zip(chains,legs):
                            _,jac=forward(chain,p['joints'][k,ids])
                            change=np.linalg.solve(jac.T@jac+.001*np.eye(6),jac.T@rhs)
                            target[ids]+=np.clip(change,-.12,.12)
                if state['recenter'] and cfg.get('wire_pressure_hold'):
                    force=env.raw.scene['pelvis_seat'].data
                    tang=force.friction_force_matrix_w.torch[i,:,:,:2].sum((0,1)).norm().item()
                    if .41<=load<=.56 and foot.min()/bw>.17 and tang<.30*load*bw and abs(state['x_speed'])<.028:
                        state['settled']+=1
                    else:
                        state['settled']=0
                    if state['hold_target'] is None and state['settled']>=6:
                        state['hold_target']=env.previous[i,:23].cpu().numpy().copy()
                    if state['hold_target'] is not None:
                        target=state['hold_target']
                state['log'].append([tick,k,err,vel,load,*foot,*state['filtered'],state['seat_trim'],
                                     float(state['hold_target'] is not None)])
            if state is not None:
                state['clock']=min(999,state['clock']+1.)
            output[i]=output.new_tensor(target)+correction
        if tick==999 and motion.rescan_feedback:
            np.savez_compressed(env.out/'rescan_balance.npz',
                                **{str(i):r['log'] for i,r in motion.rescan_feedback.items()})
        return output
    ChairContactMotion.apply=apply
    return original_attach(launcher,options)
