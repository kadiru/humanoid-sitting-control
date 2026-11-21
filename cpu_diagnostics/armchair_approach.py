"""Register a supported wooden-armchair motion to a new seat while anchoring both feet."""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from chair_contact_motion import select
from evaluate_armchair_pair import PARENT, TARGET, CONTROL
from evaluate_chair_contact_motion import signatures
from evaluate_rim_motion import actor_from
from evaluate_scan_landing import load_bundle
from hold_clearance import choose
from prepare_sitting_repairs import BodyModel, MODEL, encode, leg_fk, smoothstep
from refine_chair_contact import collect
from seat_plane_approach import foot_error
from study_live_sitting import read, sha, write


def hull_points(model,stage,row,ticks,body,origins,names):
    b=stage[ticks,row,names.index(body)].astype(float)
    return np.einsum('tij,vj->tvi',Rotation.from_quat(b[:,3:]).as_matrix(),model.vertices[body])+b[:,None,:3]-origins[row]


def seat_gap(points,g):
    local=(points-g['top_center'])@Rotation.from_quat(g['rotation_xyzw']).as_matrix()
    beyond=np.maximum(abs(local[...,:2])-.5*np.asarray([g['depth'],g['width']]),0.)
    dist=np.where(beyond.max(-1)==0,local[...,2],np.sqrt((beyond**2).sum(-1)+np.maximum(local[...,2],0.)**2))
    return dist.min(-1)


def transfer(source, target, pivot):
    a,b=np.asarray(source['normal']),np.asarray(target['normal'])
    cross=np.cross(a,b)
    angle=np.arctan2(np.linalg.norm(cross),a@b)
    rv=angle*cross/max(1e-12,np.linalg.norm(cross))
    dh=(pivot-source['top_center'])@a-(pivot-target['top_center'])@b
    return rv,dh*b


def candidates(out,model):
    base=PARENT/'rollouts'
    target=read(out/'pair_bank/geometry.json')[0]
    rows=[]
    for name in read(PARENT/'summary.json')['known']:
        stage=base/name
        p=stage/'evaluation'
        m=read(p/'manifest.json'); result=read(p/'results.json')
        geom=read(Path(m['args']['bank'])/'geometry.json')
        ids=[i for i,r in enumerate(result['episodes']) if r['clean'] and geom[i].get('family')=='wooden_armchair']
        if not ids:continue
        with np.load(p/'trajectory.npz') as z:
            body=z['body'];load=z['support_bw']
        with np.load(p/'edge_audit.npz') as z:contact=z['pelvis_contact_world_local']
        origins=np.array(result['origins']); names=m['metadata']['body_names']
        for i in ids:
            valid=np.flatnonzero((load[:,i]>.01)&np.isfinite(contact[:,i]).all(-1))
            if not len(valid):continue
            touch=int(valid[0]);pivot=contact[touch,i].astype(float)
            rv,shift=transfer(geom[i],target,pivot)
            # Compare all loaded poses and the incoming 0.4 s. A finite seat can
            # reverse the order of contact even when its infinite plane matches.
            ticks=np.unique(np.r_[np.arange(max(200,touch-20),1000,5),touch,999])
            gap=[]
            for b in ('pelvis','left_hip_pitch_link','right_hip_pitch_link'):
                points=hull_points(model,body,i,ticks,b,origins,names)
                registered=(points-pivot)@Rotation.from_rotvec(rv).as_matrix().T+pivot+shift
                gap.append(seat_gap(registered,target))
            cap,left,right=gap
            near=cap<.015
            margin=float(np.minimum(left-cap,right-cap)[near].min()) if near.any() else -1.
            if margin<.001 or abs(cap[ticks==touch].item())>.004:continue
            # A smaller rigid displacement requires less compensation at both ankles.
            # Prefer clearance beyond the gap that separated the two measured strikes.
            score=np.linalg.norm(shift)+.22*np.linalg.norm(rv)+max(0,.004-margin)*3
            rows.append((float(score),stage,i,dict(id=result['episodes'][i]['id'],contact=touch,pivot=pivot.tolist(),
                rotation_vector=rv.tolist(),shift=shift.tolist(),margin_m=margin,source=geom[i],
                original_hold_s=result['episodes'][i]['final_hold_s'])))
    return sorted(rows,key=lambda v:v[0])


def register(stage,row,info,model):
    p=stage/'evaluation'
    meta=read(p/'manifest.json')['metadata']
    f=read(stage/'evaluation.freeze.json')['command']
    trial=collect(stage,actor_from(f[f.index('--checkpoint')+1]),actor_from(f[f.index('evaluate')+1]))[row]
    target=trial['command'].copy()
    with np.load(p/'trajectory.npz') as z:
        pose=z['body'][:,row,0].astype(float)
        q=z['obs'][:,row,222:245].astype(float)
    pose[:,:3]-=read(p/'results.json')['origins'][row]
    feet,legs=leg_fk(model,meta['joint_names'])
    lo=np.array(meta['lower']);hi=np.array(meta['upper'])
    end=info['contact']-35; start=max(150,end-175)
    knots=np.unique(np.r_[np.arange(start,1000,8),end,999])
    pivot=np.asarray(info['pivot']);rv=np.asarray(info['rotation_vector']);shift=np.asarray(info['shift'])
    offset=[]; max_pos=0.;max_rot=0.
    for t in knots:
        a=float(smoothstep((t-start)/(end-start)))
        R=Rotation.from_quat(pose[t,3:]).as_matrix()
        change=Rotation.from_rotvec(a*rv).as_matrix()
        newpos=change@(pose[t,:3]-pivot)+pivot+a*shift
        newR=change@R
        measured=feet(q[t]); dp=R.T@(newpos-pose[t,:3])
        dr=R.T@newR
        desired_pos=(measured[:,:3,3]-dp)@dr
        desired_rot=dr.T@measured[:,:3,:3]
        def residual(x):
            v=q[t].copy();v[legs]=x
            return foot_error(desired_pos,desired_rot,feet(v))
        lower=np.maximum(lo[legs]+.03,q[t,legs]-.12)
        upper=np.minimum(hi[legs]-.03,q[t,legs]+.12)
        if np.any(upper<=lower):return None
        fit=least_squares(residual,np.clip(q[t,legs],lower,upper),bounds=(lower,upper),max_nfev=25)
        err=fit.fun.reshape(2,6)
        max_pos=max(max_pos,np.linalg.norm(err[:,:3],axis=1).max())
        max_rot=max(max_rot,np.linalg.norm(err[:,3:],axis=1).max()/.18)
        if max_pos>.002 or max_rot>.01:
            print('IK boundary',info['id'],int(t),max_pos,max_rot,flush=True)
            return None
        offset.append(fit.x-q[t,legs])
    t=np.arange(1000)
    for j,k in enumerate(legs):
        target[:,k]+=np.interp(t,knots,np.array(offset)[:,j],left=0.)
    latent,executed=encode(target,trial['initial'],lo,hi)
    # Exact filter and slew cap are applied by encode; screen for saturation-induced path error.
    err=np.abs(executed-target).max()
    delay=0.
    if err>.010 and 'rim_feedback' in info:
        # Allocate time for the inward shift without increasing the motor slew rate.
        steps=np.diff(target,axis=0,prepend=trial['initial'][None,:23])
        timeline=np.cumsum(np.maximum(1.,abs(steps).max(axis=1)/.002995))-1.
        clock=np.interp(t,timeline,t)
        retimed=np.column_stack([np.interp(clock,t,v) for v in target.T])
        delay=float(t[-1]-clock[-1])
        if delay*.02>info['original_hold_s']-5.:return None
        latent,executed=encode(retimed,trial['initial'],lo,hi)
        target=retimed
        err=abs(executed-target).max()
    if err>.010:
        worst=np.unravel_index(np.argmax(abs(executed-target)),target.shape)
        print('command boundary',info['id'],worst,float(err),float(target[worst]),float(executed[worst]),flush=True)
        return None
    latent[750:]-=trial['residual']
    return latent,dict(ramp_ticks=[int(start),int(end)],max_foot_position_error_m=float(max_pos),
                       max_foot_rotation_error_rad=float(max_rot),encoded_error_rad=float(err),rate_limit_delay_ticks=delay)


def rim_feedback(prior,stage,row,target,info):
    """Restore the demonstrated inset using only samples before the observed patch violation."""
    p=prior/'candidate/evaluation'
    source=stage/'evaluation'
    construction=read(prior/'construction.json')
    assert construction['source']==str(stage) and construction['row']==row
    with np.load(p/'trajectory.npz') as z:
        first=int(np.flatnonzero(z['flags'][:,0,1])[0])
        assert z['wrong_force'][first,0]<10 and not z['physical_inside_patch'][first,0,0]
        assert z['no_slip'][:first,0].all()
    src=info['source']
    insets=[]
    for directory,i,g in [(source,row,src),(p,0,target)]:
        with np.load(directory/'edge_audit.npz') as z:pts=z['pelvis_contact_world_local'][first-5:first,i]
        a=Rotation.from_quat(g['rotation_xyzw']).as_matrix()
        x=((pts-np.array(g['top_center']))@a)[:,0]
        insets.append(float(np.nanmedian(g['depth']/2-x)))
    extra=insets[0]-insets[1]
    assert .01<extra<.05 and insets[0]>.01
    tangent=Rotation.from_quat(target['rotation_xyzw']).as_matrix()[:,0]
    return dict(window=[first-5,first],insets_m=insets,prior_candidate_sha256=sha(prior/'candidate.npz'),
                inward_m=extra), -extra*tangent


def build(out,prior=None):
    pair=out/'baseline';result=read(pair/'per_case.json')
    assert [v['id'] for v in result]==[TARGET,CONTROL]
    assert result[0]['link']=='left_hip_pitch_link' and result[0]['chair_force_N']>10 and result[1]['clean']
    lib=load_bundle(PARENT/'confirmed.npz')
    sites=np.array(signatures(pair));assert np.isfinite(sites[0]).all()
    model=BodyModel(MODEL)
    ranked=candidates(out,model)
    assert ranked,'No collision-clear registered donor.'
    rejected=[]
    for cost,stage,i,info in ranked:
        if prior is not None:
            ref=read(prior/'construction.json')
            if (str(stage),i)!=(ref['source'],ref['row']):continue
            audit,shift=rim_feedback(prior,stage,i,read(out/'pair_bank/geometry.json')[0],info)
            info=dict(info,rim_feedback=audit,shift=(np.asarray(info['shift'])+shift).tolist())
        fit=register(stage,i,info,model)
        if fit is not None:break
        rejected.append(info['id'])
    else:raise ValueError('No registration meets the two-foot pose and actuator limits.')
    motion,witness=fit; slot=len(lib['paths'])
    lib['paths']=np.concatenate((lib['paths'],motion[None,:]))
    lib['ids']=np.append(lib['ids'],'registered wooden armchair '+info['id'])
    # Use the existing narrow, directly observed seat site mechanism. Do not select by case id.
    lib['sites']=np.vstack((lib['sites'],sites[0]))
    lib['site_paths']=np.append(lib['site_paths'],slot)
    lib['site_radius']=np.append(lib['site_radius'],.5)
    for test in (sites[0],signatures(PARENT/'rollouts/fresh')[14]):
        assert choose(test,select(test,lib),lib)==slot,'Corrected motion not reachable from measured camera scan.'
    old=load_bundle(PARENT/'confirmed.npz')
    for test in (sites[1],signatures(PARENT/'rollouts/previous_fresh')[14]):
        assert choose(test,select(test,lib),lib)==choose(test,select(test,old),old),'Control routing changed.'
    np.savez_compressed(out/'candidate.npz',**lib)
    write(out/'construction.json',dict(source=str(stage),row=i,baseline=sha(PARENT/'confirmed.npz'),candidate=sha(out/'candidate.npz'),
          attribution=result[0],registration=dict(info,geometry_cost=cost,**witness),slot=slot,
          excluded_by_ik=rejected,scope='Camera-selected targeted approach; teacher geometry used only for offline construction.'))
    print(info['id'],cost,witness,flush=True)


def arrest_and_hold(out,prior):
    """Cancel measured touchdown travel, then maintain an observed quiet support posture."""
    from seat_contact_arrest import construct
    source=prior/'candidate';p=source/'evaluation'
    meta=read(p/'manifest.json')['metadata'];g=read(out/'pair_bank/geometry.json')[0]
    row=read(p/'results.json')['episodes'][0]
    assert not any(row['physical_failures']) and row['illegal_ticks']==0
    f=read(source/'evaluation.freeze.json')['command']
    trial=collect(source,actor_from(f[f.index('--checkpoint')+1]),actor_from(f[f.index('evaluate')+1]))[0]
    with np.load(p/'trajectory.npz') as z:
        slip=np.flatnonzero(~z['no_slip'][:,0]);groups=np.split(slip,np.where(np.diff(slip)!=1)[0]+1)
        assert len(groups)==2
        # Require half a second of uninterrupted qualified support on both sides of
        # the capture. Choose the quietest applied command in this measured window.
        good=z['good'][:,0];ctrl=trial['command']
        options=[t for t in range(int(groups[0][-1])+25,int(groups[1][0])-25) if good[t-25:t+26].all()]
        lock=min(options,key=lambda t:np.linalg.norm(np.diff(ctrl[t-12:t+13],axis=0),axis=1).mean())
        load=float(z['support_bw'][lock,0])
    limits=[np.array(meta[k]) for k in ['lower','upper']]
    plans=construct(source,0,trial,meta,g,limits,BodyModel(MODEL))
    path,witness=next((v,w) for method,v,w in plans if method=='arrest_1')
    assert witness['burst'][0]==int(groups[0][0])
    latent=path.copy()
    # Reconstruct what the unchanged command filter will actually have reached
    # at capture. A constant latent at this value needs no corrective motor step.
    last=trial['initial'][:23].copy()
    for desired in latent[:lock+1]:
        last+=np.clip(.1*(np.clip(desired,limits[0]+.03,limits[1]-.03)-last),-.003,.003)
    latent[lock+1:]=last
    latent[750:]-=trial['residual']
    lib=load_bundle(prior/'candidate.npz')
    original=read(prior/'construction.json');slot=original['slot']
    lib['paths'][slot]=latent
    np.savez_compressed(out/'candidate.npz',**lib)
    write(out/'construction.json',dict(original,prior_candidate=sha(prior/'candidate.npz'),
            candidate=sha(out/'candidate.npz'),measured_slip_bursts=[[int(v[0]),int(v[-1])] for v in groups],
            tangential_arrest=witness,capture_tick=lock,capture_load_bw=load))
    print('measured grip correction',witness,'capture',lock,load,flush=True)


def slow_loaded_foot(out, prior):
    """Spread the pre-seat foot rotation through time and resume the observed supported posture."""
    from compare_touchdown_ppo import verify_audit
    from humanoid_sitting.resting_objective import RestingObjectiveCfg

    source = prior/'candidate/evaluation'
    _, rows = verify_audit(source)
    assert rows[0]['slip_ticks'] == 1 and rows[0]['illegal_ticks'] == 0
    assert rows[0]['final_hold_s'] >= 5 and rows[1]['clean']
    cfg = RestingObjectiveCfg()
    meta = read(source/'manifest.json')['metadata']
    with np.load(source/'trajectory.npz') as z:
        tick = int(np.flatnonzero(~z['no_slip'][:, 0]).item())
        loads = np.linalg.norm(z['physical_force_w'][tick, 0], axis=-1) / z['weight'][tick, 0]
        speed = z['physical_slip_m_s'][tick, 0]
        moving = np.flatnonzero((loads > cfg.contact_active_load_bw) & (speed > cfg.maximum_slip_m_s))
        foot = int(moving.item())
        assert foot in (5, 6)
        seat_load = int(np.flatnonzero(z['support_bw'][:, 0] > cfg.contact_active_load_bw)[0])
        commands = z['command'][:, 0, :23]
    width = seat_load - tick
    assert 0 < width < 50
    ratio = float(speed[foot] / cfg.maximum_slip_m_s)
    ticks = np.arange(len(commands))
    # Slow progressively around the observed slip, then regain normal speed by seat load.
    weight = (1 + np.cos(np.pi * np.clip((ticks-tick) / width, -1, 1))) / 2
    wall_time = ticks + np.cumsum((ratio-1) * weight)
    clock = np.interp(ticks, wall_time, ticks)
    start = tick - width
    original = read(prior/'construction.json')
    finish = int(np.ceil(wall_time[original['capture_tick'] + 3]))
    assert finish < 750 and rows[0]['final_hold_s'] - (wall_time[-1]-ticks[-1])*cfg.control_dt >= 5
    target = np.column_stack([np.interp(clock[start:finish], ticks, col) for col in commands.T])
    path, actual = encode(target, commands[start-1], np.array(meta['lower']), np.array(meta['upper']))
    np.testing.assert_allclose(actual, target, atol=1e-5, rtol=0)
    np.testing.assert_allclose(actual[-1], commands[finish], atol=1e-5, rtol=0)
    lib = load_bundle(prior/'candidate.npz')
    lib['paths'][original['slot'], start:finish] = path
    np.savez_compressed(out/'candidate.npz', **lib)
    witness = dict(tick=tick, foot_channel=foot, speed_m_s=float(speed[foot]), load_bw=float(loads[foot]),
                   threshold_m_s=cfg.maximum_slip_m_s, first_seat_load=seat_load,
                   delay_s=float((wall_time[-1]-ticks[-1])*cfg.control_dt), applied_window=[start,finish])
    write(out/'construction.json', dict(original, prior_candidate=sha(prior/'candidate.npz'),
          candidate=sha(out/'candidate.npz'), foot_slowdown=witness))
    print('Loaded foot timing', witness, flush=True)


def join_supported_phases(out, foot_source, seat_source):
    """Keep the recorded slip-free foot approach and join the grip-preserving seat arrival."""
    from compare_touchdown_ppo import verify_audit
    from humanoid_sitting.resting_objective import RestingObjectiveCfg
    cfg = RestingObjectiveCfg()
    traces = []
    for folder in (foot_source, seat_source):
        p = folder/'candidate/evaluation'
        _, rows = verify_audit(p)
        assert rows[0]['illegal_ticks'] == 0 and rows[1]['clean']
        with np.load(p/'trajectory.npz') as z:
            traces.append((z['command'][:, 0, :23], np.flatnonzero(~z['no_slip'][:, 0]),
                           int(np.flatnonzero(z['support_bw'][:, 0] > cfg.contact_active_load_bw)[0])))
    foot_commands, foot_slips, foot_touch = traces[0]
    seat_commands, seat_slips, seat_touch = traces[1]
    assert len(seat_slips) == 1 and seat_slips[0] < seat_touch and foot_slips[0] == foot_touch
    start = int(seat_slips[0]) + 1
    target_info = read(seat_source/'construction.json')
    ticks = np.arange(len(foot_commands))
    delay = foot_touch - seat_touch
    assert 0 <= delay < foot_touch - start
    end = target_info['capture_tick'] + delay + 3
    section = ticks[start:end]
    weight = smoothstep((section-start)/(foot_touch-start))
    desired = foot_commands[section] + weight[:,None] * (seat_commands[section-delay] - foot_commands[section])
    meta = read(seat_source/'candidate/evaluation/manifest.json')['metadata']
    # Enforce the actuator limit on the transition itself. Restore the exact subsequent
    # pose once the joined commands reach the demonstrated quiet hold.
    start_command = foot_commands[start-1]
    changes = np.diff(desired, axis=0, prepend=start_command[None,:])
    wall = np.cumsum(np.maximum(1., abs(changes).max(axis=1)/.002999))-1.
    finish = start+int(np.ceil(wall[-1]))+1
    assert finish < 750
    clock = np.interp(np.arange(finish-start), wall, np.arange(len(desired)))
    targets = np.column_stack([np.interp(clock, np.arange(len(desired)), c) for c in desired.T])
    path, actual = encode(targets, start_command, np.array(meta['lower']), np.array(meta['upper']))
    np.testing.assert_allclose(actual, targets, atol=1e-5, rtol=0)
    np.testing.assert_allclose(actual[-1], seat_commands[finish], atol=1e-5, rtol=0)
    foot_lib = load_bundle(foot_source/'candidate.npz')
    lib = load_bundle(seat_source/'candidate.npz')
    slot = target_info['slot']
    assert read(foot_source/'construction.json')['slot'] == slot
    lib['paths'][slot,:start] = foot_lib['paths'][slot,:start]
    lib['paths'][slot,start:finish] = path
    np.savez_compressed(out/'candidate.npz', **lib)
    record = dict(foot_candidate=sha(foot_source/'candidate.npz'), seat_candidate=sha(seat_source/'candidate.npz'),
                  join_ticks=[start, foot_touch], seat_touch_tick=seat_touch, resume_tick=finish,
                  join_delay_ticks=float(wall[-1]-(len(wall)-1)))
    write(out/'construction.json', dict(target_info, candidate=sha(out/'candidate.npz'), phase_join=record))
    print('Foot to seat transition', record, flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('out',type=Path)
    method=parser.add_mutually_exclusive_group()
    method.add_argument('--rim-feedback-from',type=Path)
    method.add_argument('--grip-from',type=Path)
    method.add_argument('--foot-from',type=Path)
    method.add_argument('--join',nargs=2,type=Path,metavar=('FOOT','SEAT'))
    args=parser.parse_args()
    if args.join:join_supported_phases(args.out.resolve(),*args.join)
    elif args.foot_from:slow_loaded_foot(args.out.resolve(),args.foot_from)
    elif args.grip_from:arrest_and_hold(args.out.resolve(),args.grip_from)
    else:build(args.out.resolve(),args.rim_feedback_from)
