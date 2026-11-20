"""Construct load-transfer trajectories from the paired, stable chair recordings."""
import numpy as np

from evaluate_rim_motion import actor_from
from prepare_sitting_repairs import BodyModel, MODEL, encode, smoothstep
from refine_chair_contact import collect
from seat_contact_arrest import construct
from study_live_sitting import read


def paths_for_targets(base, prior, targets, meta, geom):
    calibration=read(base/'evaluation.freeze.json')['command']
    original=actor_from(calibration[calibration.index('--checkpoint')+1])
    trained=actor_from(calibration[calibration.index('evaluate')+1])
    limits=[np.asarray(meta[k]) for k in ('lower','upper')]
    model=BodyModel(MODEL)
    t=np.arange(1000)
    candidates={i:{} for i in targets}
    for stage in (base,base.with_name('fresh_reference'),*sorted((prior/'rollouts').glob('alternative_*'))):
        trials=collect(stage,original,trained)
        p=stage/'evaluation'
        with np.load(p/'trajectory.npz') as z:
            active=np.linalg.norm(z['physical_force_w'],axis=3)>.01*z['physical_bodyweight_N'][:,:,None]
            speed=z['physical_slip_m_s']
        for i in targets:
            trial=trials[i];row=trial['row']
            if not row['stable'] or any(row['physical_failures']) or row['illegal_ticks']:continue
            # Choose the stable, legal recording with the smallest slip burst;
            # never use a source with a failed hold to generate a grip repair.
            choice=candidates[i]
            if choice and choice['ticks']<=row['slip_ticks']:continue
            choice.clear()
            choice.update(ticks=row['slip_ticks'],stage=str(stage),motions={})
            bad=np.flatnonzero((active[:,i]&(speed[:,i]>.02)).any(axis=1))
            if not len(bad):continue
            pivot=int(bad[0]);end=int(bad[-1])
            cmd=trial['command']
            # Give the joints time to settle through the first loaded contact.
            # Extend the time spent in the observed slip window in proportion
            # to the ratio of peak patch speed to the original scoring speed.
            peak=float(np.where(active[:,i],speed[:,i],0.).max())
            multiplier=float(np.clip(peak/.02,1.3,3.5))
            window=smoothstep((t-(pivot-50))/25.)*(1.-smoothstep((t-end-12)/30.))
            timeline=t+np.cumsum(window*(multiplier-1.))
            clock=np.interp(t,timeline,t)
            slowed=np.stack([np.interp(clock,t,v) for v in cmd.T],axis=1)
            def add(name,target):
                latent,actual=encode(target,trial['initial'],*limits)
                if abs(actual-target).max()>.012:return
                latent[750:]-=trial['residual']
                choice['motions'][name]=latent
            add('loaded_slow',slowed)
            # Bring tangential travel of the patch toward zero while keeping
            # both feet anchored to their measured stance.
            for k,path,info in construct(stage,i,trial,meta,geom[i],limits,model):
                choice['motions'][k]=path
            # Reach the demonstrated support pose earlier and let the joints
            # settle at constant target through the observed contact interval.
            contact=int(trial['touch'])
            hold_tick=min(max(contact+4,pivot),end+3)
            pause=smoothstep((t-(contact-30))/25.)*(1.-smoothstep((t-end-8)/55.))
            target=cmd+(cmd[hold_tick]-cmd)*pause[:,None]
            step=np.diff(target,axis=0,prepend=trial['initial'][None,:])
            axis=np.cumsum(np.maximum(1.,abs(step).max(1)/.00299))-1.
            settled=np.stack([np.interp(t,axis,v) for v in target.T],axis=1)
            add('contact_dwell',settled)
            choice.update(burst=[pivot,end],peak_m_s=peak,multiplier=multiplier)
    return candidates


def support_matched_donors(prior_fit, targets, obstacle_geometry, lib, rejected):
    from evaluate_chair_contact_motion import bank
    from evaluate_rim_motion import shape_features
    from pathlib import Path
    cache={}
    examples=[]
    for d in read(prior_fit)['donors']:
        stage=Path(d['stage'])
        if stage not in cache:
            ids=read(stage/'evaluation/results.json')['episodes']
            geoms=read(bank(stage)/'geometry.json')
            cache[stage]={r['id']:g for r,g in zip(ids,geoms,strict=True)}
        g=cache[stage][d['id']]
        examples.append((d['slot'],g,shape_features(g)))
    options={}
    for i in targets:
        g=obstacle_geometry[i];feat=shape_features(g)
        ranked=sorted(examples,key=lambda v:np.linalg.norm((v[2]-feat)/[.010,.020,.045,.035])+
                      .75*(v[1].get('family')!=g.get('family')))
        paths=lib['paths']
        seen=list(rejected[i]);options[i]=[]
        for slot,donor,_ in ranked:
            if slot in seen or any(np.abs(paths[slot,370:800:15]-paths[j,370:800:15]).max()<.025 for j in seen if j>=0):
                continue
            seen.append(slot);options[i].append(slot)
            if len(options[i])==4:break
    return options
