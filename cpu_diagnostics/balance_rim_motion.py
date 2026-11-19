"""Lower the pelvis along the observed surface normal before rearward pitch accelerates."""
from pathlib import Path
import argparse
import numpy as np

from compare_touchdown_ppo import verify_audit
from evaluate_rim_motion import MATCHED, bank_subset, run
from prepare_sitting_repairs import BodyModel, MODEL, encode, smoothstep
from seat_plane_approach import SeatPlaneApproach, seat_normal
from study_live_sitting import read, write, sha


def prepare(out, study):
    p=study/'rollouts/matched/evaluation'
    _,rows=verify_audit(p)
    bank=Path(read(p/'manifest.json')['args']['bank'])
    meta=read(bank/'metadata.json')
    model=BodyModel(MODEL)
    approach=SeatPlaneApproach(model,meta,len(rows))
    low,high=[np.array(meta[k]) for k in ('lower','upper')]
    with np.load(study/'motions.npz') as z:
        features,ids,priors=[z[k] for k in ('features','ids','paths')]
    with np.load(p/'rim_motion.npz') as z:
        selected=z['selected']
    with np.load(p/'trajectory.npz') as z:
        frames=z['obs'][:,:,222:333]
        load=np.linalg.norm(z['physical_force_w'][:,:,0],axis=-1)/z['physical_bodyweight_N']
        previous=z['previous'][0,:,:23]
        cmd=z['command'][:,:,:23]
    designs=[];paths=[];indices=[];features_out=[]
    ticks=np.arange(1000)
    # Freeze design against the most recent individually tested legal landing.
    for item in read(study/'selection.json'):
        i=item['target'];j=selected[i]
        assert j>=0
        prior=priors[j]
        def add(name,path,checks):
            designs.append(dict(motion=name,target=rows[i]['id'],checks=checks))
            paths.append(path);indices.append(i);features_out.append(features[j])
        add(ids[j],prior,{})
        if rows[i]['clean']: continue
        touch=int(np.flatnonzero(load[:,i]>.01)[0])
        history=frames[:41:10,i]
        normals=[seat_normal(approach.feet(frame[:23]),frame[69:72],frame[-13:]) for frame in history]
        normal=np.median([n for n in normals if n is not None],axis=0)
        normal/=np.linalg.norm(normal)
        full=touch-50
        ramp=smoothstep((ticks-(full-200))/200)
        # Early normal translation reaches at most 18 mm before touchdown. Arm
        # counterweight uses the same early ramp; unwind arms only after load settles.
        arm=ramp*(1.-smoothstep((ticks-touch-50)/230))
        arm_columns=[meta['joint_names'].index(f'{s}_{k}_pitch_joint') for s in ('left','right') for k in ('shoulder','elbow')]
        knots=np.unique(np.r_[np.arange(0,1000,10),full,999])
        for depth in (0.,.009,.018):
            offsets=[]
            for t in knots:
                shift=depth*ramp[t]
                if not shift:
                    offsets.append(np.zeros(23))
                    continue
                fit=approach.correct(cmd[t,i],frames[t,i],normal,0.,normal_m=-shift)
                if fit is None: break
                offsets.append(fit[0])
            if len(offsets)!=len(knots): continue
            offsets=np.stack([np.interp(ticks,knots,col) for col in np.array(offsets).T],axis=1)
            for angle in (0.,.16,.32,.48):
                if angle==depth==0: continue
                delta=offsets.copy()
                delta[:,arm_columns]-=angle*arm[:,None]
                # Add only the change in encoded reference. This preserves the
                # archived hold residual subtraction and the existing live actor.
                base,_=encode(cmd[:,i],previous[i],low,high)
                requested=cmd[:,i]+delta
                # Allocate wall time to every knot using the original joint step cap.
                # Faster legs cannot drag a saturated hip past its rate limit.
                increment=np.diff(requested,axis=0,prepend=previous[i,None,:])
                durations=np.maximum(1.,np.max(abs(increment),axis=1)/.00298)
                schedule=np.cumsum(durations)-1.
                clock=np.interp(ticks,schedule,ticks)
                target=np.stack([np.interp(clock,ticks,col) for col in requested.T],axis=1)
                latent,applied=encode(target,previous[i],low,high)
                error=float(abs(applied-target).max())
                if error>.004: continue
                residual_offset=prior-base
                residual_offset[:750]=0.
                add(f'{ids[j]} normal={depth:g} arms={angle:g}',latent+residual_offset,
                    dict(touch_tick=touch,full_correction_tick=full,normal_lowering_m=depth,
                         arm_advance_rad=angle,command_tracking_rad=error,extra_ticks_at_load=float(schedule[touch]-touch)))
    write(out/'source_motions.json',dict(parent_inputs={str(x):sha(x) for x in (p/'trajectory.npz',p/'rim_motion.npz',study/'motions.npz')},motions=designs))
    np.savez_compressed(out/'screen_paths.npz',paths=paths,features=features_out,indices=indices,ids=[r['motion'] for r in designs])
    bank_subset(out/'screen_bank',bank,indices)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path);parser.add_argument('previous',type=Path)
    args=parser.parse_args()
    run(args.output.resolve(),prepare=lambda out:prepare(out,args.previous.resolve()),sources=(Path(__file__),))
