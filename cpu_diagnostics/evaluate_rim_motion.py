"""Construct chair touchdown motions from saved traces, screen, then verify on untouched starts."""
import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from compare_touchdown_ppo import verify_audit
from evaluate_seat_contact_continuity import placement
from evaluate_seat_plane_approach import evaluate
from evaluate_seat_slip import reserve
from evaluate_standing_rim import comparison
from prepare_sitting_repairs import BodyModel, MODEL, encode, leg_fk
from repair_sitting_touchdown import retarget, retime
from study_live_sitting import ROOT, read, sha, write
from policy import Learner

PREVIOUS = ROOT/'outputs/guarded_seat_landing_01'
MATCHED = Path('/dev/shm/standing_rim_03/rollouts/matched')
TARGETS = ('round_stool/1','cantilever_chair/1','two_seat_sofa/2')
HERE = Path(__file__).resolve().parent


def shape_features(g):
    # Common world coordinates for design only. Deployment uses the measured rim.
    normal = np.array(g['normal'])
    slopes = normal[:2]/normal[2]
    center = np.array(g['top_center'])
    height = center[2] + slopes @ (center[:2]-[-.24,0.])
    yaw = g.get('yaw',0.)
    front = center[0]+.5*g['depth']*np.cos(np.deg2rad(g['pitch_deg']))/np.cos(yaw)+center[1]*np.tan(yaw)
    return np.r_[height,front,slopes]


def bank_subset(out, bank, indices):
    out.mkdir()
    geom = read(bank/'geometry.json')
    specs = [dict(geom[i],case=j,surface_id=j,label=f'probe/{j:02d}') for j,i in enumerate(indices)]
    write(out/'geometry.json',specs)
    metadata = read(bank/'metadata.json')
    write(out/'metadata.json',dict(metadata,cases=len(specs),ids=[g['label'] for g in specs]))
    with np.load(bank/'bank.npz') as z:
        data = {k:(v[:,indices] if k=='q_post' else v[indices]) for k,v in z.items()}
    data['reference'] = np.arange(len(indices))
    np.savez_compressed(out/'bank.npz',**data)


def actor_from(path):
    ck = torch.load(path,map_location='cpu',weights_only=False)
    learner = Learner(ck['mean'],ck['scale'],65011)
    learner.actor.load_state_dict(ck['actor'])
    return learner


def build(out):
    torch.set_num_threads(1)
    refs = [PREVIOUS/'rollouts'/s/'evaluation' for s in read(PREVIOUS/'summary.json')['known']]
    refs.append(MATCHED/'evaluation')
    base = refs[-1]
    bank = Path(read(base/'manifest.json')['args']['bank'])
    geometry = read(bank/'geometry.json')
    meta = read(bank/'metadata.json')
    models = BodyModel(MODEL)
    fk,legs = leg_fk(models,meta['joint_names'])
    limits = [np.array(meta[k]) for k in ('lower','upper')]
    command = read(MATCHED/'evaluation.freeze.json')['command']
    original = actor_from(command[command.index('--checkpoint')+1])
    trained = actor_from(command[command.index('evaluate')+1])
    donors = []
    for p in refs:
        _,rows = verify_audit(p)
        source = Path(read(p/'manifest.json')['args']['bank'])
        for i,(g,r) in enumerate(zip(read(source/'geometry.json'),rows,strict=True)):
            if r['clean']:
                donors.append((p,i,g))
    with np.load(base/'standing_rims.npz') as r,np.load(base/'standing_scan.npz') as z:
        rims,planes=r['values'],z['planes']
        f = np.nanmedian(np.stack((rims[:,:,0],rims[:,:,4],rims[:,:,5],
                                   np.where(rims[:,:,6]==1,planes[:,:,3]/planes[:,:,4],np.nan)),axis=-1),axis=0)
    @lru_cache(maxsize=2)
    def load_arrays(p):
        with np.load(p/'trajectory.npz') as z:
            return {k:z[k] for k in ('command','body','q','support_bw','previous','obs','physical_force_w','physical_bodyweight_N','physical_slip_m_s')}
    def sample(p,i):
        z=load_arrays(p)
        results = read(p/'results.json')
        body=z['body'][:,i,results['body_names'].index('pelvis')].copy()
        body[:,:3]-=results['origins'][i]
        motion=dict(body=body,q=z['q'][:,i],load=z['support_bw'][:,i])
        o=torch.as_tensor(z['obs'][750:,i])
        active=read(p/'grip_handoff.json')['active_ticks'][i]
        t=np.arange(250)
        ramp=np.clip(t/50.,0.,1.)
        alpha=torch.as_tensor(((t<=active)*ramp**2*(3-2*ramp))[:,None],dtype=o.dtype)
        with torch.no_grad():
            x=original.normalize(o)
            residual=torch.lerp(original.actor(x),trained.actor(x),alpha).numpy()
        return motion,z['command'][:,i,:23],z['previous'][0,i,:23],residual
    features,paths,sources,indices,records=[],[],[],[],[]
    def add(index,name,p,i,path,residual,checks):
        path=path.copy()
        path[750:]-=residual
        assert path.shape==(1000,23) and np.isfinite(path).all()
        features.append(f[index]);paths.append(path);sources.append(name);indices.append(index)
        records.append(dict(name=name,target=geometry[index]['label'],source=str(p),source_row=i,checks=checks))
    for index,g in enumerate(geometry):
        if not any(g['label'].endswith('/'+s) for s in TARGETS):
            continue
        # Slow the measured approach where loaded slip occurred. The earlier sofa
        # trace cleared the forbidden part and supplies a separate legal landing.
        histories=[base]
        if g['label'].endswith('two_seat_sofa/2'):
            histories.append(PREVIOUS/'rollouts/fresh/evaluation')
        for p in histories:
            verify_audit(p)
            motion,cmd,initial,residual=sample(p,index)
            z=load_arrays(p)
            loaded=np.linalg.norm(z['physical_force_w'][:,index,0],axis=-1)>.01*z['physical_bodyweight_N'][:,index]
            slip=np.flatnonzero(loaded & (z['physical_slip_m_s'][:,index,0]>.02))
            if not len(slip): continue
            for fraction in (.55,.78):
                path,_,_,checks=retime(cmd,initial,int(np.flatnonzero(loaded)[0]),int(slip[-1]),limits,fraction)
                add(index,f'{p.parent.name}/slower_{fraction}',p,index,path,residual,checks)
        goal=shape_features(g)
        def distance(d):
            # Maximum plane mismatch over the resting pelvis and centerline rim displacement.
            v=np.abs(shape_features(d[2])-goal)
            return v @ [.95,.32,.11,.10]
        ranked=sorted(donors,key=distance)
        seen=set()
        for p,i,source in ranked:
            signature=tuple(np.round(shape_features(source),5))
            if signature in seen: continue
            seen.add(signature)
            motion,cmd,initial,residual=sample(p,i)
            # Direct transfer is the control. Fully aligned transfer preserves both donor foot poses.
            path,_=encode(cmd,initial,*limits)
            add(index,source['label']+'/direct',p,i,path,residual,dict(geometry_distance=distance((p,i,source))))
            dx=np.clip(goal[1]-shape_features(source)[1],-.018,.018)
            try:
                waypoint,checks=retarget(motion,source,g,limits,cmd,models,fk,legs,fore_aft_m=float(dx))
                path,actual=encode(waypoint,initial,*limits)
                if np.max(abs(actual-waypoint))<=.015:
                    add(index,source['label']+'/aligned',p,i,path,residual,checks)
            except (ValueError,AssertionError) as exc:
                print('donor IK rejected',source['label'],str(exc)[:120],flush=True)
            if len(seen)==5: break
    assert len(set(indices))==3
    np.savez_compressed(out/'screen_paths.npz',features=features,paths=paths,ids=sources,indices=indices)
    write(out/'source_motions.json',records)
    bank_subset(out/'screen_bank',bank,indices)


def run(out,prepare=build,sources=()):
    reservation=out/'reservation.json'
    refs={k:PREVIOUS/'rollouts'/k for k in read(PREVIOUS/'summary.json')['known']}
    refs['matched']=MATCHED
    banks=[Path(read(p/'evaluation/manifest.json')['args']['bank']) for p in refs.values()]
    known=[g for b in banks for g in read(b/'geometry.json')]
    assert len(known)==476
    fresh=reserve(out,PREVIOUS,seed=31968211,prefix='rim_motion_holdout')
    places=list(map(placement,read(fresh/'geometry.json')))
    assert len(set(places))==24 and set(places).isdisjoint(map(placement,known))
    prepare(out)
    def launch(label,ref,library,bank=None,screen=False):
        dest=out/'rollouts'/label
        evaluate(dest,ref,bank=bank,rim_motion=library,screen_motion=screen,
                 inputs=[Path(__file__),reservation,out/'source_motions.json',HERE/'standing_rim.py',*sources])
        return dest
    probe=launch('screen',MATCHED,out/'screen_paths.npz',out/'screen_bank',True)
    _,rows=verify_audit(probe/'evaluation')
    _,previous=verify_audit(MATCHED/'evaluation')
    with np.load(out/'screen_paths.npz') as z:
        selected=[]
        for i in sorted(set(z['indices'])):
            options=np.flatnonzero(z['indices']==i)
            clean=[j for j in options if rows[j]['clean']]
            if clean:
                selected.append(max(clean,key=lambda j:rows[j]['final_hold_s']))
            else:
                # Carry forward an incremental repair only when the individual physical
                # flags, stability and slip/illegal counts are no worse than the incumbent.
                a=previous[i]
                def compatible(b):
                    return (b['stable']>=a['stable'] and b['illegal_ticks']<=a['illegal_ticks']
                            and b['slip_ticks']<=a['slip_ticks']
                            and all(y<=x for x,y in zip(a['physical_failures'],b['physical_failures'])))
                viable=[j for j in options if compatible(rows[j])]
                if viable:
                    selected.append(min(viable,key=lambda j:(any(rows[j]['physical_failures']),not rows[j]['stable'],rows[j]['slip_ticks'],-rows[j]['final_hold_s'])))
        if selected:
            lib=out/'motions.npz'
            np.savez_compressed(lib,**{k:z[k][selected] for k in ('paths','ids','features')})
        else:
            # Explicit empty library makes the mandatory paired retest exercise the incumbent.
            lib=out/'motions.npz'
            np.savez_compressed(lib,paths=z['paths'][:0],ids=z['ids'][:0],features=z['features'][:0])
        write(out/'selection.json',[dict(target=int(z['indices'][j]),motion=str(z['ids'][j]),probe=rows[j]) for j in selected])
    dest=launch('matched',MATCHED,lib)
    matched=comparison(dest,MATCHED)
    summary=dict(matched=matched,qualified=matched['after']['clean']>matched['before']['clean'] and not matched['regressions'])
    write(out/'summary.json',summary)
    print('matched',summary,flush=True)
    if summary['qualified']:
        summary['known']={'matched':matched}
        for k,ref in refs.items():
            if k=='matched': continue
            summary['known'][k]=comparison(launch(k,ref,lib),ref)
            write(out/'summary.json',summary)
        assert sum(r['total'] for r in summary['known'].values())==476
        assert all(sha(Path(p))==v for p,v in read(reservation).items())
        reference=out/'rollouts/fresh_reference'
        evaluate(reference,MATCHED,bank=fresh,inputs=[reservation,HERE/'standing_rim.py'])
        summary['fresh']=comparison(launch('fresh',reference,lib,fresh),reference)
        write(out/'summary.json',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    run(parser.parse_args().output.resolve())
