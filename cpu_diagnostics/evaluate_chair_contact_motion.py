"""Build legal touchdown paths from recorded chair trials, then validate by paired simulation."""
import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from chair_contact_motion import descriptor, depth_tiles, select
from compare_touchdown_ppo import verify_audit
from evaluate_rim_motion import actor_from, shape_features
from evaluate_seat_contact_continuity import placement
from evaluate_seat_plane_approach import evaluate
from evaluate_seat_slip import reserve
from evaluate_standing_rim import comparison
from prepare_sitting_repairs import BodyModel, MODEL, encode, leg_fk
from repair_sitting_touchdown import retarget
from study_live_sitting import read, write, sha

PARENT = Path('/dev/shm/approach_route_view_02')
CAMPAIGN = Path('/mnt/robot-training/humanoid-sitting/chair_shapes_20261116_campaign')
HERE = Path(__file__).resolve().parent


def bank(stage):
    return Path(read(stage/'evaluation/manifest.json')['args']['bank'])


def signatures(stage):
    p = stage/'evaluation'
    with np.load(p/'standing_scan.npz') as z, np.load(p/'standing_rims.npz') as r:
        rims = r['values']
        v = rims[:, :, [0, 4, 5]].copy()
        v[rims[:, :, 6] != 1.] = np.nan
        planes, tiles = z['planes'], depth_tiles(z['depth'])
    return [descriptor(v[:, i], planes[:, i], tiles[:, i]) for i in range(v.shape[1])]


def references():
    keys = read(PARENT/'summary.json')['known']
    stages = PARENT/'rollouts'
    refs = {('prior_matched' if k=='matched' else k): stages/('previous_fresh' if k=='fresh' else k) for k in keys}
    refs['matched'] = stages/'fresh'
    assert sum(len(read(bank(s)/'metadata.json')['ids']) for s in refs.values()) == 524
    return refs


def build(out, refs):
    torch.set_num_threads(1)
    base = refs['matched']
    target_geometry = read(bank(base)/'geometry.json')
    meta = read(bank(base)/'metadata.json')
    limits = [np.asarray(meta[k]) for k in ('lower','upper')]
    model = BodyModel(MODEL)
    fk, legs = leg_fk(model, meta['joint_names'])
    _, rows = verify_audit(base/'evaluation')
    targets = [i for i, r in enumerate(rows) if r['slip_ticks'] or r['physical_failures'][1]]
    assert (len(targets),sum(bool(r['slip_ticks']) for r in rows),sum(r['physical_failures'][1] for r in rows)) == (14,14,4)
    f = np.array(signatures(base))
    retain = np.concatenate([np.array(signatures(s)) for k,s in refs.items() if k!='matched'] +
                            [np.delete(f,targets,axis=0)])
    # Match transfer candidates by plane gap at the donor contact patch, its
    # distance from the front rim, slope and landing posture. All geometry
    # labels are confined to offline construction.
    examples = []
    files = {}
    for s in dict.fromkeys([*refs.values(), *sorted(CAMPAIGN.glob('batch_*'))]):
        p = s/'evaluation'
        _, records = verify_audit(p)
        files[str(p/'independent_audit.json')] = sha(p/'independent_audit.json')
        g = read(bank(s)/'geometry.json')
        with np.load(p/'trajectory.npz') as traces, np.load(p/'edge_audit.npz') as edge:
            load = traces['support_bw']
            q = traces['command'][[390,520,695],:,:23].transpose(1,0,2)
            patch = edge['pelvis_contact_world_local']
        for i,r in enumerate(records):
            if not r['clean']: continue
            loaded = np.flatnonzero(load[:,i]>.05)
            if not len(loaded): continue
            point = np.median(patch[loaded[:12],i],axis=0)
            if not np.isfinite(point).all(): continue
            examples.append((s,i,g[i],shape_features(g[i]),point,q[i]))
    assert examples
    np.savez_compressed(out/'observations.npz',features=f,retain=retain,targets=targets)
    command = read(base/'evaluation.freeze.json')['command']
    original = actor_from(command[command.index('--checkpoint')+1])
    trained = actor_from(command[command.index('evaluate')+1])

    @lru_cache(maxsize=2)
    def load(s):
        p = s/'evaluation'
        with np.load(p/'trajectory.npz') as z:
            data = {k:z[k] for k in ('q','command','desired','previous','support_bw','no_slip')}
            data['obs'] = z['obs'][750:]
            data['body'] = z['body'][:,:,0]
        data['origins'] = np.array(read(p/'results.json')['origins'])
        data['trained_ticks'] = read(p/'grip_handoff.json')['active_ticks']
        return data

    @lru_cache(maxsize=100)
    def sample(s,i):
        z=load(s)
        body=z['body'][:,i].copy()
        body[:,:3]-=z['origins'][i]
        obs=torch.as_tensor(z['obs'][:,i].copy())
        t=np.arange(250)
        phase=np.minimum(1.,t/50.)
        alpha=torch.as_tensor(((t<=z['trained_ticks'][i])*phase**2*(3.-2.*phase))[:,None],dtype=obs.dtype)
        with torch.no_grad():
            normalized=original.normalize(obs)
            residual=torch.lerp(original.actor(normalized),trained.actor(normalized),alpha).numpy()
        return (dict(body=body,q=z['q'][:,i].copy(),load=z['support_bw'][:,i].copy()),
                z['command'][:,i,:23].copy(),z['previous'][0,i,:23].copy(),z['desired'][:,i].copy(),residual)

    records,paths,indices,methods = [],[],[],[]
    def add(i,method,path,residual,source,source_row,info):
        path=path.astype(np.float32,copy=True)
        path[750:]-=residual
        assert path.shape==(1000,23) and np.isfinite(path).all()
        records.append(dict(target=rows[i]['id'],method=method,source=str(source),row=int(source_row),**info))
        paths.append(path)
        indices.append(i)
        methods.append(method)

    for i in targets:
        g=target_geometry[i]
        goal=shape_features(g)
        def mismatch(d):
            _,_,src,feat,point,_=d
            target_gap=float((point-np.array(g['top_center']))@g['normal'])
            source_gap=float((point-np.array(src['top_center']))@src['normal'])
            return np.linalg.norm((feat-goal)/[.015,.026,.050,.035]) + .8*abs(target_gap-source_gap)/.012
        chosen=[]
        for d in sorted(examples,key=mismatch):
            if any(np.max(abs(d[-1]-other[-1]))<.025 for other in chosen): continue
            chosen.append(d)
            if len(chosen)==3: break
        assert len(chosen)==3
        for rank,(s,j,src,feat,point,_) in enumerate(chosen):
            motion,cmd,initial,desired,residual=sample(s,j)
            info=dict(donor=src['label'],estimated_plane_gap_m=float((point-np.array(g['top_center']))@g['normal']))
            add(i,f'donor_{rank}',desired,residual,s,j,info)
            # Seat-height and edge alignment are finished before first load.
            dx=float(np.clip(goal[1]-feat[1],-.024,.024))
            for pitch,label in ((0.,'aligned'),(.060,'clearance')):
                if label=='clearance' and (rank!=0 or not rows[i]['physical_failures'][1]): continue
                try:
                    way,witness=retarget(motion,src,g,limits,cmd,model,fk,legs,fore_aft_m=dx,pitch_adjust_rad=pitch)
                except (AssertionError,ValueError):
                    continue
                latent,executed=encode(way,initial,*limits)
                err=float(abs(executed-way).max())
                if err<=.015:
                    add(i,f'{label}_{rank}',latent,residual,s,j,dict(**info,fit=witness,tracking_rad=err))
        own,cmd,initial,desired,residual=sample(base,i)
        z=load(base)
        slip=np.flatnonzero(~z['no_slip'][:,i])
        assert len(slip)
        # Slow the approach through the observed slip window without a catch-up
        # acceleration. Keep the original horizon and motor authority.
        t=np.arange(1000)
        start=max(160,int(slip[0])-45)
        stop=min(999,int(slip[-1])+22)
        phase=np.clip((t-start)/18,0,1)*np.clip((stop-t)/18,0,1)
        clock=np.interp(t,t+np.cumsum(phase**2*(3.-2.*phase)*.65),t)
        soft=np.stack([np.interp(clock,t,col) for col in cmd.T],axis=-1)
        latent,executed=encode(soft,initial,*limits)
        add(i,'soft_transfer',latent,residual,base,i,dict(slip_window=[int(slip[0]),int(slip[-1])],
                first_slow_tick=start,tracking_rad=float(abs(executed-soft).max())))
        print('constructed',rows[i]['id'],flush=True)
    write(out/'construction.json',dict(sources=files,clean_donors=len(examples),motions=records))
    np.savez_compressed(out/'candidates.npz',indices=indices,methods=methods,paths=paths,
                        ids=[r['method']+' '+r.get('donor',r['target']) for r in records])


def run(out, *, matched_only=False):
    out.mkdir(exist_ok=True)
    stages=out/'rollouts'
    stages.mkdir(exist_ok=True)
    refs=references()
    base=refs['matched']
    fresh=reserve(out,PARENT,seed=68192451,prefix='contact_transfer_holdout')
    places={placement(g) for s in refs.values() for g in read(bank(s)/'geometry.json')}
    reserved={placement(g) for g in read(fresh/'geometry.json')}
    assert len(reserved)==24 and reserved.isdisjoint(places)
    build(out,refs)
    inputs=[Path(__file__),out/'construction.json',out/'reservation.json',out/'observations.npz',out/'candidates.npz']
    def launch(label,ref,library,bank_override=None):
        d=stages/label
        evaluate(d,ref,contact_motion=library,bank=bank_override,inputs=inputs)
        return comparison(d,ref)
    with np.load(out/'observations.npz') as z:
        features,retain,targets=(z[k] for k in ('features','retain','targets'))
    with np.load(out/'candidates.npz') as z:
        indices,methods,paths,ids=(z[k] for k in ('indices','methods','paths','ids'))
    def library(label,selected):
        used=np.array(sorted(selected),dtype=int)
        slots=np.array([selected[i] for i in used],dtype=int)
        data=dict(features=features[used],retain=np.concatenate((retain,features[np.setdiff1d(targets,used)])),
                  paths=paths[slots],ids=ids[slots])
        for j,i in enumerate(used): assert select(features[i],data)==j,(label,i)
        dest=out/(label+'.npz')
        np.savez_compressed(dest,**data)
        return dest
    selected={}
    probes={}
    for method in ('donor_0','aligned_0','soft_transfer','clearance_0','donor_1','aligned_1','donor_2','aligned_2'):
        options={int(i):int(j) for j,i in enumerate(indices) if methods[j]==method and i not in selected}
        if not options: continue
        s=launch('probe_'+method,base,library(method,options))
        p=stages/('probe_'+method)/'evaluation'
        _,rows=verify_audit(p)
        with np.load(p/'chair_contact_motion.npz') as z:
            activated=z['selected']
        for i,j in options.items():
            if activated[i]>=0 and rows[i]['clean']: selected[i]=j
        probes[method]=dict(tried=list(options),clean=[i for i in options if i in selected],repairs=s['repairs'])
        write(out/'probe_results.json',probes)
        print(method,'recovered so far',len(selected),'/',len(targets),flush=True)
    final=library('confirmed',selected)
    write(out/'selection.json',{str(i):dict(motion=int(j),source=str(ids[j])) for i,j in selected.items()})
    inputs.append(out/'selection.json')
    matched=launch('matched',base,final)
    summary=dict(matched=matched,qualified=matched['after']['clean']>matched['before']['clean'] and not matched['regressions'])
    write(out/'summary.json',summary)
    print('gate',summary['qualified'],matched['after'],matched['regressions'],flush=True)
    if matched_only or not summary['qualified']: return
    summary['known']={'matched':matched}
    for name,reference in refs.items():
        if name=='matched': continue
        label='previous_fresh' if name=='fresh' else name
        summary['known'][name]=launch(label,reference,final)
        write(out/'summary.json',summary)
        print(label,summary['known'][name]['after'],flush=True)
    assert sum(s['total'] for s in summary['known'].values())==524
    assert all(sha(Path(path))==digest for path,digest in read(out/'reservation.json').items())
    control=stages/'fresh_reference'
    evaluate(control,base,bank=fresh,inputs=inputs)
    summary['fresh']=launch('fresh',control,final,fresh)
    write(out/'summary.json',summary)
    print('fresh',summary['fresh']['after'],summary['fresh']['regressions'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--matched-only', action='store_true',
                        help='Stop after the paired retest when preparing inputs for refine_chair_contact.py.')
    args=parser.parse_args()
    run(args.output.resolve(), matched_only=args.matched_only)
