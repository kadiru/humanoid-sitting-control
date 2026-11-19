"""Paired validation of two-view approach selection on lost passes and new chair placements."""
import argparse
from pathlib import Path

import numpy as np

from approach_route_view import select_view, signature
from compare_touchdown_ppo import verify_audit
from evaluate_seat_contact_continuity import placement
from evaluate_seat_plane_approach import evaluate
from evaluate_seat_slip import reserve
from evaluate_standing_rim import comparison
from study_live_sitting import ROOT, read, sha, write

PREVIOUS = Path('/dev/shm/rim_landing_final')
NOMINAL = ROOT/'outputs/guarded_seat_landing_01'
HERE = Path(__file__).resolve().parent


def features(folder):
    with np.load(folder/'standing_rims.npz') as scan, np.load(folder/'standing_scan.npz') as poses:
        v = scan['values']
        optics = v[:,:,[0,4,5]].copy()
        optics[v[:,:,6]!=1.] = np.nan
        planes = poses['planes']
        return [signature(optics[:,i],planes[:,i]) for i in range(v.shape[1])]


def bank_path(stage):
    return Path(read(stage/'evaluation/manifest.json')['args']['bank'])


def fit(out, refs):
    preferred, protected, cases, target = [], [], [], {'slabs':[], 'chairs':[]}
    provenance = {}
    for name,stage in refs.items():
        if name in ('fresh','matched'):
            continue
        historic = NOMINAL/'rollouts'/name
        old, new = (verify_audit(s/'evaluation')[1] for s in (historic,stage))
        assert bank_path(historic)==bank_path(stage)
        f = features(stage/'evaluation')
        cmd = read(stage/'evaluation.freeze.json')['command']
        renderer = 'chairs' if '--worker' in cmd[:cmd.index('evaluate')] else 'slabs'
        for i,(a,b,(frontal,x)) in enumerate(zip(old,new,f,strict=True)):
            assert a['id']==b['id']
            lost = a['clean'] and not b['clean']
            if lost:
                target[renderer].append((name,i))
            if not frontal and x is not None:
                if lost:
                    preferred.append(x)
                elif not a['clean']:
                    protected.append(x)
            cases.append((name,frontal,x,lost))
        for s in (historic,stage):
            audit = s/'evaluation/independent_audit.json'
            provenance[str(audit)] = sha(audit)
        for file in ('standing_scan.npz','standing_rims.npz'):
            p = stage/'evaluation'/file
            provenance[str(p)] = sha(p)
    assert sum(map(len,target.values()))==43
    library = dict(nominal=np.array(preferred).reshape(-1,4), protect=np.array(protected).reshape(-1,4))
    np.savez(out/'routing.npz',**library)
    projected = sum(lost and select_view(front,x,library) for _,front,x,lost in cases)
    write(out/'calibration.json',dict(inputs=provenance,cases=len(cases),lost=43,
          nominal_centers=len(preferred),protected_centers=len(protected),predicted_nominal_on_lost=int(projected),
          excluded_banks=['matched','fresh']))
    return target


def repack(destination, refs, targets):
    geometry, ids, columns = [], [], {}
    meta = None
    command = None
    fields = ('task','joint_names','joint_ids','body_names','lower','upper')
    for name,i in targets:
        stage = refs[name]
        cmd = read(stage/'evaluation.freeze.json')['command']
        cmd = cmd[:cmd.index('--mode')]
        assert command is None or command==cmd
        command = cmd
        source = bank_path(stage)
        current = read(source/'metadata.json')
        assert meta is None or all(current[k]==meta[k] for k in fields)
        meta = current
        shape = read(source/'geometry.json')[i]
        ids.append(meta['ids'][i])
        geometry.append(dict(shape,case=len(geometry),surface_id=len(geometry)))
        with np.load(source/'bank.npz') as z:
            reference = int(z['reference'][i])
            for k,v in z.items():
                if k!='reference':
                    columns.setdefault(k,[]).append(v[:,reference] if k=='q_post' else v[i])
    assert len(set(ids))==len(ids)
    destination.mkdir()
    write(destination/'geometry.json',geometry)
    write(destination/'metadata.json',dict(meta,ids=ids,cases=len(ids)))
    np.savez_compressed(destination/'bank.npz',reference=np.arange(len(ids)),
                        **{k:np.stack(v,axis=1 if k=='q_post' else 0) for k,v in columns.items()})


def run(out, repeat=None):
    out.mkdir(exist_ok=True)
    stages=out/'rollouts'
    stages.mkdir(exist_ok=True)
    previous=read(PREVIOUS/'summary.json')
    refs={name:PREVIOUS/'rollouts'/name for name in (*previous['known'],'fresh')}
    assert sum(len(read(bank_path(p)/'metadata.json')['ids']) for p in refs.values())==500
    routing=out/'routing.npz'
    targets=fit(out,refs)
    bank=reserve(out,PREVIOUS,seed=61192413,prefix='dual_view_holdout')
    prior_places={placement(g) for p in refs.values() for g in read(bank_path(p)/'geometry.json')}
    new_places={placement(g) for g in read(bank/'geometry.json')}
    assert len(new_places)==24 and prior_places.isdisjoint(new_places)
    inputs=[Path(__file__),out/'reservation.json',out/'calibration.json']
    def evaluate_pair(label,ref,bank=None):
        dest=stages/label
        evaluate(dest,ref,bank=bank,approach_view=routing,inputs=inputs)
        result=comparison(dest,ref)
        with np.load(dest/'evaluation/approach_view.npz') as z:
            result['nominal_view_rows']=np.flatnonzero(z['nominal']).tolist()
        return result
    if repeat is not None:
        # Select between separately feeding the controller the tilted rim at
        # touchdown and retaining the incumbent for contradictory examples.
        # Classifier inputs throughout are standing measurements only.
        with np.load(routing) as z:
            library={k:z[k] for k in z.files}
        conflict=[]
        extra_protected=[]
        for group in ('slabs','chairs','matched'):
            trial=repeat/'rollouts'/group
            values=features(trial/'evaluation')
            for pair,(_,x) in zip(read(trial/'comparison.json'),values,strict=True):
                a,b=pair['before'],pair['after']
                if x is not None and a['stable'] and not b['stable']:
                    conflict.append(x)
                elif x is not None and b['slip_ticks']>a['slip_ticks'] and not b['clean']:
                    extra_protected.append(x)
        library['conflict']=np.array(conflict).reshape(-1,4)
        library['protect']=np.vstack((library['protect'],np.array(extra_protected).reshape(-1,4)))
        library['use_feedback']=True
        probe=out/'feedback_probe.npz'
        np.savez(probe,**library)
        control=repeat/'rollouts/chairs_reference'
        dest=stages/'feedback_probe'
        evaluate(dest,control,approach_view=probe,inputs=inputs)
        trial=comparison(dest,control)
        with np.load(dest/'evaluation/approach_view.npz') as z:
            used=z['separate_feedback_rows']
        rows=verify_audit(dest/'evaluation')[1]
        use=bool(len(used) and all(rows[i]['clean'] for i in used) and not trial['regressions'])
        library['use_feedback']=use
        routing=out/'confirmed_routing.npz'
        np.savez(routing,**library)
        write(out/'feedback_selection.json',dict(use=use,rows=used.tolist(),comparison=trial,
              first_stage_sources={str(q):sha(q) for q in (repeat/'rollouts'/g/'comparison.json'
                                    for g in ('slabs','chairs','matched'))}))
        inputs.append(out/'feedback_selection.json')
        print('separate touchdown feedback',use,trial['after'],flush=True)

    screens={}
    for group,subset in targets.items():
        if repeat is None:
            sub_bank=out/(group+'_bank')
            repack(sub_bank,refs,subset)
            ref=refs[subset[0][0]]
            control=stages/(group+'_reference')
            evaluate(control,ref,bank=sub_bank,inputs=inputs)
        else:
            control=repeat/'rollouts'/(group+'_reference')
            sub_bank=bank_path(control)
            verify_audit(control/'evaluation')
        screens[group]=evaluate_pair(group,control,sub_bank)
        print(group,screens[group]['before'],screens[group]['after'],screens[group]['regressions'],flush=True)
    screens['matched']=evaluate_pair('matched',refs['matched'])
    qualified=(all(s['after']['clean']>=s['before']['clean'] and not s['regressions'] for s in screens.values())
               and any(s['after']['clean']>s['before']['clean'] for s in screens.values()))
    summary=dict(screens=screens,qualified=qualified)
    write(out/'summary.json',summary)
    print('gate',qualified,screens['matched']['after'],screens['matched']['regressions'],flush=True)
    if not qualified:
        return
    summary['known']={'matched':screens['matched']}
    for name,ref in refs.items():
        if name!='matched':
            summary['known'][name]=evaluate_pair('previous_fresh' if name=='fresh' else name,ref)
            write(out/'summary.json',summary)
            print(name,summary['known'][name]['after']['clean'],flush=True)
    assert sum(s['total'] for s in summary['known'].values())==500
    assert all(sha(Path(p))==d for p,d in read(out/'reservation.json').items())
    control=stages/'fresh_reference'
    evaluate(control,refs['fresh'],bank=bank,inputs=inputs)
    summary['fresh']=evaluate_pair('fresh',control,bank)
    write(out/'summary.json',summary)
    print('fresh',summary['fresh']['after'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--repeat',type=Path)
    args=parser.parse_args()
    run(args.output.resolve(),args.repeat)
