"""Validate pre-contact warnings, whole-motion selection and loaded-hold feedback."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from chair_contact_motion import select
from compare_touchdown_ppo import verify_audit
from evaluate_chair_contact_motion import CAMPAIGN, bank, signatures
from evaluate_scan_landing import load_bundle, GPU
from evaluate_seat_contact_continuity import placement
from evaluate_seat_plane_approach import evaluate
from evaluate_seat_slip import reserve
from evaluate_standing_rim import comparison
from hold_clearance import StanceHull, TICKS, choose, risk, signals, warns
from prepare_sitting_repairs import BodyModel, MODEL
from study_live_sitting import read, sha, write

PARENT=Path('/dev/shm/scan_landing_02')
FIRST=PARENT.with_name('scan_landing_01')
HERE=Path(__file__).resolve().parent


def references():
    refs={('scan_matched' if k=='matched' else k):PARENT/'rollouts'/k for k in read(PARENT/'summary.json')['known']}
    refs['matched']=PARENT/'rollouts/fresh'
    assert sum(len(read(bank(s)/'metadata.json')['ids']) for s in refs.values())==596
    return refs


def worse(a,b):
    return bool(a['clean'] and not b['clean'] or a['stable'] and not b['stable']
                or not a['slip_ticks'] and b['slip_ticks'] or not a['illegal_ticks'] and b['illegal_ticks']
                or any(y and not x for x,y in zip(a['physical_failures'],b['physical_failures'],strict=True)))


def fit(out,refs):
    lib=load_bundle(PARENT/'fitted.npz')
    meta=read(bank(refs['matched'])/'metadata.json')
    hull=StanceHull(BodyModel(MODEL),meta['joint_names'])
    jobs=defaultdict(dict)
    for d in read(FIRST/'fit.json')['donors']:
        jobs[Path(d['stage'])][d['id']]=d['slot']
    for s in refs.values():jobs.setdefault(s,{})
    safe=[];risk_rows=[];sweeps=[];audit=[];inputs={}
    for stage,donors in jobs.items():
        p=stage/'evaluation'
        _,rows=verify_audit(p)
        inputs[str(p/'independent_audit.json')]=sha(p/'independent_audit.json')
        f=np.array(signatures(stage))
        is_parent=stage in refs.values()
        pairs=read(stage/'comparison.json') if is_parent else None
        with np.load(p/'chair_contact_motion.npz') as z: actual=z['selected']
        with np.load(p/'trajectory.npz') as z:
            frames=z['obs'][TICKS,:,222:333]
            if is_parent:
                flags=z['flags']; slips=~z['no_slip']
        for i,r in enumerate(rows):
            if not np.isfinite(f[i]).all():continue
            slot=donors.get(r['id'],int(actual[i]) if is_parent else -1)
            if slot<0:continue
            sequence=signals(hull,frames[:,i],f[i],TICKS)
            if r['clean']:
                sweeps.append((slot,f[i],sequence))
            if not is_parent:continue
            a,b=pairs[i]['before'],pairs[i]['after']
            assert b==r
            if not worse(a,b):
                safe.append((slot,f[i]))
                continue
            # Physical outcomes label events offline. Only encoders, gravity
            # and the rear-camera scan enter the predictor.
            events=[]
            if not a['physical_failures'][1] and b['physical_failures'][1]:
                events.append((0,np.flatnonzero(flags[:,i,1])))
            if not a['slip_ticks'] and b['slip_ticks']:
                events.append((1,np.flatnonzero(slips[:,i])))
            for kind,times in events:
                if not len(times):continue
                onset=int(times[0])
                early=(TICKS>=onset-35)&(TICKS<=onset-5)
                warning=np.flatnonzero(warns(sequence,kind)&early)
                audit.append(dict(id=r['id'],stage=str(stage),slot=int(slot),kind=['housing','grip'][kind],
                                  event_tick=onset,lead_ticks=int(onset-TICKS[warning[0]]) if len(warning) else None))
                if len(warning):risk_rows.append((slot,f[i]))
    assert sweeps and risk_rows and safe
    lib.update(sweep_ids=np.array([i for i,f,s in sweeps]),sweep_scans=np.array([f for i,f,s in sweeps]),
               sweep_signals=np.array([s for i,f,s in sweeps],dtype=np.float32),
               safe_ids=np.array([i for i,f in safe]),safe_scans=np.array([f for i,f in safe]),
               risk_ids=np.array([i for i,f in risk_rows]),risk_scans=np.array([f for i,f in risk_rows]),
               tested_scans=np.empty((0,52)),tested_from=np.empty(0,int),tested_to=np.empty(0,int),
               alternative_rank=0)
    # Also count warnings on successful landings. A geometric warning alone
    # must not automatically switch an otherwise proven motion.
    clean_housing=sum(warns(s,0).any() for _,_,s in sweeps)
    clean_grip=sum(warns(s,1).any() for _,_,s in sweeps)
    write(out/'prediction.json',dict(inputs=inputs,events=audit,clean_sweeps=len(sweeps),
          clean_with_housing_warning=int(clean_housing),clean_with_grip_warning=int(clean_grip)))
    np.savez_compressed(out/'sensors.npz',**lib)


def run(out,repair_from=None):
    refs=references();base=refs['matched']
    stages=out/'rollouts';stages.mkdir(exist_ok=True,parents=True)
    fresh=out/'holdout_bank'
    if not fresh.exists():reserve(out,PARENT,seed=70529138,prefix='hold_clearance_holdout')
    seal=read(out/'reservation.json')
    assert all(sha(Path(p))==d for p,d in seal.items())
    occupied={placement(g) for b in [*(bank(s) for s in refs.values()),Path('/dev/shm/chair_coverage_02/nearby_bank'),
              *(p/'bank' for p in CAMPAIGN.glob('batch_*'))] for g in read(b/'geometry.json')}
    new={placement(g) for g in read(fresh/'geometry.json')}
    assert len(new)==24 and new.isdisjoint(occupied)
    if not (out/'sensors.npz').exists():fit(out,refs)
    lib=load_bundle(out/'sensors.npz')
    scans=np.array(signatures(base))
    with np.load(base/'evaluation/chair_contact_motion.npz') as z: original=z['selected']
    old=read(base/'comparison.json')
    target=[i for i,r in enumerate(old) if worse(r['before'],r['after'])]
    assert len(target)==5 and sum(r['after']['clean'] for r in old)==9
    print('advance warnings',[(r['id'],r['lead_ticks']) for r in read(out/'prediction.json')['events'] if r['stage']==str(base)],flush=True)
    inputs=[Path(__file__),HERE/'chair_contact_motion.py',HERE/'hold_clearance.py',HERE/'scan_landing.py',
            out/'sensors.npz',out/'prediction.json',out/'reservation.json',PARENT/'fitted.npz']
    def write_lib(name,rank,responses):
        z=dict(lib,alternative_rank=rank)
        if responses:
            idx=sorted(responses)
            z.update(tested_scans=scans[idx],tested_from=original[idx],tested_to=np.array([responses[i] for i in idx]))
        path=out/(name+'.npz')
        if path.exists():
            existing=load_bundle(path)
            for k,v in z.items():np.testing.assert_array_equal(existing[k],v)
        else:np.savez_compressed(path,**z)
        return path,z

    def rollout(label,ref,path=None,geom=None):
        d=stages/label
        if not (d/'evaluation.score.json').exists():
            evaluate(d,ref,bank=geom,contact_motion=path,inputs=inputs)
        freeze=read(d/'evaluation.freeze.json')
        assert freeze['gpu']==GPU and freeze['controls_per_case']==1000
        if path is None:
            verify_audit(d/'evaluation');return d
        assert sha(path)==freeze['inputs'][str(path)]
        return comparison(d,ref)

    def inspect(label,z,score):
        d=stages/label/'evaluation'
        with np.load(d/'chair_contact_motion.npz') as a: slots=a['selected']
        np.testing.assert_array_equal(slots,[choose(f,select(f,z),z) for f in scans])
        rows=verify_audit(d)[1]
        okay=[i for i in target if not worse(old[i]['before'],rows[i]) and not worse(old[i]['after'],rows[i])]
        qualified=len(okay)==5 and not score['regressions'] and all(not worse(v['before'],b) for v,b in zip(old,rows,strict=True))
        print(label,score['after'],'cleared',okay,'regressions',score['regressions'],flush=True)
        return okay,qualified,slots

    resolved={}
    if repair_from is not None:
        from load_phase_recovery import paths_for_targets, support_matched_donors
        parent=read(repair_from/'summary.json')
        assert not parent['qualified'] and sha(repair_from/'sensors.npz')==sha(out/'sensors.npz')
        confirmed=load_bundle(repair_from/'confirmed.npz')
        for f,initial,choice in zip(confirmed['tested_scans'],confirmed['tested_from'],confirmed['tested_to'],strict=True):
            distances=np.linalg.norm(scans[:,4:]-f[4:],axis=1)
            i=int(distances.argmin())
            assert distances[i]==0 and initial==original[i]
            resolved[i]=int(choice)
        pending=[i for i in target if old[i]['id'] not in parent['cleared']]
        reference=base.with_name('fresh_reference')
        with np.load(reference/'evaluation/chair_contact_motion.npz') as saved:fallback=saved['selected']
        meta=read(bank(base)/'metadata.json')
        geom=read(bank(base)/'geometry.json')
        repairs=paths_for_targets(base,repair_from,pending,meta,geom)
        proposals={i:{} for i in pending}; witnesses={}
        hull=StanceHull(BodyModel(MODEL),meta['joint_names'])
        for i,record in repairs.items():
            if not record:continue
            witness={k:v for k,v in record.items() if k!='motions'}
            witnesses[old[i]['id']]=witness
            if not worse(old[i]['after'],old[i]['before']):
                proposals[i]['prior_approach']=int(fallback[i])
            with np.load(Path(record['stage'])/'evaluation/trajectory.npz') as z:
                series=signals(hull,z['obs'][TICKS,i,222:333],scans[i],TICKS)
            for method,motion in record['motions'].items():
                slot=len(lib['paths']);proposals[i][method]=slot
                lib['paths']=np.concatenate((lib['paths'],motion.astype(np.float32)[None,:]))
                lib['ids']=np.append(lib['ids'],method+' '+old[i]['id'])
                lib['sweep_ids']=np.append(lib['sweep_ids'],slot)
                lib['sweep_scans']=np.vstack((lib['sweep_scans'],scans[i]))
                lib['sweep_signals']=np.concatenate((lib['sweep_signals'],series.astype(np.float32)[None,:]))
        rejected={i:[int(original[i])] for i in pending}
        for trial in sorted((repair_from/'rollouts').glob('alternative_*')):
            with np.load(trial/'evaluation/chair_contact_motion.npz') as a:
                for i in pending:rejected[i].append(int(a['selected'][i]))
        donors=support_matched_donors(FIRST/'fit.json',pending,geom,lib,rejected)
        for i,options in donors.items():
            for j,slot in enumerate(options):proposals[i]['support_donor_'+str(j)]=slot
        write(out/'phase_recovery.json',dict(witnesses=witnesses,proposals=proposals))
        inputs.extend([HERE/'load_phase_recovery.py',HERE/'seat_contact_arrest.py',HERE/'refine_chair_contact.py',
                       repair_from/'confirmed.npz',repair_from/'summary.json',out/'phase_recovery.json'])
        for method in ('prior_approach','loaded_slow','arrest_1','contact_dwell','arrest_0.55',
                       'support_donor_0','support_donor_1','support_donor_2','support_donor_3'):
            responses=dict(resolved)
            for i in pending:
                if i not in resolved and method in proposals[i]:responses[i]=proposals[i][method]
            if responses==resolved:continue
            label='recovery_'+method
            path,z=write_lib(label,0,responses)
            score=rollout(label,base,path)
            okay,qualified,slots=inspect(label,z,score)
            for i in okay:resolved[i]=int(slots[i])
            for i,r in enumerate(old):
                if i not in target and r['after']['id'] in score['regressions']:resolved[i]=int(original[i])
            if qualified:break
    for rank in (range(6) if repair_from is None else ()):
        label='alternative_'+str(rank)
        path,z=write_lib(label,rank,resolved)
        proposed=[choose(f,int(p),z) for f,p in zip(scans,original,strict=True)]
        if rank and not any(proposed[i]!=original[i] for i in target if i not in resolved):break
        score=rollout(label,base,path)
        okay,qualified,slots=inspect(label,z,score)
        for i in okay:resolved[i]=int(slots[i])
        for i,r in enumerate(old):
            if i not in target and r['after']['id'] in score['regressions']:resolved[i]=int(original[i])
        if qualified:break
    if repair_from is not None:
        # Retain the incumbent if all tested alternatives still regress.
        for i in target:resolved.setdefault(i,int(original[i]))
    path,z=write_lib('confirmed',0,resolved)
    score=rollout('matched',base,path)
    okay,qualified,slots=inspect('matched',z,score)
    summary=dict(matched=score,qualified=qualified,prior_regressions=[old[i]['id'] for i in target],
                 cleared=[old[i]['id'] for i in okay])
    write(out/'summary.json',summary)
    if not qualified:return
    summary['known']={'matched':score}
    for name,ref in refs.items():
        if name=='matched':continue
        summary['known'][name]=rollout(name,ref,path)
        write(out/'summary.json',summary)
        print(name,summary['known'][name]['after'],flush=True)
    assert sum(v['total'] for v in summary['known'].values())==596
    assert all(sha(Path(p))==d for p,d in seal.items())
    control=rollout('fresh_reference',base,geom=fresh)
    summary['fresh']=rollout('fresh',control,path,fresh)
    write(out/'summary.json',summary)
    print('fresh',summary['fresh']['after'],summary['fresh']['regressions'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--repair-from',type=Path)
    args=parser.parse_args()
    run(args.output.resolve(),args.repair_from)
