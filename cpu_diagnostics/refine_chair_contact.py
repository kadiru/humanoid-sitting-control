"""Combine a measured low-slip touchdown with a legal loaded posture, then test on reserved chairs."""
import argparse
from pathlib import Path

import numpy as np
import torch

from compare_touchdown_ppo import verify_audit
from evaluate_chair_contact_motion import HERE, PARENT, bank, references
from evaluate_rim_motion import actor_from
from evaluate_seat_plane_approach import evaluate
from evaluate_standing_rim import comparison
from prepare_sitting_repairs import encode, smoothstep
from study_live_sitting import read, sha, write


def collect(stage, original, trained):
    folder=stage/'evaluation'
    _,rows=verify_audit(folder)
    active=read(folder/'grip_handoff.json')['active_ticks']
    with np.load(folder/'trajectory.npz') as z:
        cmd=z['command'][:,:,:23]
        desired=z['desired']
        initial=z['previous'][0,:,:23]
        obs=z['obs'][750:]
        force=np.linalg.norm(z['physical_force_w'][:,:,0],axis=-1)/z['physical_bodyweight_N']
        slip=~z['no_slip']
        wrong=z['wrong_force']
    tick=np.arange(250)[:,None]
    alpha=torch.as_tensor((tick<=np.array(active)[None,:])*smoothstep(tick/50.),dtype=torch.float32)[:,:,None]
    with torch.no_grad():
        norm=original.normalize(torch.as_tensor(obs))
        residual=torch.lerp(original.actor(norm),trained.actor(norm),alpha).numpy()
    result=[]
    for i,r in enumerate(rows):
        loaded=np.flatnonzero(force[:,i]>.01)
        touchdown=int(loaded[0]) if len(loaded) else 999
        bad=np.flatnonzero(slip[:,i])
        path=desired[:,i].copy()
        path[750:]-=residual[:,i]
        result.append(dict(stage=stage.name,row=r,command=cmd[:,i],initial=initial[i],residual=residual[:,i],
                           path=path,touch=touchdown,load_ticks=int(sum(force[:,i]>.05)),peak_wrong=float(wrong[:,i].max()),
                           entry_slip=int(sum(slip[max(0,touchdown-25):touchdown+40,i])),slips=bad))
    return result


def partial(candidate, base):
    b,a=candidate['row'],base['row']
    return (b['stable'] and candidate['load_ticks']>=25 and b['final_hold_s']+.021>=a['final_hold_s'] and
            b['slip_ticks']<=a['slip_ticks'] and b['illegal_ticks']<=a['illegal_ticks'] and
            all(y<=x for x,y in zip(a['physical_failures'],b['physical_failures'],strict=True)) and
            (b['slip_ticks']<a['slip_ticks'] or sum(b['physical_failures'])<sum(a['physical_failures'])))


def quality(trial):
    r=trial['row']
    return (r['clean'],r['stable'],not any(r['physical_failures']),not r['illegal_ticks'],
            -r['slip_ticks'],r['final_hold_s'])


def assemble(pool, base, limits):
    """Phase alignment uses actual first load, followed by an independently tested hold."""
    loaded=[p for p in pool if 150<p['touch']<850 and p['load_ticks']>=12]
    if len(loaded)<2: return []
    post=max(loaded,key=quality)
    distinct=[p for p in loaded if np.max(abs(p['command'][360:850]-post['command'][360:850]))>.02]
    if not distinct: return []
    soft=min(distinct,key=lambda p:(p['entry_slip'],p['row']['slip_ticks'],int(p['row']['physical_failures'][1]),
                                     p['peak_wrong'],-p['row']['final_hold_s']))
    # Map source first contact onto the receiving trajectory's first contact.
    t=np.arange(1000)
    landing=post['touch']
    settle=max(850,landing+40,soft['touch']+40)
    clock=np.interp(t,[0,150,landing,settle,999],[0,150,soft['touch'],settle,999])
    src=np.stack([np.interp(clock,t,col) for col in soft['command'].T],axis=1)
    weights=[('mixed_low',np.full(1000,.30)),('mixed_high',np.full(1000,.65)),
             ('soft_then_legal',1.-smoothstep((t-landing+12)/100.)),
             ('legal_then_soft',smoothstep((t-landing-12)/145.))]
    # For a hold-only loss keep the incumbent approach and transfer to a
    # recorded stable hold ahead of the measured break in contact.
    if len(base['slips']) and base['slips'][0]>base['touch']+80:
        hold=max(loaded,key=lambda p:(p['row']['stable'],p['row']['final_hold_s'],-p['row']['slip_ticks']))
        if hold is not base:
            switch=smoothstep((t-int(base['slips'][0])+110)/100.)
            special=base['command']+switch[:,None]*(hold['command']-base['command'])
            weights.append(('hold_transfer',None))
    out=[]
    for method,w in weights:
        requested=special if w is None else post['command']+w[:,None]*(src-post['command'])
        latent,actual=encode(requested,base['initial'],*limits)
        error=float(abs(actual-requested).max())
        if error>.025: continue
        latent[750:]-=post['residual']
        out.append((method,latent,dict(post=post['stage'],soft=soft['stage'],touch=int(landing),
                   soft_touch=int(soft['touch']),settle=int(settle),tracking_rad=error)))
    return out


def run(out,first):
    out.mkdir(exist_ok=True)
    stages=out/'rollouts'
    stages.mkdir(exist_ok=True)
    refs=references()
    base=refs['matched']
    control=first/'rollouts/matched'
    gate=read(first/'summary.json')
    assert gate['qualified'] and gate['matched']['regressions']==[]
    seal=read(first/'reservation.json')
    assert all(sha(Path(p))==digest for p,digest in seal.items())
    write(out/'reservation.json',seal)
    with np.load(first/'observations.npz') as z:
        features,retain,targets=(z[k] for k in ('features','retain','targets'))
    with np.load(first/'candidates.npz') as z:
        paths=z['paths']
    # Preserve the original selected references byte-for-byte when composing the final library.
    selected={int(i):(paths[v['motion']],v['source']) for i,v in read(first/'selection.json').items()}
    command=read(base/'evaluation.freeze.json')['command']
    original=actor_from(command[command.index('--checkpoint')+1])
    trained=actor_from(command[command.index('evaluate')+1])
    torch.set_num_threads(1)
    meta=read(bank(base)/'metadata.json')
    limits=[np.asarray(meta[k]) for k in ('lower','upper')]
    baseline=collect(base,original,trained)
    verify_audit(control/'evaluation')
    recorded={i:[] for i in targets}
    receipts={}
    sources={}
    def add_evidence(stage):
        data=collect(stage,original,trained)
        file=stage/'evaluation/independent_audit.json'
        receipts[str(file)]=sha(file)
        with np.load(stage/'evaluation/chair_contact_motion.npz') as z:
            active=z['selected']
        for i in targets:
            if active[i]>=0: recorded[i].append(data[i])
        return data
    for stage in sorted((first/'rollouts').glob('probe_*')):
        if (stage/'comparison.json').exists(): add_evidence(stage)
    choices={}
    for i in targets:
        if i in selected: continue
        pool=[baseline[i],*recorded[i]]
        options=assemble(pool,baseline[i],limits)
        choices[i]={method:path for method,path,record in options}
        sources[str(i)]={method:record for method,path,record in options}
    write(out/'phase_alignment.json',dict(audits=receipts,construction=sources))
    inputs=[Path(__file__),HERE/'evaluate_chair_contact_motion.py',out/'phase_alignment.json',first/'selection.json',
            first/'confirmed.npz',first/'observations.npz',out/'reservation.json']
    def library(label,motions):
        from chair_contact_motion import select
        used=np.array(sorted(motions),dtype=int)
        keep=np.concatenate((retain,features[np.setdiff1d(targets,used)]))
        dest=out/(label+'.npz')
        bundle=dict(features=features[used],retain=keep,paths=np.stack([motions[i][0] for i in used]),
                    ids=np.array([motions[i][1] for i in used]))
        for j,i in enumerate(used): assert select(features[i],bundle)==j
        np.savez_compressed(dest,**bundle)
        return dest
    def launch(label,reference,lib,b=None):
        dest=stages/label
        evaluate(dest,reference,contact_motion=lib,bank=b,inputs=inputs)
        return comparison(dest,reference)
    for method in ('soft_then_legal','mixed_low','mixed_high','legal_then_soft','hold_transfer'):
        proposed={i:(c[method],method) for i,c in choices.items() if i not in selected and method in c}
        if not proposed:continue
        lib=library(method,proposed)
        launch('probe_'+method,base,lib)
        data=add_evidence(stages/('probe_'+method))
        for i in proposed:
            if data[i]['row']['clean']: selected[i]=proposed[i]
        print(method,'clean target motions',len(selected),flush=True)
    # Preserve measured intermediate gains only when a previous failure improves,
    # the five-second loaded hold passes, and all other categorical gates survive.
    partial_rows=[]
    for i in targets:
        if i in selected:continue
        eligible=[p for p in recorded[i] if partial(p,baseline[i])]
        if not eligible:continue
        best=max(eligible,key=quality)
        if best['stage'].startswith('probe_'):
            label=best['stage'][6:]
            home=out if label in choices.get(i,{}) else first
            stage=home/'rollouts'/best['stage']
            with np.load(stage/'evaluation/chair_contact_motion.npz') as z: j=int(z['selected'][i])
            with np.load(home/(label+'.npz')) as z:
                selected[i]=(z['paths'][j],str(z['ids'][j]))
            partial_rows.append(int(i))
    write(out/'selection.json',dict(sources={str(i):p[1] for i,p in selected.items()},partial=partial_rows,
                                     inputs=receipts))
    inputs.append(out/'selection.json')
    final=library('confirmed',selected)
    matched=launch('matched',base,final)
    # Also compare against the first candidate so recovered clean cases are retained.
    extra=comparison(stages/'matched',control)
    write(out/'vs_first.json',extra)
    matched=comparison(stages/'matched',base)
    summary=dict(matched=matched,qualified=matched['after']['clean']>matched['before']['clean'] and
                 not matched['regressions'] and not extra['regressions'])
    write(out/'summary.json',summary)
    print('gate',summary['qualified'],matched['after'],matched['regressions'],extra['regressions'],flush=True)
    if not summary['qualified']:return
    summary['known']={'matched':matched}
    for name,ref in refs.items():
        if name=='matched':continue
        label='previous_fresh' if name=='fresh' else name
        summary['known'][name]=launch(label,ref,final)
        write(out/'summary.json',summary)
        print(label,summary['known'][name]['after'],flush=True)
    assert sum(v['total'] for v in summary['known'].values())==524
    assert all(sha(Path(p))==d for p,d in seal.items())
    fresh=first/'holdout_bank'
    reference=stages/'fresh_reference'
    evaluate(reference,base,bank=fresh,inputs=inputs)
    summary['fresh']=launch('fresh',reference,final,fresh)
    write(out/'summary.json',summary)
    print('fresh',summary['fresh']['after'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('first',type=Path)
    args=parser.parse_args()
    run(args.out.resolve(),args.first.resolve())
