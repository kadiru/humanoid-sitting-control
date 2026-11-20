"""Validate the same sitting approach across nearby measured chair placements."""
import argparse
from pathlib import Path
import shutil

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from chair_contact_motion import CAPTURE, distances
from compare_touchdown_ppo import verify_audit
from evaluate_chair_contact_motion import CAMPAIGN, bank, build, signatures
from evaluate_chair_shapes import CHAIRS, pieces
from evaluate_rim_motion import actor_from
from evaluate_seat_contact_continuity import placement
from evaluate_seat_plane_approach import evaluate
from evaluate_seat_slip import reserve
from evaluate_standing_rim import comparison
from refine_chair_contact import assemble, collect, partial, quality
from seat_contact_arrest import construct
from prepare_sitting_repairs import BodyModel, MODEL
from study_live_sitting import bank_from_geometry, read, sha, write

PREVIOUS = Path('/dev/shm/chair_contact_transfer_02')
HERE = Path(__file__).resolve().parent


def reference_banks():
    root = PREVIOUS/'rollouts'
    refs = {}
    for name in read(PREVIOUS/'summary.json')['known']:
        directory = 'previous_fresh' if name=='fresh' else name
        key = 'prior_contact_matched' if name=='matched' else directory
        refs[key] = root/directory
    refs['matched'] = root/'fresh'
    assert sum(len(read(bank(s)/'metadata.json')['ids']) for s in refs.values())==548
    return refs


def neighboring_chairs(dest, source, targets):
    """Move the front rim, pitch, height and yaw together in two opposing directions."""
    shape = {v[0]:v for v in CHAIRS}
    geometry = read(source/'geometry.json')
    mapping, rows = [], []
    for i in targets:
        g=geometry[i]
        family, depth, width, form, back, arms, style=shape[g['family']]
        for sign in (-1,1):
            height=g['reference_height_cm']/100+sign*.009
            pitch=g['pitch_deg']+sign*1.0
            yaw=g['yaw_deg']+sign*1.8
            dx=g['fore_aft_offset_cm']/100+sign*.012
            dy=g['lateral_offset_cm']/100-sign*.012
            rot=Rotation.from_euler('ZY',[yaw,pitch],degrees=True)
            front=np.array([-.4+dx,dy,height])+.25*Rotation.from_euler('y',pitch,degrees=True).as_matrix()[:,0]
            top=front-.5*depth*rot.as_matrix()[:,0]
            rows.append(dict(g,label=f'coverage_offset/{family}/{i}/{sign}',x=top[0],y=top[1],height=top[2],
                    top_center=top.tolist(),pitch_deg=pitch,yaw_deg=yaw,yaw=np.deg2rad(yaw),
                    rotation_xyzw=rot.as_quat().tolist(),normal=rot.as_matrix()[:,2].tolist(),
                    reference_height_cm=100*height,fore_aft_offset_cm=100*dx,lateral_offset_cm=100*dy,
                    parts=pieces(depth,width,top[2],rot,back,arms,style)))
            mapping.append(int(i))
    bank_from_geometry(dest,source,rows)
    return mapping


def run(out, reuse=None):
    out.mkdir(exist_ok=True)
    stages=out/'rollouts'
    stages.mkdir(exist_ok=True)
    refs=reference_banks()
    base=refs['matched']
    sealed=out/'holdout_bank'
    if reuse is not None:
        # Reuse only completed experiments. Verify the frozen candidate inputs
        # before importing them into a separately versioned evaluation.
        evidence=read(reuse/'rollouts/probe_donor_0/evaluation.freeze.json')['inputs']
        for name in ('construction.json','observations.npz','candidates.npz','reservation.json'):
            src=reuse/name
            assert sha(src)==evidence[str(src)]
            if not (out/name).exists():shutil.copy2(src,out/name)
            assert sha(out/name)==sha(src)
        if not sealed.exists():shutil.copytree(reuse/'holdout_bank',sealed)
    if not sealed.exists():
        reserve(out,bank(base).parent,seed=43561027,prefix='coverage_holdout')
    seal=read(out/'reservation.json')
    assert {p.name:sha(p) for p in sealed.iterdir()}=={Path(k).name:v for k,v in seal.items()}
    assert all(sha(Path(k))==v for k,v in seal.items())
    archive=set()
    for b in [*(bank(s) for s in refs.values()), *(s/'bank' for s in CAMPAIGN.glob('batch_*'))]:
        archive.update(placement(g) for g in read(b/'geometry.json'))
    fresh={placement(g) for g in read(sealed/'geometry.json')}
    assert len(fresh)==24 and fresh.isdisjoint(archive)
    if not (out/'candidates.npz').exists():
        earlier=[s for study in (PREVIOUS,Path('/dev/shm/chair_contact_transfer_01'))
                 for s in sorted((study/'rollouts').glob('probe_*')) if (s/'comparison.json').exists()]
        build(out,refs,expected=(19,17,5),donor_count=7,extra_stages=earlier)
    def arrays(path):
        with np.load(path) as z:return {k:z[k] for k in z.files}
    legacy=arrays(PREVIOUS/'confirmed.npz')
    old_count=len(legacy['paths'])
    obs=arrays(out/'observations.npz')
    features,targets,retain=[obs[k] for k in ('features','targets','retain')]
    proposed=arrays(out/'candidates.npz')
    inputs=[Path(__file__),HERE/'chair_contact_motion.py',HERE/'seat_contact_arrest.py',HERE/'refine_chair_contact.py',HERE/'evaluate_chair_contact_motion.py',
            out/'construction.json',out/'observations.npz',out/'candidates.npz',out/'reservation.json',PREVIOUS/'confirmed.npz']

    def saved_stage(label):
        if reuse is not None and label.startswith('probe_'):
            previous=reuse/'rollouts'/label
            if (previous/'comparison.json').exists():return previous
        return stages/label

    def rollout(label,ref,lib=None,geometry=None):
        dest=saved_stage(label)
        if dest.parent!=stages:
            receipt=read(dest/'evaluation.freeze.json')
            args=receipt['command']
            source=Path(args[args.index('--chair-contact-motion')+1])
            assert sha(source)==receipt['inputs'][str(source)]
            old=arrays(source);current=arrays(lib)
            assert old.keys()==current.keys()
            for key in old:np.testing.assert_array_equal(old[key],current[key])
            verify_audit(dest/'evaluation')
            assert bank(ref)==bank(dest)
            return None
        if not (dest/'evaluation.score.json').exists():
            evaluate(dest,ref,contact_motion=lib,bank=geometry,inputs=inputs)
        freeze=read(dest/'evaluation.freeze.json')
        assert freeze['gpu']=='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
        if lib is not None:
            assert freeze['inputs'][str(lib)]==sha(lib)
        if lib is None:
            verify_audit(dest/'evaluation')
            return dest
        return comparison(dest,ref)

    def bundle(label, motions, anchors, keep):
        keys=sorted(motions)
        slots={i:old_count+j for j,i in enumerate(keys)}
        sites=np.array([f for f,i,r,clean in anchors]).reshape(-1,features.shape[1])
        row_ids=np.array([slots[i] for f,i,r,clean in anchors],dtype=int)
        links=[]
        for j,(a,i,_,clean) in enumerate(anchors):
            if not clean or not np.isfinite(a[0]):continue
            for k in range(j):
                b,h,_,other_clean=anchors[k]
                if i!=h or not other_clean or not np.isfinite(b[0]):continue
                span=abs(a[:4]-b[:4])/CAPTURE
                if .15<span.max()<=3. and distances(a,b[None])[0]<=3.:
                    links.append((j,k))
        z=dict(legacy)
        if keys:
            z['paths']=np.concatenate((legacy['paths'],np.stack([motions[i][0] for i in keys])))
            z['ids']=np.r_[legacy['ids'],[motions[i][1] for i in keys]]
        z.update(sites=sites,site_paths=row_ids,links=np.asarray(links,dtype=int).reshape(-1,2),
                 site_radius=np.array([r for f,i,r,clean in anchors]),protected=keep)
        path=out/(label+'.npz')
        if not path.exists(): np.savez_compressed(path,**z)
        else:
            existing=arrays(path)
            for k,v in z.items():np.testing.assert_array_equal(existing[k],v)
        return path,slots

    def matched_bundle(label,motions):
        keep=np.concatenate((retain,features[np.setdiff1d(targets,list(motions))]))
        return bundle(label,motions,[(features[i],i,1.,False) for i in motions],keep)

    cmd=read(base/'evaluation.freeze.json')['command']
    original=actor_from(cmd[cmd.index('--checkpoint')+1])
    trained=actor_from(cmd[cmd.index('evaluate')+1])
    torch.set_num_threads(1)
    baseline=collect(base,original,trained)
    recorded={i:[] for i in targets}
    clean={}
    def probe(method,options):
        if not options:return
        lib,slots=matched_bundle(method,options)
        label='probe_'+method
        rollout(label,base,lib)
        dest=saved_stage(label)
        data=collect(dest,original,trained)
        with np.load(dest/'evaluation/chair_contact_motion.npz') as z: actual=z['selected']
        for i,motion in options.items():
            if actual[i]!=slots[i]:continue
            recorded[i].append((data[i],motion))
            if data[i]['row']['clean']:clean[i]=motion
        print(method,'clean targeted cases',len(clean),'/',len(targets),flush=True)

    for method in ('donor_0','donor_1','donor_2','donor_3','donor_4','donor_5','donor_6',
                   'clearance_0','aligned_0','aligned_1','aligned_2','soft_transfer'):
        options={int(i):(proposed['paths'][j],str(proposed['ids'][j])) for j,i in enumerate(proposed['indices'])
                 if proposed['methods'][j]==method and i not in clean}
        probe(method,options)
    meta=read(bank(base)/'metadata.json')
    limits=[np.array(meta[k]) for k in ('lower','upper')]
    choices={}
    witnesses={}
    for i in targets:
        if i in clean:continue
        options=assemble([baseline[i],*(r for r,m in recorded[i])],baseline[i],limits)
        choices[i]={k:(p,k) for k,p,w in options}
        witnesses[str(i)]={k:w for k,p,w in options}
    write(out/'phase_alignment.json',witnesses)
    inputs.append(out/'phase_alignment.json')
    for method in ('mixed_low','mixed_high','soft_then_legal','legal_then_soft','hold_transfer'):
        probe(method,{i:c[method] for i,c in choices.items() if i not in clean and method in c})
    # Compensate the observed tangential travel of a material point at load.
    # The lower-slip alternative can require a pre-contact housing tilt.
    model=BodyModel(MODEL)
    geometry=read(bank(base)/'geometry.json')
    corrections={}
    corrections_info={}
    for i in targets:
        if i in clean:continue
        pool=[baseline[i],*(r for r,m in recorded[i])]
        post=max(pool,key=quality)
        others=[r for r in pool if r['row']['physical_failures'][1] and r is not post]
        sources=[('post',post)]
        if others:
            sources.append(('clearance',min(others,key=lambda r:(r['entry_slip'],r['peak_wrong'],-r['load_ticks']))))
        corrections[i]={}
        for name,trial in sources:
            stage=base if trial is baseline[i] else saved_stage(trial['stage'])
            for kind,path,info in construct(stage,i,trial,meta,geometry[i],limits,model):
                method=name+'_'+kind
                corrections[i][method]=(path,method)
                corrections_info.setdefault(str(i),{})[method]=info
    write(out/'arrest_geometry.json',corrections_info)
    inputs.append(out/'arrest_geometry.json')
    for method in ('post_arrest_1','clearance_arrest_1','post_arrest_0.55','clearance_arrest_0.55'):
        probe(method,{i:c[method] for i,c in corrections.items() if i not in clean and method in c})
    selected=clean.copy()
    for i in targets:
        if i in selected:continue
        better=[p for p in recorded[i] if partial(p[0],baseline[i])]
        if better:selected[i]=max(better,key=lambda p:quality(p[0]))[1]
    selected={i:selected[i] for i in sorted(selected)}
    info=dict(clean_rows=sorted(map(int,clean)),partial_rows=sorted(int(i) for i in selected if i not in clean),
              sources={str(i):m[1] for i,m in selected.items()})
    write(out/'selection.json',info)
    inputs.append(out/'selection.json')
    anchors=[(features[i],i,1. if i in clean else .25,i in clean) for i in selected]
    keep=np.concatenate((retain,features[np.setdiff1d(targets,list(selected))]))
    extra=[]
    rejected=[]
    if clean:
        # These placements develop coverage. The independent final holdout remains sealed.
        neighbors=out/'nearby_bank'
        if not neighbors.exists():
            owner=neighboring_chairs(neighbors,bank(base),sorted(clean))
            write(out/'nearby_rows.json',owner)
        owner=read(out/'nearby_rows.json')
        places={placement(g) for g in read(neighbors/'geometry.json')}
        assert len(places)==len(owner) and places.isdisjoint(archive|fresh)
        reference=rollout('nearby_reference',base,geometry=neighbors)
        scans=signatures(reference)
        probe_sites=[(f,i,1.,False) for i,f in zip(owner,scans,strict=True)]
        test,slots=bundle('nearby_test',clean,probe_sites,np.empty((0,features.shape[1])))
        nearby=rollout('nearby',reference,test,neighbors)
        _,rows=verify_audit(stages/'nearby/evaluation')
        with np.load(stages/'nearby/evaluation/chair_contact_motion.npz') as z: actual=z['selected']
        for j,(f,i,r) in enumerate(zip(scans,owner,rows,strict=True)):
            if actual[j]==slots[i] and r['clean']:extra.append((f,i,1.,True))
            else:rejected.append(f)
        info['nearby']=dict(tested=len(rows),validated=len(extra),baseline=nearby['before'],candidate=nearby['after'])
    protection=np.concatenate((keep,np.array(rejected).reshape(-1,features.shape[1])))
    final,slots=bundle('confirmed',selected,anchors+extra,protection)
    info['links']=len(arrays(final)['links'])
    write(out/'coverage.json',info)
    inputs.append(out/'coverage.json')
    matched=rollout('matched',base,final)
    summary=dict(matched=matched,qualified=matched['after']['clean']>matched['before']['clean'] and not matched['regressions'])
    write(out/'summary.json',summary)
    print('gate',summary['qualified'],matched['after'],matched['regressions'],flush=True)
    if not summary['qualified']:return
    if clean:
        summary['nearby_final']=rollout('nearby_final',reference,final,neighbors)
        write(out/'summary.json',summary)
    summary['known']={'matched':matched}
    for name,ref in refs.items():
        if name=='matched':continue
        summary['known'][name]=rollout(name,ref,final)
        write(out/'summary.json',summary)
        print(name,summary['known'][name]['after'],flush=True)
    assert sum(s['total'] for s in summary['known'].values())==548
    assert all(sha(Path(k))==v for k,v in seal.items())
    reference=rollout('fresh_reference',base,geometry=sealed)
    summary['fresh']=rollout('fresh',reference,final,sealed)
    write(out/'summary.json',summary)
    print('fresh',summary['fresh']['after'],summary['fresh']['regressions'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--reuse',type=Path,help='Import frozen preparation and completed probes from an earlier run.')
    args=parser.parse_args()
    run(args.out.resolve(),args.reuse.resolve() if args.reuse else None)
