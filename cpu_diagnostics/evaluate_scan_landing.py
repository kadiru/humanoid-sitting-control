"""Fit chair approach selection from saved contact geometry, then run paired trials."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from chair_contact_motion import select, select_incumbent
from compare_touchdown_ppo import verify_audit
from evaluate_chair_contact_motion import CAMPAIGN, bank, signatures
from evaluate_rim_motion import actor_from
from evaluate_seat_contact_continuity import placement
from evaluate_seat_plane_approach import evaluate
from evaluate_seat_slip import reserve
from evaluate_standing_rim import comparison
from prepare_sitting_repairs import BodyModel, MODEL, smoothstep
from scan_landing import ranked
from study_live_sitting import read, sha, write

PARENT = Path('/dev/shm/chair_coverage_02')
HERE = Path(__file__).resolve().parent
GPU = 'GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'


def reference_banks():
    refs = {('coverage_matched' if k == 'matched' else k): PARENT/'rollouts'/k
            for k in read(PARENT/'summary.json')['known']}
    refs['matched'] = PARENT/'rollouts/fresh'
    assert sum(len(read(bank(s)/'geometry.json')) for s in refs.values()) == 572
    return refs


def load_bundle(path):
    with np.load(path) as z:
        return {k: z[k] for k in z}


def contact_sample(model, geom, joints, pelvis, patch, feet, forward, side):
    """Contact and housing positions in the standing frame, measured offline."""
    top = np.array(geom['top_center'])
    rotation = Rotation.from_quat(geom['rotation_xyzw']).as_matrix()
    axes = np.stack((forward,side),axis=1)
    xy = [(patch-feet)@axes]
    gaps = []
    base = Rotation.from_quat(pelvis[3:]).as_matrix()
    fk = model.fk(joints)
    for suffix in ('left','right'):
        body = suffix+'_hip_pitch_link'
        link = fk[body]
        vertices = (model.vertices[body]@link[:3,:3].T+link[:3,3])@base.T+pelvis[:3]
        local = (vertices-top)@rotation
        # Conservative rectangular extent for the stool and chamfered tops.
        outside = np.maximum(abs(local[:,:2])-[geom['depth']/2,geom['width']/2],0.)
        gap = np.where(np.max(outside,axis=1)<.004, local[:,2],
                       np.hypot(np.linalg.norm(outside,axis=1),np.maximum(local[:,2],0.)))
        least = gap.argmin()
        xy.append((vertices[least]-feet)@axes)
        gaps.append(gap[least])
    return np.array(xy), gaps


def fit(out, refs):
    legacy = load_bundle(PARENT/'confirmed.npz')
    paths = list(legacy['paths'])
    ids = list(legacy['ids'])
    def key(a):
        return hashlib.sha256(np.ascontiguousarray(a,dtype=np.float32)).digest()
    path_index = {key(p):j for j,p in enumerate(paths)}
    def register(a, label):
        digest=key(a)
        if digest not in path_index:
            path_index[digest]=len(paths)
            paths.append(a.copy())
            ids.append(label)
        return path_index[digest]
    base = refs['matched']
    cmd = read(base/'evaluation.freeze.json')['command']
    original = actor_from(cmd[cmd.index('--checkpoint')+1])
    trained = actor_from(cmd[cmd.index('evaluate')+1])
    model=BodyModel(MODEL)
    donors, failed, fallback_good, fallback_bad = [], [], [], []
    seen_good, seen_bad = set(), set()
    hashes = {}
    probes = [p for study in ('chair_coverage_02','chair_coverage_01','chair_contact_transfer_02','chair_contact_transfer_01')
              for p in sorted((PARENT.parent/study/'rollouts').glob('probe_*')) if (p/'comparison.json').exists()]
    stages = list(dict.fromkeys([*refs.values(),PARENT/'rollouts/nearby_final',PARENT/'rollouts/nearby',*probes]))
    for stage in stages:
        p=stage/'evaluation'
        _, rows=verify_audit(p)
        hashes[str(p/'independent_audit.json')]=sha(p/'independent_audit.json')
        feat=np.array(signatures(stage))
        with np.load(p/'chair_contact_motion.npz') as z:
            chosen=z['selected']
        geometry=read(bank(stage)/'geometry.json')
        if stage in refs.values() or stage.name=='nearby_final':
            for f,r,j in zip(feat,rows,chosen,strict=True):
                if np.isfinite(f).all() and j<0:
                    (fallback_good if r['clean'] else fallback_bad).append(f)
        if stage == base or not any(g.get('family') for g in geometry):
            continue
        freeze=read(stage/'evaluation.freeze.json')
        opt=freeze['command']
        lib=Path(opt[opt.index('--chair-contact-motion')+1])
        assert sha(lib)==freeze['inputs'][str(lib)]
        hashes[str(lib)]=freeze['inputs'][str(lib)]
        with np.load(lib) as z:
            motors,names=z['paths'],z['ids']
        mapped={}
        for j in set(chosen[chosen>=0]):
            mapped[j]=register(motors[j],str(names[j]))
        # Preserve a record of where the same motion previously failed.
        for i,(f,r,j) in enumerate(zip(feat,rows,chosen,strict=True)):
            if j>=0 and np.isfinite(f).all() and not r['clean']:
                identity=(mapped[j],r['id'])
                if identity not in seen_bad:
                    failed.append((mapped[j],f))
                    seen_bad.add(identity)
        eligible=[i for i,r in enumerate(rows) if r['clean'] and np.isfinite(feat[i]).all()
                  and (chosen[i]>=0 or stage in refs.values() or stage.name=='nearby_final')]
        if not eligible:continue
        origins=np.asarray(read(p/'results.json')['origins'])
        with np.load(p/'standing_scan.npz') as z:
            standq,root,rs=z['q'][0],z['root'][0],z['rotation'][0]
        with np.load(p/'edge_audit.npz') as z: patches=z['pelvis_contact_world_local']
        with np.load(p/'trajectory.npz') as z:
            load=z['support_bw']; q=z['q']; body=z['body'][:,:,0]
            if any(chosen[i]<0 for i in eligible):
                desired=z['desired']
                obs=torch.as_tensor(z['obs'][750:])
                active=np.array(read(p/'grip_handoff.json')['active_ticks'])
                t=np.arange(250)[:,None]
                alpha=torch.as_tensor((t<=active)*smoothstep(t/50.),dtype=torch.float32)[:,:,None]
                with torch.no_grad():
                    normalized=original.normalize(obs)
                    residual=torch.lerp(original.actor(normalized),trained.actor(normalized),alpha).numpy()
        for i in eligible:
            r,f,j=rows[i],feat[i],chosen[i]
            if j<0:
                motion=desired[:,i].copy()
                motion[750:]-=residual[:,i]
                slot=register(motion,r['id'])
            else:
                slot=mapped[j]
            identity=(slot,r['id'])
            if identity in seen_good:continue
            touched=np.flatnonzero((load[:,i]>.05)&np.isfinite(patches[:,i]).all(axis=1))
            if len(touched)<5:continue
            tick=int(touched[0]); patch=np.median(patches[touched[:5],i],axis=0)
            fk=model.fk(standq[i])
            ankles=np.array([root[i]+rs[i]@fk[side+'_ankle_roll_link'][:3,3] for side in ('left','right')])
            side=ankles[0]-ankles[1];side[2]=0;side/=np.linalg.norm(side)
            forward=np.cross(side,[0.,0.,1.])
            feet=ankles.mean(0)
            pelvis=body[tick,i].copy();pelvis[:3]-=origins[i]
            xy,gaps=contact_sample(model,geometry[i],q[tick,i],pelvis,patch,feet,forward,side)
            # The rim is a tangent-coordinate distance above the ankle frame.
            # Project the recorded patch onto the same sagittal plane tangent.
            slope=np.tan(np.deg2rad(f[2])); lat=f[3]
            normal=forward*slope+side*lat+[0.,0.,1.]
            normal/=np.linalg.norm(normal)
            tangent=forward-normal*(forward@normal);tangent/=np.linalg.norm(tangent)
            inset=f[0]-(patch-feet)@tangent
            donors.append((slot,f,xy,gaps,inset,str(stage),r['id']))
            seen_good.add(identity)
    assert donors and failed
    library=dict(legacy,paths=np.asarray(paths),ids=np.asarray(ids),
                 landing_ids=np.array([d[0] for d in donors]),landing_scans=np.array([d[1] for d in donors]),
                 landing_xy=np.array([d[2] for d in donors]),landing_gaps=np.array([d[3] for d in donors]),
                 landing_inset=np.array([d[4] for d in donors]),fallback_clean=np.array(fallback_good),
                 fallback_fail=np.array(fallback_bad),motion_failed_ids=np.array([d[0] for d in failed]),
                 motion_failed_scans=np.array([d[1] for d in failed]))
    assert np.isfinite(library['paths']).all() and np.isfinite(library['landing_xy']).all()
    write(out/'fit.json',dict(sources=hashes,donors=[dict(slot=int(d[0]),stage=d[5],id=d[6]) for d in donors],
                              failed_motion_examples=len(failed),fallback_clean=len(fallback_good),fallback_fail=len(fallback_bad)))
    np.savez_compressed(out/'fitted.npz',**library)
    print('fit',len(donors),'clean landings',len(failed),'failed motion examples',flush=True)


def calibrate(out, first, base):
    """Use paired, complete touchdown outcomes to veto an unsafe transfer."""
    trial=first/'rollouts/matched'
    verify_audit(trial/'evaluation')
    verify_audit(base/'evaluation')
    assert bank(trial)==bank(base)
    comparison_rows=read(trial/'comparison.json')
    receipt=read(trial/'evaluation.freeze.json')
    source=first/'fitted.npz'
    assert receipt['inputs'][str(source)]==sha(source)
    library=load_bundle(source)
    scans=signatures(trial)
    with np.load(trial/'evaluation/chair_contact_motion.npz') as z: selected=z['selected']
    with np.load(base/'evaluation/chair_contact_motion.npz') as z: previous=z['selected']
    with np.load(trial/'evaluation/trajectory.npz') as z:
        flags=z['flags'][:,:,1]
        support=z['support_bw']
        slip=~z['no_slip']
    outcomes={'adverse':[], 'supported':[]}
    details=[]
    regressions=read(first/'summary.json')['matched']['regressions']
    for i,(pair,path,old,f) in enumerate(zip(comparison_rows,selected,previous,scans,strict=True)):
        if old>=0 or path<0:continue
        a,b=pair['before'],pair['after']
        assert a['id']==b['id'] and a['steps']==b['steps']==1000
        assert select(f,library)==path
        worse=b['id'] in regressions or b['slip_ticks']>a['slip_ticks']
        category='adverse' if worse else 'supported' if b['clean'] else None
        if category is None:continue
        outcomes[category].append((path,f))
        first_tick=lambda arr: int(np.flatnonzero(arr)[0]) if arr.any() else None
        details.append(dict(id=b['id'],outcome=category,path=int(path),first_load=first_tick(support[:,i]>.01),
                            first_forbidden=first_tick(flags[:,i]),first_slip=first_tick(slip[:,i])))
    assert outcomes['adverse'] and outcomes['supported']
    for kind,values in outcomes.items():
        library[kind+'_ids']=np.array([i for i,f in values],dtype=int)
        library[kind+'_scans']=np.array([f for i,f in values])
    write(out/'fit.json',dict(parent_fit=sha(first/'fit.json'),parent=sha(source),
                              comparison=sha(trial/'comparison.json'),audit=sha(trial/'evaluation/independent_audit.json'),
                              outcome_locations=details))
    np.savez_compressed(out/'fitted.npz',**library)


def run(out, first=None):
    refs=reference_banks();base=refs['matched']
    out.mkdir(exist_ok=True);stages=out/'rollouts';stages.mkdir(exist_ok=True)
    fresh=out/'holdout_bank'
    if not fresh.exists():
        reserve(out,PARENT,seed=120724614,prefix='scan_landing_holdout')
    seal=read(out/'reservation.json')
    assert all(sha(Path(p))==v for p,v in seal.items())
    archive={placement(g) for b in [*(bank(s) for s in refs.values()),PARENT/'nearby_bank',
                                    *(p/'bank' for p in CAMPAIGN.glob('batch_*'))] for g in read(b/'geometry.json')}
    new={placement(g) for g in read(fresh/'geometry.json')}
    assert len(new)==24 and new.isdisjoint(archive)
    if not (out/'fitted.npz').exists():
        if first is None:fit(out,refs)
        else:calibrate(out,first,base)
    library=load_bundle(out/'fitted.npz'); feats=np.array(signatures(base))
    rows=verify_audit(base/'evaluation')[1]
    inputs=[Path(__file__),HERE/'chair_contact_motion.py',HERE/'scan_landing.py',out/'fit.json',out/'reservation.json',out/'fitted.npz']
    plan=[]
    for f,r in zip(feats,rows,strict=True):
        choices=ranked(f,library)
        plan.append(dict(id=r['id'],before_clean=r['clean'],incumbent=select_incumbent(f,library),
                         proposed=select(f,library),ranked=choices[:3]))
    write(out/'scan_replay.json',plan)
    inputs.append(out/'scan_replay.json')
    print('scan replay',sum(v['proposed']>=0 and v['incumbent']<0 for v in plan),'new decisions',flush=True)

    def rollout(label,ref,path=None,geom=None):
        dest=stages/label
        if not (dest/'evaluation.score.json').exists():
            evaluate(dest,ref,contact_motion=path,bank=geom,inputs=inputs)
        receipt=read(dest/'evaluation.freeze.json')
        assert receipt['gpu']==GPU and receipt['controls_per_case']==1000
        if path is None:
            verify_audit(dest/'evaluation');return dest
        assert receipt['inputs'][str(path)]==sha(path)
        return comparison(dest,ref)

    path=out/'fitted.npz'
    matched=rollout('matched',base,path)
    with np.load(stages/'matched/evaluation/chair_contact_motion.npz') as z:
        np.testing.assert_array_equal(z['selected'],[v['proposed'] for v in plan])
    summary=dict(matched=matched, qualified=matched['after']['clean']>8 and not matched['regressions'])
    write(out/'summary.json',summary);print('gate',summary['qualified'],matched['after'],matched['regressions'],flush=True)
    if not summary['qualified']:return
    summary['known']={'matched':matched}
    for name,ref in refs.items():
        if name=='matched':continue
        summary['known'][name]=rollout(name,ref,path)
        write(out/'summary.json',summary)
        print(name,summary['known'][name]['after'],flush=True)
    assert sum(v['total'] for v in summary['known'].values())==572
    assert all(sha(Path(p))==v for p,v in seal.items())
    control=rollout('fresh_reference',base,geom=fresh)
    summary['fresh']=rollout('fresh',control,path,fresh)
    write(out/'summary.json',summary);print('fresh',summary['fresh']['after'],summary['fresh']['regressions'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--calibrate-from',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    run(args.output.resolve(),args.calibrate_from)
