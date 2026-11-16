"""Repair the three out-of-range camera fallbacks on their saved starts."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from align_sitting_touchdown import counterbalance
from live_sitting_edge import CAMERA_HANDOFF_CHECK_TICK, CAMERA_ROUTE_TICK, camera_route_measurement
from prepare_sitting_repairs import BodyModel, encode, leg_fk, smoothstep
from repair_fresh_sitting import checked_trace
from repair_live_touchdown import (BANKS, CHECKPOINT, HERE, LIBRARY, MODEL, PRIOR, ROOT, applied_path,
                                  load_cases, read, screen, sha, write)
from repair_sitting_touchdown import retarget
from study_live_sitting import bank_from_geometry, geometry, stage
from validate_camera_touchdown import fallback_priors, routed_score


INSTALLED = ROOT/'outputs/camera_touchdown_20261115_full'
LAST = ROOT/'outputs/camera_touchdown_20261115_unseen'
TOUCHDOWN = ROOT/'outputs/touchdown_alignment_20261115_full/touchdown.npz'
KNOWN_GROUPS = ('repair_cases', *BANKS)
TARGETS = ('camera_unseen/17', 'camera_unseen/18', 'camera_unseen/21')
SCALE = np.array([.024, .014, 3., .16])


def snapshot(folder, inputs, *, clean_only=False):
    result, geometry, trace = checked_trace(folder,inputs)
    selected = [i for i,r in enumerate(result['episodes']) if not clean_only or r['clean']]
    priors = fallback_priors(trace,selected)
    bank = Path(read(folder/'manifest.json')['args']['bank'])
    pelvis = result['body_names'].index('pelvis')
    feet = [result['body_names'].index(f'{side}_ankle_roll_link') for side in ('left','right')]
    cases = {}
    for i,prior in zip(selected,priors,strict=True):
        g = geometry[i]
        row = result['episodes'][i]
        origin = np.asarray(result['origins'][i])
        pose = trace['body'][:,i,pelvis].copy()
        pose[:,:3] -= origin
        seat = np.asarray(g['top_center'])
        forward = Rotation.from_quat(g['rotation_xyzw']).as_matrix()[:,0]
        normal = np.asarray(g['normal'])
        ankle = trace['body'][0,i,feet,:3].mean(0)-origin
        feature = np.array([(seat-ankle)@forward+g['depth']/2,
                            (seat-ankle)@normal/normal[2],g['pitch_deg'],seat[1]-ankle[1]])
        cases[row['id']] = dict(row=row,geometry=g,feature=feature,index=i,bank=bank,
                               prior=prior,previous=trace['previous'][0,i],
                               motion=dict(body=pose,q=trace['q'][:,i].copy(),load=trace['support_bw'][:,i].copy()),
                               slip=trace['loaded_slip'][:,i].copy(),bad=np.flatnonzero(~trace['no_slip'][:,i]))
    return cases


def ranked_donors(cases, target, count):
    def distance(case):
        return (np.square((case['feature']-target['feature'])/SCALE).sum()
                + .25*np.square(case['slip'].max()/.02))
    ranked = sorted(((name,c) for name,c in cases.items() if c['row']['clean']),
                    key=lambda entry:distance(entry[1]))
    distinct = []
    for name,case in ranked:
        if any(np.max(np.abs(case['prior'][:750]-other['prior'][:750]))<.001 for _,other in distinct):
            continue
        distinct.append((name,case))
        if len(distinct)==count:
            return distinct
    raise ValueError('INSUFFICIENT_CLEAN_DONORS')


def prepare(out, refinement=None):
    cases, inputs, _ = load_cases()
    inputs.update({str(p):sha(p) for p in (Path(__file__),CHECKPOINT,MODEL,PRIOR,
                                          HERE/'validate_camera_touchdown.py',HERE/'repair_sitting_touchdown.py')})
    targets = snapshot(LAST/'evaluation',inputs)
    cases.update({key:c for key,c in targets.items() if c['row']['clean']})
    cases.update(snapshot(INSTALLED/'repair_cases',inputs,clean_only=True))
    assert tuple(key for key,c in targets.items() if not c['row']['clean'])==TARGETS
    bank = LAST/'unseen'
    inputs.update({str(p):sha(p) for p in bank.iterdir()})
    meta = read(bank/'metadata.json')
    limits = tuple(np.asarray(meta[key]) for key in ('lower','upper'))
    model = BodyModel(MODEL)
    fk, legs = leg_fk(model,meta['joint_names'])
    paths, rows, indices, rejected = [], [], [], []
    unresolved = TARGETS
    if refinement:
        prior_results = refinement/'screen_results.json'
        inputs[str(prior_results)] = sha(prior_results)
        prior_rows = read(prior_results)
        unresolved = tuple(key for key in TARGETS if not any(
            r['case']==key and r['clean'] and r['choice']['method']!='fallback' for r in prior_rows))
        assert unresolved==('camera_unseen/17',),unresolved
        _, _, prior_trace = checked_trace(refinement/'screen',inputs)
        probe_path = refinement/'paths.npy'
        assert sha(probe_path)==read(refinement/'screen.freeze.json')['inputs'][str(probe_path)]
        inputs[str(probe_path)] = sha(probe_path)
        probe_positions = np.load(probe_path,allow_pickle=False)

    def append(key, choice, path, checks=None):
        assert path.shape==(1000,23) and np.isfinite(path).all()
        if any(key==r['case'] and np.array_equal(path,p) for r,p in zip(rows,paths)):
            return
        rows.append(dict(id=f'probe/{len(paths)+1:02d}',case=key,choice=choice,checks=checks or {}))
        paths.append(path)
        indices.append(targets[key]['index'])

    for key in unresolved:
        target = targets[key]
        append(key,dict(method='fallback',donor=key),target['prior'])
        if refinement:
            nearby = ('fresh/15','known_transfer/10','known_recent/15','live_holdout/14','positioned/14')
            for name in nearby:
                assert cases[name]['row']['clean']
                append(key,dict(method='replay',donor=name),cases[name]['prior'])
            # Shape the weight transfer with a temporary arm counterbalance,
            # starting from the best collision-free rebound trace.
            probe = min((r for r in prior_rows if r['case']==key and not any(r['flags'])),
                        key=lambda r:(r['slip_ticks'],r['peak_slip']))
            idx = prior_rows.index(probe)
            for shoulder,elbow in ((.24,.27),(.34,.36),(.49,.42)):
                choice = dict(method='counterbalance',donor=probe['id'],shoulder_rad=shoulder,elbow_rad=elbow)
                try:
                    path,checks = counterbalance(probe_positions[idx],target['previous'],
                        prior_trace['support_bw'][:,idx],meta['joint_names'],limits,shoulder,elbow)
                except (AssertionError,ValueError) as exc:
                    rejected.append(dict(case=key,choice=choice,reason=str(exc)))
                    continue
                append(key,choice,path,checks)
            plans = [(name,cases[name],shift,pitch)
                     for name in ('known_recent/17','alignment_unseen/20','alignment_unseen/21','live_holdout/14')
                     for shift,pitch in ((0.,0.),(-.003,.027))]
        else:
            donors = ranked_donors(cases,target,4)
            plans = initial_plans(key,target,cases,donors,meta,append)
        for name,case,shift,pitch in plans:
            choice = dict(method='retarget',donor=name,shift_m=shift,pitch_rad=pitch)
            try:
                command = applied_path(case['prior'],case['previous'],*limits)
                waypoint, checks = retarget(case['motion'],case['geometry'],target['geometry'],
                                            limits,command,model,fk,legs,fore_aft_m=shift,
                                            pitch_adjust_rad=pitch,pitch_window_ticks=(175,480))
                latent, applied = encode(waypoint,target['previous'],*limits)
                error = float(np.abs(applied-waypoint).max())
                assert error<=.004,error
                checks['encoding_error_rad'] = error
            except (AssertionError,ValueError) as exc:
                rejected.append(dict(case=key,choice=choice,reason=str(exc)))
                continue
            append(key,choice,latent,checks)

    count = len(rows)
    assert 3<count<=72
    out.mkdir()
    dst = out/'bank'
    dst.mkdir()
    with np.load(bank/'bank.npz',allow_pickle=False) as z:
        arrays = {k:z[k][indices] for k in z.files if k.startswith('reset_')}
    arrays.update(reference=np.arange(count),admitted=np.ones(count,bool),
                  q_post=np.broadcast_to(arrays['reset_q'],(1000,count,37)),
                  phase_times=np.full((count,2),15.))
    np.savez_compressed(dst/'bank.npz',**arrays)
    np.save(out/'paths.npy',np.stack(paths).astype(np.float32))
    geo = read(bank/'geometry.json')
    write(dst/'geometry.json',[dict(geo[i],case=j,surface_id=j) for j,i in enumerate(indices)])
    write(dst/'metadata.json',dict(meta,cases=count,ids=[r['id'] for r in rows]))
    write(out/'candidates.json',dict(rows=rows,rejected=rejected,source_traces=inputs))
    write(out/'reservation.json',dict(files={str(f):sha(f) for f in dst.iterdir()}))


def initial_plans(key,target,cases,donors,meta,append):
    relevant = {
        TARGETS[0]: ('alignment_unseen/17','alignment_unseen/18','known_recent/17'),
        TARGETS[1]: ('alignment_unseen/15','positioned/04','positioned/12'),
        TARGETS[2]: ('alignment_unseen/18','alignment_unseen/21','positioned/08'),
    }[key]
    for name,case in [*donors,*[(name,cases[name]) for name in relevant]]:
        assert case['row']['clean']
        append(key,dict(method='replay',donor=name),case['prior'])
    plans = [(name,case,0.,0.) for name,case in donors[:2]]
    if any(target['row']['physical_failures']):
        # Shift the pelvis inboard while keeping both foot transforms.
        return plans+[(name,cases[name],shift,pitch) for name in relevant
                      for shift,pitch in ((0.,0.),(-.004,.043))]
    tick = int(target['bad'][0])
    window = np.arange(1000)
    pulse = smoothstep((window-tick+42)/24)*(1-smoothstep((window-tick-12)/25))
    ankles = [meta['joint_names'].index(f'{side}_ankle_pitch_joint') for side in ('left','right')]
    for amount in (-.009,.009):
        motion = target['prior'].copy()
        motion[:,ankles] += (amount*pulse[:,None]).astype(np.float32)
        append(key,dict(method='ankle_rebound',donor=key,rad=amount,tick=tick),motion)
    return plans


def diagnostic(out, refinement=None):
    prepare(out,refinement)
    for file,digest in read(out/'candidates.json')['source_traces'].items():
        assert sha(file)==digest,file
    screen(out)


def reserve_fresh(bank, known):
    previous = [g for path in known.values() for g in read(Path(path)/'geometry.json')]
    seats = {g['label']:g for g in previous}
    occupied = {tuple(g['top_center']+[g['pitch_deg']]) for g in previous}
    rows = []
    local = ((.002,-.004,.001,.18),(-.002,.004,-.001,-.18),
             (.003,.002,-.0006,.12),(-.003,-.002,.0006,-.12))
    wider = ((.009,.012,.002,.35),(-.007,-.012,-.002,-.35),
             (.004,-.017,.0015,-.23))
    templates = ('matched/01/02','alignment_unseen/15','alignment_unseen/19','camera_unseen/23')
    for key in (*TARGETS,*templates):
        seat = seats[key]
        x,y,z = seat['top_center']
        for dx,dy,dz,dp in (local if key in TARGETS else wider):
            rows.append(geometry(seat,f'fallback_fresh/{len(rows)+1:02d}',z+dz,
                                 seat['pitch_deg']+dp,x+.4+dx,y+dy))
    assert len(rows)==24 and not occupied.intersection(tuple(g['top_center']+[g['pitch_deg']]) for g in rows)
    bank_from_geometry(bank,Path(known['candidate']),rows)


def freeze(out, diagnostics):
    banks = {**read(INSTALLED/'routes.json')['banks'],'target_bank':str(LAST/'unseen')}
    inputs = {str(p):sha(p) for p in (Path(__file__),HERE/'live_sitting_edge.py',CHECKPOINT,LIBRARY,
                                      PRIOR,TOUCHDOWN,INSTALLED/'camera_touchdown.npz')}
    with np.load(INSTALLED/'camera_touchdown.npz',allow_pickle=False) as z:
        camera = dict(z)
    assert inputs[str(INSTALLED/'camera_touchdown.npz')]==read(INSTALLED/'routes.json')['payload_sha256']
    protected = []
    for folder in [*(INSTALLED/k for k in KNOWN_GROUPS),LAST/'evaluation']:
        audit = read(folder/'independent_audit.json')
        evidence = folder/'trajectory.npz'
        results = folder/'results.json'
        assert audit['passed'] and audit['files']['trajectory.npz']==sha(evidence)
        assert audit['files']['results.json']==sha(results)
        inputs.update({str(p):sha(p) for p in (evidence,results,folder/'independent_audit.json')})
        selected = [i for i,r in enumerate(read(results)['episodes']) if r['clean']]
        with np.load(evidence,allow_pickle=False) as z:
            protected.extend(z['obs'][0,selected,336:])
    assert len(protected)==233
    edge = LAST/'evaluation/edge_audit.npz'
    inputs[str(edge)] = sha(edge)
    with np.load(edge,allow_pickle=False) as z:
        early = z['values'][z['ticks']<=CAMERA_ROUTE_TICK]
    names = read(LAST/'unseen/metadata.json')['ids']
    features = {}
    for key in TARGETS:
        frames = early[:,names.index(key)]
        valid = (frames[:,2]<=.012)&(frames[:,3]>=4)
        features[key] = camera_route_measurement(frames[valid][:,[0,4,5]])
        assert features[key] is not None,key
    trials = []
    for folder in diagnostics:
        actions = folder/'paths.npy'
        evidence = folder/'screen/trajectory.npz'
        results = folder/'screen/results.json'
        audit = folder/'screen/independent_audit.json'
        receipt = folder/'screen_results.json'
        frozen = folder/'screen.freeze.json'
        check = read(audit)
        assert check['passed'] and read(frozen)['inputs'][str(actions)]==sha(actions)
        assert check['files']['trajectory.npz']==sha(evidence) and check['files']['results.json']==sha(results)
        inputs.update({str(p):sha(p) for p in (actions,evidence,results,audit,receipt,frozen)})
        with np.load(evidence,allow_pickle=False) as z:
            applied = z['command'][CAMERA_HANDOFF_CHECK_TICK,:,:23]
        for row,actual,path,handoff in zip(read(receipt),read(results)['episodes'],
                                          np.load(actions,allow_pickle=False),applied,strict=True):
            assert row['id']==actual['id'] and row['clean']==actual['clean']
            if row['choice']['method']!='fallback':
                trials.append((row,path,handoff,hashlib.sha256(path.tobytes()).hexdigest()))
    approved = []
    for row,path,handoff,digest in sorted(trials,key=lambda x:x[0]['peak_slip']):
        if not row['clean']:
            continue
        key = row['case']
        local = min((np.square((features[key]-features[r['case']])/camera['scales']).sum()
                     for r,_,_,tag in trials if tag==digest and not r['clean']),default=np.inf)
        radius = min(1.,.36*local)
        if radius>1e-6:
            approved.append((row,path,handoff,radius))
    assert approved
    additions = dict(ids=np.asarray([f"{row['case']} via {row['choice']['donor']}" for row,*_ in approved]),
                     paths=np.stack([p for _,p,_,_ in approved]),
                     handoff=np.stack([q for _,_,q,_ in approved]),
                     radius=np.asarray([r for *_,r in approved]),
                     features=np.stack([features[row['case']] for row,*_ in approved]))
    camera['baseline_count'] = np.asarray(len(camera['ids']))
    camera['extension_protected'] = np.stack(protected)
    for key,addition in additions.items():
        camera[key] = np.concatenate((camera[key],addition))
    out.mkdir()
    fresh = out/'fresh_bank'
    reserve_fresh(fresh,banks)
    banks['untouched'] = str(fresh)
    payload = out/'camera_touchdown.npz'
    np.savez_compressed(payload,**camera)
    write(out/'reservation.json',dict(files={str(f):sha(f) for name in banks.values() for f in Path(name).iterdir()}))
    write(out/'routes.json',dict(banks=banks,payload_sha256=sha(payload),inputs=inputs,
                               successful_probes=[row for row,*_ in approved],
                               unverified=sorted(set(TARGETS)-{row['case'] for row,*_ in approved}),
                               protected_extension_starts=len(protected),
                               fresh_used_for_design=False))


def evaluate(out, mode):
    protocol = out/'routes.json'
    design = read(protocol)
    camera = out/'camera_touchdown.npz'
    assert sha(camera)==design['payload_sha256']
    for name,digest in design['inputs'].items():
        assert sha(name)==digest,name
    arguments = ['--adapt-library',str(LIBRARY),'--touchdown-library',str(TOUCHDOWN),
                 '--camera-touchdown',str(camera)]
    comparison = out/'comparison.json'
    summaries = read(comparison) if comparison.exists() else {}
    groups = ('target_bank',) if mode=='target' else (*KNOWN_GROUPS,'untouched')
    for label in groups:
        score = stage(out,label,Path(design['banks'][label]),PRIOR,worker=HERE/'live_sitting_edge.py',
                      extra=arguments,inputs=[LIBRARY,TOUCHDOWN,camera,Path(__file__)],protocol=protocol)
        before = []
        if label!='untouched':
            folder = LAST/'evaluation' if label=='target_bank' else INSTALLED/label
            before = [r['id'] for r in read(folder/'results.json')['episodes'] if r['clean']]
        summaries[label] = routed_score(out/label,score,before)
        write(comparison,summaries)
    if mode=='full':
        known = [summaries[k] for k in KNOWN_GROUPS]
        assert sum(r['total'] for r in known)==212 and summaries['untouched']['total']==24
        write(out/'ppo_gate.json',dict(passed=summaries['untouched']['clean']==24,
              known_clean=sum(r['clean'] for r in known),fresh_clean=summaries['untouched']['clean'],
              fresh_total=24,known_regressions=sum(len(r['regressions']) for r in known),
              original_action_limits_and_scoring=True))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('diagnostic','freeze','target','full'))
    parser.add_argument('out',type=Path)
    parser.add_argument('screens',type=Path,nargs='*')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.mode=='diagnostic':
        diagnostic(out,args.screens[0].resolve() if args.screens else None)
    elif args.mode=='freeze':
        freeze(out,[p.resolve() for p in args.screens])
    else:
        evaluate(out,args.mode)
