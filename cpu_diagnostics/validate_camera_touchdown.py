"""Compare touchdown approaches on the same starts, then verify camera routing."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from prepare_sitting_repairs import BodyModel, encode, leg_fk
from repair_fresh_sitting import checked_trace
from repair_live_touchdown import (BANKS, CHECKPOINT, EVAL_SEED, HERE, LIBRARY, MODEL, PRIOR, ROOT,
                                  applied_path, load_cases, nearest, read, screen, sha, write)
from repair_sitting_touchdown import retarget
from live_sitting_edge import (CAMERA_HANDOFF_CHECK_TICK, CAMERA_ROUTE_TICK,
                               camera_route_measurement, choose_camera_route)
from common import OBS
from policy import Learner
from study_live_sitting import bank_from_geometry, geometry as seat_geometry, stage


PREVIOUS = ROOT/'outputs/touchdown_alignment_20261115_unseen'
INSTALLED = ROOT/'outputs/touchdown_alignment_20261115_full'
ALIGNED = ('matched/01/02', 'live_holdout/01')
NEAR = ROOT/'outputs/camera_touchdown_20261115_near'
DISTANT = ROOT/'outputs/camera_touchdown_20261115_distant'


def audited_references():
    inputs = {str(p):sha(p) for p in (CHECKPOINT, PRIOR, MODEL, Path(__file__),
              HERE/'repair_sitting_touchdown.py', HERE/'prepare_sitting_repairs.py')}
    result, geometry, trace = checked_trace(PREVIOUS/'evaluation', inputs)
    gate = PREVIOUS/'gate.json'
    bank = PREVIOUS/'unseen'
    for p in (gate, INSTALLED/'routes.json', PREVIOUS/'evaluation/edge_audit.npz', *bank.iterdir()):
        inputs[str(p)] = sha(p)
    return result, geometry, trace, bank, inputs


def fallback_priors(trace, ids):
    torch.set_num_threads(1)
    learner = Learner(np.zeros(OBS), np.ones(OBS), EVAL_SEED)
    learner.load(CHECKPOINT)
    with np.load(PRIOR,allow_pickle=False) as z:
        learner.bind_sitting({**learner.sitting,**dict(z)})
    path = trace['desired'][:, ids].copy()
    observed = torch.from_numpy(trace['obs'][750:, ids].copy()).flatten(0, 1)
    with torch.no_grad():
        residual = learner.actor(learner.normalize(observed))*learner.residual_mask(observed)
    path[750:] -= residual.numpy().reshape(250,len(ids),23)
    return path.transpose(1,0,2)


def prepare(out, group):
    result, geometry, trace, bank, inputs = audited_references()
    if group=='refine':
        earlier = DISTANT/'paired.json'
        inputs[str(earlier)] = sha(earlier)
        outstanding = {name for name,r in read(earlier).items() if not r['repaired']}
        assert outstanding=={'alignment_unseen/18'}
        selected = [i for i,r in enumerate(result['episodes']) if r['id'] in outstanding]
    else:
        selected = [i for i,r in enumerate(result['episodes']) if not r['clean'] and (i<12)==(group=='near')]
        assert len(selected)==(5 if group=='near' else 4)
    unresolved = []
    if group=='distant':
        near_file = NEAR/'paired.json'
        inputs[str(near_file)] = sha(near_file)
        unresolved = [name for name,r in read(near_file).items() if not r['repaired']]
        selected.extend(i for i,r in enumerate(result['episodes']) if r['id'] in unresolved)
    names = [result['episodes'][i]['id'] for i in selected]
    assert all(read(PREVIOUS/'gate.json')['runtime_routes'][i] is None for i in selected)
    fallback = fallback_priors(trace, selected)
    indices, paths, rows, rejected = [], [], [], []

    def append(i, action, source, method, checks=None):
        assert action.shape==(1000,23) and np.isfinite(action).all()
        name = geometry[i]['label']
        if any(r['case']==name and np.array_equal(action,p) for r,p in zip(rows,paths)):
            return
        rows.append(dict(id=f'probe/{len(paths)+1:02d}',case=name,
                         choice=dict(method=method,donor=source),checks=checks or {}))
        paths.append(action)
        indices.append(i)

    original = INSTALLED/'touchdown.npz'
    inputs[str(original)] = sha(original)
    assert inputs[str(original)] == read(INSTALLED/'routes.json')['payload_sha256']
    with np.load(original,allow_pickle=False) as z:
        aligned = {str(key): path for key,path in zip(z['ids'],z['paths']) if key in ALIGNED}

    if group!='near':
        donors, known_inputs, _ = load_cases()
        inputs.update(known_inputs)
        meta = read(bank/'metadata.json')
        limits = tuple(np.asarray(meta[k]) for k in ('lower','upper'))
        model = BodyModel(MODEL)
        fk, legs = leg_fk(model, meta['joint_names'])
        feet = [result['body_names'].index(f'{side}_ankle_roll_link') for side in ('left','right')]
    if group=='distant':
        repaired = {}
        for key, label in zip(ALIGNED,('candidate','fresh_once')):
            r, g, data = checked_trace(INSTALLED/label,inputs)
            k = next(j for j,s in enumerate(r['episodes']) if s['id']==key)
            assert r['episodes'][k]['clean']
            pose = data['body'][:,k,r['body_names'].index('pelvis')].copy()
            pose[:,:3] -= np.asarray(r['origins'][k])
            repaired[key] = dict(prior=aligned[key],previous=data['previous'][0,k],geometry=g[k],
                                 motion=dict(body=pose,q=data['q'][:,k],load=data['support_bw'][:,k]))

    def shift_path(i, donor, ref, fore_aft=0., pitch=0.):
        geo = geometry[i]
        initial = trace['previous'][0,i]
        try:
            command = applied_path(ref['prior'],ref['previous'],*limits)
            waypoint, checks = retarget(ref['motion'],ref['geometry'],geo,limits,command,model,fk,legs,
                                        fore_aft_m=fore_aft,pitch_adjust_rad=pitch,
                                        pitch_window_ticks=(175,480))
            path, actual = encode(waypoint,initial,*limits)
            error = float(np.abs(actual-waypoint).max())
            assert error<=.004,error
            checks.update(encoding_error_rad=error, fore_aft_m=fore_aft,pitch_rad=pitch)
        except (ValueError, AssertionError) as exc:
            rejected.append(dict(case=geo['label'],donor=donor,fore_aft_m=fore_aft,pitch_rad=pitch,reason=str(exc)))
            return
        append(i,path,donor,'retarget',checks)

    for i, base in zip(selected,fallback):
        append(i,base,geometry[i]['label'],'fallback')
        if group=='near':
            for key in ALIGNED:
                append(i,aligned[key],key,'aligned')
            continue
        geo = geometry[i]
        if group=='refine':
            # The first screen found a slip-free but forbidden contact with
            # holdout/5/0. Bring the torso forward and preserve both feet,
            # keeping its pelvis-to-plane gap while clearing the hips.
            for donor in ('positioned/08','positioned/04','fresh/12','known_old/6','known_recent/17'):
                append(i,donors[donor]['prior'],donor,'replay')
            for donor in ('holdout/5/0','positioned/08'):
                for shift,pitch in ((0.,0.),(0.,.034),(-.006,.052)):
                    shift_path(i,donor,donors[donor],shift,pitch)
            continue
        if geo['label'] in unresolved:
            for donor, ref in repaired.items():
                for shift in (0.,-.0015,.0015):
                    shift_path(i,donor,ref,shift)
            balance = ROOT/'outputs/touchdown_alignment_20261115_balance_v1'
            path = balance/'paths.npy'
            inputs[str(path)] = sha(path)
            assert inputs[str(path)]==read(balance/'screen.freeze.json')['inputs'][str(path)]
            alt = next(j for j,r in enumerate(read(balance/'screen_results.json')) if r['id']=='probe/17' and r['clean'])
            append(i,np.load(path,allow_pickle=False)[alt],'probe/17','alternate_counterbalance')
            continue
        # Scene geometry constructs the diagnostic counterfactuals. The later
        # live controller selects only from measured camera features.
        center = np.asarray(geo['top_center'])
        normal = np.asarray(geo['normal'])
        edge_axis = Rotation.from_quat(geo['rotation_xyzw']).as_matrix()[:,0]
        ankle_center = trace['body'][0,i,feet,:3].mean(0)-np.asarray(result['origins'][i])
        feature = np.array([(center-ankle_center)@edge_axis+geo['depth']/2,
                            (center-ankle_center)@normal/normal[2],geo['pitch_deg'],center[1]-ankle_center[1]])
        donors[geo['label']] = dict(row=dict(clean=False),feature=feature)
        candidates = nearest(donors,geo['label'],4)
        for donor in candidates:
            append(i,donors[donor]['prior'],donor,'replay')
        for donor in candidates[:2]:
            shift_path(i,donor,donors[donor])

    out.mkdir()
    dst = out/'bank'
    dst.mkdir()
    count = len(rows)
    with np.load(bank/'bank.npz',allow_pickle=False) as z:
        arrays = {k:z[k][indices] for k in z.files if k.startswith('reset_')}
    arrays.update(reference=np.arange(count), admitted=np.ones(count,bool),
                  q_post=np.broadcast_to(arrays['reset_q'],(1000,count,37)),
                  phase_times=np.full((count,2),15.))
    np.savez_compressed(dst/'bank.npz',**arrays)
    np.save(out/'paths.npy',np.stack(paths).astype(np.float32))
    write(dst/'geometry.json',[dict(geometry[i],case=j,surface_id=j) for j,i in enumerate(indices)])
    write(dst/'metadata.json',dict(read(bank/'metadata.json'),cases=count,ids=[r['id'] for r in rows]))
    write(out/'candidates.json',dict(rows=rows,rejected=rejected,comparison_cases=names,source_traces=inputs))
    write(out/'reservation.json',dict(files={str(f):sha(f) for f in dst.iterdir()}))


def paired(out, group):
    prepare(out,group)
    for name,digest in read(out/'candidates.json')['source_traces'].items():
        assert sha(name)==digest,name
    screen(out)
    entries = read(out/'screen_results.json')
    baseline = read(PREVIOUS/'evaluation/results.json')['episodes']
    paired_rows = {}
    for name in read(out/'candidates.json')['comparison_cases']:
        trials = [row for row in entries if row['case']==name]
        reference = next(row for row in baseline if row['id']==name)
        assert trials[0]['choice']['method']=='fallback'
        paired_rows[name] = dict(fallback=trials[0],saved_fallback=reference,
                                 repaired=[r for r in trials[1:] if r['clean']],
                                 alternatives=trials[1:])
    write(out/'paired.json',paired_rows)


def camera_calibration(out, screens):
    runs = [*(INSTALLED/k for k in BANKS),PREVIOUS/'evaluation']
    inputs = {str(p):sha(p) for p in (Path(__file__),HERE/'live_sitting_edge.py',
                                      CHECKPOINT,LIBRARY,PRIOR,INSTALLED/'touchdown.npz')}
    protected = []
    known = read(INSTALLED/'routes.json')['banks']
    banks = {**known,'repair_cases':str(PREVIOUS/'unseen')}
    reservations = {str(p):sha(p) for bank in banks.values() for p in Path(bank).iterdir()}
    for run in runs:
        audit = read(run/'independent_audit.json')
        path = run/'trajectory.npz'
        assert audit['passed'] and sha(path)==audit['files']['trajectory.npz']
        inputs.update({str(p):sha(p) for p in (path,run/'independent_audit.json',run/'results.json')})
        rows = read(run/'results.json')['episodes']
        with np.load(path,allow_pickle=False) as z:
            protected.extend(z['obs'][0,[i for i,r in enumerate(rows) if r['clean']],336:])
    assert len(protected)==203
    geometry = read(PREVIOUS/'unseen/geometry.json')
    names = [g['label'] for g in geometry]
    edge_file = PREVIOUS/'evaluation/edge_audit.npz'
    inputs[str(edge_file)] = sha(edge_file)
    with np.load(edge_file,allow_pickle=False) as z:
        early = z['values'][z['ticks']<=CAMERA_ROUTE_TICK]
        feature = {}
        for i,name in enumerate(names):
            frames = early[:,i]
            valid = (frames[:,2]<=.012)&(frames[:,3]>=4)
            feature[name] = camera_route_measurement(frames[valid][:,[0,4,5]])
    evaluations = []
    for folder in screens:
        path = folder/'paths.npy'
        audit = folder/'screen/independent_audit.json'
        receipts = folder/'screen_results.json'
        frozen = folder/'screen.freeze.json'
        evidence = folder/'screen/trajectory.npz'
        audit_receipt = read(audit)
        assert audit_receipt['passed'] and sha(path)==read(frozen)['inputs'][str(path)]
        assert sha(evidence)==audit_receipt['files']['trajectory.npz']
        for p in (path,audit,receipts,frozen,evidence,folder/'paired.json'):
            inputs[str(p)] = sha(p)
        with np.load(evidence,allow_pickle=False) as z:
            replayed = z['command'][CAMERA_HANDOFF_CHECK_TICK,:,:23]
        physical = read(folder/'screen/results.json')['episodes']
        for row,trajectory,check,command_end in zip(read(receipts),np.load(path,allow_pickle=False),
                                                    physical,replayed,strict=True):
            assert row['id']==check['id'] and row['clean']==check['clean']
            if row['choice']['method']=='fallback':
                continue
            sample = feature[row['case']]
            if sample is not None:
                digest = hashlib.sha256(trajectory.tobytes()).hexdigest()
                evaluations.append((row,trajectory,sample,digest,command_end))

    # A nearby failed rollout with the same motion limits where a successful
    # rollout can support use of that motion. Calibration uses camera units.
    scales = np.asarray([.012,.007,.9])
    support = []
    for row,path,sample,digest,command_end in evaluations:
        if not row['clean']:
            continue
        nearest_bad = min((np.square((x-sample)/scales).sum()
                           for r,_,x,key,_ in evaluations if key==digest and not r['clean']),default=np.inf)
        radius = min(1.,.36*nearest_bad)
        if radius>1e-6:
            support.append((row,path,sample,radius,command_end))
    assert support
    support.sort(key=lambda item:item[0]['peak_slip'])
    with np.load(INSTALLED/'touchdown.npz',allow_pickle=False) as z:
        signature_scale = z['scale']
    camera = dict(ids=np.asarray([f"{row['case']} via {row['choice']['donor']}" for row,*_ in support]),
                  paths=np.stack([path for _,path,_,_,_ in support]),
                  features=np.stack([x for _,_,x,_,_ in support]),
                  radius=np.asarray([r for _,_,_,r,_ in support]),
                  handoff=np.stack([command_end for *_,command_end in support]),
                  scales=scales,protected=np.stack(protected),signature_scale=signature_scale)
    out.mkdir()
    payload = out/'camera_touchdown.npz'
    np.savez_compressed(payload,**camera)
    picks = {}
    for name,sample in feature.items():
        if sample is None:
            continue
        j,distance = choose_camera_route(sample,camera)
        picks[name] = dict(features=sample.tolist(),distance=distance,
                           route=None if j<0 else str(camera['ids'][j]))
    write(out/'routes.json',dict(banks=banks,inputs=inputs,payload_sha256=sha(payload),
         proven_paths=[dict(row,acceptance_squared_distance=r) for row,_,_,r,_ in support],
         archived_feature_decisions=picks,protected_clean_starts=len(protected)))
    write(out/'reservation.json',dict(files=reservations))


def run_cameras(out, only_repair=False):
    receipt = out/'routes.json'
    routes = read(receipt)
    camera = out/'camera_touchdown.npz'
    assert sha(camera)==routes['payload_sha256']
    for file,digest in routes['inputs'].items():
        assert sha(file)==digest,file
    touched = INSTALLED/'touchdown.npz'
    arguments = ['--adapt-library',str(LIBRARY),'--touchdown-library',str(touched),
                 '--camera-touchdown',str(camera)]
    sources = [LIBRARY,touched,camera,Path(__file__)]
    names = ('repair_cases',) if only_repair else BANKS
    summaries = read(out/'comparison.json') if (out/'comparison.json').exists() else {}
    for name in names:
        old_folder = PREVIOUS/'evaluation' if name=='repair_cases' else INSTALLED/name
        score = stage(out,name,Path(routes['banks'][name]),PRIOR,worker=HERE/'live_sitting_edge.py',
                      extra=arguments,inputs=sources,protocol=receipt)
        baseline = read(old_folder/'results.json')['episodes']
        reference_clean = {r['id'] for r in baseline if r['clean']}
        summaries[name] = routed_score(out/name,score,reference_clean)
        write(out/'comparison.json',summaries)
    if not only_repair:
        assert sum(summaries[k]['total'] for k in BANKS)==188


def routed_score(folder, score, baseline_clean=()):
    rows = read(folder/'results.json')['episodes']
    clean = {r['id'] for r in rows if r['clean']}
    with np.load(folder/'camera_touchdown_selection.npz',allow_pickle=False) as selection:
        decisions = {r['id']:str(selection['ids'][k]) for r,k in zip(rows,selection['selected']) if k>=0}
    return dict(score,baseline_clean=len(baseline_clean),regressions=sorted(set(baseline_clean)-clean),
                repairs=sorted(clean-set(baseline_clean)) if baseline_clean else [],
                camera_choices=decisions,camera_clean=len(clean.intersection(decisions)))


def unseen_validation(out, installed):
    routes = read(installed/'routes.json')
    banks = {key:Path(path) for key,path in routes['banks'].items()}
    past = [g for bank in banks.values() for g in read(bank/'geometry.json')]
    previous = {g['label']:g for g in past}
    seen = {tuple(g['top_center']+[g['pitch_deg']]) for g in past}
    samples = []
    for key in ('matched/01/02','live_holdout/01','alignment_unseen/15',
                'alignment_unseen/18','alignment_unseen/19','alignment_unseen/21'):
        seat = previous[key]
        for dx,dy,dz,dp in ((.0015,-.003,.0014,.18),(-.0015,.003,-.0014,-.18)):
            x,y,z = seat['top_center']
            samples.append(seat_geometry(seat,f'camera_unseen/{len(samples)+1:02d}',
                                        z+dz,seat['pitch_deg']+dp,x+.4+dx,y+dy))
    for i,height in enumerate((.368,.386,.405,.429)):
        for j,pitch in enumerate((8.75,13.25,18.25)):
            dx,dy = ((-.018,-.026),(.008,.028),(.017,-.016))[(i+2*j)%3]
            samples.append(seat_geometry(samples[0],f'camera_unseen/{len(samples)+1:02d}',
                                         height,pitch,dx,dy))
    assert len(samples)==24 and not seen.intersection(tuple(g['top_center']+[g['pitch_deg']]) for g in samples)
    out.mkdir()
    bank = out/'unseen'
    bank_from_geometry(bank,banks['candidate'],samples)
    write(out/'reservation.json',dict(files={str(p):sha(p) for p in bank.iterdir()}))
    comparison = read(installed/'comparison.json')
    assert sum(comparison[k]['total'] for k in BANKS)==188
    camera = installed/'camera_touchdown.npz'
    assert sha(camera)==routes['payload_sha256']
    touched = INSTALLED/'touchdown.npz'
    score = stage(out,'evaluation',bank,PRIOR,worker=HERE/'live_sitting_edge.py',
                  extra=['--adapt-library',str(LIBRARY),'--touchdown-library',str(touched),
                         '--camera-touchdown',str(camera)],
                  inputs=[LIBRARY,touched,camera,Path(__file__),installed/'comparison.json'],
                  protocol=installed/'routes.json')
    score = routed_score(out/'evaluation',score)
    known_clean = sum(comparison[k]['clean'] for k in BANKS)
    write(out/'gate.json',dict(passed=known_clean==188 and score['clean']==24 and bool(score['camera_choices']),
          known_clean=known_clean,repair_set_clean=comparison['repair_cases']['clean'],
          unseen=score,used_for_route_selection=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group',choices=('near','distant','refine','fit','repair','known','unseen'))
    parser.add_argument('out',type=Path)
    parser.add_argument('inputs',type=Path,nargs='*')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.group in ('near','distant','refine'):
        paired(out,args.group)
    elif args.group=='fit':
        camera_calibration(out,[p.resolve() for p in args.inputs])
    elif args.group=='unseen':
        unseen_validation(out,args.inputs[0].resolve())
    else:
        run_cameras(out,only_repair=args.group=='repair')
