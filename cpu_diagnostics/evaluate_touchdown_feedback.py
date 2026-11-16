"""Compare encoder and rear-depth touchdown feedback on the archived starts."""

import argparse
from pathlib import Path

import numpy as np

from align_sitting_touchdown import counterbalance
from live_sitting_edge import CAMERA_HANDOFF_CHECK_TICK
from prepare_sitting_repairs import BodyModel, encode, leg_fk
from repair_camera_fallback import KNOWN_GROUPS, TOUCHDOWN, snapshot
from repair_live_touchdown import CHECKPOINT, HERE, LIBRARY, MODEL, PRIOR, ROOT, applied_path, read, sha, write
from repair_sitting_touchdown import retarget, retime
from study_live_sitting import bank_from_geometry, geometry, stage
from validate_camera_touchdown import routed_score


BASE = ROOT/'outputs/touchdown_neighborhoods_20261115_full'
FAILED = (2,3,5,13,19,22)
NEIGHBORS = (1,4,6,14,20,23)
PROFILES = [
    dict(inset=.017,edge_gain=.65,damping=.08,normal_damping=.05,edge_lift=.24),
    dict(inset=.018,edge_gain=.76,damping=.16,normal_damping=.09,edge_lift=.28),
    dict(inset=.020,edge_gain=.94,damping=.12,normal_damping=.08,edge_lift=.42),
]
EARLY_EDGE_PROFILES = [
    dict(PROFILES[1],edge_bias_m=[.004,.006]),
    dict(PROFILES[1],edge_bias_m=[.007,.010]),
]


def base_paths():
    library = BASE/'camera_touchdown.npz'
    existing = read(BASE/'routes.json')
    assert sha(library)==existing['payload_sha256']
    return library,Path(existing['banks']['untouched'])


def arguments(library,config=None,protection=None,routes=None):
    extra = ['--adapt-library',str(LIBRARY),'--touchdown-library',str(TOUCHDOWN),
             '--camera-touchdown',str(library)]
    inputs = [LIBRARY,TOUCHDOWN,library,Path(__file__)]
    if config is not None:
        extra += ['--touchdown-feedback',str(config)]
        inputs += [config,HERE/'touchdown_feedback.py']
    if protection is not None:
        extra += ['--feedback-protected',str(protection)]
        inputs.append(protection)
    if routes is not None:
        extra += ['--feedback-routes',str(routes)]
        inputs.append(routes)
    return extra,inputs


def pair_bank(out, archived, profiles, subset):
    old = read(archived/'geometry.json')
    metadata = read(archived/'metadata.json')
    cases = []
    selections = []
    for profile in (-1,*range(len(profiles))):
        for index in subset:
            selections.append(index-1)
            cases.append(dict(id=f'pair/{profile+1}/{index:02d}',index=index,
                              archived=metadata['ids'][index-1],profile=profile))
    out.mkdir()
    bank = out/'pairs'
    bank.mkdir()
    with np.load(archived/'bank.npz',allow_pickle=False) as z:
        arrays = {k:(z[k][:,selections] if k=='q_post' else z[k][selections])
                  for k in z.files if k!='reference'}
    arrays['reference'] = np.arange(len(cases))
    np.savez_compressed(bank/'bank.npz',**arrays)
    write(bank/'geometry.json',[dict(old[i],case=j,surface_id=j) for j,i in enumerate(selections)])
    write(bank/'metadata.json',dict(metadata,cases=len(cases),ids=[r['id'] for r in cases]))
    write(out/'feedback.json',dict(profiles=profiles,profile_by_row=[r['profile'] for r in cases]))
    write(out/'paired_design.json',dict(cases=cases,source=str(archived),fresh_used_for_design=False))
    write(out/'reservation.json',dict(files={str(f):sha(f) for f in bank.iterdir()}))
    return cases,bank


def screen(out, profiles=PROFILES, subset=tuple(sorted((*FAILED,*NEIGHBORS))), routes=None):
    library,archived = base_paths()
    cases,bank = pair_bank(out,archived,profiles,subset)
    extra,inputs = arguments(library,out/'feedback.json',routes=routes)
    if routes is not None:
        inputs.append(routes.with_suffix('.json'))
    stage(out,'screen',bank,PRIOR,worker=HERE/'live_sitting_edge.py',extra=extra,
          inputs=inputs,protocol=out/'paired_design.json')
    recorded = read(out/'screen/results.json')['episodes']
    assert len(recorded)==len(cases)
    data = []
    for case,result in zip(cases,recorded,strict=True):
        assert case['id']==result['id']
        data.append(dict(case,clean=result['clean'],stable=result['stable'],
             hold_s=result['final_hold_s'],slip_ticks=result['slip_ticks'],
             illegal_ticks=result['illegal_ticks'],flags=result['physical_failures']))
    write(out/'paired_results.json',data)


def edge_routes(folder):
    """Retarget two verified nearby approaches while preserving both feet."""
    folder.mkdir()
    inputs = {}
    source = BASE/'untouched'
    cases = snapshot(source,inputs)
    camera = source/'camera_touchdown_selection.npz'
    inputs.update({str(path):sha(path) for path in
                   (camera,MODEL,HERE/'repair_sitting_touchdown.py',HERE/'prepare_sitting_repairs.py')})
    trajectories = [('neighborhood_fresh/04','neighborhood_fresh/05'),
                    ('neighborhood_fresh/14','neighborhood_fresh/13')]
    with np.load(camera,allow_pickle=False) as z:
        features = np.stack([z['features'][cases[target]['index']] for _,target in trajectories])
    assert np.isfinite(features).all()
    metadata = read(Path(read(source/'manifest.json')['args']['bank'])/'metadata.json')
    limits = [np.asarray(metadata[k]) for k in ('lower','upper')]
    model = BodyModel(MODEL)
    fk,legs = leg_fk(model,metadata['joint_names'])
    paths,handoffs,checks = [],[],[]
    for use_slowdown in (False,True):
        variant,applied_handoffs = [],[]
        for n,(donor_id,target_id) in enumerate(trajectories):
            donor,target = cases[donor_id],cases[target_id]
            assert donor['row']['clean'] and not target['row']['clean']
            command = applied_path(donor['prior'],donor['previous'],*limits)
            waypoint,geometry_checks = retarget(donor['motion'],donor['geometry'],target['geometry'],
                    limits,command,model,fk,legs,fore_aft_m=(-.004,-.003)[n],
                    pitch_adjust_rad=(0.,.020)[n])
            latent,applied = encode(waypoint,target['previous'],*limits)
            landing = int(round(geometry_checks['donor_landing_s']/.02))-1
            assert np.abs(applied[landing-28:]-waypoint[landing-28:]).max()<=.004
            timing = {}
            if use_slowdown and n==0:
                latent,applied,waypoint,timing = retime(applied,target['previous'],landing,landing+18,limits,.60)
                assert np.abs(applied-waypoint).max()<=.004
            variant.append(latent)
            applied_handoffs.append(applied[CAMERA_HANDOFF_CHECK_TICK])
            checks.append(dict(donor=donor_id,target=target_id,geometry=geometry_checks,timing=timing))
        paths.append(variant)
        handoffs.append(applied_handoffs)
    for path,digest in inputs.items():
        assert sha(path)==digest,path
    payload = folder/'edge_routes.npz'
    np.savez_compressed(payload,paths=np.asarray(paths),handoff=np.asarray(handoffs),
                        features=features,scale=[.007,.005,.35],radius=[.85,1.],
                        ids=['retargeted shallow touchdown','retargeted steep touchdown'])
    write(payload.with_suffix('.json'),dict(sources=inputs,checks=checks,payload_sha256=sha(payload)))
    return payload


def balanced_routes(folder, previous):
    """Counter the shallow-seat rebound using the measured load-transfer time."""
    audit = previous/'screen/independent_audit.json'
    trajectories = previous/'screen/trajectory.npz'
    results = previous/'screen/results.json'
    launch = previous/'screen.freeze.json'
    source = previous.with_name(previous.name+'_routes')/'edge_routes.npz'
    verified = read(audit)
    assert verified['passed'] and verified['files']['trajectory.npz']==sha(trajectories)
    assert verified['files']['results.json']==sha(results)
    assert read(launch)['inputs'][str(source)]==sha(source)
    rows = read(results)['episodes']
    index = next(i for i,row in enumerate(rows) if row['id']=='pair/1/05')
    row = rows[index]
    assert row['slip_ticks']>0 and row['illegal_ticks']==0 and not any(row['physical_failures'])
    meta = read(previous/'pairs/metadata.json')
    limits = [np.asarray(meta[k]) for k in ('lower','upper')]
    with np.load(trajectories,allow_pickle=False) as data:
        load = data['support_bw'][:,index]
        initial = data['previous'][0,index]
    with np.load(source,allow_pickle=False) as data:
        library = {k:data[k] for k in data.files}
    paths,handoffs,checks,rejected = [],[],[],[]
    for shoulder,elbow in ((.10,.11),(.14,.15),(.18,.20)):
        try:
            corrected,check = counterbalance(library['paths'][0,0],initial,load,meta['joint_names'],
                                             limits,shoulder,elbow)
        except AssertionError as error:
            rejected.append(dict(shoulder_rad=shoulder,elbow_rad=elbow,reason=str(error)))
            continue
        handoff = applied_path(corrected[:CAMERA_HANDOFF_CHECK_TICK+1],initial,*limits)[-1]
        paths.append(np.stack([corrected,library['paths'][0,1]]))
        handoffs.append(np.stack([handoff,library['handoff'][0,1]]))
        checks.append(check)
    assert paths
    folder.mkdir()
    payload = folder/'edge_routes.npz'
    np.savez_compressed(payload,**dict(library,paths=np.stack(paths),handoff=np.stack(handoffs)))
    sources = [source,source.with_suffix('.json'),trajectories,results,audit,launch,HERE/'align_sitting_touchdown.py']
    write(payload.with_suffix('.json'),dict(sources={str(p):sha(p) for p in sources},checks=checks,rejected=rejected,
                        payload_sha256=sha(payload)))
    return payload


def reserve_fresh(bank,banks):
    known = [seat for path in banks.values() for seat in read(Path(path)/'geometry.json')]
    by_name = {seat['label']:seat for seat in known}
    occupied = {tuple(seat['top_center']+[seat['pitch_deg']]) for seat in known}
    variations = ((.0022,-.0045,.0012,.19),(-.0022,.0045,-.0012,-.19))
    wider = ((.0085,.015,.0025,.34),(-.0085,-.015,-.0025,-.34),(.005,-.019,.0013,-.28))
    examples = [(f'neighborhood_fresh/{i:02d}',variations) for i in FAILED]
    examples += [(name,wider) for name in ('alignment_unseen/15','fallback_fresh/19',
                                          'matched/03/05','fresh/24')]
    fresh = []
    for name, changes in examples:
        original = by_name[name]
        x,y,z = original['top_center']
        for dx,dy,dz,dp in changes:
            fresh.append(geometry(original,f'feedback_fresh/{len(fresh)+1:02d}',z+dz,
                                  original['pitch_deg']+dp,x+.4+dx,y+dy))
    assert len(fresh)==24 and not occupied.intersection(tuple(g['top_center']+[g['pitch_deg']]) for g in fresh)
    bank_from_geometry(bank,Path(banks['candidate']),fresh)


def freeze(out, trial):
    design_path = trial/'paired_design.json'
    launch_path = trial/'screen.freeze.json'
    results_path = trial/'paired_results.json'
    config_path = trial/'feedback.json'
    audit_path = trial/'screen/independent_audit.json'
    actual_path = trial/'screen/results.json'
    audit = read(audit_path)
    assert audit['passed'] and audit['files']['results.json']==sha(actual_path)
    recorded = read(actual_path)['episodes']
    pairs = read(results_path)
    for pair,row in zip(pairs,recorded,strict=True):
        assert pair['id']==row['id'] and pair['clean']==row['clean'] and row['full_standing_start']
    launch = read(launch_path)
    for name in ('live_sitting_edge.py','touchdown_feedback.py'):
        matches = [digest for path,digest in launch['inputs'].items() if Path(path).name==name]
        assert matches==[sha(HERE/name)],name
    assert launch['inputs'][str(config_path)]==sha(config_path)
    routes = None
    if '--feedback-routes' in launch['command']:
        routes = Path(launch['command'][launch['command'].index('--feedback-routes')+1])
        assert launch['inputs'][str(routes)]==sha(routes)
        assert launch['inputs'][str(routes.with_suffix('.json'))]==sha(routes.with_suffix('.json'))
    profiles = read(config_path)['profiles']
    candidates = []
    for variant,config in enumerate(profiles):
        rows = [r for r in pairs if r['profile']==variant]
        repaired = {r['index'] for r in rows if r['index'] in FAILED and r['clean']}
        retained = {r['index'] for r in rows if r['index'] in NEIGHBORS and r['clean']}
        if repaired==set(FAILED) and retained==set(NEIGHBORS):
            candidates.append((min(r['hold_s'] for r in rows),variant,config))
    out.mkdir()
    if not candidates:
        write(out/'paired_gate.json',dict(passed=False,source=str(results_path),
                                          reason='No feedback setting passed all six failures and six nearby starts.'))
        return False
    _,variant,setting = max(candidates,key=lambda r:r[0])
    library,archived = base_paths()
    previous_routes = read(BASE/'routes.json')
    banks = {**{k:v for k,v in previous_routes['banks'].items() if k!='untouched'},
             'feedback_bank':str(archived)}
    inputs = {str(p):sha(p) for p in (Path(__file__),HERE/'live_sitting_edge.py',HERE/'touchdown_feedback.py',
                                     CHECKPOINT,LIBRARY,TOUCHDOWN,PRIOR,library,BASE/'routes.json',
                                     design_path,launch_path,results_path,config_path,audit_path,actual_path)}
    protected = []
    for label in (*KNOWN_GROUPS,'target_bank','adjustment_bank'):
        folder = BASE/label
        audit = read(folder/'independent_audit.json')
        trace = folder/'trajectory.npz'
        results = folder/'results.json'
        assert audit['passed'] and audit['files']['trajectory.npz']==sha(trace)
        assert audit['files']['results.json']==sha(results)
        inputs.update({str(p):sha(p) for p in (trace,results,folder/'independent_audit.json')})
        clean = [i for i,r in enumerate(read(results)['episodes']) if r['clean']]
        with np.load(trace,allow_pickle=False) as data:
            protected.extend(data['obs'][0,clean,336:])
    assert len(protected)==260
    with np.load(library,allow_pickle=False) as z:
        scale = z['signature_scale']
    holdout = out/'fresh_bank'
    reserve_fresh(holdout,banks)
    banks['untouched'] = str(holdout)
    guard = out/'protected.npz'
    np.savez_compressed(guard,signature=np.stack(protected),scale=scale)
    config = out/'feedback.json'
    write(config,dict(profiles=[setting]))
    inputs.update({str(p):sha(p) for p in (guard,config)})
    if routes is not None:
        inputs.update({str(p):sha(p) for p in (routes,routes.with_suffix('.json'))})
    write(out/'reservation.json',dict(files={str(f):sha(f) for p in banks.values() for f in Path(p).iterdir()}))
    write(out/'paired_gate.json',dict(passed=True,variant=variant,profiles=profiles,banks=banks,inputs=inputs,
                                     routes=str(routes) if routes is not None else None,
                                     protected_starts=len(protected),fresh_used_for_design=False))
    return True


def evaluate(out, mode):
    protocol = out/'paired_gate.json'
    design = read(protocol)
    assert design['passed']
    for name,digest in design['inputs'].items():
        assert sha(name)==digest,name
    library,_ = base_paths()
    routes = Path(design['routes']) if design['routes'] is not None else None
    extra,inputs = arguments(library,out/'feedback.json',out/'protected.npz',routes)
    comparison = out/'comparison.json'
    data = read(comparison) if comparison.exists() else {}
    labels = ('feedback_bank',) if mode=='confirm' else (*KNOWN_GROUPS,'target_bank','adjustment_bank','untouched')
    if mode=='full':
        assert data['feedback_bank']['total']==24
    for label in labels:
        score = stage(out,label,Path(design['banks'][label]),PRIOR,worker=HERE/'live_sitting_edge.py',
                      extra=extra,inputs=inputs,protocol=protocol)
        before = []
        if label!='untouched':
            old = BASE/('untouched' if label=='feedback_bank' else label)
            before = [r['id'] for r in read(old/'results.json')['episodes'] if r['clean']]
        data[label] = routed_score(out/label,score,before)
        write(comparison,data)
    if mode=='full':
        ppo_gate(out,data,out)


def ppo_gate(out,data,unchanged):
    known = [data[k] for k in KNOWN_GROUPS]
    repairs = [data[k] for k in ('target_bank','adjustment_bank')]
    fresh = data['untouched']
    assert sum(r['total'] for r in known)==212 and fresh['total']==24
    assert all(r['total']==24 for r in repairs)
    retained = all(r['clean']==r['total'] and not r['regressions']
                   for r in (*known,*repairs,data['feedback_bank']))
    evidence = {label:str(unchanged/label) for label in (*KNOWN_GROUPS,'target_bank','adjustment_bank','untouched')}
    evidence['feedback_bank'] = str(out/'feedback_bank')
    write(out/'ppo_gate.json',dict(passed=retained and fresh['clean']==24,
          known_clean=sum(r['clean'] for r in known),fresh_clean=fresh['clean'],fresh_total=24,
          original_bank_clean=data['feedback_bank']['clean'],
          repair_bank_clean=[r['clean'] for r in repairs],
          known_regressions=sum(len(r['regressions']) for r in (*known,*repairs)),
          evidence=evidence))


def retain_archived_pass(out,qualified):
    """Restore the prior route on one archived regression and verify scope."""
    previous_gate = qualified/'paired_gate.json'
    protocol = read(previous_gate)
    for name,digest in protocol['inputs'].items():
        assert sha(name)==digest,name
    comparison = qualified/'comparison.json'
    data = read(comparison)
    archived = BASE/'untouched'
    prior_cases = data['feedback_bank']
    regressed, = prior_cases['regressions']
    assert prior_cases['failed']==[regressed]
    inputs = {str(p):sha(p) for p in (comparison,previous_gate,qualified/'ppo_gate.json',Path(__file__))}
    prior_results = archived/'results.json'
    prior_trace = archived/'trajectory.npz'
    prior_audit = archived/'independent_audit.json'
    audit = read(prior_audit)
    assert audit['passed'] and audit['files']['trajectory.npz']==sha(prior_trace)
    assert audit['files']['results.json']==sha(prior_results)
    index, = [i for i,r in enumerate(read(prior_results)['episodes']) if r['id']==regressed and r['clean']]
    with np.load(prior_trace,allow_pickle=False) as z:
        added = z['obs'][0,index,336:]
    with np.load(qualified/'protected.npz',allow_pickle=False) as z:
        original,scale = z['signature'],z['scale']
    inputs.update({str(p):sha(p) for p in (prior_results,prior_trace,prior_audit)})
    verified = {}
    for name,summary in data.items():
        folder = qualified/name
        trace,results,receipt = (folder/k for k in ('trajectory.npz','results.json','independent_audit.json'))
        audit = read(receipt)
        assert audit['passed'] and audit['clean_count']==summary['clean']
        assert audit['files']['trajectory.npz']==sha(trace) and audit['files']['results.json']==sha(results)
        rows = read(results)['episodes']
        with np.load(trace,allow_pickle=False) as z:
            signatures = z['obs'][0,:,336:]
        distances = np.square((signatures-added)/scale).mean(-1)
        touched = [r['id'] for r,distance in zip(rows,distances,strict=True) if distance<=5e-7]
        if name=='feedback_bank':
            assert touched==[regressed]
        else:
            assert not touched and summary['clean']==summary['total'] and not summary['regressions']
        verified[name] = dict(signatures=len(signatures),matches=touched,min_distance=float(distances.min()))
        inputs.update({str(p):sha(p) for p in (trace,results,receipt)})
    out.mkdir()
    guard = out/'protected.npz'
    np.savez_compressed(guard,signature=np.concatenate((original,added[None])),scale=scale)
    config = out/'feedback.json'
    config.write_bytes((qualified/'feedback.json').read_bytes())
    reservation = out/'reservation.json'
    reservation.write_bytes((qualified/'reservation.json').read_bytes())
    proof = out/'retention.json'
    write(proof,dict(restored=regressed,signatures=verified,archived_passing_result=str(prior_results),
                     unchanged_holdout=True,holdout_used_for_training=False))
    inputs.update({str(p):sha(p) for p in (guard,config,reservation,proof)})
    protocol.update(inputs={**protocol['inputs'],**inputs},protected_starts=len(original)+1)
    write(out/'paired_gate.json',protocol)
    write(out/'comparison.json',data)
    evaluate(out,'confirm')
    ppo_gate(out,read(out/'comparison.json'),qualified)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('screen','early-edge','retarget','balance','freeze','confirm','full','retain'))
    parser.add_argument('out',type=Path)
    parser.add_argument('source',type=Path,nargs='?')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.mode=='screen':
        screen(out)
    elif args.mode=='early-edge':
        screen(out,EARLY_EDGE_PROFILES)
    elif args.mode=='retarget':
        routes = edge_routes(out.with_name(out.name+'_routes'))
        profiles = [dict(PROFILES[1],preserve_shallow=True,route_set=i) for i in range(2)]
        screen(out,profiles,routes=routes)
    elif args.mode=='balance':
        routes = balanced_routes(out.with_name(out.name+'_routes'),args.source.resolve())
        profiles = [dict(PROFILES[1],preserve_shallow=True,route_set=i)
                    for i,_ in enumerate(read(routes.with_suffix('.json'))['checks'])]
        screen(out,profiles,routes=routes)
    elif args.mode=='freeze':
        freeze(out,args.source.resolve())
    elif args.mode=='retain':
        retain_archived_pass(out,args.source.resolve())
    else:
        evaluate(out,args.mode)
