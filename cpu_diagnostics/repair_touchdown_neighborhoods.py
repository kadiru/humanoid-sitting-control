"""Fit and validate seat-dependent touchdowns on archived neighborhoods."""

import argparse
from pathlib import Path

import numpy as np

from live_sitting_edge import (CAMERA_HANDOFF_CHECK_TICK, CAMERA_ROUTE_TICK,
                               camera_route_measurement, interpolate_touchdown)
from prepare_sitting_repairs import BodyModel, encode, leg_fk
from repair_camera_fallback import (KNOWN_GROUPS, TOUCHDOWN, ranked_donors, snapshot)
from repair_fresh_sitting import checked_trace
from repair_live_touchdown import (CHECKPOINT, HERE, LIBRARY, MODEL, PRIOR, ROOT, applied_path,
                                  load_cases, read, screen, sha, write)
from repair_sitting_touchdown import retarget
from study_live_sitting import bank_from_geometry, geometry, stage
from validate_camera_touchdown import routed_score


BASE = ROOT/'outputs/camera_fallback_20261115_full'
NEIGHBORHOODS = {
    'rebound': (1,2,3,4),
    'seat_edge': (13,14,15),
    'steep': (16,17,18),
    'high': (22,23,24),
}
FAILED = {2,3,14,16,24}
LABELS = tuple(f'fallback_fresh/{i:02d}' for i in sorted(FAILED))


def case_name(index):
    return f'fallback_fresh/{index:02d}'


def reference_cases(inputs):
    cases, previous_inputs, _ = load_cases()
    inputs.update(previous_inputs)
    for label in ('candidate','repair_cases','target_bank','untouched'):
        cases.update(snapshot(BASE/label,inputs))
    probe = ROOT/'outputs/camera_fallback_20261115_refine'
    candidates = read(probe/'screen_results.json')
    paths = probe/'paths.npy'
    assert sha(paths)==read(probe/'screen.freeze.json')['inputs'][str(paths)]
    inputs.update({str(p):sha(p) for p in (paths,probe/'screen_results.json',probe/'screen.freeze.json')})
    results, seats, trace = checked_trace(probe/'screen',inputs)
    pulses = np.load(paths,allow_pickle=False)
    for i,row in enumerate(candidates):
        if not row['clean']:
            continue
        assert results['episodes'][i]['clean']
        pose = trace['body'][:,i,results['body_names'].index('pelvis')].copy()
        pose[:,:3] -= np.asarray(results['origins'][i])
        # These candidates moved the original rebound case without slipping.
        cases[f"rebound/{row['id']}"] = dict(cases['camera_unseen/17'],prior=pulses[i],
                                             geometry=seats[i],previous=trace['previous'][0,i],
                    motion=dict(body=pose,q=trace['q'][:,i],load=trace['support_bw'][:,i]))
    return cases


def geometry_adjusted(donor, target, model, fk, legs, limits, inset, pitch):
    command = applied_path(donor['prior'],donor['previous'],*limits)
    # Track some of the change in edge position, keeping the target support
    # patch inset. retarget compensates the resulting normal-plane change
    # and holds the world poses of both feet at every path knot.
    dx = np.clip(.55*(target['feature'][0]-donor['feature'][0])+inset,-.018,.018)
    turn = np.clip(pitch+np.deg2rad(.25*(target['geometry']['pitch_deg']-
                                        donor['geometry']['pitch_deg'])),-.055,.085)
    waypoint, checks = retarget(donor['motion'],donor['geometry'],target['geometry'],
                        limits,command,model,fk,legs,fore_aft_m=float(dx),
                        pitch_adjust_rad=float(turn),pitch_window_ticks=(175,480))
    path, applied = encode(waypoint,target['previous'],*limits)
    error = np.abs(applied-waypoint)
    landing = int(round(checks['donor_landing_s']/.02))-1
    at_contact = float(error[max(0,landing-28):].max())
    if not (at_contact<=.004 and error.max()<=.055):
        raise ValueError(('command_tracking',float(error.max()),at_contact))
    return path,dict(checks,fore_aft_adjustment_m=float(dx),pitch_adjustment_rad=float(turn),
                     max_path_error_rad=float(error.max()),contact_and_hold_error_rad=at_contact)


def prepare(out, edge_refinement=False):
    inputs = {str(p):sha(p) for p in (Path(__file__),MODEL,CHECKPOINT,PRIOR,
                                HERE/'repair_sitting_touchdown.py',HERE/'repair_camera_fallback.py')}
    cases = reference_cases(inputs)
    assert {key for key,c in cases.items() if key.startswith('fallback_fresh/') and not c['row']['clean']}==set(LABELS)
    source = BASE/'fresh_bank'
    inputs.update({str(p):sha(p) for p in source.iterdir()})
    meta = read(source/'metadata.json')
    limits = tuple(np.asarray(meta[key]) for key in ('lower','upper'))
    model = BodyModel(MODEL)
    fk, legs = leg_fk(model,meta['joint_names'])
    profiles = {
        'rebound': [('rebound/probe/02',0.,0.),('rebound/probe/05',0.,0.),
                    ('rebound/probe/07',-.002,.012),('rebound/probe/09',-.002,0.)],
        'seat_edge': [('fallback_fresh/15',-.006,.020),('matched/01/02',-.008,.037),
                      ('fallback_fresh/13',-.007,.018),('fallback_fresh/15',-.011,.038)],
        'steep': [('fallback_fresh/17',0.,.01),('fallback_fresh/18',-.002,.025),
                  ('positioned/04',0.,.012),('alignment_unseen/18',-.002,.030)],
        'high': [('fallback_fresh/22',0.,0.),('fallback_fresh/23',0.,.012)],
    }
    high = cases[case_name(24)]
    profiles['high'].extend((name,0.,.008) for name,_ in ranked_donors(cases,high,2))
    groups = NEIGHBORHOODS
    if edge_refinement:
        groups = {'seat_edge':NEIGHBORHOODS['seat_edge']}
        profiles['seat_edge'] = [
            ('alignment_unseen/06',0.,0.),
            ('alignment_unseen/06',-.002,0.),
            ('alignment_unseen/06',-.004,.014),
            ('alignment_unseen/10',0.,0.),
            ('alignment_unseen/10',-.002,.012),
            ('matched/01/02',0.,0.),
            ('fallback_fresh/15',0.,0.),
        ]
    rows, paths, indices, rejected = [], [], [], []

    def append(key,profile,choice,path,checks=None):
        assert path.shape==(1000,23) and np.isfinite(path).all()
        rows.append(dict(id=f'probe/{len(rows)+1:02d}',case=key,profile=profile,
                         choice=choice,checks=checks or {}))
        paths.append(path)
        indices.append(cases[key]['index'])

    for group,indices_in_group in groups.items():
        for idx in indices_in_group:
            key = case_name(idx)
            target = cases[key]
            if idx in FAILED and not edge_refinement:
                append(key,f'{group}/original',dict(method='original',donor=key),target['prior'])
            for j,(name,inset,pitch) in enumerate(profiles[group]):
                suffix = f'edge_refinement_{j+1}' if edge_refinement else str(j+1)
                profile = f'{group}/{suffix}'
                donor = cases[name]
                assert donor['row']['clean']
                choice = dict(method='seat_geometry',donor=name,inset_m=inset,forward_pitch_rad=pitch)
                try:
                    path,checks = geometry_adjusted(donor,target,model,fk,legs,limits,inset,pitch)
                except (AssertionError,ValueError) as exc:
                    rejected.append(dict(case=key,profile=profile,choice=choice,reason=str(exc)))
                    continue
                append(key,profile,choice,path,checks)
            if idx in FAILED and not edge_refinement:
                # A replay is a control for the geometry adjustment. It also
                # tests whether the control change is needed at this start.
                for name,_,_ in profiles[group][:2]:
                    append(key,f'{group}/replay/{name}',dict(method='replay',donor=name),cases[name]['prior'])
    count = len(rows)
    assert 5<count<=72
    out.mkdir()
    bank = out/'bank'
    bank.mkdir()
    with np.load(source/'bank.npz',allow_pickle=False) as z:
        arrays = {k:z[k][indices] for k in z.files if k.startswith('reset_')}
    arrays.update(reference=np.arange(count),admitted=np.ones(count,bool),
                  q_post=np.broadcast_to(arrays['reset_q'],(1000,count,37)),
                  phase_times=np.full((count,2),15.))
    np.savez_compressed(bank/'bank.npz',**arrays)
    np.save(out/'paths.npy',np.stack(paths).astype(np.float32))
    geometries = read(source/'geometry.json')
    write(bank/'geometry.json',[dict(geometries[i],case=j,surface_id=j) for j,i in enumerate(indices)])
    write(bank/'metadata.json',dict(meta,cases=count,ids=[r['id'] for r in rows]))
    write(out/'candidates.json',dict(rows=rows,rejected=rejected,source_traces=inputs))
    write(out/'reservation.json',dict(files={str(f):sha(f) for f in bank.iterdir()}))


def probe(out, edge_refinement=False):
    prepare(out,edge_refinement)
    for file,digest in read(out/'candidates.json')['source_traces'].items():
        assert sha(file)==digest,file
    screen(out)


def fresh_cases(bank, banks):
    previous = [g for path in banks.values() for g in read(Path(path)/'geometry.json')]
    seats = {g['label']:g for g in previous}
    occupied = {tuple(g['top_center']+[g['pitch_deg']]) for g in previous}
    samples = []
    nearby = ((.0025,-.004,.0011,.16),(-.0025,.004,-.0011,-.16),(.0015,-.003,-.0007,.24))
    wider = ((.008,.014,.0022,.37),(-.008,-.014,-.0022,-.37),(.004,-.018,.0012,-.27))
    local = (case_name(2),case_name(14),case_name(16),case_name(24))
    other = ('camera_unseen/21','alignment_unseen/19','matched/03/05','fresh/24')
    for key in (*local,*other):
        seat = seats[key]
        x,y,z = seat['top_center']
        for dx,dy,dz,ds in (nearby if key in local else wider):
            label = f'neighborhood_fresh/{len(samples)+1:02d}'
            samples.append(geometry(seat,label,z+dz,seat['pitch_deg']+ds,x+.4+dx,y+dy))
    assert len(samples)==24 and not occupied.intersection(tuple(g['top_center']+[g['pitch_deg']]) for g in samples)
    bank_from_geometry(bank,Path(banks['candidate']),samples)


def camera_coordinates(folder, bank, inputs):
    source = folder/'edge_audit.npz'
    inputs[str(source)] = sha(source)
    with np.load(source,allow_pickle=False) as z:
        early = z['values'][z['ticks']<=CAMERA_ROUTE_TICK]
    names = read(bank/'metadata.json')['ids']
    coordinates = {}
    for i,key in enumerate(names):
        frames = early[:,i]
        usable = (frames[:,2]<=.012)&(frames[:,3]>=4)
        coordinates[key] = camera_route_measurement(frames[usable][:,[0,4,5]])
    return coordinates


def screen_rows(screens, coordinates, inputs):
    trials = []
    for folder in screens:
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
            row['screen'] = str(folder)
            if row['choice']['method']=='original':
                continue
            feature = coordinates[row['case']]
            if feature is not None:
                trials.append((row,path,handoff,feature))
    return trials


def supported_paths(trials, scales):
    groups = {}
    for row,*rest in trials:
        groups.setdefault(row['profile'],[]).append((row,*rest))
    preferred = {}
    for family, placements in NEIGHBORHOODS.items():
        expected = {case_name(i) for i in placements}
        complete = [(profile,entries) for profile,entries in groups.items()
                    if profile.startswith(f'{family}/') and {r[0]['case'] for r in entries}==expected
                    and all(r[0]['clean'] and r[0]['choice']['method']=='seat_geometry' for r in entries)]
        if complete:
            preferred[family] = min(complete,key=lambda entry:(
                max(r[0]['peak_slip'] for r in entry[1]),
                -min(r[0]['hold_s'] for r in entry[1])))[0]
    accepted = []
    for profile, entries in groups.items():
        family = profile.split('/',1)[0]
        if family in preferred and profile!=preferred[family]:
            continue
        positives = [t for t in entries if t[0]['clean']]
        negative = [t[3] for t in entries if not t[0]['clean']]
        breadth = len({t[0]['case'] for t in positives})
        for row,path,handoff,feature in positives:
            nearest_bad = min((float(np.square((feature-other)/scales).sum())
                               for other in negative),default=np.inf)
            radius = min(1.2,.36*nearest_bad)
            if breadth==1:
                # A single passing start supports a local repair only.
                radius = min(radius,.006)
            if radius>1e-8:
                accepted.append((row,path,handoff,feature,radius,breadth))
    return sorted(accepted,key=lambda t:(-t[-1],t[0]['peak_slip'],t[0]['profile']))


def freeze(out,screens):
    installed = BASE/'camera_touchdown.npz'
    old_routes = read(BASE/'routes.json')
    assert sha(installed)==old_routes['payload_sha256']
    banks = {**{k:v for k,v in old_routes['banks'].items() if k!='untouched'},
             'adjustment_bank':str(BASE/'fresh_bank')}
    inputs = {str(p):sha(p) for p in (Path(__file__),HERE/'live_sitting_edge.py',CHECKPOINT,
                                      LIBRARY,PRIOR,TOUCHDOWN,installed)}
    protected = []
    folders = [*(BASE/k for k in (*KNOWN_GROUPS,'target_bank')),BASE/'untouched']
    for folder in folders:
        audit = read(folder/'independent_audit.json')
        evidence = folder/'trajectory.npz'
        result = folder/'results.json'
        assert audit['passed'] and audit['files']['trajectory.npz']==sha(evidence)
        assert audit['files']['results.json']==sha(result)
        inputs.update({str(p):sha(p) for p in (evidence,result,folder/'independent_audit.json')})
        chosen = [i for i,r in enumerate(read(result)['episodes']) if r['clean']]
        with np.load(evidence,allow_pickle=False) as z:
            protected.extend(z['obs'][0,chosen,336:])
    assert len(protected)==255
    coordinates = camera_coordinates(BASE/'untouched',BASE/'fresh_bank',inputs)
    scales = np.array([.012,.006,.9])
    accepted = supported_paths(screen_rows(screens,coordinates,inputs),scales)
    assert accepted
    with np.load(installed,allow_pickle=False) as z:
        camera = dict(z)
    camera.update(
        neighborhood_ids=np.asarray([f"{r['profile']} at {r['case']}" for r,*_ in accepted]),
        neighborhood_groups=np.asarray([r['profile'] for r,*_ in accepted]),
        neighborhood_features=np.stack([f for _,_,_,f,_,_ in accepted]),
        neighborhood_handoff=np.stack([q for _,_,q,_,_,_ in accepted]),
        neighborhood_paths=np.stack([p for _,p,_,_,_,_ in accepted]),
        neighborhood_radius=np.asarray([radius for _,_,_,_,radius,_ in accepted]),
        neighborhood_scales=scales,neighborhood_protected=np.stack(protected))
    predictions = {}
    for label in LABELS:
        assert coordinates[label] is not None,label
        prediction = interpolate_touchdown(coordinates[label],camera)
        predictions[label] = None if prediction is None else str(camera['neighborhood_ids'][prediction[0]])
    out.mkdir()
    case_bank = out/'fresh_bank'
    fresh_cases(case_bank,banks)
    banks['untouched'] = str(case_bank)
    payload = out/'camera_touchdown.npz'
    np.savez_compressed(payload,**camera)
    write(out/'reservation.json',dict(files={str(f):sha(f) for p in banks.values() for f in Path(p).iterdir()}))
    write(out/'routes.json',dict(banks=banks,payload_sha256=sha(payload),inputs=inputs,
          protected_starts=len(protected),archived_camera_decisions=predictions,
          positive_witnesses=[dict(row,squared_radius=radius,passing_profile_placements=count)
                              for row,_,_,_,radius,count in accepted],
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
    groups = ('adjustment_bank',) if mode=='repair' else (*KNOWN_GROUPS,'target_bank','untouched')
    for label in groups:
        results = stage(out,label,Path(design['banks'][label]),PRIOR,worker=HERE/'live_sitting_edge.py',
                        extra=arguments,inputs=[LIBRARY,TOUCHDOWN,camera,Path(__file__)],protocol=protocol)
        baseline = []
        if label!='untouched':
            old = BASE/('untouched' if label=='adjustment_bank' else label)
            baseline = [r['id'] for r in read(old/'results.json')['episodes'] if r['clean']]
        summaries[label] = routed_score(out/label,results,baseline)
        write(comparison,summaries)
    if mode=='full':
        known = [summaries[k] for k in KNOWN_GROUPS]
        fresh = summaries['untouched']
        assert sum(r['total'] for r in known)==212 and fresh['total']==24
        write(out/'ppo_gate.json',dict(passed=fresh['clean']==24,
              known_clean=sum(r['clean'] for r in known),fresh_clean=fresh['clean'],
              fresh_total=24,known_regressions=sum(len(r['regressions']) for r in known)))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('probe','edge','freeze','repair','full'))
    parser.add_argument('out',type=Path)
    parser.add_argument('screens',type=Path,nargs='*')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.mode in ('probe','edge'):
        probe(out,edge_refinement=args.mode=='edge')
    elif args.mode=='freeze':
        freeze(out,[p.resolve() for p in args.screens])
    else:
        evaluate(out,args.mode)
