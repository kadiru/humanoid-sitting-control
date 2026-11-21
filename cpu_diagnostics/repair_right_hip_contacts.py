"""Repair chair contact using saved bilateral hip geometry and matched full standing starts."""
import argparse
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import sys

if len(sys.argv) > 2 and sys.argv[1] == '--probe-worker':
    launcher_dir = Path(sys.argv[2]).parent
    sys.path[1:1] = [str(launcher_dir), str(launcher_dir/'trainer')]

import numpy as np
from scipy.spatial.transform import Rotation

from armchair_approach import hull_points, register, seat_gap, transfer
from chair_contact_motion import select
from compare_touchdown_ppo import verify_audit
from evaluate_chair_contact_motion import CAMPAIGN, bank, signatures
from evaluate_rim_motion import actor_from, bank_subset, shape_features
from evaluate_scan_landing import load_bundle
from evaluate_seat_plane_approach import DISK_RESERVE
from evaluate_standing_rim import comparison
from hold_clearance import choose
from prepare_sitting_repairs import BodyModel, MODEL
from refine_chair_contact import collect
from study_live_sitting import PREPARED, read, sha, stage, write

BASE = Path('/dev/shm/armchair_approach_05/full')
KNOWN = Path('/dev/shm/hold_clearance_02')
HIP = 'right_hip_pitch_link'
CONTACTS = ('pelvis', 'left_hip_pitch_link', HIP)


def contact_report(run):
    folder = run/'evaluation'
    _, rows = verify_audit(folder)
    with np.load(folder/'contact_attribution.npz') as probes, np.load(folder/'trajectory.npz') as trace:
        forces = probes['chair_history_peak_N'][:, :, probes['probe_links'].tolist().index(HIP)]
        # Reconstruct the original forbidden-force magnitude from independent link probes.
        names = probes['body_names'].tolist()
        allowed = probes['filtered_body_names'].tolist()
        other = [j for j, name in enumerate(names) if name not in allowed]
        net = probes['net_history_peak_N'][:, :, other].max(axis=2)
        wrong = np.maximum(net, probes['wrong_pair_N'].max(axis=2))
        np.testing.assert_allclose(wrong, trace['wrong_force'], atol=.001, rtol=1e-5)
        assert np.abs(trace['command']-trace['previous']).max() <= .0030003
        return [dict(id=r['id'], clean=r['clean'], stable=r['stable'], hold_s=r['final_hold_s'],
                     slips=r['slip_ticks'], forbidden=bool(r['physical_failures'][1]),
                     hip_peak_N=float(forces[:,i].max()), hip_ticks=int((forces[:,i]>10).sum()),
                     flags=r['physical_failures'], illegal_ticks=r['illegal_ticks'])
                for i,r in enumerate(rows)]


def targets():
    return [i for i,r in enumerate(read(BASE/'per_case.json')) if r.get('link') == HIP]


def source_actors():
    args = read(BASE/'evaluation.freeze.json')['command']
    return actor_from(args[args.index('--checkpoint')+1]), actor_from(args[args.index('evaluate')+1])


def component_gap(vertices, geometry):
    rot = Rotation.from_quat(geometry['rotation_xyzw']).as_matrix()
    centered = (vertices-geometry['top_center']) @ rot + [0,0,.05]
    parts = [dict(name='seat',center=[0,0,0],size=[geometry['depth'],geometry['width'],.1],
                  xyzw=[0,0,0,1]), *geometry['parts']]
    gaps = {}
    for p in parts:
        delta = (centered-p['center']) @ Rotation.from_quat(p['xyzw']).as_matrix()
        outside = abs(delta)-np.asarray(p['size'])/2
        gaps[p['name']] = float((np.linalg.norm(np.maximum(outside,0),axis=-1) +
                                 np.minimum(outside.max(axis=-1),0)).min())
    return min(gaps, key=gaps.get), min(gaps.values())


def build(out):
    ids = targets()
    baseline = contact_report(BASE)
    assert len(ids) == 7 and sum(r['clean'] for r in baseline) == 9
    geo = read(bank(BASE)/'geometry.json')
    model = BodyModel(MODEL)
    res = read(BASE/'evaluation/results.json')
    names = res['body_names']
    origins = np.array(res['origins'])
    evidence = []
    with np.load(BASE/'evaluation/trajectory.npz') as z:
        bodies = z['body']
    for i in ids:
        record = read(BASE/'per_case.json')[i]
        t = record['tick']
        vertices = hull_points(model,bodies,i,[t],HIP,origins,names)[0]
        part,gap = component_gap(vertices,geo[i])
        evidence.append(dict(target=record['id'],tick=t,force_N=record['chair_force_N'],
                             nearest_component=part,vertex_gap_m=gap))
    print('First contact',evidence,flush=True)
    write(out/'events.json',evidence)

    sources = [BASE, *(KNOWN/'rollouts'/n for n in read(KNOWN/'summary.json')['known']),
               *sorted(CAMPAIGN.glob('batch_*'))]
    examples = []
    for s in dict.fromkeys(sources):
        _, rows = verify_audit(s/'evaluation')
        gs = read(bank(s)/'geometry.json')
        for j,r in enumerate(rows):
            if r['clean'] and r['final_hold_s'] >= 5 and 'mesh_file' not in gs[j]:
                examples.append((s,j,gs[j],r))
    jobs = defaultdict(list)
    for i in ids:
        goal = shape_features(geo[i])
        def error(example):
            delta = abs(shape_features(example[2])-goal)
            return float(delta @ [.95,.32,.11,.10])
        ordered = sorted(examples,key=error)
        near = ordered[:42] + [e for e in ordered if e[2].get('family') == geo[i]['family']][:12]
        seen = set()
        for s,j,g,r in near:
            if (s,j) in seen:
                continue
            seen.add((s,j))
            jobs[s].append((i,j,g,r,error((s,j,g,r))))
    ranked = defaultdict(list)
    for s, work in jobs.items():
        p = s/'evaluation'
        results = read(p/'results.json')
        with np.load(p/'trajectory.npz') as z, np.load(p/'edge_audit.npz') as edge:
            body, load = z['body'], z['support_bw']
            motor = z['command'][350:850:20,:,:23]
            patch = edge['pelvis_contact_world_local']
        for i,j,g,r,error in work:
            legal = np.flatnonzero((load[:,j]>.01) & np.isfinite(patch[:,j]).all(axis=1))
            if not len(legal):
                continue
            tick = int(legal[0])
            t = np.unique(np.r_[np.arange(max(200,tick-100),1000,10),tick,999])
            gap = [seat_gap(hull_points(model,body,j,t,b,results['origins'],results['body_names']),geo[i])
                   for b in CONTACTS]
            cap,left,right = gap
            close = cap < cap.min()+.02
            margin = np.minimum(left-cap,right-cap)[close].min()
            inset = geo[i]['depth']/2 - ((patch[legal[0],j]-geo[i]['top_center']) @
                                        Rotation.from_quat(geo[i]['rotation_xyzw']).as_matrix())[0]
            # Evaluate the actual donor poses on the target seat: signed hip separation,
            # pelvic reach and the measured front rim; use both hips throughout the landing.
            cost = (error + 2*max(0.,.005-margin)+1.5*max(0.,.015-inset)+abs(cap[t==tick].item()))
            ranked[i].append((cost,s,j,g,r,tick,patch[tick,j],float(margin),motor[:,j].copy()))
    records, paths = [], []
    inputs = {}
    actors = source_actors()
    @lru_cache(maxsize=2)
    def trials(s):
        return collect(s,*actors)
    for i in ids:
        distinct = []
        for example in sorted(ranked[i],key=lambda e:e[0]):
            if any(abs(example[-1]-d[-1]).max()<.018 for d in distinct):
                continue
            distinct.append(example)
            if len(distinct)==6:
                break
        assert distinct
        for rank,(cost,s,j,g,r,tick,pivot,margin,_) in enumerate(distinct):
            info = dict(target=i, donor=r['id'], source=str(s), source_row=j,
                        approach_cost=cost, predicted_relative_hip_margin_m=margin)
            inputs[str(s/'evaluation/independent_audit.json')] = sha(s/'evaluation/independent_audit.json')
            paths.append(trials(s)[j]['path'])
            records.append(dict(info,method='demonstrated'))
            if rank < 2:
                angles,shift = transfer(g,geo[i],pivot)
                witness = dict(id=r['id'],contact=tick,pivot=pivot.tolist(),rotation_vector=angles.tolist(),
                               shift=shift.tolist(),original_hold_s=r['final_hold_s'])
                fit = register(s,j,witness,model)
                if fit is not None:
                    paths.append(fit[0])
                    records.append(dict(info,method='registered',fit=fit[1]))
    save_screen(out,records,paths,inputs)


def save_screen(out,records,paths,inputs):
    write(out/'construction.json', dict(sources=inputs,probes=records))
    np.savez_compressed(out/'paths.npz',paths=np.stack(paths))
    bank_subset(out/'probe_bank', bank(BASE), [rec['target'] for rec in records])
    print('Prepared',len(records),'motion trials',flush=True)


def library_for(out,selected,screen):
    lib = load_bundle(BASE.parent/'candidate.npz')
    records = read(out/'construction.json')['probes']
    with np.load(out/'paths.npz') as z:
        motion = z['paths'][selected]
    offset = len(lib['paths'])
    slots = offset + np.arange(len(selected))
    lib['paths'] = np.concatenate((lib['paths'],motion))
    lib['ids'] = np.append(lib['ids'],[records[j]['method']+' '+records[j]['donor'] for j in selected])
    if screen:
        lib['diagnostic_slots'] = slots
    else:
        scans = np.array(signatures(BASE))
        chosen = [records[j]['target'] for j in selected]
        assert len(chosen)==len(set(chosen))
        lib['sites'] = np.vstack((lib['sites'],scans[chosen]))
        lib['site_paths'] = np.append(lib['site_paths'],slots)
        lib['site_radius'] = np.append(lib['site_radius'],np.full(len(selected),.5))
        stage_ids = [choose(f,select(f,lib),lib) for f in scans]
        with np.load(BASE/'evaluation/chair_contact_motion.npz') as z:
            expected = z['selected'].copy()
        expected[chosen] = slots
        np.testing.assert_array_equal(stage_ids,expected)
    path = out/('screen.npz' if screen else 'controller.npz')
    np.savez_compressed(path,**lib)
    return path


def combine_screens(out, screens):
    """Prefer demonstrated clearance and a repeatable hold when two batches tested the same path."""
    by_target = {i: defaultdict(list) for i in targets()}
    sources = {}
    for root in screens:
        run = root/'screen'
        scored = contact_report(run)
        assert read(run/'evaluation.freeze.json')['inputs'][str(root/'screen.npz')] == sha(root/'screen.npz')
        records = read(root/'construction.json')['probes']
        for p in (root/'paths.npz', root/'construction.json', run/'evaluation/independent_audit.json',
                  run/'evaluation/contact_attribution.npz'):
            sources[str(p)] = sha(p)
        with np.load(root/'paths.npz') as z:
            paths = z['paths']
        for j,(r,info,path) in enumerate(zip(scored,records,paths,strict=True)):
            by_target[info['target']][sha256(path.tobytes()).hexdigest()].append((r,info,root,j,path))

    chosen, motions = [], []
    for i,groups in by_target.items():
        eligible = [v for v in groups.values() if all(x[0]['clean'] and x[0]['hip_ticks']==0 for x in v)]
        if not eligible:
            continue
        def rank(v):
            return (-max(x[0]['hip_peak_N'] for x in v), len(v), min(x[0]['hold_s'] for x in v))
        best = max(eligible,key=rank)
        _,info,_,_,path = best[0]
        chosen.append(dict(info,tests=[dict(run=str(root),probe=j,score=r) for r,_,root,j,_ in best]))
        motions.append(path)
    assert chosen
    np.savez_compressed(out/'paths.npz',paths=np.stack(motions))
    write(out/'construction.json',dict(sources=sources,probes=chosen))
    write(out/'selection.json',chosen)


def evaluate(out,screen,combined=False):
    records = read(out/'construction.json')['probes']
    if screen or combined:
        picks = list(range(len(records)))
    else:
        options = contact_report(out/'screen')
        picks = []
        for i in targets():
            choices = [j for j,info in enumerate(records) if info['target']==i and options[j]['clean']
                       and options[j]['hip_ticks']==0]
            if choices:
                picks.append(max(choices,key=lambda j: options[j]['hold_s']))
        write(out/'selection.json',[dict(probe=j,**records[j],result=options[j]) for j in picks])
        assert picks
    path = library_for(out,picks,screen)
    f = read(BASE/'evaluation.freeze.json')
    sources = [Path(k) for k in f['inputs']]
    assert all(sha(p)==f['inputs'][str(p)] for p in sources)
    options = list(f['command'][3:f['command'].index('--mode')])
    options[options.index('--chair-contact-motion')+1] = str(path)
    geometry = out/'probe_bank' if screen else bank(BASE)
    runner = Path(f['command'][2])
    if screen:
        runner = Path(__file__)
        options[0] = '--probe-worker'
    study = out/('screen' if screen else 'matched')
    study.mkdir()
    write(study/'reservation.json',dict(files={str(p):sha(p) for p in geometry.iterdir()}))
    required = [Path(__file__),frozen_armchair(),path,out/'construction.json',out/'paths.npz',BASE/'evaluation.freeze.json']
    stage(study,'evaluation',geometry,PREPARED/'library.npz',worker=runner,extra=options,
          inputs=sources+required,disk_reserve=DISK_RESERVE)
    rows = contact_report(study)
    write(study/'per_case.json',rows)
    if screen:
        with np.load(study/'evaluation/chair_contact_motion.npz') as z,np.load(path) as lib:
            np.testing.assert_array_equal(z['selected'],lib['diagnostic_slots'])
        for i in targets():
            print(i,[(j,r['clean'],r['hold_s'],r['slips'],round(r['hip_peak_N'])) for j,r in enumerate(rows)
                     if records[j]['target']==i],flush=True)
    else:
        baseline = contact_report(BASE)
        score = comparison(study,BASE)
        protected = [(a['id'],b['clean']) for a,b in zip(baseline,rows,strict=True) if a['clean']]
        summary = dict(comparison=score,protected=protected,controller_sha256=sha(path),
                       targets=[dict(before=baseline[i],after=rows[i]) for i in targets()])
        write(out/'summary.json',summary)
        print(summary,flush=True)


def frozen_armchair():
    return Path(__file__).with_name('armchair_approach.py')


def probe_worker():
    import chair_contact_motion
    from evaluate_armchair_pair import trace_contacts
    # The diagnostic bank repeats a physical placement with different motions.
    # Only diagnostic workers accept a row-indexed path; the final test uses the original camera worker.
    indices = None
    def select_probe(observed,library):
        nonlocal indices
        if indices is None:
            indices = iter(library['diagnostic_slots'])
        return int(next(indices))
    chair_contact_motion.select = select_probe
    trace_contacts(Path(sys.argv[2]),sys.argv[3:])


if __name__=='__main__':
    if sys.argv[1]=='--probe-worker':
        probe_worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('out',type=Path)
        parser.add_argument('phase',choices=('build','screen','matched','combine'))
        parser.add_argument('--screens',nargs='+',type=Path)
        args = parser.parse_args()
        if args.phase=='build':
            build(args.out)
        elif args.phase=='combine':
            assert args.screens
            combine_screens(args.out,args.screens)
            evaluate(args.out,False,combined=True)
        else:
            evaluate(args.out,args.phase=='screen')
