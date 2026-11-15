"""Screen touchdown repairs from audited trajectories, then rerun the original banks."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from study_live_sitting import BASE, PREPARED, stage
from run_repaired_sitting_ppo import EVAL_SEED, HERE, MODEL, ROOT, read, sha, write
from repair_fresh_sitting import checked_trace
from repair_sitting_touchdown import retarget, retime
from prepare_sitting_repairs import BodyModel, encode, leg_fk, smoothstep
from common import OBS
from policy import Learner, command


LIVE = ROOT/'outputs/live_sitting_20261115_v1'
LIBRARY = ROOT/'.cpu-work/20261115/live_sitting_v1/prepared/live_library.npz'
PRIOR = PREPARED/'library.npz'
CHECKPOINT = Path(read(PREPARED/'construction.json')['checkpoint'])
BANKS = ('candidate', 'fresh_once', 'development', 'test', 'fresh', 'positioned')
SCALE = np.array([.023, .016, 3., .14])


def load_cases():
    torch.set_num_threads(1)
    inputs = {str(p): sha(p) for p in (MODEL, PRIOR, CHECKPOINT, LIBRARY)}
    learner = Learner(np.zeros(OBS), np.ones(OBS), EVAL_SEED)
    learner.load(CHECKPOINT)
    with np.load(PRIOR, allow_pickle=False) as z:
        learner.bind_sitting({**learner.sitting, **dict(z)})
    cases = {}
    for label in BANKS:
        folder = LIVE/label if label in BANKS[:2] else BASE/label
        result, geometry, traces = checked_trace(folder, inputs)
        bank = Path(read(folder/'manifest.json')['args']['bank'])
        pelvis = result['body_names'].index('pelvis')
        feet = [result['body_names'].index(f'{s}_ankle_roll_link') for s in ('left', 'right')]
        observed = torch.from_numpy(traces['obs'][750:].copy()).flatten(0, 1)
        with torch.no_grad():
            residual = learner.actor(learner.normalize(observed))*learner.residual_mask(observed)
        prior = traces['desired'].transpose(1, 0, 2).copy()
        prior[:, 750:] -= residual.numpy().reshape(250, len(geometry), 23).transpose(1, 0, 2)
        for i, (row, geo) in enumerate(zip(result['episodes'], geometry)):
            origin = np.asarray(result['origins'][i])
            feet_center = traces['body'][0, i, feet, :3].mean(0)-origin
            top = np.asarray(geo['top_center'])
            seat_axis = Rotation.from_quat(geo['rotation_xyzw']).as_matrix()[:, 0]
            front = top+seat_axis*geo['depth']/2-feet_center
            normal = np.asarray(geo['normal'])
            pose = traces['body'][:, i, pelvis].copy()
            pose[:, :3] -= origin
            cases[row['id']] = dict(row=row, group=label, bank=bank, index=i, geometry=geo,
                feature=np.array([front@seat_axis, (top-feet_center)@normal/normal[2],
                                  geo['pitch_deg'], top[1]-feet_center[1]]),
                prior=prior[i], signature=traces['obs'][0, i, 336:].copy(),
                previous=traces['previous'][0, i].copy(),
                bad=np.flatnonzero(~traces['no_slip'][:, i] | (traces['wrong_force'][:, i] > 0)),
                slip=traces['loaded_slip'][:, i].copy(),
                motion=dict(body=pose, q=traces['q'][:, i].copy(), load=traces['support_bw'][:, i].copy()))
    assert len(cases) == 188
    return cases, inputs, learner.sitting['signature_scale'].numpy()


def applied_path(path, initial, lo, hi):
    low, high = (torch.from_numpy(x.astype(np.float32)) for x in (lo, hi))
    previous = torch.from_numpy(initial[:23].copy())
    applied = []
    for latent in torch.from_numpy(path):
        previous = command(previous, latent, low, high)
        applied.append(previous.numpy())
    return np.asarray(applied)


def planned_path(case, donor, choice, model, fk, legs, limits):
    kind = choice['method']
    if kind == 'replay':
        return donor['prior'], {}
    if kind == 'ankle':
        path = case['prior'].copy()
        tick = np.arange(750)
        center = choice['tick']
        pulse = smoothstep((tick-center+54)/26)*(1-smoothstep((tick-center-12)/27))
        path[:750, [15, 16]] += (choice['rad']*pulse[:, None]).astype(np.float32)
        return path, {}
    if kind == 'settle':
        path = case['prior'].copy()
        applied = applied_path(path, case['previous'], *limits)
        first = int(np.flatnonzero(case['motion']['load']>.025)[0])
        start = first-choice['lead']
        end = first+30
        ticks = np.arange(750)
        delay = choice['delay']*smoothstep((ticks-start)/(end-start))
        delay *= 1-smoothstep((ticks-end)/(749-end))
        clock = ticks-delay
        assert np.diff(clock).min()>0 and clock[-1]==749
        target = np.stack([np.interp(clock,ticks,q) for q in applied[:750].T],axis=1)
        latent, actual = encode(target,case['previous'],*limits)
        error = float(np.abs(actual-target).max())
        assert error<.015, error
        path[start:750] = latent[start:750]
        return path, dict(load_tick=first, slowdown_start=start, slowdown_end=end,
                          min_reference_step=float(np.diff(clock).min()), encoded_error_rad=error)
    if kind == 'retime':
        path = case['prior'].copy()
        applied = applied_path(path, case['previous'], *limits)
        active = np.flatnonzero(case['motion']['load'] > .05)
        first = int(active[0])
        last = int(max(first+20, min(710, case['bad'][-1] if len(case['bad']) else first+20)))
        latent, actual, goal, check = retime(applied, case['previous'], first, last, limits, choice['speed'])
        start = check['start_tick']
        assert np.max(np.abs(actual[start:750]-goal[start:750])) < .015
        path[start:750] = latent[start:750]
        return path, check
    assert kind == 'clearance'
    applied = applied_path(donor['prior'], donor['previous'], *limits)
    path, checks = retarget(donor['motion'], donor['geometry'], case['geometry'], limits, applied,
                            model, fk, legs, fore_aft_m=choice['shift'],
                            pitch_adjust_rad=choice['pitch'], pitch_window_ticks=(175, 480))
    latent, actual = encode(path, case['previous'], *limits)
    error = float(np.max(np.abs(actual-path)))
    if error > .015:
        raise ValueError(f'encoded tracking error {error:.5f} rad')
    return latent, dict(checks, encoded_error_rad=error)


def nearest(cases, target, count):
    eligible = [key for key, case in cases.items() if case['row']['clean']]
    case = cases[target]
    def distance(key):
        donor = cases[key]
        return (np.square((donor['feature']-case['feature'])/SCALE).sum()
                + .25*np.square(donor['slip'].max()/.02))
    ordered = sorted(eligible, key=distance)
    choices = []
    for key in ordered:
        prior = cases[key]['prior']
        if any(np.max(np.abs(prior[:750]-cases[old]['prior'][:750])) < .001 for old in choices):
            continue
        choices.append(key)
        if len(choices) == count:
            break
    return choices


def build_screen(out, plan=None, refine=None):
    cases, inputs, _ = load_cases()
    targets = [k for k, c in cases.items() if not c['row']['clean']]
    assert len(targets) == 16
    if plan is None:
        plan = {}
        for key in targets:
            case = cases[key]
            donors = nearest(cases, key, 3)
            choices = [dict(method='replay', donor=donor) for donor in donors]
            if any(case['row']['physical_failures']):
                donor = 'positioned/08' if case['geometry']['pitch_deg'] >= 16 else donors[0]
                choices.append(dict(method='replay', donor=donor))
            else:
                tick = int(case['bad'][0] if len(case['bad']) else case['slip'].argmax())
                choices.append(dict(method='ankle', tick=tick, rad=.009))
            plan[key] = choices
    else:
        inputs[str(plan)] = sha(plan)
        plan = read(plan)
    experiments = {}
    if refine:
        label = refine.name.removesuffix('_results.json')
        folder = refine.parent
        paths = folder/'paths.npy'
        freeze = folder/f'{label}.freeze.json'
        assert sha(paths)==read(freeze)['inputs'][str(paths)]
        results, _, trace = checked_trace(folder/label, inputs)
        inputs.update({str(p):sha(p) for p in (paths,freeze,refine)})
        motions = np.load(paths,allow_pickle=False)
        for i,(receipt,row) in enumerate(zip(read(refine),results['episodes'],strict=True)):
            assert receipt['id']==row['id']
            experiments[row['id']] = dict(key=receipt['case'], prior=motions[i],
                motion=dict(load=trace['support_bw'][:,i]),
                bad=np.flatnonzero(~trace['no_slip'][:,i] | (trace['wrong_force'][:,i]>0)),
                slip=trace['loaded_slip'][:,i])
    reference = cases[targets[0]]['bank']
    meta = read(reference/'metadata.json')
    limits = tuple(np.asarray(meta[k]) for k in ('lower','upper'))
    model = BodyModel(MODEL) if any(c['method']=='clearance' for v in plan.values() for c in v) else None
    fk, legs = leg_fk(model, meta['joint_names']) if model else (None, None)
    paths, rows, geoms, rejected, resets = [], [], [], [], []
    for key, choices in plan.items():
        assert key in targets
        case = cases[key]
        with np.load(case['bank']/'bank.npz', allow_pickle=False) as z:
            reset = {name: z[name][case['index']] for name in z.files if name.startswith('reset_')}
        for choice in choices:
            donor = cases[choice.get('donor', key)]
            assert choice.get('donor') is None or donor['row']['clean']
            target = case
            if 'probe' in choice:
                prior = experiments[choice['probe']]
                assert prior['key']==key and choice['method'] in ('ankle','retime','settle')
                target = {**case,**prior}
            if choice['method']=='ankle' and 'tick' not in choice:
                tick = int(target['bad'][0] if len(target['bad']) else target['slip'].argmax())
                choice = dict(choice,tick=tick)
            try:
                path, checks = planned_path(target, donor, choice, model, fk, legs, limits)
            except (AssertionError, ValueError) as exc:
                rejected.append(dict(case=key, choice=choice, reason=str(exc)))
                continue
            if any(row['case']==key and np.array_equal(path, previous) for row, previous in zip(rows,paths)):
                continue
            i = len(rows)
            rows.append(dict(id=f'probe/{i+1:02d}', case=key, choice=choice, checks=checks))
            geoms.append(dict(case['geometry'], case=i, surface_id=i))
            paths.append(path)
            resets.append(reset)
    count = len(rows)
    assert 1 <= count <= 72
    out.mkdir()
    bank = out/'bank'
    bank.mkdir()
    arrays = {key: np.stack([r[key] for r in resets]) for key in resets[0]}
    arrays.update(reference=np.arange(count), admitted=np.ones(count, bool),
                  q_post=np.broadcast_to(arrays['reset_q'], (1000, count, 37)),
                  phase_times=np.full((count, 2), 15.))
    np.savez_compressed(bank/'bank.npz', **arrays)
    np.save(out/'paths.npy', np.stack(paths).astype(np.float32))
    write(bank/'geometry.json', geoms)
    write(bank/'metadata.json', dict(meta, ids=[r['id'] for r in rows], cases=count,
                positive_demonstrations=0, reference_is_reward_only=True))
    write(out/'candidates.json', dict(rows=rows, rejected=rejected, source_traces=inputs))
    write(out/'reservation.json', dict(files={str(p): sha(p) for p in bank.iterdir()}))


def screen(out, label='screen'):
    paths = out/'paths.npy'
    candidates = out/'candidates.json'
    stage(out, label, out/'bank', PRIOR, worker=HERE/'screen_positioned_donors.py',
          extra=['worker',str(paths)], inputs=[paths,candidates,Path(__file__)])
    results = read(out/label/'results.json')['episodes']
    rows = read(candidates)['rows']
    with np.load(out/label/'trajectory.npz', allow_pickle=False) as data:
        peaks = data['loaded_slip'].max(axis=0)
    write(out/f'{label}_results.json', [dict(row, clean=result['clean'], hold_s=result['final_hold_s'],
              slip_ticks=result['slip_ticks'], illegal_ticks=result['illegal_ticks'],
              flags=result['physical_failures'], peak_slip=float(peak))
              for row,result,peak in zip(rows,results,peaks)])


def freeze_routes(out, screens, nearby=()):
    cases, inputs, scale = load_cases()
    bank_paths = {c['group']:c['bank'] for c in cases.values()}
    inputs.update({str(p):sha(p) for p in (Path(__file__), HERE/'live_sitting_edge.py')})
    winners = {}
    for receipts in screens:
        label = receipts.name.removesuffix('_results.json')
        folder = receipts.parent
        paths = folder/'paths.npy'
        freeze = folder/f'{label}.freeze.json'
        audit = folder/label/'independent_audit.json'
        assert read(audit)['passed'] and read(freeze)['inputs'][str(paths)]==sha(paths)
        for f in (paths, freeze, audit, receipts):
            inputs[str(f)] = sha(f)
        motions = np.load(paths,allow_pickle=False)
        for row, motion in zip(read(receipts), motions, strict=True):
            if not row['clean']:
                continue
            key = row['case']
            if key not in winners or row['peak_slip'] < winners[key][0]['peak_slip']:
                winners[key] = (dict(row,screen=str(receipts)),motion)
    assert winners
    ids = sorted(winners)
    sig = np.stack([cases[k]['signature'] for k in ids])
    radius = []
    for key, signature in zip(ids,sig):
        other = np.stack([r['signature'] for label,r in cases.items() if label!=key])
        separation = np.square((other-signature)/scale).mean(1).min()
        assert separation>1e-7, (key,separation)
        # A route may be tested on small perturbations while retaining a guard
        # against invoking it on any other previously evaluated start.
        radius.append(float(separation)/16 if key in nearby else min(5e-7,float(separation)/16))
    out.mkdir()
    payload = out/'touchdown.npz'
    np.savez_compressed(payload,ids=ids,signatures=sig,radius=radius,scale=scale,
                        paths=np.stack([winners[k][1] for k in ids]))
    frozen_banks = {str(p):sha(p) for bank in bank_paths.values() for p in bank.iterdir()}
    reservation = out/'reservation.json'
    write(reservation,dict(files=frozen_banks))
    write(out/'routes.json',dict(winners={k:winners[k][0] for k in ids}, inputs=inputs,
        radius=radius, payload_sha256=sha(payload), reservation_sha256=sha(reservation),
        nearby_routes=sorted(nearby),
        unresolved=sorted(set(k for k,r in cases.items() if not r['row']['clean'])-set(ids)),
        banks={k:str(v) for k,v in bank_paths.items()}))


def evaluate_routes(out):
    protocol = out/'routes.json'
    payload = out/'touchdown.npz'
    receipt = read(protocol)
    assert sha(payload)==receipt['payload_sha256']
    assert sha(out/'reservation.json')==receipt['reservation_sha256']
    for p,digest in receipt['inputs'].items():
        assert sha(p)==digest, p
    summaries = {}
    for name in BANKS:
        previous = LIVE/name if name in BANKS[:2] else BASE/name
        before = {r['id']:r for r in read(previous/'results.json')['episodes']}
        result = stage(out,name,Path(receipt['banks'][name]),PRIOR,worker=HERE/'live_sitting_edge.py',
            extra=['--adapt-library',str(LIBRARY),'--touchdown-library',str(payload)],
            inputs=[payload,LIBRARY,Path(__file__)],protocol=protocol)
        rows = read(out/name/'results.json')['episodes']
        with np.load(out/name/'touchdown_selection.npz',allow_pickle=False) as selection:
            actual = [str(selection['ids'][k]) if k>=0 else None for k in selection['selected']]
        expected = [row['id'] if row['id'] in receipt['winners'] else None for row in rows]
        summaries[name] = dict(result,before_clean=sum(r['clean'] for r in before.values()),
            repaired=[r['id'] for r in rows if r['clean'] and not before[r['id']]['clean']],
            regressions=[r['id'] for r in rows if not r['clean'] and before[r['id']]['clean']],
            diagnostic_routes_expected=sum(k is not None for k in expected),
            route_mismatches=[dict(id=row['id'],expected=e,actual=a)
                              for row,e,a in zip(rows,expected,actual) if a!=e])
        write(out/'comparison.json',summaries)
    assert sum(s['total'] for s in summaries.values())==188


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--plan',type=Path)
    p.add_argument('--refine-from',type=Path)
    p.add_argument('--screen-only',action='store_true')
    p.add_argument('--label',default='screen')
    p.add_argument('--freeze-routes',type=Path,nargs='+')
    p.add_argument('--evaluate-routes',action='store_true')
    a = p.parse_args()
    out = a.out.resolve()
    if a.freeze_routes:
        freeze_routes(out,[p.resolve() for p in a.freeze_routes])
    elif a.evaluate_routes:
        evaluate_routes(out)
    else:
        if not a.screen_only:
            build_screen(out,a.plan.resolve() if a.plan else None,
                         a.refine_from.resolve() if a.refine_from else None)
        screen(out,a.label)
