"""Run paired sitting skill rollouts on the archived four-scale furniture catalog."""
import argparse
import csv
import os
from pathlib import Path
import shutil

import numpy as np

from fresh_sitting_skills_study import GPU, bank_digest
from furniture_scale_suite import SIZES, SUITE
from run import DISK_RESERVE, P, PREPARED, read, sha, stage, write
from scale_approach_study import bank_at
from sitting_skills import DESCEND, TRANSFER, REST


STUDY = P/'furniture_skill_comparison_01'
APPROACH = P/'scale_approach_ppo_01'
SKILLS = P/'sitting_skills_01'
REFERENCE = APPROACH/'test_100_0'
WORKER = P/'furniture_skill_comparison_worker.py'
METHODS = ('piecewise', 'unified')
CHECKPOINTS = dict(zip(METHODS, ('staged', 'shared'), strict=True))


def prepare():
    source = SUITE/'cases.json'
    geometry = read(source)
    roster = [c['asset'] for c in geometry if c['scale_pct'] == 100]
    assert len(roster) == len(set(roster)) == 32
    copies = {scale:[c for c in geometry if c['scale_pct'] == scale] for scale in SIZES}
    assert all([c['asset'] for c in group] == roster for group in copies.values())
    controls = read(APPROACH/'final.json')
    assert sha(controls['checkpoint']) == controls['sha256']
    assert sha(controls['library']) == controls['library_sha256']
    weights = {method:SKILLS/(name+'.pt') for method,name in CHECKPOINTS.items()}
    for method, path in weights.items():
        fit = read(SKILLS/(CHECKPOINTS[method]+'_fit.json'))
        assert sha(path) == fit['checkpoint_sha256'] and fit['phases'] == [0,1,2,3]
    STUDY.mkdir()
    base = STUDY/'banks'
    base.mkdir()
    banks = {}
    placement_sources = {f:sha(f) for f in {source, *(Path(c['bank'])/f for c in geometry
                         for f in ('geometry.json','metadata.json','bank.npz'))}}
    for scale, group in copies.items():
        bank = base/str(scale)
        for c in group:
            assert sha(c['mesh']) == c['mesh_sha256']
        bank_at(bank, group)
        banks[str(scale)] = bank_digest(bank)
    inputs = [Path(__file__), WORKER, P/'sitting_skills.py', P/'scale_approach.py',
              P/'rescan_sitting.py', P/'pelvis_marker_worker.py', APPROACH/'final.json',
              *(SKILLS/(n+'_fit.json') for n in CHECKPOINTS.values()), *weights.values(),
              Path(controls['checkpoint']), Path(controls['library'])]
    protocol = dict(roster=roster, grid=roster[:30], supplement=roster[30:], scales=SIZES,
                    methods=METHODS, gpu=GPU, cases=copies, banks=banks,
                    inputs={str(p.resolve()):sha(p) for p in inputs},
                    placement_sources={str(p):h for p,h in placement_sources.items()},
                    approach=dict(mode='test',checkpoint=controls['checkpoint'],library=controls['library']),
                    actor={method:str(path) for method,path in weights.items()},
                    phases=[DESCEND,TRANSFER,REST], standing_aligned=True)
    write(STUDY/'protocol.json', protocol)
    print(len(roster), 'objects at', len(SIZES), 'scales, two methods,', len(geometry)*2, 'rollouts',flush=True)


def score(folder):
    from publish_confirmed import assess

    rows, witness = assess(folder)
    ev = folder/'evaluation'
    stamp = read(ev/'independent_audit.json')
    assert stamp['passed'] and stamp['strict_predicates_exact'] and not stamp['physical_tolerances_changed']
    assert read(ev/'manifest.json')['action_limits_unchanged']
    assert read(ev/'results.json')['success_criteria_unchanged']
    with np.load(ev/'trajectory.npz') as z, np.load(ev/'skill_decisions.npz') as actions:
        ground = z['physical_force_w'][:,:,5:]
        bw = z['physical_bodyweight_N'][:,:,None]
        slippage = (np.linalg.norm(ground,axis=-1)>.01*bw) & (z['physical_slip_m_s'][:,:,5:]>.02)
        planted = (ground[...,2]>.05*bw).all(axis=(0,2)) & ~slippage.any(axis=(0,2))
        assert np.abs(z['command'][:,:,:23]-z['previous'][:,:,:23]).max() < .0030003
        forbidden = z['flags'][:,:,1].any(axis=0)
        skill = [(int(i),int(p),int(a),bool(e)) for i,p,a,e in zip(
            *(actions[k] for k in ('row','phase','action','enabled')), strict=True)]
        stored = []
        for i,r in enumerate(rows):
            accepted = bool(r['supported_sit'] and planted[i])
            reason = r['reason']
            if r['supported_sit'] and not planted[i]:
                reason = 'Loaded foot slipped' if slippage[:,i].any() else 'Foot lost floor load'
            choices = {str(p):a for k,p,a,e in skill if k==i and e}
            fields = {key:(value.item() if isinstance(value,np.generic) else value) for key,value in r.items()}
            stored.append(dict(fields, passed=accepted, descent_feet_planted=bool(planted[i]),
                               forbidden_contact=bool(forbidden[i]), reason=reason, skill_actions=choices))
    write(folder/'witness_counts.json', witness)
    write(folder/'scores.json', stored)
    print(folder.name, 'passed', sum(r['passed'] for r in stored), '/', len(stored),flush=True)
    return stored


def rollout(method, scale):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == GPU
    protocol = read(STUDY/'protocol.json')
    bank_id = str(scale)
    source = STUDY/'banks'/bank_id
    assert bank_digest(source) == protocol['banks'][bank_id]
    base_inputs = []
    for path,digest in protocol['inputs'].items():
        p = Path(path)
        if p == Path(__file__).resolve() and sha(p) != digest:
            p = STUDY/'initial_source'/p.name
        assert sha(p) == digest, p
        base_inputs.append(p)
    frozen = read(REFERENCE/'evaluation.freeze.json')
    assert all(sha(p)==h for p,h in frozen['inputs'].items())
    target = STUDY/f'{method}_{scale}'
    target.mkdir()
    bank = target/'bank'
    shutil.copytree(source, bank)
    assert bank_digest(bank) == protocol['banks'][bank_id]
    shutil.copyfile(REFERENCE/'landing_profile.json', target/'landing_profile.json')
    write(target/'approach.json', protocol['approach'])
    n = len(protocol['roster'])
    write(target/'skills.json', dict(mode='evaluate', phases=protocol['phases'],choices=[[0]*4 for _ in range(n)],
                                    hold_rows=list(range(n)), seed=73281, weights=protocol['actor'][method]))
    write(target/'reservation.json', dict(files={str(f):sha(f) for f in bank.iterdir()}))
    meshes = [Path(c['mesh']) for c in protocol['cases'][bank_id]]
    assert all(sha(m)==c['mesh_sha256'] for m,c in zip(meshes,protocol['cases'][bank_id],strict=True))
    cmd = frozen['command']
    inputs = [*map(Path,frozen['inputs']), *base_inputs, Path(__file__), STUDY/'protocol.json',*meshes,
              target/'skills.json',target/'approach.json',target/'landing_profile.json']
    stage(target,'evaluation',bank,PREPARED/'library.npz',worker=WORKER,extra=cmd[3:cmd.index('--mode')],
          inputs=inputs,disk_reserve=DISK_RESERVE)
    score(target)


def report():
    protocol = read(STUDY/'protocol.json')
    rows, results = [], {}
    for method in METHODS:
        results[method] = {}
        for scale in SIZES:
            dest = STUDY/f'{method}_{scale}'
            assert bank_digest(dest/'bank') == protocol['banks'][str(scale)]
            measures = read(dest/'scores.json')
            assert [r['asset'] for r in measures] == protocol['roster']
            assert read(dest/'evaluation.freeze.json')['gpu'] == GPU
            with np.load(dest/'evaluation'/'trajectory.npz') as trace:
                f = trace['physical_force_w'][-250:,:,:5]
                normal = trace['physical_normal_w'][-250:,:,:5]
                load = (f*normal).sum(-1).clip(0).sum(-1)/trace['physical_bodyweight_N'][-250:]
                watts = np.abs(trace['physical_torque'][-250:,:,:23]*trace['physical_joint_velocity'][-250:,:,:23]).sum(-1)
            for r,seat,watts in zip(measures,load.mean(0),watts.mean(0),strict=True):
                r.update(method=method,scale_pct=scale,
                         hold_seat_normal_pct=round(float(seat)*100,2) if r['passed'] else None,
                         hold_body_mechanical_power_W=round(float(watts),3) if r['passed'] else None)
                rows.append(r)
            main = [r for r in measures if r['asset'] in protocol['grid']]
            results[method][scale] = dict(checked=len(main), passed=sum(r['passed'] for r in main),
                                         full_catalog_passed=sum(r['passed'] for r in measures))
    assert len(rows) == len({(r['method'],r['scale_pct'],r['asset']) for r in rows}) == 256
    with (STUDY/'comparison.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(STUDY/'summary.json',dict(gpu=GPU,rollouts=len(rows),methods=results))
    print(results,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','run','score','report'))
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--scale',type=int,choices=SIZES)
    args = parser.parse_args()
    if args.operation == 'prepare':
        prepare()
    elif args.operation == 'run':
        rollout(args.method,args.scale)
    elif args.operation == 'score':
        score(STUDY/f'{args.method}_{args.scale}')
    else:
        report()
