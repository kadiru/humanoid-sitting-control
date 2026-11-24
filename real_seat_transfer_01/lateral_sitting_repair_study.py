"""Repair identified start families and require retention on both existing banks."""
import argparse
import csv
import os
from pathlib import Path
import shutil

import numpy as np

from fresh_sitting_skills_study import GPU, BANK_FILES, bank_digest, verified_inputs, stats
import planted_descent_study as scorer
from run import DISK_RESERVE, P, PREPARED, read, sha, stage, write

BEFORE = P / 'fresh_sitting_skills_24_01'
OLD = P / 'sitting_skills_01'
STUDY = P / 'lateral_sitting_repair_01'
WORKER = 'lateral_sitting_repair_worker.py'

WIRE = dict(recenter=True, hold_start=582, hold_load=.24, hold_gyro=.20,
            hold_tilt=.692, hold_samples=3, brake_start=568, brake_ticks=36, setback=-.003)
WHITECLIFF = dict(hold_start=580, hold_load=.29)
BASIC_ANGLE = dict(hold_start=568, hold_load=.27, hold_tilt=.692,
                   slow_start=520, slow_end=628, slow_fraction=.29)
DEFAULT = {
    0: dict(WIRE,lateral_m=.025,roll=-.041),
    1: dict(WIRE,standing=True,lateral_m=-.021,roll=.030),
    2: dict(WIRE,brake_start=562,brake_ticks=48,hold_start=577),
    4: dict(WIRE,standing=True),
    6: dict(WHITECLIFF,standing=True,lateral_m=-.023),
    8: dict(WHITECLIFF,standing=True,lateral_m=-.016),
    12: dict(lateral_m=.028,roll=-.022),
    16: dict(standing=True),
    18: dict(lateral_m=-.024,roll=.047),
    19: dict(lateral_m=.033,roll=-.047,slow_start=506,slow_end=624,slow_fraction=.28),
    20: dict(lateral_m=-.034,roll=.052),
    21: BASIC_ANGLE,
}


def passed(row):
    return row['supported'] and row['descent_feet_planted']


def audited(folder):
    ev = folder/'evaluation'
    a = read(ev/'independent_audit.json')
    assert a['passed'] and a['strict_predicates_exact'] and not a['physical_tolerances_changed']
    assert a['files']['trajectory.npz'] == sha(ev/'trajectory.npz')
    assert a['files']['results.json'] == sha(ev/'results.json')
    return read(folder/'measures.json')


def prepare():
    STUDY.mkdir()
    before = {m:audited(BEFORE/m) for m in ('staged','shared','retention')}
    protocol = read(BEFORE/'protocol.json')
    assert bank_digest(BEFORE/'staged'/'bank') == bank_digest(BEFORE/'shared'/'bank') == protocol['bank']
    assert all(passed(x) for x in before['retention']) and len(before['retention'])==12
    rows = []
    preserved = []
    for i,(a,b) in enumerate(zip(before['staged'],before['shared'],strict=True)):
        assert a['case']==b['case']==protocol['trials'][i]['case']
        if passed(a) or passed(b):
            preserved.append(dict(row=i, case=a['case'], source='staged' if passed(a) else 'shared'))
        else:
            rows.append(dict(row=i, case=a['case'], staged_failure=a['reason'], shared_failure=b['reason']))
    assert len(preserved)==len(rows)==12
    assert set(DEFAULT)=={r['row'] for r in rows}
    write(STUDY/'protection.json',dict(
        preserved=preserved, repairs=rows, original=[r['case'] for r in before['retention']],
        fresh_bank=protocol['bank'], known_bank=bank_digest(BEFORE/'retention'/'bank'),
        source_sha256={m:sha(BEFORE/m/'measures.json') for m in before}))


def previous_sources(frozen):
    # Completed experiments keep versions of evaluators that have since been
    # revised. Reuse their exact content and never relax the hashes.
    paths = []
    for path, digest in frozen['inputs'].items():
        p = Path(path)
        if sha(p)!=digest:
            if p.name in ('fresh_sitting_skills_study.py','fresh_sitting_skills_worker.py'):
                p = BEFORE/'initial_source'/p.name
            elif p.name in ('sitting_skills.py','sitting_skills_study.py'):
                p = OLD/'matched_raw_source'/p.name
            else:
                raise AssertionError(path)
            assert sha(p)==digest, p
        paths.append(p)
    return paths


def launch(name, original=False, overrides=None, retained_from=None, prior_fresh=None):
    assert os.environ['CUDA_VISIBLE_DEVICES']==GPU
    source = BEFORE/('retention' if original else 'staged')
    frozen = read(source/'evaluation.freeze.json')
    prior = previous_sources(frozen)
    plan = read(STUDY/'protection.json')
    for arm,digest in plan['source_sha256'].items():
        assert sha(BEFORE/arm/'measures.json')==digest
    folder = STUDY/name
    folder.mkdir()
    bank = folder/'bank'
    shutil.copytree(source/'bank',bank)
    assert bank_digest(bank)==plan['known_bank' if original else 'fresh_bank']
    shutil.copyfile(source/'landing_profile.json',folder/'landing_profile.json')
    config = read(source/'step_config.json')
    skills = read(source/'skills.json')
    extras = [Path(__file__).resolve(),STUDY/'protection.json']
    if original:
        worker = P/'sitting_skills_worker.py'
    else:
        worker = P/WORKER
        extras.extend(previous_sources(read((BEFORE/'shared'/'evaluation.freeze.json'))))
        repair = {str(k):v for k,v in DEFAULT.items()}
        if prior_fresh:
            previous = read(STUDY/prior_fresh/'repair.json')
            repair = previous['routes']
            extras.append(STUDY/prior_fresh/'repair.json')
        for k,v in (overrides or {}).items():
            assert k in repair
            repair[k] = {**repair[k], **v}
        shared = read(BEFORE/'shared'/'skills.json')
        implementation = OLD/'matched_raw_source'/'sitting_skills.py'
        shared['implementation'] = str(implementation)
        extras.extend([worker,implementation])
        if retained_from:
            checked = audited(STUDY/retained_from)
            assert bank_digest(STUDY/retained_from/'bank')==plan['fresh_bank']
            extras.extend([STUDY/retained_from/'measures.json', STUDY/retained_from/'evaluation'/'independent_audit.json'])
            sets = [r['row'] for r in plan['repairs'] if passed(checked[r['row']])]
            for row in sets:
                assert repair[str(row)]==previous['routes'][str(row)], row
        write(folder/'repair.json',dict(routes=repair,
                                        use_shared=[r['row'] for r in plan['preserved'] if r['source']=='shared'],
                                        shared=shared))
        extras.append(folder/'repair.json')
    write(folder/'skills.json',skills)
    write(folder/'step_config.json',config)
    write(folder/'reservation.json',dict(files={str(f):sha(f) for f in bank.iterdir()}))
    cmd = frozen['command']
    stage(folder,'evaluation',bank,PREPARED/'library.npz',worker=worker,
          extra=cmd[3:cmd.index('--mode')],
          inputs=[*prior,*extras,folder/'skills.json',folder/'step_config.json',folder/'landing_profile.json'],
          disk_reserve=DISK_RESERVE)
    scorer.STUDY = STUDY
    scorer.score(name)


def report(name):
    measure = audited(STUDY/name)
    known = audited(STUDY/'regression')
    plan = read(STUDY/'protection.json')
    assert len(measure)==24 and len(known)==12
    assert [x['case'] for x in known]==plan['original']
    assert all(measure[x['row']]['case']==x['case'] for x in plan['preserved']+plan['repairs'])
    old = read(BEFORE/'protocol.json')['trials']
    lost = [x['case'] for x in plan['preserved'] if not passed(measure[x['row']])]
    new = [x['case'] for x in plan['repairs'] if passed(measure[x['row']])]
    s = dict(fresh=stats(measure),original=stats(known),fresh_baseline_union=len(plan['preserved']),
             lost_fresh=lost, lost_original=[x['case'] for x in known if not passed(x)], repaired=new,
             lateral=stats([x for p,x in zip(old,measure,strict=True) if p['family']=='lateral']),
             angled=stats([x for p,x in zip(old,measure,strict=True) if p['family']=='angled']),
             remaining=[dict(case=x['case'],reason=x['reason'],hold_s=x['hold_s']) for x in measure if not passed(x)])
    write(STUDY/(name+'_summary.json'),s)
    origins = {x['case']:x['source'] for x in plan['preserved']}
    retained = [dict(x,previous='original',passed=passed(x)) for x in known]
    fresh = [dict(x,previous=origins.get(x['case'],'failed'),passed=passed(x),start_family=p['family'])
             for p,x in zip(old,measure,strict=True)]
    with (STUDY/(name+'_paired.csv')).open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in fresh+retained for k in row)))
        w.writeheader()
        w.writerows(fresh+retained)
    print(s,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','run','report'))
    parser.add_argument('--name',default='signed_stance')
    parser.add_argument('--overrides',type=Path)
    parser.add_argument('--retain')
    args=parser.parse_args()
    if args.operation=='prepare':
        prepare()
    elif args.operation=='run':
        launch(args.name,original=args.name=='regression',
               overrides=read(args.overrides) if args.overrides else None,
               retained_from=args.retain,prior_fresh=args.retain)
    else:
        report(args.name)
