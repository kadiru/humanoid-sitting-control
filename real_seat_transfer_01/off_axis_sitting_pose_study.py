"""Paired physical comparison with the camera alignment measurements restored."""
import argparse
from collections import Counter
import csv
import os
from pathlib import Path
import shutil

import numpy as np

from off_axis_sitting_pose import STUDY
from off_axis_sitting_study import STUDY as BASELINE
from off_axis_sitting_study import GPU, audit_decisions, bank_digest, catalog, checkpoint
from run import DISK_RESERVE, P, PREPARED, read, sha, stage, write
from sitting_skills import ALIGN


WORKER = P/'off_axis_sitting_pose_worker.py'


def freeze():
    original = BASELINE/'protocol.json'
    protocol = read(original)
    assert all(sha(path)==digest for path,digest in protocol['inputs'].items())
    frozen = read(STUDY/'camera_inputs.json')
    assert all(sha(path)==digest for path,digest in frozen.items())
    sources = [Path(__file__), WORKER, P/'off_axis_sitting_pose.py', P/'off_axis_sitting_study.py',
               P/'off_axis_sitting_worker.py', P/'seat_alignment.py', P/'stance23_planner.py', original]
    protocol['inputs'].update(frozen|{str(path):sha(path) for path in sources})
    for scale in catalog.SIZES:
        audit = STUDY/f'camera_audit_{scale}.json'
        receipt = read(audit)
        assert len(receipt['cases'])==64
        assert all(sha(path)==digest for path,digest in receipt['source'].items())
        protocol['inputs'].update(receipt['source'])
        protocol['inputs'][str(audit)] = sha(audit)
    protocol['camera_pose_enabled'] = True
    protocol['camera_pose_only'] = True
    protocol['alignment_enabled'] = False
    file = STUDY/'protocol.json'
    assert not file.exists()
    write(file,protocol)


def rollout(method, scale):
    assert os.environ['CUDA_VISIBLE_DEVICES']==GPU
    config = STUDY/'protocol.json'
    protocol = read(config)
    assert all(sha(path)==digest for path,digest in protocol['inputs'].items())
    spec = protocol['banks'][str(scale)]
    old = BASELINE/f'{method}_{scale}'
    reference = old/'evaluation.freeze.json'
    frozen = read(reference)
    assert all(sha(path)==digest for path,digest in frozen['inputs'].items())
    assert bank_digest(old/'bank')==spec['hashes']
    target = STUDY/old.name
    target.mkdir()
    bank = target/'bank'
    shutil.copytree(old/'bank',bank)
    assert bank_digest(bank)==spec['hashes']
    shutil.copyfile(old/'landing_profile.json',target/'landing_profile.json')
    write(target/'approach.json',protocol['approach'])
    settings = read(old/'skills.json')
    assert settings['weights']==protocol['actor'][method]
    settings.update(pose_template=str(STUDY/f'seat_clouds_{scale}.npz'),
                    pose_assets=[case['asset'] for case in spec['cases']])
    write(target/'skills.json',settings)
    write(target/'reservation.json',dict(files={str(path):sha(path) for path in bank.iterdir()}))
    cmd = frozen['command']
    stage(target,'evaluation',bank,PREPARED/'library.npz',worker=WORKER,extra=cmd[3:cmd.index('--mode')],
          inputs=[*map(Path,frozen['inputs']),*map(Path,protocol['inputs']),config,reference,
                  target/'skills.json',target/'approach.json',target/'landing_profile.json'],
          disk_reserve=DISK_RESERVE)
    catalog.score(target)


def verify(folder, scale):
    ev = folder/'evaluation'
    observations = read(ev/'camera_pose_observation.json')
    witnesses = read(STUDY/f'camera_audit_{scale}.json')['cases']
    assert len(observations)==len(witnesses)==64
    assert [item['asset'] for item in observations]==[case['asset'] for case in witnesses]
    validity_changes = sum(item['valid']!=case['valid'] for item,case in zip(observations,witnesses))
    values = np.asarray([item['inputs'] for item in observations],dtype=np.float32)
    # The repeated start can have submillimetre depth differences on some meshes.
    prior = np.asarray([item['features'] for item in witnesses])
    max_change = np.abs(values-prior).max(axis=0)
    with np.load(ev/'skill_decisions.npz',allow_pickle=False) as data:
        aligned = data['phase']==ALIGN
        np.testing.assert_array_equal(np.sort(data['row'][aligned]),np.arange(64))
        np.testing.assert_array_equal(values[data['row']],data['features'][:,4:7])
        assert not data['enabled'][aligned].any()
    startup = read(ev/'startup.json')
    assert all(startup['sensor_valid']) and not any(startup['warm_failure'])
    return dict(valid=sum(item['valid'] for item in observations), validity_changes=validity_changes,
                max_abs_difference_from_saved_feature=max_change.tolist())


def report(scales):
    protocol = read(STUDY/'protocol.json')
    models = {name:checkpoint(path) for name,path in protocol['actor'].items()}
    summaries, receipts, rows = {}, {}, []
    for scale in scales:
        specs = protocol['banks'][str(scale)]['cases']
        scores, receipts[scale] = {}, {}
        for method in catalog.METHODS:
            folder = STUDY/f'{method}_{scale}'
            assert bank_digest(folder/'bank')==protocol['banks'][str(scale)]['hashes']
            receipt = audit_decisions(folder, method, protocol, models, True)
            receipts[scale][method] = dict(receipt, camera=verify(folder,scale))
            data = read(folder/'scores.json')
            prior = read(BASELINE/folder.name/'scores.json')
            assert [s['asset'] for s in data]==[c['asset'] for c in specs]
            assert [s['asset'] for s in prior]==[c['asset'] for c in specs]
            scores[method] = [dict(a,**case,method=method,prior_passed=b['passed'],
                                   prior_reason=b['reason'],prior_skill_actions=b['skill_actions'])
                              for case,a,b in zip(specs,data,prior,strict=True)]
            rows.extend(scores[method])
        both = list(zip(*(scores[m] for m in catalog.METHODS),strict=True))
        summaries[scale] = {}
        for family in ('lateral','angled'):
            pairs = [(a,b) for a,b in both if a['asset'] in protocol['grid'] and a['family']==family]
            record = {m:Counter(passed=0,prior_passed=0,repaired=0,regressed=0) for m in catalog.METHODS}
            for pair in pairs:
                for item in pair:
                    before, after = item['prior_passed'],item['passed']
                    record[item['method']].update(passed=int(after), prior_passed=int(before),
                                                  repaired=int(after and not before), regressed=int(before and not after))
            summaries[scale][family] = dict(starts=len(pairs), methods=record,
                outcomes_differ=sum(a['passed']!=b['passed'] for a,b in pairs),
                choices_differ=sum(a['skill_actions']!=b['skill_actions'] for a,b in pairs))
    assert len(rows)==len({(row['case'],row['method']) for row in rows})==128*len(scales)
    suffix = '' if tuple(scales)==tuple(catalog.SIZES) else '_partial'
    with (STUDY/f'comparison{suffix}.csv').open('w',newline='') as stream:
        csvout = csv.DictWriter(stream,fieldnames=list(rows[0]))
        csvout.writeheader()
        csvout.writerows(rows)
    write(STUDY/f'actor_audit{suffix}.json',receipts)
    write(STUDY/f'summary{suffix}.json',dict(gpu=GPU,rollouts=len(rows),main=summaries))
    print(summaries,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('freeze','run','report'))
    parser.add_argument('--method',choices=catalog.METHODS)
    parser.add_argument('--scale',type=int,choices=catalog.SIZES)
    args = parser.parse_args()
    if args.operation=='freeze':
        freeze()
    elif args.operation=='run':
        rollout(args.method,args.scale)
    else:
        report((args.scale,) if args.scale else catalog.SIZES)
