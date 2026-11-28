"""Freeze, run and score the 64 matched 88% furniture starts."""
import argparse
import csv
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from off_axis_sitting_pose import STUDY as PARENT
from off_axis_sitting_pose_study import verify
from off_axis_sitting_study import GPU, audit_decisions, bank_digest, catalog, checkpoint
from run import DISK_RESERVE, P, PREPARED, read, sha, stage, write
from sitting_clearance_descent import Corridor, FEET, START
from sitting_forward_balance_audit import MODEL, diagnose


STUDY = P/'sitting_clearance_descent_01'
BASELINE = PARENT/'piecewise_88'
WORKER = P/'sitting_clearance_worker.py'
TARGETS = ('Armchair','Crestwood_Chair','Hazelwood_Chair')


def prepare():
    protocol = read(PARENT/'protocol.json')
    assert all(sha(p)==d for p,d in protocol['inputs'].items())
    cases = protocol['banks']['88']['cases']
    baseline = read(BASELINE/'scores.json')
    assert len(cases)==len(baseline)==64 and sum(x['passed'] for x in baseline)==17
    scene = read(BASELINE/'bank'/'geometry.json')
    results = read(BASELINE/'evaluation'/'results.json')
    trace = BASELINE/'evaluation'/'trajectory.npz'
    audit = read(BASELINE/'evaluation'/'independent_audit.json')
    assert audit['passed'] and sha(trace)==audit['files']['trajectory.npz']
    with np.load(trace,allow_pickle=False) as data:
        q, body = data['q'][START-1],data['body'][START-1].astype(float)
    body[:,:,:3] -= np.asarray(results['origins'])[:,None]
    feet = [results['body_names'].index(n) for n in FEET]
    arms = read(P/'folded_arms.json')['arms']
    STUDY.mkdir()
    proposed, receipt = {}, []
    for case,geometry,old in zip(cases,scene,baseline,strict=True):
        row = case['row']
        assert case['asset']==geometry['name']==old['asset']
        if old['passed'] or case['asset'] not in TARGETS:
            continue
        assert sha(geometry['mesh_file'])==geometry['mesh_sha256']
        assert old['reason']=='Physical guard [0, 0, 0, 0, 1]'
        root = body[row,0]
        planner = Corridor(geometry,q[row].astype(float),root[:3],Rotation.from_quat(root[3:]).as_matrix(),body[row,feet],arms)
        endpoints, filters = planner.endpoints()
        result = planner.path(endpoints)
        proof = None if result is None else result.pop('proof')
        if result is not None:
            for key,value in result.items():
                proposed[f'{row}_{key}'] = value
        evidence = dict(case,endpoint_counts=filters,path=proof,mesh_sha256=geometry['mesh_sha256'])
        receipt.append(evidence)
        print(case['case'],filters,proof,flush=True)
    assert len(receipt)==6
    output=STUDY/'paths.npz'
    np.savez_compressed(output,**proposed)
    witnesses=STUDY/'geometry_witness.json'
    write(witnesses,receipt)
    frozen = [Path(__file__),WORKER,P/'sitting_clearance_descent.py',P/'stance23_planner.py',
              P/'rearstep_model.py',P/'stance23_mass_cal.npz',P/'rearstep_kin.py',
              P/'rearstep_cal.npz',P/'surface.py',P/'folded_arms.json',MODEL,
              PARENT/'protocol.json',BASELINE/'scores.json',trace,output,witnesses]
    protocol['planned_rows'] = [item['row'] for item in receipt if item['path'] is not None]
    protocol['protected_rows'] = [case['row'] for case,score in zip(cases,baseline,strict=True) if score['passed']]
    protocol['inputs'].update({str(path):sha(path) for path in frozen})
    write(STUDY/'protocol.json',protocol)


def run():
    assert os.environ['CUDA_VISIBLE_DEVICES']==GPU
    protocol = STUDY/'protocol.json'
    plan = read(protocol)
    assert plan['planned_rows'] and not set(plan['planned_rows'])&set(plan['protected_rows'])
    assert all(sha(p)==d for p,d in plan['inputs'].items())
    reference = BASELINE/'evaluation.freeze.json'
    prior = read(reference)
    assert all(sha(p)==d for p,d in prior['inputs'].items())
    bank_hash = plan['banks']['88']['hashes']
    assert bank_digest(BASELINE/'bank')==bank_hash
    target = STUDY/'piecewise_88'
    target.mkdir()
    bank = target/'bank'
    shutil.copytree(BASELINE/'bank',bank)
    assert bank_digest(bank)==bank_hash
    for name in ('approach.json','skills.json','landing_profile.json'):
        shutil.copyfile(BASELINE/name,target/name)
    write(target/'clearance_route.json',dict(paths=str(STUDY/'paths.npz'),rows=plan['planned_rows']))
    write(target/'reservation.json',dict(files={str(p):sha(p) for p in bank.iterdir()}))
    cmd = prior['command']
    stage(target,'evaluation',bank,PREPARED/'library.npz',worker=WORKER,extra=cmd[3:cmd.index('--mode')],
          inputs=[*map(Path,prior['inputs']),*map(Path,plan['inputs']),protocol,reference,
                  *(target/n for n in ('approach.json','skills.json','landing_profile.json','clearance_route.json'))],
          disk_reserve=DISK_RESERVE)
    catalog.score(target)


def report():
    plan = read(STUDY/'protocol.json')
    folder = STUDY/'piecewise_88'
    ev = folder/'evaluation'
    models = {name:checkpoint(path) for name,path in plan['actor'].items()}
    decisions = audit_decisions(folder,'piecewise',plan,models,True)
    decisions['camera'] = verify(folder,88)
    predicted = read(STUDY/'geometry_witness.json')
    offered = {x['row']:x['path'] for x in predicted}
    capture = read(ev/'clearance_capture.json')
    assert set(plan['planned_rows'])=={x['row'] for x in capture}
    cases = plan['banks']['88']['cases']
    labels = [case['case'] for case in cases]
    diagnosis = {row['asset']:row for row in diagnose(folder,labels,88,read(MODEL))}
    previous, current = read(BASELINE/'scores.json'),read(folder/'scores.json')
    rows = []
    for case,old,new in zip(cases,previous,current,strict=True):
        row = case['row']
        label = case['case']
        stop = diagnosis.get(label,{})
        rows.append(dict(case,protected=old['passed'],passes=new['passed'],old_reason=old['reason'],
                         new_reason=new['reason'],hold_s=new['terminal_hold_s'],contact=new['contact_detail'],
                         both_feet_planted=new['descent_feet_planted'],target=row in offered,
                         path_admitted=row in plan['planned_rows'],seat_weight_pct=new['terminal_seat_weight_pct'],
                         stop_s=stop.get('stop_s'),com_ahead_of_toe_cm=stop.get('com_ahead_of_loaded_toe_cm')))
    repairs = [item['case'] for item in rows if item['passes'] and not item['protected']]
    regressions = [item['case'] for item in rows if item['protected'] and not item['passes']]
    summary = dict(trials=64,baseline_passes=17,passes=sum(row['passes'] for row in rows),
                   repairs=repairs,regressions=regressions,geometry_paths=len(plan['planned_rows']),
                   runtime_admitted=sum(x['admitted'] for x in capture),
                   targets=[x for x in rows if x['target']],actor=decisions)
    with (STUDY/'comparison.csv').open('w',newline='') as stream:
        out=csv.DictWriter(stream,fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    write(STUDY/'summary.json',summary)
    print(summary,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','run','report'))
    args = parser.parse_args()
    {'prepare':prepare,'run':run,'report':report}[args.operation]()
