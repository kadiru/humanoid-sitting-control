"""Evaluate the frozen grip controller on a new grid of 24 shifted seats."""

import argparse
import csv
from pathlib import Path

from compare_touchdown_ppo import BASELINE, verify_audit
from run_repaired_sitting_ppo import ROOT, read, sha, write
from study_live_sitting import PREPARED, bank_from_geometry, geometry, stage


REFERENCE = ROOT/'outputs/touchdown_grip_ppo_comparison_20261115_v1'
HEIGHTS_CM = (36.3, 37.6, 38.9, 40.4, 42.1, 43.6)
PITCH_DEG = (6.25, 10.25, 14.25, 18.75)
FORE_AFT_CM = (-3., -1.8, -.6, .6, 1.8, 3.)
LATERAL_CM = (-4.1, -2.7, -1.1, 1.5, 3.2, 5.2)


def reserve(out, previous_banks):
    source = Path(previous_banks['candidate'])
    template = read(source/'geometry.json')[0]
    catalogues = {Path(path)/'geometry.json' for path in previous_banks.values()}
    for pattern in ('*/geometry.json','*/bank/geometry.json','*/fresh_bank/geometry.json',
                    '*/reserved_bank/geometry.json','*/unseen/geometry.json'):
        catalogues.update((ROOT/'outputs').glob(pattern))
    historical = set()
    for path in sorted(catalogues):
        for seat in read(path):
            if isinstance(seat,dict) and 'pitch_deg' in seat and 'height' in seat:
                historical.add((round(float(seat['height'])*100,5),round(float(seat['pitch_deg']),5)))
    cases = []
    for row,height in enumerate(HEIGHTS_CM):
        for col,pitch in enumerate(PITCH_DEG):
            assert (height,pitch) not in historical,(height,pitch)
            dx = FORE_AFT_CM[(2*row+col)%len(FORE_AFT_CM)]
            dy = LATERAL_CM[(row+2*col+1)%len(LATERAL_CM)]
            seat = geometry(template,f'grip_fresh/{len(cases)+1:02d}',height/100,pitch,dx/100,dy/100)
            cases.append(dict(seat,split='fresh_grip_evaluation'))
    out.mkdir()
    bank = out/'bank'
    bank_from_geometry(bank,source,cases)
    write(out/'reservation.json',dict(files={str(p):sha(p) for p in bank.iterdir()},
          prior_catalogues=len(catalogues),historical_height_pitch_pairs=len(historical),
          heights_cm=HEIGHTS_CM,pitches_deg=PITCH_DEG,reference_is_reward_only=True))
    return bank,cases


def run(out):
    reference = REFERENCE/'untouched.freeze.json'
    frozen = read(reference)
    assert read(REFERENCE/'status.json')==dict(state='complete',ppo_clean=284,total=284)
    for name,digest in frozen['inputs'].items():
        assert sha(name)==digest,name
    command = frozen['command']
    worker = Path(command[2])
    options = command[3:command.index('--mode')]
    assert options[0]=='evaluate' and worker.name=='grip_hold_ppo.py'
    checkpoint = Path(options[1])
    prior_protocol = REFERENCE/'comparison_protocol.json'
    prior = read(prior_protocol)
    assert prior['guarded_handoff'] and prior['checkpoint_sha256']==sha(checkpoint)
    bank,cases = reserve(out,read(BASELINE/'paired_gate.json')['banks'])
    protocol = out/'protocol.json'
    write(protocol,dict(previous_experiment=str(prior_protocol),checkpoint_sha256=sha(checkpoint),
          frozen_controller=True,policy_tuning=False,design_frozen_before_rollouts=True))
    inputs = [reference,prior_protocol,Path(__file__),*(Path(p) for p in frozen['inputs'])]
    score = stage(out,'evaluation',bank,PREPARED/'library.npz',worker=worker,extra=options,
                  protocol=protocol,inputs=inputs)
    ran = read(out/'evaluation.freeze.json')
    expected = command.copy()
    expected[expected.index('--bank')+1] = str(bank)
    expected[expected.index('--output')+1] = str(out/'evaluation')
    assert ran['command']==expected and ran['gpu']==frozen['gpu']
    audit,episodes = verify_audit(out/'evaluation')
    handoff = read(out/'evaluation/grip_handoff.json')
    rows = []
    for index,(seat,result,power) in enumerate(zip(cases,episodes,audit['final_absolute_mechanical_power_W'],strict=True)):
        assert seat['label']==result['id']
        rows.append(dict(id=result['id'],height_cm=round(seat['height']*100,3),slope_deg=seat['pitch_deg'],
             fore_aft_offset_cm=round(seat['fore_aft_offset_cm'],3),lateral_offset_cm=round(seat['lateral_offset_cm'],3),
             passed=result['clean'],hold_s=round(result['final_hold_s'],2),power_w=power,
             slip_ticks=result['slip_ticks'],illegal_ticks=result['illegal_ticks'],
             physical_failures=[k for k,flag in zip(('fall','forbidden','torque','impact','joint_limit'),
                                                   result['physical_failures'],strict=True) if flag],
             ppo_active_ticks=handoff['active_ticks'][index]))
    assert len(rows)==24
    write(out/'comparison.json',dict(score=score,rows=rows,power_window_s=[15,20],
          power_measure='Mean sum of absolute joint mechanical power',action_limits_unchanged=True,
          physical_scoring_unchanged=True))
    with (out/'results.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    run(parser.parse_args().output.resolve())
