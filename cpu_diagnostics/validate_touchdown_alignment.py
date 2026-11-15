"""Verify both aligned approaches, then score an untouched bank of 24 starts."""

import argparse
from pathlib import Path

import numpy as np

from align_sitting_touchdown import TARGETS
from repair_live_touchdown import (BANKS, HERE, LIBRARY, PRIOR, evaluate_routes, freeze_routes,
                                  read, sha, write)
from study_live_sitting import bank_from_geometry, geometry, stage


def reserve_unseen(out, installed):
    receipt = read(installed/'routes.json')
    banks = {key: Path(path) for key,path in receipt['banks'].items()}
    known_geometry = [g for bank in banks.values() for g in read(bank/'geometry.json')]
    rows = {g['label']:g for g in known_geometry}
    seen = {tuple(g['top_center']+[g['pitch_deg']]) for g in known_geometry}
    cases = []
    # Six small, predetermined perturbations of each repaired start, followed
    # by twelve broader points. All predictions are scored on a single pass.
    variations = ((.001, 0., .001, .25), (-.001, 0., -.001, -.25),
                  (0., .002, .001, -.25), (0., -.002, -.001, .25),
                  (.002, .002, 0., -.125), (-.002, -.002, 0., .125))
    for target in TARGETS:
        base = rows[target]
        for dx, dy, dz, dp in variations:
            center = base['top_center']
            cases.append(geometry(base, f'alignment_unseen/{len(cases)+1:02d}',
                                  center[2]+dz, base['pitch_deg']+dp,
                                  center[0]+.4+dx, center[1]+dy))
    for i,height in enumerate((.366,.381,.402,.427)):
        for j,pitch in enumerate((8.25,13.75,17.75)):
            dx,dy = ((-.017,.028),(.011,-.031),(.021,.016))[(i+j)%3]
            cases.append(geometry(cases[0], f'alignment_unseen/{len(cases)+1:02d}',
                                  height,pitch,dx,dy))
    assert len(cases)==24 and not seen.intersection(tuple(g['top_center']+[g['pitch_deg']]) for g in cases)
    out.mkdir()
    source = banks['candidate']
    bank_from_geometry(out/'unseen', source, cases)
    write(out/'reservation.json', dict(files={str(f):sha(f) for f in (out/'unseen').iterdir()}))


def validate(full, unseen, screens):
    freeze_routes(full, screens, nearby=TARGETS)
    protocol = full/'routes.json'
    routes = read(protocol)
    assert not routes['unresolved']
    reserve_unseen(unseen, full)
    evaluate_routes(full)
    comparison = read(full/'comparison.json')
    retained = all(comparison[k]['clean']==comparison[k]['total'] and
                   not comparison[k]['regressions'] and not comparison[k]['route_mismatches'] for k in BANKS)
    if not retained:
        write(unseen/'gate.json',dict(passed=False,reason='known cases did not all pass'))
        return
    payload = full/'touchdown.npz'
    assert sha(payload) == routes['payload_sha256']
    result = stage(unseen,'evaluation',unseen/'unseen',PRIOR,worker=HERE/'live_sitting_edge.py',
                   extra=['--adapt-library',str(LIBRARY),'--touchdown-library',str(payload)],
                   inputs=[payload, LIBRARY, Path(__file__), full/'comparison.json'],protocol=protocol)
    with np.load(unseen/'evaluation/touchdown_selection.npz',allow_pickle=False) as data:
        selected = [None if k<0 else str(data['ids'][k]) for k in data['selected']]
    exercised = sorted(set(TARGETS).intersection(selected))
    passed = result['clean']==24 and set(exercised)==set(TARGETS)
    write(unseen/'gate.json',dict(passed=passed,known_clean=188,unseen=result,
                                runtime_routes=selected,aligned_routes_exercised=exercised,
                                used_for_route_selection=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('full',type=Path)
    parser.add_argument('unseen',type=Path)
    parser.add_argument('screens',nargs='+',type=Path)
    args=parser.parse_args()
    validate(args.full.resolve(),args.unseen.resolve(),[p.resolve() for p in args.screens])
