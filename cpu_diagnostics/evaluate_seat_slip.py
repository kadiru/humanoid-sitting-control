"""Paired retest of seat slip followed by conditional retention and a new chair bank."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from compare_touchdown_ppo import verify_audit
from evaluate_chair_shapes import CHAIRS, pieces
from evaluate_seat_plane_approach import compare, evaluate
from study_live_sitting import bank_from_geometry, read, sha, write


def reserve(out, previous):
    source = previous/'holdout_bank'
    base = read(source/'geometry.json')
    rng = np.random.default_rng(674031)
    rows = []
    for i, (family, depth, width, shape, back, arms, style) in enumerate(CHAIRS):
        for j, sign in enumerate((-1, 1)):
            height = rng.uniform(.361, .437)
            pitch = rng.uniform(13.6, 19.1)
            yaw = sign*rng.uniform(8, 16)
            dx, dy = rng.uniform([-.024, -.034], [.026, .039])
            rot = Rotation.from_euler('ZY', [yaw,pitch], degrees=True)
            front = np.array([-.4+dx,dy,height]) + .25*Rotation.from_euler('y',pitch,degrees=True).as_matrix()[:,0]
            top = front - .5*depth*rot.as_matrix()[:,0]
            rows.append(dict(base[2*i+j], label=f'shear_holdout/{family}/{j+1}',shape=shape,
                             x=top[0],y=top[1],height=top[2],top_center=top.tolist(),pitch_deg=pitch,
                             yaw_deg=yaw,yaw=np.deg2rad(yaw),rotation_xyzw=rot.as_quat().tolist(),
                             normal=rot.as_matrix()[:,2].tolist(),reference_height_cm=height*100,
                             fore_aft_offset_cm=100*dx,lateral_offset_cm=100*dy,
                             parts=pieces(depth,width,top[2],rot,back,arms,style)))
    bank = out/'holdout_bank'
    bank_from_geometry(bank, source, rows)
    write(out/'reservation.json', {str(p):sha(p) for p in bank.iterdir()})
    return bank


def paired(out, baseline, method, bank=None):
    evaluate(out, baseline, bank=bank, braking=method, inputs=[Path(__file__)])
    result = compare(baseline/'evaluation',out/'evaluation',out)
    with np.load(baseline/'evaluation/trajectory.npz') as a, np.load(out/'evaluation/trajectory.npz') as b:
        commands = (a['command'],b['command'])
        both_after_load = np.maximum.accumulate(
            np.linalg.norm(a['physical_force_w'][:,:,0],axis=-1) > .01*a['physical_bodyweight_N'],axis=0)
        protected = ~np.roll(both_after_load,1,axis=0)
        protected[0] = True
        assert np.array_equal(commands[0][protected],commands[1][protected])
    with np.load(baseline/'evaluation/loaded_clearance_hold.npz') as a, np.load(out/'evaluation/loaded_clearance_hold.npz') as b:
        assert np.array_equal(a['active'],b['active'])
        assert np.array_equal(a['held_leg_target'][a['active']],b['held_leg_target'][b['active']])
    rows = result.pop('cases')
    result['slip_repairs'] = [r['id'] for r in rows if r['old']['slip'] and not r['new']['slip']]
    result['new_slips'] = [r['id'] for r in rows if r['new']['slip'] and not r['old']['slip']]
    result['new_forbidden'] = [r['id'] for r in rows if r['new']['forbidden'] and not r['old']['forbidden']]
    result['lost_stable'] = [r['id'] for r in rows if r['old']['stable'] and not r['new']['stable']]
    return result


def run(out, previous, method):
    audit, rows = verify_audit(previous/'holdout/evaluation')
    assert len(rows)==24 and sum(bool(r['slip_ticks']) for r in rows)==19
    out.mkdir()
    fresh = reserve(out,previous)
    summary = dict(baseline=audit['files'])
    trial = paired(out/'matched', previous/'holdout',method)
    summary['matched'] = trial
    gate = bool(trial['repairs']) and not any(trial[k] for k in ('regressions','new_slips','new_forbidden','lost_stable'))
    summary['qualified'] = gate
    write(out/'summary.json',summary)
    print('matched',trial,flush=True)
    if not gate:
        return
    summary['retention'] = {}
    for group in read(previous/'summary.json')['retention']:
        name = 'retain_'+group
        result = paired(out/name,previous/name,method)
        summary['retention'][group] = result
        write(out/'summary.json',summary)
        print(name,result['counts'],flush=True)
    assert sum(g['total'] for g in summary['retention'].values())==356
    assert all(sha(Path(p))==digest for p,digest in read(out/'reservation.json').items())
    evaluate(out/'fresh_reference',previous/'holdout',bank=fresh,inputs=[out/'reservation.json'])
    result = paired(out/'fresh',out/'fresh_reference',method,fresh)
    assert result['total']==24
    summary['fresh'] = result
    write(out/'summary.json',summary)
    print('fresh',result,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('previous',type=Path)
    parser.add_argument('--method',choices=('shear','normal_transfer','follow_descent'),required=True)
    args = parser.parse_args()
    run(args.output,args.previous,args.method)
