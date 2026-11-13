"""Build a feet-planted clearance approach for the two 20-degree seats."""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from prepare_sitting_repairs import DATASET, DONOR, FEET, HOUSINGS, MODEL, leg_fk
from common import ROOT, read, sha, write
from humanoid_sitting.shared_frame import BodyModel, apply, transform


PRIOR = ROOT/'outputs/sitting_repairs_20261113_v3'
INERTIA = ROOT/'outputs/full_body_contact_imitation_v4/capture_known_old/static_inertial_model.json'
STATICS = ROOT/'experiments/continuous_clearance_20261108_v1/static_model.py'
sys.path.insert(0, str(STATICS.parent))
from static_model import FullRest


CASES = [6, 7]
BRANCH = 24
TICKS = (650, 709, 748)
COOKED = Path('/data/humanoid-sitting-perception/low_seat_contact_01/cooked_model_50/model.json')


def plan(out, cooked):
    meta, geometry = (read(DATASET/name) for name in ('metadata.json', 'geometry.json'))
    g = geometry[CASES[0]]
    for name in ('top_center', 'normal', 'rotation_xyzw', 'depth', 'width'):
        assert g[name] == geometry[CASES[1]][name]
    body = BodyModel(MODEL)
    collision = BodyModel(COOKED) if cooked else body
    fk, leg_ids = leg_fk(body, meta['joint_names'])
    inertial = read(INERTIA)
    center, normal = (np.asarray(g[k]) for k in ('top_center', 'normal'))
    seat_R = Rotation.from_quat(g['rotation_xyzw']).as_matrix()
    lo, hi = (np.asarray(meta[k], np.float32) for k in ('lower', 'upper'))
    old_library = PRIOR/'repair_library.npz'
    results = read(PRIOR/'full72/results.json')
    audit = read(PRIOR/'full72/independent_audit.json')
    trace = PRIOR/'full72/trajectory.npz'
    assert audit['passed'] and sha(trace) == audit['files']['trajectory.npz']
    assert [i for i,r in enumerate(results['episodes']) if not r['clean']] == CASES
    with np.load(trace, allow_pickle=False) as z:
        actual = z['body'][BRANCH-1,DONOR,0].astype(float)
        actual[:3] -= results['origins'][DONOR]
        q0 = z['q'][BRANCH-1,DONOR].astype(float)
        last = z['command'][BRANCH-1,DONOR].astype(float)
        standing = z['previous'][0,DONOR].copy()
        prefix = z['desired'][:BRANCH,DONOR].copy()
        signatures = z['obs'][0,:,336:].copy()
    root = transform(Rotation.from_quat(actual[3:]).as_matrix(), actual[:3])
    foot = root@fk(q0)
    foot_reference = foot.copy()
    foot_reference[:, :3, :3] = Rotation.from_euler('z', .30, degrees=True).as_matrix()
    r_end = Rotation.from_euler('xyz', [.7392,18.9978,2.5567] if cooked else [0,39.4,.30], degrees=True).as_matrix()
    p_end = np.array([-.12857658,.060,0.] if cooked else [-.103,actual[1],0.])
    verts = collision.vertices['pelvis']
    markers = verts@r_end.T + p_end
    marker_index = np.argmin((markers-center)@normal)
    p_end[2] = -((markers[marker_index]-center)@normal)/normal[2]
    contact = r_end@verts[marker_index]+p_end
    inset = np.array([g['depth']/2,g['width']/2])-abs(((contact-center)@seat_R)[:2])
    assert (inset > .01).all()
    hover = p_end+.013*normal
    shape = [{'body':n,'vertices':v} for n,v in body.vertices.items()]
    patch = np.array([center+seat_R@np.array([x*g['depth']/2,y*g['width']/2,0.]) for x in (-1,1) for y in (-1,1)])
    with np.load(DATASET/'bank.npz', allow_pickle=False) as bank:
        standing_q = bank['reset_q'][CASES[0]].astype(float)
    left_foot = fk(standing_q)[0,:3,:3]
    stat = FullRest(body, inertial, shape, standing_q, -left_foot[:,2], standing, normal, -normal@center,
                    patch, 'pelvis', verts[marker_index], contact, 'sitting')
    stat.feet = foot_reference
    stat.floor = 0.
    foot_y = []
    for name, pose in zip(FEET, foot_reference):
        vertices = apply(pose, body.vertices[name])
        foot_y.append(vertices[vertices[:,2] < vertices[:,2].min()+.004,1].mean())
    stat.foot_y = foot_y
    end_q = q0.copy()
    end_q[[5,6,9,10,17,18]] = [-.6,-.6,.12,-.12,.2,.2] if cooked else [-1.25,-1.25,.3,-.3,1.4,1.4]
    interp = Slerp([0,1], Rotation.from_matrix(np.stack((root[:3,:3],r_end))))
    guess = q0[leg_ids].copy()
    commands, records = [], []
    for tick in np.r_[np.linspace(BRANCH,TICKS[0],32).astype(int),
                       np.linspace(TICKS[0],TICKS[1],16).astype(int)[1:],
                       np.linspace(TICKS[1],TICKS[2],16).astype(int)[1:]]:
        approach = min(1., (tick-BRANCH)/(TICKS[0]-BRANCH))
        descent = np.clip((tick-TICKS[0])/(TICKS[1]-TICKS[0]),0,1)
        load = .16*np.clip((tick-TICKS[1])/(TICKS[2]-TICKS[1]),0,1)
        q = (1-approach)*q0+approach*end_q
        rotation = interp(approach).as_matrix()
        pos = (1-approach)*actual[:3]+approach*hover-descent*.013*normal
        anchor = foot.copy()
        for i in range(len(foot)):
            anchor[i,:3,:3] = Slerp([0,1], Rotation.from_matrix(
                np.stack((foot[i,:3,:3],foot_reference[i,:3,:3]))))(approach).as_matrix()
        wanted = np.linalg.inv(transform(rotation,pos)) @ anchor

        def residual(angles):
            q[leg_ids] = angles
            f = fk(q)
            return np.c_[(f[:,:3,3]-wanted[:,:3,3]), .2*Rotation.from_matrix(
                f[:,:3,:3] @ wanted[:,:3,:3].transpose(0,2,1)).as_rotvec()].ravel()

        fit = least_squares(residual, guess, bounds=(lo[leg_ids]+.03, hi[leg_ids]-.03), max_nfev=32,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if abs(fit.fun).max() >= 1e-6:
            raise ValueError(('feet', int(tick), float(abs(fit.fun).max())))
        guess = fit.x.copy()
        q[leg_ids] = guess
        x = np.r_[q[:23],pos,Rotation.from_matrix(stat.R0.T@rotation).as_rotvec(),0.]
        model = stat.state(x,load)
        nominal = q[:23]+model['tau']/stat.K
        if tick == BRANCH:
            bias = last[:23]-nominal
        cmd = nominal+bias*(1-approach)
        command_full = np.r_[cmd, standing[23:]]
        clearance = {}
        for name in ('pelvis',*HOUSINGS):
            v = apply(model['root']@model['fk'][name],collision.vertices[name])
            clearance[name] = float(((v-center)@normal).min())
        if min(clearance[n] for n in HOUSINGS) <= 0:
            raise ValueError(('housing',int(tick),clearance))
        commands.append(command_full)
        records.append(dict(tick=int(tick), load_bw=float(load), pose=q.tolist(), clearance_m=clearance,
                            foot_fit=float(abs(fit.fun).max()), com=model['com'].tolist(), cop=model['cop'].tolist()))
    commands = np.asarray(commands)
    path_ticks = np.array([r['tick'] for r in records])
    first = np.searchsorted(path_ticks,TICKS[0])+1
    arc = np.r_[0,np.cumsum(np.max(abs(np.diff(commands[:first,:23],axis=0)),axis=1))]
    needed = int(np.ceil(arc[-1]/.00299))
    arrival = BRANCH + max(TICKS[0]-BRANCH,needed)
    landing, settled = max(TICKS[1],arrival+50), TICKS[-1]
    if settled <= landing:
        raise ValueError(('approach cannot fit the original horizon', arrival, landing, arc[-1]))
    approach_clock = BRANCH+(arrival-BRANCH)*arc/arc[-1]
    landing_clock = np.interp(path_ticks[first:],[TICKS[0],TICKS[1],TICKS[2]],[arrival,landing,settled])
    schedule = np.r_[approach_clock,landing_clock]
    waypoint = np.stack([np.interp(np.arange(BRANCH+1,1001),schedule,commands[:,j]) for j in range(37)],axis=1)
    low, high = lo+np.float32(.03), hi-np.float32(.03)
    prior = standing[:23].copy()
    for row in prefix:
        prior += np.clip(np.float32(.1)*(np.clip(row,low,high)-prior),-.003,.003)
    np.testing.assert_array_equal(prior,last[:23].astype(np.float32))
    latent = [*prefix]
    max_error = 0.
    max_error_location = None
    for i,row in enumerate(waypoint[:,:23]):
        goal = np.clip(prior.astype(float)+(row-prior)/.1,low,high).astype(np.float32)
        prior += np.clip(np.float32(.1)*(goal-prior),-.003,.003)
        error = abs(row-prior)
        if error.max() > max_error:
            max_error = float(error.max())
            j = int(error.argmax())
            max_error_location = (i+BRANCH,meta['joint_names'][j],float(row[j]),float(prior[j]))
        latent.append(goal)
    if max_error > .002:
        raise ValueError(('rate',max_error,max_error_location))
    latent = np.array(latent,np.float32)
    assert latent.shape == (1000,23)
    out.mkdir(exist_ok=False)
    with np.load(old_library,allow_pickle=False) as z:
        existing = dict(z)
    # Reuse exactly the six verified motions. New routes use only the
    # latched camera/proprioception signature and the existing distance scale.
    ckpt=ROOT/'outputs/hold_effort_ppo_20261113_v1/training/final.pt'
    import torch
    scale=torch.load(ckpt,map_location='cpu',weights_only=False)['sitting']['signature_scale'].numpy()
    others=[i for i in range(len(signatures)) if i not in CASES]
    distance=np.square((signatures[:,None]-signatures[CASES][None])/scale).mean(-1)
    radius=distance[others].min(0)/4
    library = dict(repair_signature=np.concatenate((existing['repair_signature'],signatures[CASES])),
                   repair_radius=np.r_[existing['repair_radius'],radius],
                   repair_desired=np.concatenate((existing['repair_desired'],np.repeat(latent[None],2,axis=0))))
    error=np.square((signatures[:,None]-library['repair_signature'][None])/scale).mean(-1)
    route=error.argmin(1)
    matches=error[np.arange(len(error)),route] < library['repair_radius'][route]
    old_error=error[:,:len(existing['repair_radius'])]
    old_route=old_error.argmin(1)
    old_matches=old_error[np.arange(len(error)),old_route] < existing['repair_radius'][old_route]
    np.testing.assert_array_equal(matches[others],old_matches[others])
    np.testing.assert_array_equal(route[others][old_matches[others]],old_route[others][old_matches[others]])
    assert matches[CASES].all() and (route[CASES] >= len(existing['repair_radius'])).all()
    np.savez(out/'library.npz',**library)
    write(out/'construction.json',dict(cases=CASES, donor=DONOR, branch_control=BRANCH,
          phase_controls=[arrival,landing,settled], command_error_rad=max_error, planned_cap_inset_m=inset.tolist(), nodes=records,
          source_inputs={str(p):sha(p) for p in (trace,MODEL,INERTIA,STATICS,old_library,ckpt,Path(__file__),*([COOKED] if cooked else []))},
          library_sha256=sha(out/'library.npz')))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--cooked',action='store_true')
    args=parser.parse_args()
    plan(args.output,args.cooked)
