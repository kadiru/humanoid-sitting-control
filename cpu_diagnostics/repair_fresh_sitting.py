"""Retarget clean approaches to the six failed fresh seat conditions."""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from evaluate_fresh_sitting import CHECKPOINT_SHA, REPAIR_SHA
from repair_sitting_touchdown import HOLD_TICK, STUDY, retarget
from prepare_sitting_repairs import BodyModel, HOUSINGS, MODEL, encode, leg_fk
from run_repaired_sitting_ppo import DATA, EVAL_SEED, ROOT, read, sha, write
from common import MARGIN, OBS, STEP
from policy import Learner, command


FRESH = ROOT/'outputs/fresh_sitting_20261113_v1'
REPAIRED = ROOT/'outputs/sitting_touchdown_20261113_v2_full'
COOKED = DATA/'cooked_model_50/model.json'
SPLITS = dict(development=REPAIRED/'development', test=REPAIRED/'test', fresh=FRESH/'evaluation')
DONORS = {
    'fresh/03': 'fresh/04',
    'fresh/08': 'fresh/12',
    'fresh/11': 'fresh/07',
    'fresh/14': 'known_transfer/8',
    'fresh/15': 'known_transfer/10',
    'fresh/21': 'known_transfer/18',
}


def checked_trace(folder, inputs):
    audit = read(folder/'independent_audit.json')
    assert audit['passed']
    for name, digest in audit['files'].items():
        p = folder/name
        assert sha(p) == digest
        inputs[str(p)] = digest
    bank = Path(read(folder/'manifest.json')['args']['bank'])
    for p in (folder/'independent_audit.json',folder/'manifest.json',bank/'geometry.json',bank/'metadata.json'):
        inputs[str(p)] = sha(p)
    results = read(folder/'results.json')
    geometry = read(bank/'geometry.json')
    assert results['active_transitions'] == len(geometry)*1000
    with np.load(folder/'trajectory.npz',allow_pickle=False) as z:
        data = {k:z[k] for k in ('obs','desired','command','previous','q','body','support_bw',
                                  'loaded_slip','no_slip','wrong_force','first_failure')}
    return results, geometry, data


def housing_gaps(cooked, poses, names, origin, geometry):
    center, normal = (np.asarray(geometry[k]) for k in ('top_center','normal'))
    result = {}
    for name in HOUSINGS:
        pose = poses[names.index(name)]
        vertices = cooked.vertices[name]@Rotation.from_quat(pose[3:]).as_matrix().T
        result[name] = float(((vertices+pose[:3]-origin-center)@normal).min())
    return result


def prepare(previous_library, out, donors):
    torch.set_num_threads(1)
    selection = read(STUDY/'development_selection.json')
    checkpoint = STUDY/f"train_{selection['selected_seed']}/final.pt"
    assert sha(checkpoint) == CHECKPOINT_SHA and sha(previous_library) == REPAIR_SHA
    inputs = {str(path): sha(path) for path in (checkpoint,previous_library,MODEL,COOKED,Path(__file__))}
    snapshots = [checked_trace(path,inputs) for path in SPLITS.values()]
    entries = [r for results,_,_ in snapshots for r in results['episodes']]
    index = {r['id']:(j,i) for j,(results,_,_) in enumerate(snapshots)
             for i,r in enumerate(results['episodes'])}
    assert len(index) == len(entries) == 120
    assert {r['id'] for r in entries if not r['clean']} == set(donors)
    metadata = read(FRESH/'bank/metadata.json')
    model, cooked = BodyModel(MODEL), BodyModel(COOKED)
    fk, legs = leg_fk(model,metadata['joint_names'])
    limits = [np.asarray(metadata[k]) for k in ('lower','upper')]
    lo, hi = [torch.from_numpy(x.astype(np.float32)) for x in limits]
    old = Learner(np.zeros(OBS),np.ones(OBS),EVAL_SEED)
    old.load(checkpoint)
    assert int(old.sitting['hold_start_tick']) == HOLD_TICK
    with np.load(previous_library,allow_pickle=False) as z:
        old.bind_sitting({**old.sitting,**dict(z)})
    initial = torch.from_numpy(np.concatenate([data['obs'][0] for _,_,data in snapshots]))
    saved = np.concatenate([data['desired'][:HOLD_TICK] for _,_,data in snapshots],axis=1)
    with torch.no_grad():
        priors = []
        for tick in range(1000):
            initial[:,335] = tick/1000
            priors.append(old.prior(initial).numpy())
    priors = np.stack(priors)
    np.testing.assert_array_equal(priors[:HOLD_TICK],saved)
    ids = [r['id'] for r in entries]
    latent_paths, design = [], []
    for target, donor in donors.items():
        j, i = index[target]
        t_results, t_geometry, t_data = snapshots[j]
        s, k = index[donor]
        d_results, d_geometry, d_data = snapshots[s]
        assert d_results['episodes'][k]['clean']
        before = np.flatnonzero(~t_data['no_slip'][:,i])
        assert len(before) and before[-1] < HOLD_TICK
        donor_pose = d_data['body'][:,k,0].copy()
        donor_pose[:,:3] -= np.asarray(d_results['origins'][k])
        source = dict(body=donor_pose,q=d_data['q'][:,k],load=d_data['support_bw'][:,k])
        nominal = d_data['command'][:,k].astype(float)
        base = torch.from_numpy(d_data['previous'][HOLD_TICK,k,:23].copy())
        for tick in range(HOLD_TICK,1000):
            base = command(base,torch.from_numpy(priors[tick,ids.index(donor)]),lo,hi)
            nominal[tick,:23] = base.numpy()
        waypoints, geometry = retarget(source,d_geometry[k],t_geometry[i],limits,nominal,model,fk,legs)
        start = t_data['previous'][0,i]
        desired, applied = encode(waypoints,start,*limits)
        assert np.isfinite(desired).all()
        command_error = float(abs(applied-waypoints[:,:23]).max())
        assert command_error <= .015, (target, command_error)
        assert np.abs(np.diff(np.vstack((start[:23],applied)),axis=0)).max() <= STEP+2e-7
        assert (applied >= lo.numpy()+MARGIN-2e-7).all() and (applied <= hi.numpy()-MARGIN+2e-7).all()
        latent_paths.append(desired)
        wrong = np.flatnonzero(t_data['first_failure'][:,i])
        gaps = housing_gaps(cooked,t_data['body'][wrong[0],i],t_results['body_names'],
                           np.asarray(t_results['origins'][i]),t_geometry[i]) if len(wrong) else {}
        design.append(dict(target=target,donor=donor,measured_donor_peak_slip_m_s=float(d_data['loaded_slip'][:,k].max()),
                           original_slip_times_s=((before+1)*.02).tolist(),original_wrong_contact_peak_N=float(t_data['wrong_force'][:,i].max()),
                           original_contact_time_cooked_housing_gaps_m=gaps,command_error_rad=command_error,**geometry))

    paths = np.stack(latent_paths)
    selected = np.array([ids.index(name) for name in donors])
    keep = np.ones(120,bool)
    keep[selected] = False
    signatures = initial[:,336:].numpy()
    scale = old.sitting['signature_scale'].numpy()
    distance = np.square((signatures[:,None]-signatures[selected][None])/scale).mean(-1)
    radius = distance[keep].min(0)/4
    assert (radius > 0).all()
    original = {k:v.numpy() for k,v in old.sitting.items()}
    library = {k:original[k] for k in ('repair_signature','repair_radius','repair_desired')}
    library.update(touchdown_signature=signatures[selected],touchdown_radius=radius,touchdown_desired=paths)
    old.bind_sitting({**original,**library})
    with torch.no_grad():
        for tick in range(1000):
            initial[:,335] = tick/1000
            predicted = old.prior(initial).numpy()
            np.testing.assert_array_equal(predicted[selected],paths[:,tick])
            np.testing.assert_array_equal(predicted[keep],priors[tick,keep])
    out.mkdir(exist_ok=False)
    np.savez(out/'library.npz',**library)
    write(out/'construction.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=CHECKPOINT_SHA,
                                      library_sha256=sha(out/'library.npz'),design=design,inputs=inputs,
                                      previously_passing_priors_verified=114,case_ids_used_by_policy=False,
                                      original_fresh_cases_used_for_repair=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous_library',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--donor-map',type=Path)
    args = parser.parse_args()
    prepare(args.previous_library.resolve(),args.output.resolve(),
            read(args.donor_map) if args.donor_map else DONORS)
