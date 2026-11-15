"""Repair loaded pelvis slip by retargeting verified sitting approaches."""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'trainer'))
from common import ROOT, read, sha, write
from policy import Learner
from run_repaired_sitting_ppo import EVAL_SEED
from prepare_sitting_repairs import HOUSINGS, MODEL, BodyModel, encode, leg_fk, smoothstep


STUDY = ROOT/'outputs/repaired_sitting_ppo_20261113_v1'
TARGETS = (6, 7, 10, 11, 20, 21)
DONORS = (24, 28, 70)
HOLD_TICK = 750


def retime(command, initial, first_load, last_slip, limits, speed):
    """Slow the recorded commanded path through first touch and load transfer."""
    clock = np.arange(len(command), dtype=float)
    start, end = first_load-30, last_slip+12
    assert 0 <= start < end < HOLD_TICK
    phase = np.clip((clock-start)/12, 0, 1)*np.clip((end-clock)/12, 0, 1)
    slowdown = (phase**2*(3-2*phase))*(1-speed)
    # Redistribute elapsed time before the PPO takeover. Do not compress beyond
    # the original joint-rate authority when rebuilding the command sequence.
    schedule = clock+np.cumsum(slowdown)/speed
    delay = schedule[-1]-clock[-1]
    catchup = np.clip((clock-end)/(HOLD_TICK-1-end), 0, 1)
    schedule -= delay*catchup
    assert np.all(np.diff(schedule) > 0)
    target = np.stack([np.interp(clock, schedule, signal) for signal in command.T], axis=1)
    desired, applied = encode(target, initial, *limits)
    return desired, applied, target, dict(start_tick=start, end_tick=end, peak_delay_ticks=float(delay))


def retarget(donor, source, target, limits, command, model, fk, legs, *, fore_aft_m=0., pitch_adjust_rad=0.,
             pitch_window_ticks=None):
    """Align a clean loaded approach to the new plane while preserving both feet."""
    body = donor['body']
    rotation = Rotation.from_quat(body[:,3:]).as_matrix()
    root = body[:,:3]
    vertices = model.vertices['pelvis']
    anchor = vertices@rotation[-1].T+root[-1]
    old_n, new_n = (np.asarray(g['normal']) for g in (source,target))
    old_c, new_c = (np.asarray(g['top_center']) for g in (source,target))
    old_gap = ((anchor-old_c)@old_n).min()
    new_gap = ((anchor+[fore_aft_m,0,0]-new_c)@new_n).min()
    dz = (old_gap-new_gap)/new_n[2]
    translation = np.array([fore_aft_m,0.,dz])
    landing = int(np.flatnonzero(donor['load'] > .05)[0])
    start = 150
    ticks = np.unique(np.r_[np.arange(start,1000,10),landing,999])
    amount = smoothstep((ticks-start)/(landing-40-start))
    pitch_start, pitch_finish = pitch_window_ticks or (max(start,landing-180),landing-40)
    assert start <= pitch_start < pitch_finish <= landing
    pitch_timing = smoothstep((ticks-pitch_start)/(pitch_finish-pitch_start))
    changes = []
    max_fit = 0.
    for t, shift, pitch_fraction in zip(ticks, amount, pitch_timing):
        q = donor['q'][t].astype(float)
        wanted = fk(q)
        final_rotation = rotation[t]
        final_position = root[t]+shift*translation
        if pitch_adjust_rad:
            # Keep the same seat-normal gap at every knot while pitching the
            # trunk forward and preserving both recorded foot transforms.
            final_rotation = Rotation.from_rotvec([0,pitch_fraction*pitch_adjust_rad,0]).as_matrix()@rotation[t]
            old_gap = ((vertices@rotation[t].T+final_position-new_c)@new_n).min()
            new_gap = ((vertices@final_rotation.T+final_position-new_c)@new_n).min()
            final_position[2] += (old_gap-new_gap)/new_n[2]
            wanted[:,:3,3] = (wanted[:,:3,3]@rotation[t].T+root[t]-final_position)@final_rotation
            wanted[:,:3,:3] = final_rotation.T@rotation[t]@wanted[:,:3,:3]
        else:
            wanted[:,:3,3] -= rotation[t].T@(shift*translation)

        def residual(angles):
            q[legs] = angles
            f = fk(q)
            twist = Rotation.from_matrix(f[:,:3,:3]@wanted[:,:3,:3].transpose(0,2,1)).as_rotvec()
            return np.c_[f[:,:3,3]-wanted[:,:3,3], .2*twist].ravel()

        bounds = (limits[0][legs]+.03,limits[1][legs]-.03)
        fit = least_squares(residual,q[legs],bounds=bounds,max_nfev=32,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        max_fit = max(max_fit,float(abs(fit.fun).max()))
        changes.append(fit.x-donor['q'][t,legs])
    assert max_fit < 1e-6
    q = donor['q'][-1].copy()
    q[legs] += changes[-1]
    poses = model.fk(q)
    housing = {}
    for name in HOUSINGS:
        pose = poses[name]
        geom = model.vertices[name]@pose[:3,:3].T+pose[:3,3]
        world = geom@final_rotation.T+final_position
        housing[name] = float(((world-new_c)@new_n).min())
    assert min(housing.values()) > 0, housing
    delta = PchipInterpolator(ticks,changes,axis=0)(np.clip(np.arange(1000),start,999))
    trajectory = command.copy()
    trajectory[:,legs] += delta
    return trajectory,dict(donor_landing_s=(landing+1)*.02,shift_m=float(dz),
                           foot_fit_m=max_fit,housing_plane_clearance_m=housing)


def prepare(out, speed=None):
    torch.set_num_threads(1)
    selection = read(STUDY/'development_selection.json')
    checkpoint = STUDY/f"train_{selection['selected_seed']}/final.pt"
    assert sha(checkpoint) == selection['selected_checkpoint_sha256']
    dev = STUDY/f"development_{selection['selected_seed']}"
    test = STUDY/'holdout_selected'
    metadata = read(test/'manifest.json')['metadata']
    limits = [np.array(metadata[k]) for k in ('lower', 'upper')]
    if speed is None:
        dev_geometry_path = Path(read(dev/'manifest.json')['args']['bank'])/'geometry.json'
        test_geometry_path = STUDY/'reserved_bank/geometry.json'
        dev_rows, dev_geometry, test_geometry = map(read,(dev/'results.json',dev_geometry_path,test_geometry_path))
        model = BodyModel(MODEL)
        fk, legs = leg_fk(model, metadata['joint_names'])
    traces = [dev/'trajectory.npz', test/'trajectory.npz']
    for trace in traces:
        audit = read(trace.parent/'independent_audit.json')
        assert audit['passed'] and sha(trace) == audit['files']['trajectory.npz']
    rows = read(test/'results.json')['episodes']
    assert tuple(i for i,r in enumerate(rows) if not r['clean']) == TARGETS

    agent = Learner(np.zeros(447), np.ones(447), EVAL_SEED)
    agent.load(checkpoint)
    with np.load(traces[0], allow_pickle=False) as d, np.load(traces[1], allow_pickle=False) as z:
        signatures = np.concatenate((d['obs'][0, :, 336:], z['obs'][0, :, 336:]))
        all_initial = torch.from_numpy(np.concatenate((d['obs'][0], z['obs'][0])))
        goals = []
        for tick in range(1000):
            all_initial[:,335] = tick/1000
            goals.append(agent.prior(all_initial))
        pristine = torch.stack(goals).numpy()
        np.testing.assert_array_equal(pristine[:HOLD_TICK], np.concatenate(
            (d['desired'][:HOLD_TICK], z['desired'][:HOLD_TICK]), axis=1))
        force, slip, bw = (z[f'physical_{k}'] for k in ('force_w','slip_m_s','bodyweight_N'))
        commands, commands_before = z['command'], z['previous']
        if speed is None:
            dev_command, dev_previous, dev_q, dev_body, dev_load, dev_slip = (
                d[k] for k in ('command','previous','q','body','support_bw','loaded_slip'))
        desired = pristine[:,72:].copy()
        design = []
        low, high = limits[0].astype(np.float32)+np.float32(.03), limits[1].astype(np.float32)-np.float32(.03)
        for i in TARGETS:
            loaded = np.linalg.norm(force[:,i], axis=-1) > .01*bw[:,i,None]
            bad = np.argwhere(loaded & (slip[:,i] > .02))
            assert np.all(bad[:,1] == 0) and bad[-1,0] < HOLD_TICK
            first_load = int(np.flatnonzero(loaded[:,0])[0])
            if speed is None:
                donor = DONORS[TARGETS.index(i)//2]
                assert dev_rows['episodes'][donor]['clean']
                pose = dev_body[:,donor,0].copy()
                pose[:,:3] -= dev_rows['origins'][donor]
                motion = dict(body=pose,q=dev_q[:,donor],load=dev_load[:,donor])
                cmd = dev_command[:,donor].astype(float)
                base = dev_previous[HOLD_TICK,donor,:23].copy()
                fresh_prior = pristine[:,donor]
            else:
                cmd = commands[:,i].copy()
                base = commands_before[HOLD_TICK,i,:23].copy()
                fresh_prior = pristine[:,72+i]
            # Continue with the unmodified demonstration prior, not with
            # previously sampled actor outputs baked into the new prior.
            for j in range(HOLD_TICK, len(cmd)):
                base += np.clip(np.float32(.1)*(np.clip(fresh_prior[j], low, high)-base), -.003, .003)
                cmd[j,:23] = base
            if speed is None:
                target, timing = retarget(motion,dev_geometry[donor],test_geometry[i],limits,cmd,model,fk,legs)
                latent, applied = encode(target, commands_before[0,i], *limits)
                desired[:,i] = latent
                timing.update(donor_id=dev_rows['episodes'][donor]['id'],donor_slip_m_s=float(dev_slip[:,donor].max()))
            else:
                latent, applied, target, timing = retime(cmd,commands_before[0,i],first_load,int(bad[-1,0]),limits,speed)
                start = timing['start_tick']
                desired[start:HOLD_TICK,i] = latent[start:HOLD_TICK]
            err = float(abs(applied-target[:,:23]).max())
            assert err <= .015, (i, err)
            design.append(dict(id=rows[i]['id'], first_load_s=.02*(first_load+1),
                               slip_times_s=(.02*(bad[:,0]+1)).tolist(), contact='pelvis',
                               slip_peak_m_s=float(slip[bad[:,0],i,0].max()), planned_command_error_rad=err,
                               **timing))
        old = {k:v.numpy() for k,v in agent.sitting.items()}
        targets = list(TARGETS)
        target = signatures[72+np.array(targets)]
        scale = old['signature_scale']
        distance = np.square((signatures[:,None]-target[None])/scale).mean(-1)
        keep = np.ones(96, bool)
        keep[72+np.array(targets)] = False
        radius = distance[keep].min(0)/4
        library = dict(repair_signature=np.concatenate((old['repair_signature'],target)),
                       repair_radius=np.r_[old['repair_radius'],radius],
                       repair_desired=np.concatenate((old['repair_desired'],desired[:,targets].transpose(1,0,2))))
        for tick in range(1000):
            all_initial[:,335] = tick/1000
            agent.bind_sitting(old)
            expected = agent.prior(all_initial).numpy()
            agent.bind_sitting({**old,**library})
            actual = agent.prior(all_initial).numpy()
            np.testing.assert_array_equal(actual[72+np.array(targets)],desired[tick,targets])
            np.testing.assert_array_equal(actual[keep], expected[keep])
    out.mkdir(exist_ok=False)
    np.savez(out/'library.npz', **library)
    write(out/'construction.json', dict(
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        selected_seed=selection['selected_seed'], speed=speed,
        targets=targets, design=design,
        library_sha256=sha(out/'library.npz'),
        inputs={str(p):sha(p) for p in [
            *traces,*([MODEL, dev_geometry_path, test_geometry_path] if speed is None else [])]},
        original_test_cases_used_for_repair=True, action_limits_unchanged=True, success_criteria_unchanged=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('--retiming-only-speed', type=float,
                   help='Reproduce the timing-only diagnostic instead of the donor repair.')
    a = p.parse_args()
    assert a.retiming_only_speed is None or 0 < a.retiming_only_speed < 1
    prepare(a.out, a.retiming_only_speed)
