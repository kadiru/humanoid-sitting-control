"""Prepare the camera-based lean PPO inputs after the six CPU holds pass."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path('/home/kadir/humanoid-sitting')
SOURCE_BANKS = ROOT/'outputs/full_body_contact_imitation_v4/dataset'
SITTING_PRIOR = ROOT/'outputs/motion_prior_ppo_20261029_v1/prepared/initial_64111.pt'
SEED = 65011
sys.path.insert(0, str(Path(__file__).resolve().parent/'trainer'))
from audit import physical
from common import read, sha, write
from policy import Learner, command, teacher_targets


def checked_trials(evidence):
    receipt = read(evidence/'results.json')
    manifest = read(evidence/'manifest.json')
    rows = {r['case_id']: r for r in receipt['rows']}
    assert receipt['all_completed'] and len(rows) == len(manifest['cases']) == 6
    checked = {}
    for case in manifest['cases']:
        folder = Path(case['folder'])
        assert folder.parent == evidence
        contract_path = folder/'contract.json'
        assert sha(contract_path) == case['contract_sha256']
        c = read(contract_path)
        name, task = c['case_id'], c['task']
        row = rows[name]
        assert row['clean'] and row['standing_reference_valid'] and row['final_hold_s'] >= 5
        assert row['exit_code'] == 0 and row['physics_steps'] == 5300 and not row['timed_out']
        for path, digest in c['source_bank_sha256'].items():
            assert sha(path) == digest
        trace = folder/'result/rollout/score_inputs.npz'
        with np.load(trace, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        assert data['command'].shape == (1025, 1, 37)
        assert all(np.isfinite(v).all() for v in data.values() if v.dtype.kind in 'fc')
        assert not physical(data)['failure'].any()
        score = physical({k: v[25:] for k, v in data.items() if v.shape[0] == 1025})
        assert score['clean'].tolist() == [True]
        cpu_hold_tick = 1000-int(np.cumprod(score['good'][::-1, 0]).sum())
        with np.load(folder/'motion.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(data['command'][:25, 0], np.repeat(z['initial'][None], 25, axis=0))
            np.testing.assert_allclose(data['command'][25:, 0], z['command'], atol=2e-7, rtol=0)
        assert (np.abs(np.diff(data['command'][:, 0, :23], axis=0)) <= .0030002).all()
        assert (data['command'][1:, :, 23:] == data['command'][:1, :, 23:]).all()
        checked.setdefault(task, []).append(dict(id=name, index=c['bank_index'], trace=str(trace),
                                                  trace_sha256=sha(trace), contract_sha256=case['contract_sha256'],
                                                  cpu_hold_tick=cpu_hold_tick))
    assert set(checked) == {'sitting', 'leaning'} and all(len(v) == 3 for v in checked.values())
    return checked


def bank(task):
    folder = SOURCE_BANKS/task
    with np.load(folder/'bank.npz', allow_pickle=False) as z:
        arrays = dict(z)
    return read(folder/'metadata.json'), read(folder/'geometry.json'), arrays


def observations(data, indices):
    obs = data['obs'][:, indices]
    assert obs.shape[0] == 1000 and obs.shape[-1] == 336 and np.isfinite(obs).all()
    signature = obs[0, :, 222:333]
    return np.concatenate((obs, np.broadcast_to(signature[None], (1000, *signature.shape))), axis=-1)


def prepare(evidence, out):
    checked = checked_trials(evidence)
    meta, geometry, lean = bank('leaning')
    smeta, _, sit = bank('sitting')
    selected = {task: np.array([r['index'] for r in checked[task]]) for task in checked}
    for task, data, m in [('sitting', sit, smeta), ('leaning', lean, meta)]:
        i = selected[task]
        assert [m['ids'][j] for j in i] == [r['id'] for r in checked[task]]
        assert data['admitted'][i].all() and data['good'][-250:, i].all() and not data['failure'][:, i].any()
    assert meta['joint_names'] == smeta['joint_names'] and meta['joint_ids'] == list(range(23))
    assert meta['lower'] == smeta['lower'] and meta['upper'] == smeta['upper']
    keep = selected['leaning']
    np.testing.assert_array_equal(lean['reference'][keep], keep)
    lo, hi = (torch.tensor(meta[k], dtype=torch.float32) for k in ('lower', 'upper'))
    previous = torch.from_numpy(lean['previous'][:, keep, :23])
    applied = torch.from_numpy(lean['command'][:, keep, :23])
    targets = teacher_targets(previous, applied, lo, hi)
    running = torch.from_numpy(lean['reset_command'][keep, :23])
    errors = []
    for latent, expected in zip(targets, applied):
        running = command(running, latent, lo, hi)
        errors.append(float((running-expected).abs().max()))
    assert max(errors) <= 2e-7

    lean_obs = observations(lean, keep)
    sit_obs = observations(sit, selected['sitting'])
    all_obs = np.concatenate((lean_obs, sit_obs), axis=1).reshape(6000, -1)
    signature = lean_obs[0, :, 336:]
    saved_hold_start = 1000-np.cumprod(lean['good'][::-1, keep], axis=0).sum(axis=0)
    residual_start = np.maximum(saved_hold_start, [r['cpu_hold_tick'] for r in checked['leaning']])
    agent = Learner(all_obs.mean(0), all_obs.std(0), SEED)
    agent.bind_library(dict(signature=signature, signature_scale=np.maximum(signature.std(0), .05),
                            desired=targets.transpose(0, 1).contiguous(), new_skill=np.ones(3, np.float32),
                            residual_start_tick=residual_start))
    prior = torch.load(SITTING_PRIOR, map_location='cpu', weights_only=False)
    assert all(torch.count_nonzero(prior['actor'][k]) == 0 for k in ('4.weight', '4.bias'))
    agent.bind_sitting(prior['library'])
    agent.bind_demonstrations(all_obs)
    with torch.no_grad():
        inputs = torch.from_numpy(lean_obs.reshape(3000, -1))
        torch.testing.assert_close(agent.choose(inputs, False)[0], targets.reshape(3000, 23), rtol=0, atol=0)
        before_hold = (np.arange(1000)[:, None] < residual_start).reshape(-1)
        np.testing.assert_array_equal(agent.residual_mask(inputs)[:, 0].bool(), ~before_hold)
        torch.testing.assert_close(agent.choose(inputs[before_hold], True)[0],
                                   agent.prior(inputs[before_hold]), rtol=0, atol=0)
        assert not agent.residual_mask(torch.from_numpy(sit_obs.reshape(3000, -1))).any()

    out.mkdir(exist_ok=False)
    fit = out/'leaning_train'
    fit.mkdir()
    reduced = {k: lean[k][keep] for k in ('reset_q', 'reset_qd', 'reset_command', 'reset_root_pose',
                                         'reset_root_velocity', 'phase_times', 'admitted')}
    reduced.update(q_post=lean['q_post'][:, keep], reference=np.arange(3, dtype=np.int64))
    np.savez_compressed(fit/'bank.npz', **reduced)
    write(fit/'geometry.json', [geometry[i] for i in keep])
    write(fit/'metadata.json', dict(meta, cases=3, positive_demonstrations=3, ids=[meta['ids'][i] for i in keep]))
    agent.save(out/'initial.pt')
    np.savez_compressed(out/'demonstrations.npz', obs=all_obs)
    write(out/'preparation.json', dict(
        passed=True, selected=checked, initial_source=str(SITTING_PRIOR), initial_source_sha256=sha(SITTING_PRIOR),
        seed=SEED, new_demonstrations=6, trainable_leaning_demonstrations=3, preserved_sitting_demonstrations=64,
        residual_start_ticks=residual_start.tolist(), approach_exploration_disabled=True,
        recurrence_max_error_rad=max(errors), imitation_observation_count=len(all_obs),
        existing_recorded_camera_observations=True, live_camera_checked=False,
        sitting_residual_enabled=False, fresh_holdout_used=False, ppo_updates=0,
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepare(args.evidence, args.out)
