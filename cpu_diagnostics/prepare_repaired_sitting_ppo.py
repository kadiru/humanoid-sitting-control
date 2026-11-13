"""Build sitting PPO inputs from the independently scored 72 development demos."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'trainer'))
from audit import evaluate
from common import OBS, ROOT, read, sha, write
from policy import Learner, command


SEEDS = (63111, 63117)
HOLD_TICK = 750
UPDATES = 8
BASELINE = ROOT/'outputs/steep_clearance_20261113_v3_full/evaluation'
DEVELOPMENT = ROOT/'outputs/full_body_contact_imitation_v4/dataset/sitting'
REPAIRS = ROOT/'.cpu-work/20261113/steep_clearance_v3/prepared/library.npz'
PREVIOUS = ROOT/'outputs/hold_effort_ppo_20261113_v1/training/final.pt'
RESET_FIELDS = ('q', 'qd', 'command', 'root_pose', 'root_velocity')


def prepare(out):
    torch.set_num_threads(1)
    audit = read(BASELINE/'independent_audit.json')
    assert audit['passed'] and audit['clean_count'] == audit['rows'] == 72
    for name, digest in audit['files'].items():
        assert sha(BASELINE/name) == digest
    # Recompute the independent physical and action checks on a private receipt
    # directory so the accepted evidence and its audit remain immutable.
    out.mkdir(exist_ok=False)
    review = out/'baseline_review'
    review.mkdir()
    for name in ('trajectory.npz', 'results.json', 'startup.json', 'manifest.json'):
        (review/name).symlink_to(BASELINE/name)
    assert evaluate(review) == audit
    meta = read(DEVELOPMENT/'metadata.json')
    assert meta == read(BASELINE/'manifest.json')['metadata']
    assert meta['ids'] == [r['id'] for r in read(BASELINE/'results.json')['episodes']]
    assert meta['joint_ids'] == list(range(23))
    with np.load(DEVELOPMENT/'bank.npz', allow_pickle=False) as original:
        resets = {f'reset_{name}': original[f'reset_{name}'] for name in RESET_FIELDS}
        repaired_ids = np.asarray(meta['ids'])[~original['admitted']]
    with np.load(BASELINE/'trajectory.npz', allow_pickle=False) as z:
        obs, desired, q = z['obs'], z['desired'], z['q']
        good, load, previous, applied = z['good'], z['support_bw'], z['previous'], z['command']
    assert obs.shape == (1000, 72, OBS) and desired.shape == (1000, 72, 23)
    assert good[HOLD_TICK-1:].all() and len(repaired_ids) == 8
    np.testing.assert_array_equal(previous[0], resets['reset_command'])
    np.testing.assert_array_equal(previous[1:], applied[:-1])
    start = np.argmax(load >= .05, axis=0)
    settled = 1000-np.cumprod(good[::-1], axis=0).sum(0)
    assert (start <= settled).all()

    # Use the validated repair routing verbatim. Demonstrations also supply the
    # imitation observations and the reward-only reference for every row.
    source = torch.load(PREVIOUS, map_location='cpu', weights_only=False)
    with np.load(REPAIRS, allow_pickle=False) as overlay:
        sitting = {**source['sitting'], **dict(overlay), 'hold_start_tick': HOLD_TICK}
    assert len(sitting['repair_signature']) == len(repaired_ids)
    sample = obs[HOLD_TICK:].reshape(-1, OBS)
    mean, scale = sample.mean(0), sample.std(0)
    lo, hi = (torch.tensor(meta[k], dtype=torch.float32) for k in ('lower', 'upper'))
    actor_receipts = {}
    for seed in SEEDS:
        agent = Learner(mean, scale, seed)
        agent.bind_library(source['library'])
        agent.bind_sitting(sitting)
        running = torch.from_numpy(resets['reset_command'][:, :23])
        max_error = 0.
        with torch.no_grad():
            for tick in range(1000):
                inputs = torch.from_numpy(obs[tick])
                action = agent.choose(inputs, False)[0]
                torch.testing.assert_close(action, torch.from_numpy(desired[tick]), rtol=0, atol=0)
                assert (agent.residual_mask(inputs) == (tick >= HOLD_TICK)).all()
                running = command(running, action, lo, hi)
                max_error = max(max_error, float((running-torch.from_numpy(applied[tick, :, :23])).abs().max()))
                if tick in (0, HOLD_TICK-1, HOLD_TICK):
                    stochastic = agent.choose(inputs, True)[0]
                    assert torch.equal(stochastic, action) == (tick < HOLD_TICK)
        assert max_error <= 2e-7
        path = out/f'initial_{seed}.pt'
        agent.save(path)
        actor_receipts[str(seed)] = dict(initial_sha256=sha(path), prior_exact=True,
                                         command_replay_max_error_rad=max_error)

    bank = out/'development'
    bank.mkdir()
    np.savez_compressed(bank/'bank.npz', **resets, q_post=q, reference=np.arange(72),
                        admitted=np.ones(72, bool), phase_times=.02*(1+np.stack((start, settled), axis=1)))
    write(bank/'geometry.json', read(DEVELOPMENT/'geometry.json'))
    write(bank/'metadata.json', dict(meta, positive_demonstrations=72,
                                     demonstration_audit_sha256=sha(BASELINE/'independent_audit.json')))
    np.savez_compressed(out/'demonstrations.npz', obs=sample)
    inputs = [PREVIOUS, REPAIRS, BASELINE/'independent_audit.json',
              *(BASELINE/name for name in audit['files']),
              *(DEVELOPMENT/name for name in ('bank.npz', 'metadata.json', 'geometry.json'))]
    write(out/'preparation.json', dict(passed=True, baseline_clean=72, repaired_ids=repaired_ids.tolist(),
                                      imitation_observations=len(sample), hold_start_tick=HOLD_TICK,
                                      final_hold_ticks=(1000-settled).tolist(), initial=actor_receipts,
                                      source_inputs={str(path): sha(path) for path in inputs},
                                      fresh_holdout_used=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    prepare(parser.parse_args().out)
