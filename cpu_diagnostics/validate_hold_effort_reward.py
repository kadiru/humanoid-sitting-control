"""Check the hold-effort objective against saved physical rollouts before PPO."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
TRAINER = HERE/'trainer'
if not TRAINER.exists():
    TRAINER = HERE.parent/'armchair_rest_trainer_20261105_v1'
sys.path.insert(0, str(TRAINER))
from audit import physical, reward_numpy
from common import read, sha, write
from rewards import hold_effort_credit, staged_reward


def check(folder):
    trace = folder/'trajectory.npz'
    audit = read(folder/'independent_audit.json')
    assert audit['passed'] and sha(trace) == audit['files']['trajectory.npz']
    with np.load(trace, allow_pickle=False) as archive:
        data = dict(archive)
    scores = physical(data)
    np.testing.assert_array_equal(scores['clean'], audit['clean'])
    assert data['reward'].shape[0] == 1000
    data['hold_effort_reward'] = np.ones_like(data['good'])
    torque = np.square(data['physical_torque'][..., :23]).sum(-1)
    upward = data['physical_force_w'][..., :5, 2].sum(-1)
    good, clean, end = (torch.from_numpy(data[k]) for k in ('good', 'clean_completion', 'horizon'))
    first = torch.from_numpy(data['first_failure'])
    balance = torch.zeros(len(clean[0]), dtype=torch.float64)
    values, credits = [], []
    for i in range(len(clean)):
        x = {k: torch.from_numpy(v[i]) for k, v in data.items() if k in (
            'alive', 'legal', 'phase', 'good', 'q_error', 'speed', 'gyro', 'support_bw', 'loaded_slip',
            'wrong_force', 'peak_bw', 'normalized_command_delta', 'held_ticks', 'first_failure', 'horizon',
            'clean_completion', 'stable_completion', 'previous_potential', 'mean_qualifying_effort')}
        credit, balance = hold_effort_credit(torch.from_numpy(torque[i]), torch.from_numpy(upward[i]),
                                            good[i], i+1, clean[i], end[i] | first[i], balance)
        credits.append(credit.numpy())
        values.append(staged_reward(x, hold_effort=credit)[0].numpy())
    values = np.asarray(values)*data['learn']
    expected = reward_numpy(data)*data['learn']
    np.testing.assert_allclose(values, expected, atol=3e-6, rtol=3e-6)
    discount = .999**np.arange(1000)
    effort_returns = discount@np.asarray(credits)
    np.testing.assert_allclose(effort_returns[~scores['clean']], 0, atol=5e-5, rtol=0)
    np.testing.assert_array_equal(np.asarray(credits)[:750], 0)
    return dict(path=str(trace), sha256=sha(trace), clean=scores['clean'].tolist(),
                old_return=(discount@data['reward']).tolist(), new_return=(discount@values).tolist(),
                hold_effort_return=effort_returns.tolist(), mean_final_torque_squared=torque[-250:].mean(0).tolist(),
                maximum_step_error=float(abs(values-expected).max()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    before, after, sitting = (check(args.previous/name) for name in ('prior_leaning3', 'learned_leaning3', 'sitting72'))
    assert before['clean'] == after['clean'] == [True]*3 and sum(sitting['clean']) == 64
    assert all(a>b for a,b in zip(before['new_return'], after['new_return']))
    assert all(a>b for a,b in zip(before['hold_effort_return'], after['hold_effort_return']))
    clean, unclean = (np.array(sitting['new_return'])[np.asarray(sitting['clean']) == v] for v in (True, False))
    assert clean.min() > unclean.max()
    # Isolate the refund on a rollout with a credited hold but no clean completion.
    balance = torch.zeros(1, dtype=torch.float64)
    bonuses = []
    for i in range(751, 1001):
        bonus, balance = hold_effort_credit(torch.tensor([400.]), torch.tensor([10.]), torch.tensor([True]),
                                            i, torch.tensor([False]), torch.tensor([i==1000]), balance)
        bonuses.append(bonus.item())
    assert abs(.999**np.arange(250)@bonuses) < 5e-5
    write(args.output, dict(passed=True, prior=before, learned=after,
                            sitting_clean=int(len(clean)), sitting_unclean=int(len(unclean)),
                            minimum_clean_sitting_return=float(clean.min()), maximum_unclean_sitting_return=float(unclean.max()),
                            late_refund_discounted_return=float(.999**np.arange(250)@bonuses)))
