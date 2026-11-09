"""Audit recaptured demonstrations and form per-task training/evaluation banks."""
import argparse
from pathlib import Path
import numpy as np
from common import read, write, sha, STEP, FILTER


def audit_capture(root):
    meta = read(root/'demonstration_capture.json')
    with np.load(root/'demonstrations_50hz.npz', allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    n = d['obs'].shape[1]
    assert d['obs'].shape == (1000, n, 336)
    loaded = np.linalg.norm(d['force'], axis=-1) > .01*d['weight'][..., None]
    no_slip = ~(loaded & (d['slip'] > .02)).any(-1)
    np.testing.assert_array_equal(no_slip, d['no_slip'])
    alive = ~np.maximum.accumulate(d['failure'], axis=0)
    stable = alive[-1] & (d['good'][-250:] & alive[-250:]).all(0)
    clean = stable & no_slip.all(0) & d['legal'].all(0)
    ids = meta['joint_ids']
    np.testing.assert_array_equal(d['obs'][:, :, 2*111:2*111+23], d['q_pre'][:, :, ids])
    np.testing.assert_array_equal(d['obs'][:, :, 2*111+75:2*111+98], d['previous'][:, :, ids])
    np.testing.assert_array_equal(d['previous'][1:], d['command'][:-1])
    np.testing.assert_array_equal(d['q_pre'][1:], d['q_post'][:-1])
    delta = d['command'][:, :, ids]-d['previous'][:, :, ids]
    # This is a data inclusion check, not permission to relax controller limits.
    reachable = (abs(delta) <= STEP+2e-7).all((0, 2))
    outside = ((d['command'][:, :, ids] < np.array(meta['lower'])+.03-2e-7) |
               (d['command'][:, :, ids] > np.array(meta['upper'])-.03+2e-7)).any((0, 2))
    admitted = clean & reachable & ~outside
    rows = read(root/'results.json')['episodes']
    assert len(rows) == n
    assert [bool(x['success']) for x in rows] == stable.tolist()
    for x in rows:
        i = x['case']
        if 'slip_ticks' in x:
            assert x['slip_ticks'] == int((~no_slip[:, i]).sum())
    audit = dict(passed=True, task=meta['task'], stable=stable.tolist(), strict_clean=clean.tolist(),
                 admitted=admitted.tolist(), excluded_action_envelope=(~reachable | outside).tolist(),
                 full_rate_alignment=True, physical_success_criteria_changed=False,
                 files={p.name: sha(p) for p in root.iterdir() if p.name in ['demonstrations_50hz.npz', 'original_reset.npz', 'results.json', 'geometry.json', 'demonstration_capture.json']})
    write(root/'demonstration_audit.json', audit)
    return d, meta, audit


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); p.add_argument('--captures', nargs='+', required=True)
    args = p.parse_args(); out = Path(args.output); out.mkdir(exist_ok=False)
    groups = {}
    for path in args.captures:
        root = Path(path); d, meta, audit = audit_capture(root)
        groups.setdefault(meta['task'], []).append((root, d, meta, audit))
    summary = {}
    for task, items in groups.items():
        dest = out/task; dest.mkdir()
        data = {k: np.concatenate([x[1][k] for x in items], axis=1) for k in items[0][1]}
        admitted = np.concatenate([x[3]['admitted'] for x in items]).astype(bool)
        geometry = []; identifiers = []; resets = []
        for root, d, meta, audit in items:
            gs = read(root/'geometry.json')
            assert len(meta['geometry_indices']) == d['obs'].shape[1]
            for i, geometry_index in enumerate(meta['geometry_indices']):
                g = gs[geometry_index]
                geometry.append(dict(g, case=len(geometry), start_stage=0))
                identifiers.append(f"{meta['bank']}/{i}")
            with np.load(root/'original_reset.npz', allow_pickle=False) as z:
                resets.append({k: z[k] for k in z.files})
        count = len(geometry)
        assert count == (72 if task == 'sitting' else 18)
        assert admitted.sum() >= (60 if task == 'sitting' else 8), (task, int(admitted.sum()))
        # Reward-only reference association. Never an actor feature or reset
        # from another surface. Non-admitted cases always start standing.
        features = np.array([[g['height']/.05, g['pitch_deg']/10, g['depth'], g['width']] for g in geometry])
        choices = np.flatnonzero(admitted)
        reference = np.array([i if admitted[i] else choices[np.argmin(((features[choices]-features[i])**2).sum(1))] for i in range(count)])
        meta = items[0][2]
        for _, _, m, _ in items:
            assert m['joint_ids'] == meta['joint_ids'] and m['joint_names'] == meta['joint_names']
            assert m['lower'] == meta['lower'] and m['upper'] == meta['upper']
        ids = meta['joint_ids']
        # Applied command deltas are paired with the PRE-action observation.
        target = ((data['command']-data['previous'])[:, :, ids]/STEP).clip(-1, 1)
        phase_times = []
        for j in reference:
            c = np.flatnonzero((data['support_bw'][:, j] >= .05) & data['legal'][:, j])[0]*.02
            s = np.flatnonzero(data['good'][:, j] & (np.arange(1000)*.02 >= c))[0]*.02
            assert 0 <= c <= s <= 15., (task, int(j), c, s)
            phase_times.append([float(c), float(s)])
        np.savez_compressed(dest/'bank.npz', **data, admitted=admitted, reference=reference,
                            bc_target=target, phase_times=np.array(phase_times),
                            **{'reset_'+k: np.concatenate([r[k] for r in resets]) for k in resets[0]})
        write(dest/'geometry.json', geometry)
        write(dest/'metadata.json', dict(task=task, ids=identifiers, joint_ids=ids,
              joint_names=meta['joint_names'], body_names=meta['body_names'], lower=meta['lower'], upper=meta['upper'],
              cases=count, positive_demonstrations=int(admitted.sum()), initial_reset_warmup_ticks=25,
              phase_reset_times_s=[0, 6, 10, 15], reference_is_reward_only=True,
              fresh_holdout_used=False, source_audits={str(x[0]/'demonstration_audit.json'):sha(x[0]/'demonstration_audit.json') for x in items}))
        summary[task] = dict(cases=count, positive_demonstrations=int(admitted.sum()), references=reference.tolist())
    assert set(summary) == {'sitting', 'leaning'}
    write(out/'dataset_audit.json', dict(passed=True, tasks=summary, full_rate_teacher_validation=True, holdout_opened=False))


if __name__ == '__main__':
    main()
