"""Build motion references from completed DEVELOPMENT trajectories, without Isaac.

This produces pose/command references, not state-action pairs for behavior cloning.
No interpolation is certified as a physically valid trajectory or reset state.
"""
from pathlib import Path
import ast
import hashlib
import json
import time
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
START = time.monotonic()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def body_joints():
    source = PROJECT / 'ubuntu_audit/sitting_task/src/humanoid_sitting/deployment.py'
    tree = ast.parse(source.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BODY_JOINT_NAMES' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing explicit joint map')


def phases(trace, case):
    """Coarse reference annotations only, never a live truth-driven actor input."""
    t = trace['t']
    legal = trace['legal'][:, case].astype(bool)
    good = trace['good'][:, case].astype(bool) & trace['alive'][:, case].astype(bool)
    contact = np.flatnonzero((trace['support'][:, case] >= .05) & legal)
    if not len(contact):
        raise ValueError('No legal supported-contact sample')
    c = int(contact[0])
    settling = np.flatnonzero(good & (np.arange(len(t)) >= c))
    if not len(settling):
        raise ValueError('No qualifying supported-rest sample')
    s = int(settling[0])
    label = np.zeros(len(t), dtype=np.int8)
    label[c:] = 1
    label[s:] = 2
    # The final five seconds are a hold reference; verify every saved sample.
    hold = t > 15.
    if not good[hold].all():
        raise ValueError('Final hold reference is not continuously good at saved cadence')
    label[hold] = 3
    return label, {'contact_sample_s': float(t[c]), 'first_good_sample_s': float(t[s]), 'hold_interval_s': [15., 20.]}


def main():
    names = body_joints()
    assert len(names) == len(set(names)) == 23
    sources = [('sitting', b, HERE/'inputs'/b) for b in ['known_old', 'known_recent', 'known_transfer']]
    sources.append(('leaning', 'lean_reference_dev18', PROJECT/'validated_transfer_effort_01/assessment_results/leaning/reference'))
    accepted, excluded, pending, bindings, packs = [], [], [], {}, {'sitting': [], 'leaning': []}
    for task, bank, root in sources:
        result = json.loads((root/'results.json').read_text())
        manifest = json.loads((root/'manifest.json').read_text())
        source_bank = manifest['args']['bank']
        assert 'holdout' not in source_bank.lower() and 'holdout' not in str(root).lower()
        assert manifest['args']['start_stage'] == '0'
        with np.load(root/'trace.npz', allow_pickle=False) as trace:
            ids = [result['joint_names'].index(n) for n in names]
            assert len(set(ids)) == 23
            t = trace['t']
            np.testing.assert_allclose(t, np.arange(1, 201)*.1, rtol=0, atol=1e-12)
            for key in ['q', 'command', 'support', 'geometry']:
                assert np.isfinite(trace[key]).all(), (bank, key)
            ticks = np.load(root/'every_tick.npz', allow_pickle=False) if task == 'sitting' else None
            for row in result['episodes']:
                case = row['case']
                identity = f'{bank}/{case}'
                stable = bool(row['success'] and row['steps'] == 1000 and row['duration_s'] == 20 and row['final_hold_s'] >= 5 and not any(row['failures'].values()))
                if task == 'sitting':
                    clean = stable and row['slip_ticks'] == 0 and row['illegal_ticks'] == 0
                    # Independent recorded tick predicates must match the result.
                    assert int((~ticks['no_slip'][:, case].astype(bool)).sum()) == row['slip_ticks']
                    assert int((~ticks['legal'][:, case].astype(bool)).sum()) == row['illegal_ticks']
                    if not clean:
                        excluded.append({'id': identity, 'task': task, 'reason': 'Not strict-clean; retain for failure evaluation, never positive imitation', 'slip_ticks': row['slip_ticks'], 'illegal_ticks': row['illegal_ticks'], 'physical_failures': row['failures']})
                        continue
                    assert not ticks['failure'][:, case].any()
                elif not stable:
                    excluded.append({'id': identity, 'task': task, 'reason': 'Historical stable-completion gate failed'})
                    continue
                phase, boundaries = phases(trace, case)
                item = {'id': identity, 'task': task, 'case': case, 'source': str(root), 'source_bank': source_bank, 'phase_annotation': boundaries, 'full_state_action_training_ready': False}
                if task == 'leaning':
                    item.update(status='CANDIDATE_ONLY_REQUIRES_FULL_RATE_CONTACT_REVALIDATION', sampled_slip_count=int((~trace['no_slip'][:, case]).sum()), sampled_illegal_count=int((~trace['legal'][:, case]).sum()))
                    pending.append(item)
                else:
                    item['status'] = 'STRICT_CLEAN_RECORDED_MOTION_REFERENCE'
                    accepted.append(item)
                packs[task].append({'id': identity, 'q': trace['q'][:, case, :][:, ids], 'command': trace['command'][:, case, :][:, ids], 'phase': phase, 'geometry': trace['geometry'][:, case], 'support': trace['support'][:, case]})
            if ticks is not None:
                ticks.close()
        for name in ['results.json', 'manifest.json', 'trace.npz', 'geometry.json', 'startup.npz', 'every_tick.npz', 'independent_audit_v2.json']:
            p = root/name
            if p.exists():
                bindings[str(p)] = sha(p)
    assert len(accepted) == 64 and len(excluded) == 8 and len(pending) == 18
    for task, rows in packs.items():
        np.savez_compressed(HERE/f'{task}_motion_references.npz', time_s=np.arange(1, 201)*.1,
                            joint_names=np.array(names), ids=np.array([r['id'] for r in rows]),
                            **{key: np.stack([r[key] for r in rows]) for key in ['q', 'command', 'phase', 'geometry', 'support']})
    output = {'classification': 'Offline development-data preparation; no policy trained or simulation launched',
              'joint_names': names, 'reference_sample_period_s': .1, 'original_control_sample_period_s': .02,
              'accepted_sitting': accepted, 'candidate_leaning': pending, 'excluded_positive_examples': excluded,
              'source_sha256': bindings,
              'limitations': ['Ten-Hz joint/reference samples cannot certify intermediate motion or reconstruct missing velocities.',
                             'Recorded commands are post-step; do not pair them with same-frame observations for behavior cloning.',
                             'Only startup observation exists here; complete aligned sensor/action training pairs are absent.',
                             'Leaning stable completion is not equivalent to the current whole-trial zero-slip criterion.',
                             'Training from these known development cases cannot be reported as held-out generalization. Fresh holdout remains closed.',
                             'No demonstration of the eight failed geometries being repaired is present.'],
              'training_launch_ready': False, 'new_simulator_steps': 0, 'new_optimizer_steps': 0,
              'preparation_seconds': time.monotonic()-START}
    (HERE/'demonstration_manifest.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'clean_sitting_references': len(accepted), 'pending_leaning_references': len(pending), 'excluded_failures': len(excluded), 'seconds': output['preparation_seconds']}))


if __name__ == '__main__':
    main()
