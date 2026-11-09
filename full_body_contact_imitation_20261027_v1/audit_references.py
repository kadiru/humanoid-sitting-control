"""Separate read-only checker of reference packaging and eligibility evidence."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
m = json.loads((ROOT/'demonstration_manifest.json').read_text())
for path, digest in m['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path

accepted = {x['id'] for x in m['accepted_sitting']}
rejected = {x['id'] for x in m['excluded_positive_examples']}
assert len(accepted) == 64 and len(rejected) == 8 and not accepted & rejected
for bank in ['known_old', 'known_recent', 'known_transfer']:
    p = ROOT/'inputs'/bank
    with np.load(p/'every_tick.npz', allow_pickle=False) as z:
        loaded = np.linalg.norm(z['force'], axis=-1) > .01*z['weight'][..., None]
        slipped = (loaded & (z['slip'] > .02)).any(axis=-1).any(axis=0)
        illegal = (~z['legal'].astype(bool)).any(axis=0)
        failed = z['failure'].astype(bool).any(axis=0)
    d = json.loads((p/'results.json').read_text())
    for row in d['episodes']:
        i = row['case']
        eligible = row['steps'] == 1000 and row['success'] and row['final_hold_s'] >= 5 and not (slipped[i] or illegal[i] or failed[i])
        assert eligible == (f'{bank}/{i}' in accepted)

checked = 0
for task, entries in [('sitting', m['accepted_sitting']), ('leaning', m['candidate_leaning'])]:
    with np.load(ROOT/f'{task}_motion_references.npz', allow_pickle=False) as out:
        assert out['q'].shape == (len(entries), 200, 23)
        assert list(out['joint_names']) == m['joint_names']
        assert len(set(out['ids'])) == len(entries)
        for row_index, item in enumerate(entries):
            p = Path(item['source'])
            assert 'holdout' not in str(p).lower() and 'holdout' not in item['source_bank'].lower()
            result = json.loads((p/'results.json').read_text())
            ids = [result['joint_names'].index(n) for n in out['joint_names']]
            with np.load(p/'trace.npz', allow_pickle=False) as original:
                np.testing.assert_array_equal(out['q'][row_index], original['q'][:, item['case'], :][:, ids])
                np.testing.assert_array_equal(out['command'][row_index], original['command'][:, item['case'], :][:, ids])
                np.testing.assert_array_equal(out['time_s'], original['t'])
                hold = out['time_s'] > 15
                assert original['good'][hold, item['case']].all() and original['alive'][hold, item['case']].all()
            phase = out['phase'][row_index]
            assert (np.diff(phase) >= 0).all() and (phase[hold] == 3).all()
            if task == 'leaning':
                assert item['sampled_contact_screen_passed'] == (item['sampled_slip_count'] == item['sampled_illegal_count'] == 0)
                expected = 'CANDIDATE_ONLY_REQUIRES_FULL_RATE_CONTACT_REVALIDATION' if item['sampled_contact_screen_passed'] else 'REJECTED_POSITIVE_SAMPLED_CONTACT_FAILURE'
                assert item['status'] == expected
            assert not item['full_state_action_training_ready']
            checked += 1
assert not m['training_launch_ready'] and m['new_simulator_steps'] == m['new_optimizer_steps'] == 0
result = {'pass': True, 'scope': 'Independent data-packaging check; not simulator validation or learning success',
          'references_checked': checked, 'strict_clean_sitting': 64, 'leaning_candidates_recorded': 18,
          'leaning_pending_revalidation': sum(x['sampled_contact_screen_passed'] for x in m['candidate_leaning']),
          'leaning_rejected_for_positive_imitation': sum(not x['sampled_contact_screen_passed'] for x in m['candidate_leaning']),
          'excluded_sitting_failures': 8, 'direct_force_and_slip_threshold_check': True,
          'exact_joint_and_time_alignment': True, 'no_holdout_inputs': True,
          'training_launch_ready': False}
(ROOT/'reference_audit.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
