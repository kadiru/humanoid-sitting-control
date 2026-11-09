"""Development-only review of existing block-0 recordings; no simulation/training."""
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
from earlier_control_common import DATA, OLD, ROOT, SEEDS, PARENTS, read, write, sha, check
from earlier_control_gate import gate
from verify_earlier_control_study import source_check, actor_checks, matched_start
from verify_seat_load_study import evaluation, compare
from verify_posture_credit_study import contacts
from verify_earlier_control_video import verify


def main():
    output = DATA / 'interim_development_review_01'
    output.mkdir(exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    study = DATA / 'ppo_01'
    protocol = read(study / 'protocol.json')
    assert sha(study / 'protocol.json') == (study / 'protocol.sha256').read_text().strip()
    check(protocol['hashes'])
    paths = {'reference': str(OLD / 'preflight/calibrate')}
    paths.update({f'parent_{PARENTS[s]}': str(OLD / f'ppo_01/seed_{PARENTS[s]}/development_final') for s in SEEDS})
    paths.update({f'Checkpoint {s} (6 updates)': str(study / f'seed_{s}/block_00/current_eval') for s in SEEDS})
    verification = dict(scope='INTERIM development only, block 0 after six updates; not final or unseen-surface results', paths={'interim development': paths}, evaluations={}, contacts={}, startups={}, actors={}, gates={}, comparisons={})
    rows = {}
    geometries = {}
    for key, folder in paths.items():
        folder = Path(folder)
        summary, rows[key], geometries[key] = evaluation(folder, 18)
        verification['evaluations'][key] = summary
        verification['contacts'][key] = contacts(folder)
        verification['startups'][key] = matched_start(Path(paths['reference']), folder)
        assert all(x['steps'] == 1000 for x in rows[key])
    for seed in SEEDS:
        key = f'Checkpoint {seed} (6 updates)'
        folder = Path(paths[key])
        checkpoint = study / f'seed_{seed}/block_00/train/final.pt'
        source_check(folder, protocol)
        verification['actors'][key] = actor_checks(folder, seed, checkpoint, 10.)
        recorded = np.load(folder / 'episode_decisions.npz')['early_observation'][0]
        expected = np.load(DATA / f'preflight/calibrate_{seed}/stage_observations.npz')['10.0'][0]
        assert np.array_equal(recorded, expected)
        reference = np.load(Path(paths['reference']) / 'trace.npz')
        policy = np.load(folder / 'trace.npz')
        for field in ['q', 'body', 'command', 'reference', 'torque', 'geometry']:
            assert np.array_equal(reference[field][reference['t'] <= 10], policy[field][policy['t'] <= 10]), field
        verification['gates'][key] = gate(folder, seed)
        assert verification['gates'][key] == read(study / f'seed_{seed}/curriculum.json')[0]['current_gate']
        clean = lambda rr: [dict(x, success=bool(x['success'] and not any(x['failures'].values()) and not x['slip_ticks'] and not x['illegal_ticks'])) for x in rr]
        verification['comparisons'][key] = {baseline: compare(clean(rows[key]), clean(rows[baseline]), geometries[key]) for baseline in ['reference', 'parent_59111', 'parent_59117']}
    write(output / 'verification.json', verification)
    original = ROOT / 'scripts/render_earlier_control_video.py'
    text = original.read_text()
    edits = {
        'Earlier PPO sitting: every final comparison episode': 'INTERIM: six-update development comparison',
        'All final comparison episodes included': 'All 90 interim development episodes included',
        'See the verified report for matched stability and effort comparisons.': 'Training continues. Wider unseen-surface tests have not run.',
        'Every final comparison episode; training exploration excluded': 'Interim block-0 development recordings; training exploration and unseen surfaces excluded',
        "f\"{page['bank'].upper()} | case": "f\"INTERIM / 6 UPDATES | {page['bank'].upper()} | case",
    }
    for before, after in edits.items():
        assert text.count(before) == 1, before
        text = text.replace(before, after)
    local_renderer = output / 'interim_renderer.py'
    local_renderer.write_text(text)
    write(output / 'render_provenance.json', dict(original_source=str(original), original_sha256=sha(original), label_only_edits=edits, interim_renderer_sha256=sha(local_renderer), simulations_added=0, training_added=0))
    spec = importlib.util.spec_from_file_location('interim_renderer', local_renderer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.render(verification['paths'], output / 'video')
    result = verify(output / 'video', verification['paths'])
    write(output / 'video/verification.json', result)
    check(protocol['hashes'])
    write(output / 'SHA256.json', {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'SHA256.json'})
    print(json.dumps(result))


if __name__ == '__main__':
    main()
