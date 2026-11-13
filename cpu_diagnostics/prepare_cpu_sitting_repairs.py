"""Bind the eight sitting candidates to the isolated CPU rollout runner."""

import argparse
from pathlib import Path

import numpy as np

from run_cpu_supported_hold_batch import HERE, PREVIOUS_RUN, PROPERTIES, SCORER, WALL_CAP, read, sha, write
from prepare_sitting_repairs import DATASET, TARGETS


def prepare(work, prepared, indices):
    parent = read(PREVIOUS_RUN/'contract.json')
    usd = Path(parent['usd'])
    assert sha(usd) == parent['input_sha256'][str(usd)]
    construction = prepared/'construction.json'
    candidate = prepared/'candidate_paths.npy'
    assert sha(candidate) == read(construction)['candidate_sha256']
    meta, geometry = (read(DATASET/name) for name in ('metadata.json', 'geometry.json'))
    lower, upper = (np.array(meta[k], dtype=np.float32) for k in ('lower', 'upper'))
    assert read(PROPERTIES)['joint_names'] == meta['joint_names']
    paths = np.load(candidate, allow_pickle=False)
    sources = [p for p in HERE.iterdir() if p.suffix in ('.py', '.kit')]
    inputs = {str(p): sha(p) for p in [*sources, usd, PROPERTIES, SCORER, SCORER.with_name('common.py'),
                                      construction, candidate, *(DATASET/k for k in ('bank.npz','metadata.json','geometry.json'))]}
    cases = []
    with np.load(DATASET/'bank.npz', allow_pickle=False) as bank:
        reset = {k: bank['reset_'+k] for k in ('q', 'qd', 'command', 'root_pose', 'root_velocity')}
    assert set(indices) <= set(TARGETS)
    for i in indices:
        case, g = meta['ids'][i], geometry[i]
        folder = work/case.replace('/', '_')
        folder.mkdir()
        original = reset['command'][i]
        previous = original.copy()
        commands = []
        for desired in paths[:, i]:
            np.testing.assert_array_equal(desired, np.clip(desired, lower+.03, upper-.03))
            previous[:23] += np.clip(np.float32(.1)*(desired-previous[:23]), -.003, .003)
            commands.append(previous.copy())
        np.savez(folder/'reset.npz', **{k: v[None, i:i+1] for k,v in reset.items()})
        np.savez(folder/'motion.npz', initial=original, command=commands, desired=paths[:, i], lower=lower, upper=upper)
        center = (np.array(g['top_center'])-.05*np.array(g['normal'])).tolist()
        contract = folder/'contract.json'
        write(contract, dict(case_id=case, task='sitting', bank_index=i, usd=str(usd), properties=str(PROPERTIES),
                             reset=str(folder/'reset.npz'), motion=str(folder/'motion.npz'), scorer=str(SCORER),
                             floor_size=[2.8, 2.8, .1], support=dict(size=[g['depth'],g['width'],.1], mass_kg=50.,
                                 position=center, quaternion_xyzw=g['rotation_xyzw'], top_center=g['top_center'], normal=g['normal']),
                             input_sha256={**inputs, **{str(folder/k): sha(folder/k) for k in ('reset.npz','motion.npz')}},
                             paired_standing_controls=300, original_warmup_controls=25, active_controls=1000,
                             cpu_only=True, open_loop_teacher_replay=True, max_wall_s=WALL_CAP))
        cases.append(dict(id=case, folder=str(folder), contract_sha256=sha(contract)))
    write(work/'manifest.json', dict(cases=cases, physics_steps_max=5300*len(cases), wall_cap_s_per_case=WALL_CAP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('--indices', type=int, nargs='+', default=TARGETS)
    args = parser.parse_args()
    prepare(args.work, args.prepared, args.indices)
