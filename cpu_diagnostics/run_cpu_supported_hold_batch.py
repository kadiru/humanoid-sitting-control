"""Freeze and run the six selected support trials in one bounded CPU queue."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from cycle_executor import encode_float32_path


ROOT = Path('/home/kadir/humanoid-sitting')
PYTHON = ROOT/'IsaacLab/.venv/bin/python'
SCORER = ROOT/'experiments/armchair_rest_trainer_20261105_v1/audit.py'
PROPERTIES = ROOT/'outputs/rest_cycle_contact_fix_20261109_v1/runtime_inspection/runtime_properties.json'
PREVIOUS_RUN = ROOT/'.cpu-work/20261109/g1_loaded_reference'
HERE = Path(__file__).resolve().parent
WALL_CAP = 45


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(work, selected):
    previous = read(PREVIOUS_RUN/'contract.json')
    usd = Path(previous['usd'])
    properties = read(PROPERTIES)
    cases = []
    sources = [HERE/name for name in ('run_cpu_supported_hold.py', 'run_cpu_supported_hold_batch.py',
                                     'import_cpu_g1.py', 'isolated_runtime.py', 'physics_cpu.kit',
                                     'cycle_executor.py', 'contact_gate.py')]
    base_inputs = [*sources, usd, PROPERTIES, SCORER, SCORER.with_name('common.py'), selected]
    for task, data in read(selected).items():
        if len(data['starts']) != 3:
            raise ValueError('Expected three starts for each task')
        for path, digest in data['sha256'].items():
            if sha(path) != digest:
                raise ValueError(f'Selected input changed: {path}')
        bank_path = Path(data['source'])/'bank.npz'
        meta_path = bank_path.with_name('metadata.json')
        geom_path = bank_path.with_name('geometry.json')
        meta = read(meta_path)
        geometry = read(geom_path)
        if properties['joint_names'] != meta['joint_names']:
            raise ValueError('Reset joint order differs from the CPU importer')
        lower, upper = (np.asarray(meta[k], np.float32) for k in ('lower', 'upper'))
        with np.load(bank_path, allow_pickle=False) as archive:
            data_arrays = {key: archive[key] for key in (
                'admitted', 'good', 'failure', 'command', 'previous', 'reset_q', 'reset_qd',
                'reset_command', 'reset_root_pose', 'reset_root_velocity',
            )}
        for start in data['starts']:
            case, i = start['id'], start['bank_index']
            g = geometry[i]
            if (meta['ids'][i] != case or g != start['surface'] or g.get('shape', 'rect') != 'rect'
                    or not data_arrays['admitted'][i] or data_arrays['failure'][:, i].any()
                    or not data_arrays['good'][-250:, i].all()):
                raise ValueError(f'Saved case is not the selected clean rectangular support: {case}')
            folder = work/case.replace('/', '_')
            folder.mkdir(exist_ok=False)
            fields = ('q', 'qd', 'command', 'root_pose', 'root_velocity')
            reset = {key: data_arrays['reset_'+key][None, i:i+1].copy() for key in fields}
            initial = reset['command'][0, 0]
            np.testing.assert_array_equal(initial, data_arrays['previous'][0, i])
            np.testing.assert_array_equal(reset['root_pose'][0, 0], np.array(start['reset_root_pose'], np.float32))
            target = data_arrays['command'][:, i].copy()
            if target.shape != (1000, 37) or (target[:, 23:] != initial[23:]).any():
                raise ValueError(f'Incomplete commands or finger motion: {case}')
            desired = encode_float32_path(target, initial, lower.astype(float), upper.astype(float))
            np.savez(folder/'reset.npz', **reset)
            np.savez(folder/'motion.npz', initial=initial, command=target, desired=desired,
                     lower=lower, upper=upper)
            center = (np.asarray(g['top_center']) - .05*np.asarray(g['normal'])).tolist()
            contract = dict(case_id=case, task=task, bank_index=i, usd=str(usd), properties=str(PROPERTIES),
                            reset=str(folder/'reset.npz'), motion=str(folder/'motion.npz'), scorer=str(SCORER),
                            floor_size=[2.8, 2.8, .1], support=dict(
                                size=[g['depth'], g['width'], .1], mass_kg=50., position=center,
                                quaternion_xyzw=g['rotation_xyzw'], top_center=g['top_center'], normal=g['normal']),
                            input_sha256={str(p): sha(p) for p in [*base_inputs, folder/'reset.npz', folder/'motion.npz']},
                            source_bank_sha256={str(p): data['sha256'][str(p)] for p in (bank_path, meta_path, geom_path)},
                            paired_standing_controls=300, original_warmup_controls=25, active_controls=1000,
                            cpu_only=True, open_loop_teacher_replay=True, max_wall_s=WALL_CAP)
            write(folder/'contract.json', contract)
            cases.append(dict(id=case, folder=str(folder), contract_sha256=sha(folder/'contract.json'),
                              max_original_command_increment_rad=float(abs(target[:, :23]-data_arrays['previous'][:, i, :23]).max())))
    write(work/'manifest.json', dict(cases=cases, selected_sha256=sha(selected), physics_steps_max=31800,
                                     wall_cap_s_per_case=WALL_CAP, n=6))
    print(json.dumps(cases, indent=2), flush=True)


def launch(work):
    manifest = read(work/'manifest.json')
    template = read(PREVIOUS_RUN/'launch.json')['command']
    # Use the same GPU-inaccessible image, resource limits and device masks.
    prefix = template[:-3]
    previous_mount = 'type=bind,src={0},dst={0}'.format(PREVIOUS_RUN)
    prefix[prefix.index(previous_mount)] = 'type=bind,src={0},dst={0}'.format(work)
    name_slot = prefix.index('--name')+1
    results = []
    for index, case in enumerate(manifest['cases']):
        folder = Path(case['folder'])
        contract = folder/'contract.json'
        if sha(contract) != case['contract_sha256']:
            raise ValueError(f'Case contract changed: {case["id"]}')
        prefix[name_slot] = f'sitting-cpu-supported-20261112-{index}'
        args = [*prefix, str(HERE/'run_cpu_supported_hold.py'), str(contract), str(folder/'result')]
        write(folder/'launch.json', dict(command=args, wall_cap_s=WALL_CAP))
        begun = time.monotonic()
        timed_out = False
        with (folder/'process.log').open('x') as log:
            with subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT) as child:
                try:
                    child.wait(timeout=WALL_CAP)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    subprocess.run(['docker', 'kill', prefix[name_slot]], capture_output=True, timeout=5, check=False)
                    child.kill()
                    child.wait(timeout=5)
        receipt = dict(case_id=case['id'], exit_code=child.returncode, timed_out=timed_out,
                       wall_s=time.monotonic()-begun)
        write(folder/'process.json', receipt)
        if child.returncode == 0 and (folder/'result/result.json').is_file():
            receipt.update(read(folder/'result/result.json'))
        results.append(receipt)
        print(json.dumps(receipt), flush=True)
        # A runtime failure may affect every start. Fix the adapter before
        # consuming the rest of the queue. Physical failures still count.
        if child.returncode != 0:
            break
    write(work/'results.json', dict(rows=results, all_completed=len(results)==6 and all(
        'clean' in row for row in results)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'launch'))
    p.add_argument('work', type=Path)
    p.add_argument('--selected', type=Path)
    args = p.parse_args()
    if args.mode == 'prepare':
        if not args.selected:
            p.error('--selected is required for preparation')
        prepare(args.work, args.selected)
    else:
        launch(args.work)


if __name__ == '__main__':
    main()
