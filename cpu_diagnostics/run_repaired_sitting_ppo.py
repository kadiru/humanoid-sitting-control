"""Run the two repaired sitting PPO seeds and the sealed evaluations on the 4090."""

import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'trainer'))
from audit import evaluate, weights
from common import DATA, ROOT, read, sha, write
from prepare_repaired_sitting_ppo import BASELINE, HOLD_TICK, RESET_FIELDS, SEEDS, UPDATES


GPU = 'GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
LAUNCHER = ROOT/'scripts/run_supported_gpu0.sh'
MODEL = ROOT/'runs/g1_kinematics_geometry_04/model.json'
RESERVATION = DATA/'reservation.json'
EVAL_SEED = 65011  # Same as the repaired 72-case reference evaluation.
MIN_DISK = 20_000_000_000
TRAIN_WALL = 5400
EVAL_WALL = 1500
GPU_WAIT = 7*3600
SHARED_GPU_LOCK = ROOT.parent/'sircus-isaac/scene.lock'


def gpu(query):
    return subprocess.check_output(['nvidia-smi', '-i', GPU, query, '--format=csv,noheader,nounits'],
                                   text=True, timeout=10).strip()


def freeze(prepared, path):
    assert not path.exists()
    prep = read(prepared/'preparation.json')
    assert prep['passed'] and not prep['fresh_holdout_used'] and prep['baseline_clean'] == 72
    files = [Path(__file__), HERE/'prepare_repaired_sitting_ppo.py', LAUNCHER, MODEL, RESERVATION,
             *(HERE/'trainer').glob('*.py'), *(prepared/'development').iterdir(),
             prepared/'demonstrations.npz', prepared/'preparation.json',
             *(prepared/f'initial_{seed}.pt' for seed in SEEDS),
             *(ROOT/'scripts'/f'run_{n}.py' for n in ('resting_posture', 'supported_postures', 'supported_transition')),
             *(ROOT/'sitting_task/src/humanoid_sitting').rglob('*.py')]
    for name, digest in prep['source_inputs'].items():
        assert sha(name) == digest, name
    write(path, dict(files={str(p): sha(p) for p in files}, gpu=GPU, seeds=SEEDS,
                     evaluation_seed=EVAL_SEED, updates_per_seed=UPDATES, training_cases=72,
                     training_active_transitions_per_seed=UPDATES*72*1000,
                     training_optimized_controls_per_case_per_update=1000-HOLD_TICK,
                     training_warmup_transitions_per_seed=UPDATES*72*25,
                     maximum_active_transitions=2*UPDATES*72*1000+2*72*1000+2*24*1000,
                     maximum_warmup_transitions=2*UPDATES*72*25+2*72*25+2*24*25,
                     physics_substeps_per_control=4, simulation_processes=6,
                     process_initialization_physics_steps_max=1000,
                     training_wall_s=TRAIN_WALL, evaluation_wall_s=EVAL_WALL,
                     gpu_wait_s=GPU_WAIT, shared_gpu_lock=str(SHARED_GPU_LOCK),
                     minimum_free_output_bytes=MIN_DISK, gpu_temperature_max_C=85,
                     output_filesystem=str(ROOT/'outputs'), fresh_bank_read_after_selection=True,
                     checkpoint_selection='Final weights ranked by clean development count, then final-five-second squared torque on cases clean for both seeds, then seed ID.',
                     holdout_policy='Development-selected trained checkpoint and its zero-residual initial checkpoint, fixed before opening bank.',
                     action_limits_unchanged=True, physical_criteria_unchanged=True,
                     success='Complete standing-start 20 seconds, no physical failure, no slip or illegal contact, and at least 5 seconds of continuously qualifying final support.'))


def open_holdout(development, destination):
    reserved = read(RESERVATION)
    source = DATA/'banks/holdout'
    assert reserved['holdout_simulations'] == 0
    assert reserved['fresh_holdout_excluded_from_repair_calibration_training_selection']
    for name in ('geometry.json', 'selection.json', 'reset_candidates.npz', 'candidates.npz'):
        p = source/name
        assert sha(p) == reserved['hashes'][str(p)], p
    geometry = read(source/'geometry.json')
    assert [[g['height'], g['pitch_deg']] for g in geometry] == reserved['fresh_holdout']
    selection = read(source/'selection.json')
    assert selection['standing_only'] and selection['indices'] == [0]*12
    with np.load(source/'reset_candidates.npz', allow_pickle=False) as z:
        if 'validated' in z:
            assert z['validated'][:, 0].all()
        resets = {f'reset_{k}': np.repeat(z[k][:, 0], 2, axis=0) for k in RESET_FIELDS}
    # Match the original two-replica bank layout. The evaluator's tracking
    # reward needs a reference array; use a constant standing pose for it.
    # Neither the reward-only reference nor any reserved target reaches PPO.
    resets.update(q_post=np.repeat(resets['reset_q'][None], 1000, axis=0),
                  reference=np.arange(24), admitted=np.ones(24, bool),
                  phase_times=np.full((24, 2), 15.))
    rows = [dict(g, case=2*i+replica, surface_id=i, start_stage=0)
            for i, g in enumerate(geometry) for replica in range(2)]
    meta = read(development/'metadata.json')
    meta.pop('demonstration_audit_sha256')
    meta.pop('source_audits')
    ids = [f'holdout/{i}/{j}' for i in range(12) for j in range(2)]
    destination.mkdir()
    np.savez_compressed(destination/'bank.npz', **resets)
    write(destination/'metadata.json', dict(meta, cases=24, ids=ids, positive_demonstrations=0,
                                            fresh_holdout_used=True, reward_is_diagnostic_only=True))
    write(destination/'geometry.json', rows)
    write(destination/'reservation_audit.json', dict(
        original_files={str(p): reserved['hashes'][str(p)] for p in source.iterdir() if p.is_file()},
        geometry_pairs=reserved['fresh_holdout'], replicas_per_geometry=2, standing_only=True))


def report(path, expected_ids):
    score = evaluate(path)
    rows = read(path/'results.json')['episodes']
    assert [r['id'] for r in rows] == expected_ids
    results = dict(clean=score['clean_count'], total=score['rows'],
                   stable=sum(r['stable'] for r in rows),
                   slips=[r['id'] for r in rows if r['slip_ticks']],
                   illegal_contacts=[r['id'] for r in rows if r['illegal_ticks']],
                   physical_failures=[r['id'] for r in rows if any(r['physical_failures'])],
                   failed=[r['id'] for r in rows if not r['clean']])
    return results


def effort_and_authority(result, reference, expected_rows):
    joint_names = read(result/'results.json')['joint_names']
    legs = [i for i, name in enumerate(joint_names) if any(part in name for part in ('hip', 'knee', 'ankle'))]
    upper = [i for i in range(len(joint_names)) if i not in legs]
    assert len(legs) == 12 and len(joint_names) == 37
    with np.load(result/'trajectory.npz') as learned, np.load(reference/'trajectory.npz') as prior:
        assert learned['desired'].shape == (1000, expected_rows, 23)
        max_shift = np.max(np.abs(learned['desired'][HOLD_TICK:]-prior['desired'][HOLD_TICK:]))
        assert max_shift > 0 and np.isfinite(max_shift)
        prefix = {k: bool(np.array_equal(learned[k][:HOLD_TICK], prior[k][:HOLD_TICK]))
                  for k in ('desired', 'command', 'good', 'legal', 'no_slip', 'failure')}
        totals = {}
        for z, label in ((learned, 'learned'), (prior, 'reference')):
            torque = z['physical_torque'].astype(np.float64)
            assert torque.shape[-1] == 37
            totals[label] = {}
            for seconds in (20, 10, 5):
                window = torque[-50*seconds:]
                totals[label][str(seconds)] = dict(
                    squared=np.square(window).sum(-1).mean(0).tolist(),
                    absolute=np.abs(window).sum(-1).mean(0).tolist(),
                    rms=np.sqrt(np.square(window).mean(-1)).mean(0).tolist(),
                    legs_squared=np.square(window[..., legs]).sum(-1).mean(0).tolist(),
                    upper_squared=np.square(window[..., upper]).sum(-1).mean(0).tolist())
    return dict(prefix_exact=prefix, max_hold_desired_difference_rad=float(max_shift), effort=totals)


@contextmanager
def gpu_slot(status):
    start = time.monotonic()
    lock_paths = (SHARED_GPU_LOCK, DATA/'contact_parallel_gpu_v1/transition.lock',
                  DATA/'contact_ppo_scientific_owner.lock')
    while time.monotonic()-start < GPU_WAIT:
        with ExitStack() as locks:
            try:
                for path in lock_paths:
                    lock = locks.enter_context(path.open('r'))
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                busy = gpu('--query-compute-apps=pid')
            except BlockingIOError:
                busy = 'shared GPU lock'
            if not busy:
                yield
                return
        write(status, dict(state='waiting_for_4090', owner_pid=os.getpid(),
                           wait_s=time.monotonic()-start, reason=busy))
        time.sleep(15)
    raise TimeoutError('4090 remained busy through the bounded wait')


def run(prepared, output, frozen_path, resume=False):
    frozen = read(frozen_path)
    for name, digest in frozen['files'].items():
        assert sha(name) == digest, ('FROZEN_INPUT_CHANGED', name)
    if resume:
        original = read(output/'freeze.json')
        for name, digest in original['files'].items():
            assert sha(name) == digest, ('ORIGINAL_FROZEN_INPUT_CHANGED', name)
        assert {Path(name).name: digest for name, digest in original['files'].items()
                if Path(name).parent.name == 'trainer'} == {
                    p.name: sha(p) for p in (HERE/'trainer').glob('*.py')}
        assert all(frozen[k] == v for k, v in original.items() if k != 'files')
        assert (output/'train_63111.exit.json').is_file()
        assert not (output/'train_63117').exists()
        assert not (output/'development_63111').exists()
        assert not (output/'reserved_bank').exists()
        shutil.copy2(frozen_path, output/'continuation_freeze.json')
    else:
        output.mkdir(exist_ok=False)
        shutil.copy2(frozen_path, output/'freeze.json')
    assert shutil.disk_usage(output.parent).free > MIN_DISK
    with gpu_slot(output/'status.json'):
        development = prepared/'development'
        dev_ids = read(development/'metadata.json')['ids']

        def stage(label, bank, checkpoint, seed, train=False):
            assert not gpu('--query-compute-apps=pid'), '4090_BUSY'
            assert shutil.disk_usage(output).free > MIN_DISK
            assert int(gpu('--query-gpu=temperature.gpu')) < 85
            args = ['bash', str(LAUNCHER), str(HERE/'trainer/run_training.py'),
                    '--mode', 'train' if train else 'evaluate', '--bank', str(bank),
                    '--output', str(output/label), '--checkpoint', str(checkpoint), '--seed', str(seed),
                    '--device', 'cuda:0', '--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
            if train:
                args += ['--updates', str(UPDATES), '--demonstrations', str(prepared/'demonstrations.npz')]
            write(output/f'{label}.command.json', args)
            write(output/'status.json', dict(state=label, owner_pid=os.getpid()))
            start = time.monotonic()
            with (output/f'{label}.log').open('x') as log:
                child = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                try:
                    while child.poll() is None:
                        assert time.monotonic()-start < (TRAIN_WALL if train else EVAL_WALL), ('TIMEOUT', label)
                        assert shutil.disk_usage(output).free > MIN_DISK, 'DISK_FREE'
                        assert int(gpu('--query-gpu=temperature.gpu')) < 85, 'GPU_TEMPERATURE'
                        time.sleep(5)
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait(timeout=20)
                    write(output/f'{label}.exit.json',
                          dict(returncode=child.returncode, wall_s=time.monotonic()-start))
            assert child.returncode == 0, ('CHILD_FAILED', label, child.returncode)
            return output/label

        def comparison(path, reference, rows, ids):
            measured = report(path, ids)
            measured.update(effort_and_authority(path, reference, rows))
            return measured

        results = {}
        for seed in SEEDS:
            initial = prepared/f'initial_{seed}.pt'
            if resume and seed == SEEDS[0]:
                training = output/f'train_{seed}'
                assert read(output/f'train_{seed}.exit.json')['returncode'] == 0
                trained = read(training/'final_audit.json')
                assert trained['passed'] and trained['checkpoint_sha256'] == sha(training/'final.pt')
            else:
                training = stage(f'train_{seed}', development, initial, seed, train=True)
                trained = weights(training, 'final.pt')
            accounting = read(training/'results.json')
            assert trained['updates'] == UPDATES
            assert accounting['active_transitions'] == frozen['training_active_transitions_per_seed']
            assert accounting['warmup_transitions'] == frozen['training_warmup_transitions_per_seed']
            assert len(accounting['episodes']) == UPDATES*72
            path = stage(f'development_{seed}', development, training/'final.pt', EVAL_SEED)
            results[seed] = comparison(path, BASELINE, 72, dev_ids)
            results[seed]['weights'] = trained

        common = np.array([not any(i in r['failed'] for r in results.values()) for i in dev_ids], bool)
        selected = min(SEEDS, key=lambda seed: (
            -results[seed]['clean'],
            float(np.mean(np.array(results[seed]['effort']['learned']['5']['squared'])[common])) if common.any() else 0.,
            seed))
        checkpoint = output/f'train_{selected}/final.pt'
        initial = prepared/f'initial_{selected}.pt'
        receipt = dict(selected_seed=selected, selected_checkpoint_sha256=sha(checkpoint),
                       initial_checkpoint_sha256=sha(initial), development=results,
                       development_regressions=results[selected]['failed'], all_72_retained=results[selected]['clean']==72,
                       common_development_clean=int(common.sum()), fresh_holdout_opened=False, promoted=False)
        write(output/'development_selection.json', receipt)

        held = output/'reserved_bank'
        open_holdout(development, held)
        ids = read(held/'metadata.json')['ids']
        reference = stage('holdout_reference', held, initial, EVAL_SEED)
        reference_score = report(reference, ids)
        learned = stage('holdout_selected', held, checkpoint, EVAL_SEED)
        learned_score = comparison(learned, reference, 24, ids)
        assert sha(checkpoint) == receipt['selected_checkpoint_sha256']
        assert sha(initial) == receipt['initial_checkpoint_sha256']
        write(output/'comparison.json', dict(
            receipt, fresh_holdout_opened=True, holdout_reference=reference_score, holdout_learned=learned_score,
            holdout_gains=sorted(set(reference_score['failed'])-set(learned_score['failed'])),
            holdout_regressions=sorted(set(learned_score['failed'])-set(reference_score['failed']))))
        write(output/'status.json', dict(state='complete', selected_seed=selected,
                                        development_clean=results[selected]['clean'],
                                        heldout_clean=learned_score['clean'], promoted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'run'))
    parser.add_argument('prepared', type=Path)
    parser.add_argument('freeze', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.prepared.resolve(), args.freeze.resolve())
    elif args.output:
        run(args.prepared.resolve(), args.output.resolve(), args.freeze.resolve(), resume=args.resume)
    else:
        parser.error('--output is required for run')
