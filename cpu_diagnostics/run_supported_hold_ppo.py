"""Run one bounded 4090 lean PPO comparison with original full-72 evaluation."""

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'trainer'))
from audit import evaluate
from common import DATA, ROOT, read, sha, write


GPU = 'GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
TRAINER = HERE/'trainer/run_training.py'
LAUNCHER = ROOT/'scripts/run_supported_gpu0.sh'
SITTING = ROOT/'outputs/full_body_contact_imitation_v4/dataset/sitting'
PROTECTED = ROOT/'experiments/motion_prior_ppo_20261029_v1/protected_sitting_cases.json'
UPDATES = 128
SEED = 65011


def gpu_field(query, field):
    return subprocess.check_output(['nvidia-smi', '-i', GPU, f'--query-{query}={field}',
                                    '--format=csv,noheader,nounits'], text=True, timeout=10).strip()


def freeze(prepared):
    assert read(prepared/'trainer_preflight.json')['passed']
    prep = read(prepared/'preparation.json')
    assert prep['passed'] and prep['new_demonstrations'] == 6 and not prep['fresh_holdout_used']
    paths = [Path(__file__), HERE/'prepare_supported_hold_ppo.py', LAUNCHER, PROTECTED,
             *(HERE/'trainer').glob('*.py'), *(p for p in prepared.rglob('*') if p.is_file()),
             *(SITTING/k for k in ('bank.npz', 'metadata.json', 'geometry.json')),
             *(ROOT/'scripts'/f'run_{name}.py' for name in ('resting_posture', 'supported_postures', 'supported_transition')),
             *(ROOT/'sitting_task/src/humanoid_sitting').rglob('*.py')]
    out = prepared.parent/'ppo_freeze.json'
    if out.exists():
        raise FileExistsError(out)
    write(out, dict(files={str(p): sha(p) for p in paths}, gpu=GPU, seed=SEED, updates=UPDATES,
                    standing_start=True, leaning_baseline=3, leaning_final=3, sitting_final=72,
                    training_approach_controls_max=128*max(prep['residual_start_ticks']),
                    training_update_controls=8192, training_reset_warmup_max=128*25,
                    physics_substeps_per_control=4, each_evaluation_s=3600, training_wall_s=30000,
                    resource_disk_bytes=20_000_000_000, resource_temperature_C=85))


def run(prepared, out):
    frozen_path = prepared.parent/'ppo_freeze.json'
    frozen = read(frozen_path)
    for path, digest in frozen['files'].items():
        assert sha(path) == digest, ('FROZEN_INPUT_CHANGED', path)
    locks = [(DATA/name).open('a') for name in ('contact_parallel_gpu_v1/transition.lock',
                                                'contact_ppo_scientific_owner.lock')]
    try:
        for lock in locks:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not gpu_field('compute-apps', 'pid'), '4090_BUSY'
        out.mkdir(exist_ok=False)
        write(out/'receipt.json', dict(owner_pid=os.getpid(), freeze_sha256=sha(frozen_path)))

        def stage(label, bank, checkpoint, *, train=False):
            assert not gpu_field('compute-apps', 'pid'), '4090_BUSY'
            args = ['bash', str(LAUNCHER), str(TRAINER), '--mode', 'train' if train else 'evaluate',
                    '--bank', str(bank), '--output', str(out/label), '--checkpoint', str(checkpoint),
                    '--seed', str(SEED), '--device', 'cuda:0',
                    '--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
            if train:
                args += ['--updates', str(UPDATES), '--demonstrations', str(prepared/'demonstrations.npz')]
            write(out/(label+'.command.json'), args)
            started = time.monotonic()
            with (out/(label+'.log')).open('x') as log:
                child = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                try:
                    while child.poll() is None:
                        assert time.monotonic()-started < (30000 if train else 3600), ('TIMEOUT', label)
                        assert shutil.disk_usage(out).free > 20_000_000_000, 'DISK_FREE'
                        assert int(gpu_field('gpu', 'temperature.gpu')) < 85, 'GPU_TEMPERATURE'
                        time.sleep(5)
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait(timeout=20)
                    write(out/(label+'.exit.json'), dict(exit_code=child.returncode, wall_s=time.monotonic()-started))
            assert child.returncode == 0, ('CHILD_FAILED', label, child.returncode)
            return None if train else evaluate(out/label)

        lean = prepared/'leaning_train'
        initial = prepared/'initial.pt'
        baseline = stage('prior_leaning3', lean, initial)
        assert baseline['clean_count'] > 0, 'NO_CLEAN_CAMERA_BASELINE'
        stage('training', lean, initial, train=True)
        checkpoint = out/'training/final.pt'
        final = stage('learned_leaning3', lean, checkpoint)
        sitting = stage('sitting72', SITTING, checkpoint)
        assert sitting['rows'] == 72
        rows = read(out/'sitting72/results.json')['episodes']
        protected = read(PROTECTED)
        clean = {r['id'] for r in rows if r['clean']}
        assert {r['id'] for r in rows} == set(protected['protected']+protected['failed'])
        write(out/'comparison.json', dict(leaning_prior=baseline['clean'], leaning_learned=final['clean'],
                                         sitting_clean=len(clean), sitting_total=72,
                                         regressions=sorted(set(protected['protected'])-clean),
                                         repairs=sorted(clean & set(protected['failed'])),
                                         weights=read(out/'training/final_audit.json'), promoted=False))
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'run'))
    parser.add_argument('prepared', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.prepared)
    elif args.output:
        run(args.prepared, args.output)
    else:
        parser.error('--output is required for a run')
