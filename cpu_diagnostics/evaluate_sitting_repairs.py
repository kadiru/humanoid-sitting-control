"""Run one frozen 72-case sitting repair comparison on the RTX 4090."""

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'trainer'))
from audit import evaluate
from common import DATA, ROOT, read, sha, write
from prepare_sitting_repairs import BASELINE, DATASET, MODEL, TARGETS


GPU = 'GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
CHECKPOINT = ROOT/'outputs/hold_effort_ppo_20261113_v1/training/final.pt'
CAP = 3600


def gpu(query):
    return subprocess.check_output(['nvidia-smi', '-i', GPU, query, '--format=csv,noheader,nounits'],
                                   text=True, timeout=10).strip()


def launch(prepared, out, cpu_results=None):
    candidate = prepared/'candidate_paths.npy'
    construction = read(prepared/'construction.json')
    assert sha(candidate) == construction['candidate_sha256']
    assert sha(MODEL) == construction['model_sha256'] and construction['targets'] == TARGETS
    source = prepared.parent/'source/prepare_sitting_repairs.py'
    assert sha(source) == construction['source_sha256']
    assert sha(BASELINE/'trajectory.npz') == construction['baseline_sha256']
    assert sha(CHECKPOINT) == read(CHECKPOINT.with_name('final_audit.json'))['checkpoint_sha256']
    launcher = ROOT/'scripts/run_supported_gpu0.sh'
    config = [Path(__file__), source, launcher, candidate, CHECKPOINT,
              prepared/'construction.json', *(HERE/'trainer').glob('*.py'),
              *(DATASET/n for n in ('bank.npz', 'metadata.json', 'geometry.json')),
              *(ROOT/'scripts'/f'run_{name}.py' for name in ('resting_posture','supported_postures','supported_transition')),
              *(ROOT/'sitting_task/src/humanoid_sitting').rglob('*.py')]
    args = ['bash', str(launcher), str(HERE/'trainer/run_training.py'), '--mode', 'evaluate',
            '--bank', str(DATASET), '--output', str(out/'full72'), '--checkpoint', str(CHECKPOINT),
            '--candidate-library', str(out/'repair_library.npz'), '--seed', '65011', '--device', 'cuda:0',
            '--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
    with (DATA/'contact_ppo_scientific_owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not gpu('--query-compute-apps=pid'), '4090_BUSY'
        out.mkdir(exist_ok=False)
        signatures = np.load(prepared/'signatures.npy', allow_pickle=False)
        scale = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)['sitting']['signature_scale'].numpy()
        error = np.square((signatures[:, None]-signatures[None, TARGETS])/scale).mean(-1)
        protected = [i for i in range(72) if i not in TARGETS]
        radius = error[protected].min(0)/4
        nearest = error.argmin(1)
        assert np.flatnonzero(error[np.arange(72), nearest] < radius[nearest]).tolist() == TARGETS
        paths = np.load(candidate, allow_pickle=False)
        np.testing.assert_array_equal(paths[:, np.array(TARGETS)[nearest[TARGETS]]], paths[:, TARGETS])
        active_targets = TARGETS
        if cpu_results:
            screen = read(cpu_results)
            assert screen['all_completed']
            eligible = {r['case_id'] for r in screen['rows'] if r['clean']}
            ids = read(DATASET/'metadata.json')['ids']
            active_targets = [i for i in TARGETS if ids[i] in eligible]
            assert len(active_targets) == len(eligible) > 0
            config.append(cpu_results)
        active_slots = [TARGETS.index(i) for i in active_targets]
        library = out/'repair_library.npz'
        np.savez(library, repair_signature=signatures[active_targets], repair_radius=radius[active_slots],
                 repair_desired=paths[:, active_targets].transpose(1, 0, 2).copy())
        config += [library, prepared/'signatures.npy']
        write(out/'freeze.json', dict(inputs={str(p): sha(p) for p in config}, command=args, gpu=GPU,
                                     max_wall_s=CAP, max_control_steps=1025, max_physics_steps=5100,
                                     training_updates=0, only_development=True, active_targets=active_targets))
        started = time.monotonic()
        with (out/'simulator.log').open('x') as log:
            child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
            try:
                while child.poll() is None:
                    assert time.monotonic()-started < CAP, 'TIMEOUT'
                    assert shutil.disk_usage(out).free > 20_000_000_000, 'DISK_FREE'
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
                write(out/'process.json', dict(returncode=child.returncode, wall_s=time.monotonic()-started))
    assert child.returncode == 0
    score(out, candidate, active_targets)


def score(out, candidate, active_targets):
    episode = out/'full72'
    audit = evaluate(episode)
    baseline = read(BASELINE/'results.json')['episodes']
    final = read(episode/'results.json')['episodes']
    assert audit['rows'] == len(baseline) == len(final) == 72
    assert [r['id'] for r in baseline] == [r['id'] for r in final]
    protected = [i for i in range(72) if i not in TARGETS]
    with np.load(BASELINE/'trajectory.npz', allow_pickle=False) as old, np.load(episode/'trajectory.npz', allow_pickle=False) as new:
        desired = old['desired'].copy()
        desired[:, active_targets] = np.load(candidate, allow_pickle=False)[:, active_targets]
        routing_exact = np.array_equal(new['desired'], desired)
        commands_exact = np.array_equal(new['command'][:, protected], old['command'][:, protected])
        flags_exact = all(np.array_equal(new[key][:, protected], old[key][:, protected])
                          for key in ('good', 'no_slip', 'legal', 'failure', 'clean_completion'))
    write(out/'comparison.json', dict(clean=audit['clean_count'], total=72,
                                      repairs=[r['id'] for i,r in enumerate(final) if not baseline[i]['clean'] and r['clean']],
                                      regressions=[r['id'] for i,r in enumerate(final) if baseline[i]['clean'] and not r['clean']],
                                      targets=[final[i] for i in TARGETS], protected_commands_exact=commands_exact,
                                      protected_physical_flags_exact=flags_exact, intended_routing_exact=routing_exact,
                                      sensor_selected=True, active_targets=active_targets, promoted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cpu-results', type=Path)
    args = parser.parse_args()
    launch(args.prepared, args.output, args.cpu_results)
