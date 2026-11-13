"""Validate a touchdown correction on the original banks with the frozen PPO."""

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from repair_sitting_touchdown import STUDY, TARGETS
from run_repaired_sitting_ppo import EVAL_SEED, EVAL_WALL, GPU, HERE, LAUNCHER, MIN_DISK, MODEL, gpu, gpu_slot, read, report, sha, write, ROOT


def evaluate(prepared, out, screen):
    construction = read(prepared/'construction.json')
    library = prepared/'library.npz'
    assert sha(library) == construction['library_sha256']
    checkpoint = Path(construction['checkpoint'])
    assert sha(checkpoint) == construction['checkpoint_sha256']
    for path, digest in construction['inputs'].items():
        assert sha(path) == digest, path
    prior_bank = STUDY/'reserved_bank'
    training_bank = Path(read(STUDY/'development_63111/manifest.json')['args']['bank'])
    out.mkdir(exist_ok=False)
    bank = prior_bank
    if screen:
        indices = construction['targets']
        bank = out/'screen_bank'
        bank.mkdir()
        meta = read(prior_bank/'metadata.json')
        geometry = read(prior_bank/'geometry.json')
        write(bank/'metadata.json',dict(meta,cases=len(indices),ids=[meta['ids'][i] for i in indices]))
        write(bank/'geometry.json',[dict(geometry[i],case=j,surface_id=j) for j,i in enumerate(indices)])
        with np.load(prior_bank/'bank.npz',allow_pickle=False) as z:
            arrays = {k:(v[:,indices] if k == 'q_post' else v[indices]) for k,v in z.items()}
        arrays['reference'] = np.arange(len(indices))
        np.savez_compressed(bank/'bank.npz',**arrays)
    stages = [('test', bank, STUDY/'holdout_selected')] if screen else [
        ('development', training_bank, STUDY/'development_63111'),
        ('test', prior_bank, STUDY/'holdout_selected')]
    files = [Path(__file__), HERE/'repair_sitting_touchdown.py', HERE/'prepare_sitting_repairs.py',
             HERE/'run_repaired_sitting_ppo.py', LAUNCHER, MODEL,
             checkpoint, library, prepared/'construction.json', *(HERE/'trainer').glob('*.py'),
             *(ROOT/'scripts'/f'run_{n}.py' for n in ('resting_posture','supported_postures','supported_transition')),
             *(ROOT/'sitting_task/src/humanoid_sitting').rglob('*.py')]
    commands = {}
    for label, bank, prior in stages:
        files.extend(bank/k for k in ('bank.npz','metadata.json','geometry.json'))
        commands[label] = ['bash',str(LAUNCHER),str(HERE/'trainer/run_training.py'),'--mode','evaluate',
                           '--bank',str(bank),'--output',str(out/label),'--checkpoint',str(checkpoint),
                           '--candidate-library',str(library),'--seed',str(EVAL_SEED),'--device','cuda:0',
                           '--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
    write(out/'freeze.json',dict(sources={str(p):sha(p) for p in files},commands=commands,gpu=GPU,
                               controls_per_stage=1025,wall_s_per_stage=EVAL_WALL,screen=screen))
    results = {}
    with gpu_slot(out/'status.json'):
        for name,digest in read(out/'freeze.json')['sources'].items():
            assert sha(name) == digest, name
        for label, bank, prior in stages:
            assert not gpu('--query-compute-apps=pid')
            assert int(gpu('--query-gpu=temperature.gpu')) < 85
            assert shutil.disk_usage(out).free > MIN_DISK
            write(out/'status.json',dict(state=label,owner_pid=os.getpid()))
            start = time.monotonic()
            with (out/f'{label}.log').open('x') as log:
                child = subprocess.Popen(commands[label],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    while child.poll() is None:
                        assert time.monotonic()-start < EVAL_WALL
                        assert shutil.disk_usage(out).free > MIN_DISK
                        assert int(gpu('--query-gpu=temperature.gpu')) < 85
                        time.sleep(5)
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:
                            child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL)
                            child.wait(timeout=20)
                    write(out/f'{label}.exit.json',dict(returncode=child.returncode,wall_s=time.monotonic()-start))
            assert child.returncode == 0
            ids = read(bank/'metadata.json')['ids']
            measured = report(out/label,ids)
            old = {r['id']:r for r in read(prior/'results.json')['episodes'] if r['id'] in ids}
            rows = read(out/label/'results.json')['episodes']
            measured.update(repairs=[r['id'] for r in rows if r['clean'] and not old[r['id']]['clean']],
                            regressions=[r['id'] for r in rows if not r['clean'] and old[r['id']]['clean']],
                            holds_s={r['id']:r['final_hold_s'] for r in rows})
            if not screen:
                protected = np.array([True]*len(ids))
                if label == 'test':
                    protected[list(TARGETS)] = False
                with np.load(out/label/'trajectory.npz') as a, np.load(prior/'trajectory.npz') as b:
                    measured['protected_exact'] = {k:bool(np.array_equal(a[k][:,protected], b[k][:,protected]))
                                                   for k in ('desired','command','good','legal','no_slip','failure')}
            results[label] = measured
    write(out/'comparison.json',dict(results=results, used_original_test_for_repair=True,
                                     action_limits_unchanged=True, success_criteria_unchanged=True,
                                     ppo_checkpoint_unchanged=True))
    write(out/'status.json',dict(state='complete'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('prepared',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--screen',action='store_true')
    a = p.parse_args()
    evaluate(a.prepared.resolve(),a.out.resolve(),a.screen)
