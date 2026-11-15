"""Evaluate a fresh-case repair with the frozen scorer on the original banks."""

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np

from repair_fresh_sitting import FRESH, SPLITS
from run_repaired_sitting_ppo import (
    EVAL_SEED, EVAL_WALL, GPU, HERE, LAUNCHER, MIN_DISK, ROOT, gpu, gpu_slot, read, report, sha, write,
)


def run(prepared, out, fresh_only, positioned=False):
    construction = read(prepared/'construction.json')
    checkpoint = Path(construction['checkpoint'])
    library = prepared/'library.npz'
    assert sha(checkpoint) == construction['checkpoint_sha256']
    assert sha(library) == construction['library_sha256']
    targets = {row['target'] for row in construction['design']}
    # Reuse the original runtime and scorer with the audited routing extension.
    positioned_base = ROOT/'outputs/positioned_sitting_20261114_v1'
    original = read((positioned_base if positioned else FRESH)/'freeze.json')['sources']
    runtime = [LAUNCHER, *(HERE/'trainer').glob('*.py'),
               *(ROOT/'scripts'/f'run_{name}.py' for name in ('supported_postures','supported_transition','resting_posture')),
               *(ROOT/'sitting_task/src/humanoid_sitting').rglob('*.py')]
    trainer = {Path(name).name:value for name,value in original.items() if Path(name).parent.name == 'trainer'}
    for path in runtime:
        if path == HERE/'trainer/policy.py':
            continue
        digest = trainer[path.name] if path.parent == HERE/'trainer' else original[str(path)]
        assert sha(path) == digest, path
    frozen = {**construction['inputs'], **{str(p):sha(p) for p in
              [Path(__file__),prepared/'construction.json',library,HERE/'run_repaired_sitting_ppo.py',*runtime]}}
    stages = {}
    splits = SPLITS
    screen = 'fresh'
    if positioned:
        prior = ROOT/'outputs/fresh_sitting_repair_20261114_v2_full'
        splits = {k:prior/k for k in SPLITS}
        splits['positioned'] = positioned_base/'evaluation'
        screen = 'positioned'
    for label,prior in ({screen:splits[screen]} if fresh_only else splits).items():
        bank = Path(read(prior/'manifest.json')['args']['bank'])
        for name in ('bank.npz','metadata.json','geometry.json'):
            frozen[str(bank/name)] = sha(bank/name)
        stages[label] = ['bash',str(LAUNCHER),str(HERE/'trainer/run_training.py'),'--mode','evaluate',
                         '--bank',str(bank),'--output',str(out/label),'--checkpoint',str(checkpoint),
                         '--candidate-library',str(library),'--seed',str(EVAL_SEED),'--device','cuda:0',
                         '--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
    out.mkdir(exist_ok=False)
    write(out/'freeze.json',dict(inputs=frozen,commands=stages,gpu=GPU,controls_per_trial=1000,
                                 warmup_controls=25,wall_s_per_stage=EVAL_WALL))
    scores = {}
    with gpu_slot(out/'status.json'):
        for name,digest in frozen.items():
            assert sha(name) == digest, name
        for label,args in stages.items():
            assert not gpu('--query-compute-apps=pid')
            assert shutil.disk_usage(out).free > MIN_DISK
            assert int(gpu('--query-gpu=temperature.gpu')) < 85
            write(out/'status.json',dict(state=label,owner_pid=os.getpid()))
            start = time.monotonic()
            with (out/f'{label}.log').open('x') as log:
                child = subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
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
            assert child.returncode == 0, label
            folder = out/label
            prior = splits[label]
            old = read(prior/'results.json')['episodes']
            ids = [r['id'] for r in old]
            measured = report(folder,ids)
            results = read(folder/'results.json')
            assert results['active_transitions'] == 1000*len(ids)
            assert results['warmup_transitions'] == 25*len(ids)
            rows = results['episodes']
            keep = [i for i,r in enumerate(old) if r['clean']]
            with np.load(folder/'trajectory.npz') as a, np.load(prior/'trajectory.npz') as b:
                measured['protected_exact'] = {
                    k:bool(np.array_equal(a[k][:,keep],b[k][:,keep]))
                    for k in ('desired','command','good','legal','no_slip','failure')}
                slip, wrong = a['loaded_slip'], a['wrong_force']
            measured.update(repairs=[r['id'] for r in rows if r['clean'] and r['id'] in targets],
                            regressions=[rows[i]['id'] for i in keep if not rows[i]['clean']],
                            cases=[dict(id=r['id'],passed=r['clean'],hold_s=round(r['final_hold_s'],2),
                                        peak_loaded_slip_m_s=float(slip[:,i].max()),
                                        peak_wrong_contact_N=float(wrong[:,i].max()),
                                        slip_ticks=r['slip_ticks'],failure_flags=r['physical_failures'])
                                   for i,r in enumerate(rows)])
            scores[label] = measured
    assert sha(checkpoint) == construction['checkpoint_sha256']
    write(out/'comparison.json',dict(splits=scores,action_limits_unchanged=True,success_criteria_unchanged=True,
                                     ppo_weights_unchanged=True,original_fresh_cases_used_for_repair=True,
                                     original_positioned_cases_used_for_repair=positioned))
    write(out/'status.json',dict(state='complete',clean=sum(s['clean'] for s in scores.values()),
                                 total=sum(s['total'] for s in scores.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--fresh-only',action='store_true')
    parser.add_argument('--positioned',action='store_true')
    args = parser.parse_args()
    run(args.prepared.resolve(),args.output.resolve(),args.fresh_only,args.positioned)
