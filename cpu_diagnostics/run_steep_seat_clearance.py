"""Screen both clearance paths on the 4090, then audit all 72 starts."""

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


HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'trainer'))
from audit import evaluate
from common import DATA, ROOT, read, sha, write
from evaluate_sitting_repairs import CHECKPOINT, DATASET, GPU, gpu
from prepare_steep_seat_clearance import CASES, PRIOR


CAP = 1200


def subset(out):
    out.mkdir()
    meta = read(DATASET/'metadata.json')
    geometry = read(DATASET/'geometry.json')
    with np.load(DATASET/'bank.npz',allow_pickle=False) as bank:
        names = ('reset_q','reset_qd','reset_command','reset_root_pose','reset_root_velocity','phase_times','admitted')
        arrays = {k:bank[k][CASES] for k in names}
        arrays['q_post'] = bank['q_post'][:,bank['reference'][CASES]]
    arrays['reference'] = np.arange(len(CASES))
    np.savez_compressed(out/'bank.npz',**arrays)
    write(out/'metadata.json',dict(meta,cases=len(CASES),ids=[meta['ids'][i] for i in CASES],positive_demonstrations=0))
    write(out/'geometry.json',[geometry[i] for i in CASES])


def run(prepared, output, full):
    plan = read(prepared/'construction.json')
    library = prepared/'library.npz'
    assert sha(library) == plan['library_sha256']
    for path, digest in plan['source_inputs'].items():
        assert sha(path) == digest, path
    with (DATA/'contact_ppo_scientific_owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not gpu('--query-compute-apps=pid'), '4090_BUSY'
        output.mkdir(exist_ok=False)
        bank = DATASET if full else output/'bank'
        if not full:
            subset(bank)
        launcher = ROOT/'scripts/run_supported_gpu0.sh'
        args = ['bash',str(launcher),str(HERE/'trainer/run_training.py'),'--mode','evaluate','--bank',str(bank),
                '--output',str(output/'evaluation'),'--checkpoint',str(CHECKPOINT),'--candidate-library',str(library),
                '--seed','65011','--device','cuda:0','--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
        sources = [Path(__file__),launcher,prepared/'construction.json',library,CHECKPOINT,
                   *(HERE/'trainer').glob('*.py'),*(bank/k for k in ('bank.npz','metadata.json','geometry.json')),
                   *(ROOT/'scripts'/f'run_{n}.py' for n in ('resting_posture','supported_postures','supported_transition')),
                   *(ROOT/'sitting_task/src/humanoid_sitting').rglob('*.py')]
        write(output/'freeze.json',dict(command=args,gpu=GPU,maximum_wall_s=CAP,maximum_controls=1025,
                                       full=full,inputs={str(p):sha(p) for p in sources}))
        begun = time.monotonic()
        with (output/'simulator.log').open('x') as log:
            child = subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
            try:
                while child.poll() is None:
                    assert time.monotonic()-begun < CAP, 'TIMEOUT'
                    assert shutil.disk_usage(output).free > 20_000_000_000, 'DISK_FREE'
                    assert int(gpu('--query-gpu=temperature.gpu')) < 85, 'TEMPERATURE'
                    time.sleep(5)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL)
                        child.wait(timeout=20)
                write(output/'process.json',dict(returncode=child.returncode,wall_s=time.monotonic()-begun))
    assert child.returncode == 0
    folder = output/'evaluation'
    audit = evaluate(folder)
    rows = read(folder/'results.json')['episodes']
    with np.load(library,allow_pickle=False) as routes, np.load(folder/'trajectory.npz',allow_pickle=False) as trace:
        target_rows = CASES if full else [0,1]
        np.testing.assert_array_equal(trace['desired'][:,target_rows],routes['repair_desired'][-2:].transpose(1,0,2))
        protected = [i for i in range(72) if i not in CASES]
        exact = None
        if full:
            with np.load(PRIOR/'full72/trajectory.npz',allow_pickle=False) as old:
                exact = {k:bool(np.array_equal(trace[k][:,protected],old[k][:,protected])) for k in (
                    'desired','command','good','no_slip','legal','failure','clean_completion')}
    prior = read(PRIOR/'full72/results.json')['episodes']
    write(output/'comparison.json',dict(total=len(rows),clean=audit['clean_count'],targets=[rows[i] for i in target_rows],
                                       repairs=[r['id'] for r in rows if r['clean'] and r['id'] in {prior[i]['id'] for i in CASES}],
                                       regressions=[r['id'] for r in rows if not r['clean'] and r['id'] in {prior[i]['id'] for i in range(72) if i not in CASES}],
                                       protected_exact=exact,training_updates=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--full',action='store_true')
    args=parser.parse_args()
    run(args.prepared,args.output,args.full)
