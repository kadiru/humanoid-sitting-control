"""One frozen 24-case evaluation of the repaired sitting policy."""

import argparse
import csv
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np
from scipy.spatial.transform import Rotation

from repair_sitting_touchdown import STUDY
from run_repaired_sitting_ppo import (
    DATA, EVAL_SEED, EVAL_WALL, GPU, HERE, LAUNCHER, MIN_DISK, MODEL, ROOT,
    gpu, gpu_slot, read, report, sha, write,
)

HEIGHTS_MM = (361, 372, 381, 392, 408, 437)
SLOPES_DEG = (6.5, 10.75, 15.5, 19.25)
CHECKPOINT_SHA = 'b5c5c44c584b83b1e072527db65ff515841cfaf7e1bdda36f49118c5cdaea6d3'
REPAIR_SHA = 'a855d988d1802de26c1ecaca9c8f306d2113a2b91e79318642658381642eddc6'


def prepare_bank(bank):
    prior = STUDY/'reserved_bank'
    dev = Path(read(STUDY/'development_63111/manifest.json')['args']['bank'])
    catalogues = [dev/'geometry.json', prior/'geometry.json',
                  *sorted((DATA/'banks').glob('*/geometry.json'))]
    old = {(round(float(g['height'])*1000, 6), float(g['pitch_deg']))
           for path in catalogues for g in read(path) if 'pitch_deg' in g}
    grid = [(h, s) for h in HEIGHTS_MM for s in SLOPES_DEG]
    assert len(set(grid)) == 24 and not set(grid) & old
    geometry = []
    template = read(prior/'geometry.json')[0]
    for i, (height, slope) in enumerate(grid):
        angle = Rotation.from_euler('y', slope, degrees=True)
        row = {k: template[k] for k in ('kind', 'family', 'x', 'y', 'yaw', 'depth', 'width')}
        geometry.append(dict(row, case=i, surface_id=i, start_stage=0,
                             height=height/1000, pitch_deg=slope,
                             top_center=[*template['top_center'][:2], height/1000],
                             rotation_xyzw=angle.as_quat().tolist(), normal=angle.as_matrix()[:, 2].tolist(),
                             split='fresh_test', label=f'fresh/{i+1:02d}'))
    with np.load(prior/'bank.npz', allow_pickle=False) as z:
        reset = {k: np.repeat(z[k][:1], 24, axis=0) for k in z.files if k.startswith('reset_')}
    # Standing pose is a placeholder for the diagnostic reward. The trained
    # policy and the physical success predicates do not use reward references.
    reset.update(q_post=np.broadcast_to(reset['reset_q'], (1000,24,37)), reference=np.arange(24),
                 admitted=np.ones(24,bool), phase_times=np.full((24,2),15.))
    meta = read(prior/'metadata.json')
    bank.mkdir()
    write(bank/'geometry.json',geometry)
    write(bank/'metadata.json',dict(meta, ids=[g['label'] for g in geometry],
                                   evaluation_split='fresh_test', positive_demonstrations=0))
    np.savez_compressed(bank/'bank.npz',**reset)
    return geometry, [*catalogues, prior/'metadata.json', prior/'bank.npz']


def run(library, out):
    selection = read(STUDY/'development_selection.json')
    checkpoint = STUDY/f"train_{selection['selected_seed']}/final.pt"
    assert sha(checkpoint) == CHECKPOINT_SHA and sha(library) == REPAIR_SHA
    out.mkdir(exist_ok=False)
    bank = out/'bank'
    geometry, inputs = prepare_bank(bank)
    destination = out/'evaluation'
    args = ['bash',str(LAUNCHER),str(HERE/'trainer/run_training.py'),'--mode','evaluate',
            '--bank',str(bank),'--output',str(destination),'--checkpoint',str(checkpoint),
            '--candidate-library',str(library),'--seed',str(EVAL_SEED),'--device','cuda:0',
            '--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
    sources = [Path(__file__),HERE/'run_repaired_sitting_ppo.py',LAUNCHER,MODEL,checkpoint,library,
               *bank.iterdir(),*inputs,*(HERE/'trainer').glob('*.py'),
               *(ROOT/'scripts'/f'run_{n}.py' for n in ('resting_posture','supported_postures','supported_transition')),
               *(ROOT/'sitting_task/src/humanoid_sitting').rglob('*.py')]
    frozen = {str(p): sha(p) for p in sources}
    write(out/'freeze.json',dict(sources=frozen, command=args, gpu=GPU, cases=24, controls=1000,
                                 warmup_controls=25, wall_s=EVAL_WALL, policy_tuning=False))
    with gpu_slot(out/'status.json'):
        for path,digest in frozen.items():
            assert sha(path) == digest, path
        assert not gpu('--query-compute-apps=pid')
        assert shutil.disk_usage(out).free > MIN_DISK
        assert int(gpu('--query-gpu=temperature.gpu')) < 85
        write(out/'status.json',dict(state='evaluation',owner_pid=os.getpid()))
        start = time.monotonic()
        with (out/'evaluation.log').open('x') as log:
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
                write(out/'exit.json',dict(returncode=child.returncode,wall_s=time.monotonic()-start))
        assert child.returncode == 0
    ids = [g['label'] for g in geometry]
    score = report(destination,ids)
    results = read(destination/'results.json')
    assert results['active_transitions'] == 24000 and results['warmup_transitions'] == 600
    rows = [dict(id=g['label'],height_cm=round(g['height']*100,3),slope_deg=g['pitch_deg'],
                 passed=r['clean'],hold_s=round(r['final_hold_s'],2),slip_ticks=r['slip_ticks'],
                 illegal_ticks=r['illegal_ticks'],
                 physical_failures=[name for name,value in zip(
                     ('fall','forbidden','torque','impact','joint_limit'),r['physical_failures']) if value])
            for g,r in zip(geometry,results['episodes'])]
    write(out/'comparison.json',dict(score=score,rows=rows,policy_unchanged=True,
                                     action_limits_unchanged=True,success_criteria_unchanged=True))
    with (out/'results.csv').open('x',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(out/'status.json',dict(state='complete',clean=score['clean'],total=24))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    run(args.library.resolve(),args.output.resolve())
