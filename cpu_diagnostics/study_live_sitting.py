"""Matched seat shifts and a sealed bank for live sitting approach validation."""

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from run_repaired_sitting_ppo import (
    EVAL_SEED, EVAL_WALL, GPU, HERE, LAUNCHER, MIN_DISK, MODEL, ROOT,
    gpu, gpu_slot, read, report, sha, write,
)


BASE = ROOT/'outputs/positioned_repair_20261114_full'
PREPARED = ROOT/'.cpu-work/20261114/positioned_repair_v1/final_prepared'
BASE_IDS = ('fresh/01', 'fresh/08', 'fresh/14', 'fresh/24')
SHIFTS = ((0.,0.),(-.025,0.),(.025,0.),(0.,-.04),(0.,.04))
FRESH_HEIGHTS = (.359,.369,.383,.398,.413,.435)
FRESH_SLOPES = (7.,11.75,16.75,18.5)


def bank_from_geometry(path, source, rows):
    meta = read(source/'metadata.json')
    n = len(rows)
    assert len({tuple(g['top_center']+[g['pitch_deg']]) for g in rows}) == n
    path.mkdir()
    with np.load(source/'bank.npz',allow_pickle=False) as z:
        states = {k:np.repeat(z[k][:1],n,axis=0) for k in z.files if k.startswith('reset_')}
    states.update(q_post=np.broadcast_to(states['reset_q'],(1000,n,37)),reference=np.arange(n),
                  admitted=np.ones(n,bool),phase_times=np.full((n,2),15.))
    write(path/'geometry.json',[dict(g,case=i,surface_id=i,start_stage=0) for i,g in enumerate(rows)])
    write(path/'metadata.json',dict(meta,cases=n,ids=[g['label'] for g in rows],
                                   positive_demonstrations=0,reference_is_reward_only=True))
    np.savez_compressed(path/'bank.npz',**states)


def geometry(template, label, height, pitch, dx, dy):
    rot = Rotation.from_euler('y',pitch,degrees=True)
    xyz = [-.4+dx,dy,height]
    return dict(template,label=label,top_center=xyz,x=xyz[0],y=xyz[1],height=height,
                pitch_deg=pitch,rotation_xyzw=rot.as_quat().tolist(),normal=rot.as_matrix()[:,2].tolist(),
                fore_aft_offset_cm=100*dx,lateral_offset_cm=100*dy)


def reserve(out):
    out.mkdir(exist_ok=False)
    bases = {k:Path(read(BASE/k/'manifest.json')['args']['bank'])
             for k in ('development','test','fresh','positioned')}
    previous = read(bases['fresh']/'geometry.json')
    templates = {g['label']:g for g in previous}
    matched = [geometry(templates[key],f'matched/{i+1:02d}/{j+1:02d}',
                         templates[key]['height'],templates[key]['pitch_deg'],dx,dy)
               for i,key in enumerate(BASE_IDS) for j,(dx,dy) in enumerate(SHIFTS)]
    fresh = [geometry(previous[0],f'live_holdout/{4*i+j+1:02d}',height,pitch,
                      (-.021,-.008,.013,.024)[(i+j)%4],(-.035,.018,.035,-.018)[(2*i+j)%4])
             for i,height in enumerate(FRESH_HEIGHTS) for j,pitch in enumerate(FRESH_SLOPES)]
    past = {(g['height'],g['pitch_deg']) for bank in bases.values() for g in read(bank/'geometry.json')}
    assert not past.intersection((g['height'],g['pitch_deg']) for g in fresh)
    bank_from_geometry(out/'matched',bases['fresh'],matched)
    bank_from_geometry(out/'reserved',bases['fresh'],fresh)
    names = ('bank.npz','geometry.json','metadata.json')
    write(out/'reservation.json',dict(matched=20,reserved=24,standing_starts=True,
          files={str(p):sha(p) for bank in ('matched','reserved') for p in (out/bank/k for k in names)}))


def stage(study, label, bank, library, *, worker=None, extra=(), protocol=None, inputs=(),
          disk_reserve=MIN_DISK):
    prep = read(PREPARED/'construction.json')
    checkpoint = Path(prep['checkpoint'])
    assert sha(checkpoint)==prep['checkpoint_sha256'] and sha(library)==prep['library_sha256']
    receipt = read(study/'reservation.json')
    for p,digest in receipt['files'].items():
        # The reserved input is verified only when its stage is authorized.
        if Path(p).parent == bank:
            assert sha(p)==digest
    if bank==study/'reserved':
        assert protocol is not None and worker is not None
        gate = read(protocol)
        assert gate['development_regression_passed'] and gate['matched_improved']
        assert sha(worker)==gate['policy_source_sha256']
        assert sha(extra[1])==gate['live_payload_sha256']
    out = study/label
    args = ['bash',str(LAUNCHER),str(worker or HERE/'trainer/run_training.py'),*extra,
            '--mode','evaluate','--bank',str(bank),'--output',str(out),'--checkpoint',str(checkpoint),
            '--candidate-library',str(library),'--seed',str(EVAL_SEED),'--device','cuda:0',
            '--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
    sources = [Path(__file__),LAUNCHER,MODEL,checkpoint,library,*bank.iterdir(),*inputs,*(HERE/'trainer').glob('*.py'),
               *(ROOT/'scripts'/f'run_{n}.py' for n in ('supported_postures','supported_transition','resting_posture')),
               *(ROOT/'sitting_task/src/humanoid_sitting').rglob('*.py')]
    if worker:
        sources.append(worker)
    if protocol is not None:
        sources.append(protocol)
    frozen = {str(p):sha(p) for p in sources}
    write(study/f'{label}.freeze.json',dict(inputs=frozen,command=args,gpu=GPU,
          controls_per_case=1000,warmup_controls=25,wall_seconds=EVAL_WALL,disk_reserve_bytes=disk_reserve))
    with gpu_slot(study/'status.json'):
        for p,digest in frozen.items():
            assert sha(p)==digest,p
        assert not gpu('--query-compute-apps=pid')
        assert int(gpu('--query-gpu=temperature.gpu'))<85 and shutil.disk_usage(study).free>disk_reserve
        write(study/'status.json',dict(state=label,owner_pid=os.getpid()))
        start = time.monotonic()
        with (study/f'{label}.log').open('x') as log:
            child = subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while child.poll() is None:
                    assert time.monotonic()-start<EVAL_WALL
                    assert int(gpu('--query-gpu=temperature.gpu'))<85 and shutil.disk_usage(study).free>disk_reserve
                    time.sleep(5)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL)
                        child.wait(timeout=20)
                write(study/f'{label}.exit.json',dict(returncode=child.returncode,wall_s=time.monotonic()-start))
        assert child.returncode==0,label
    ids = read(bank/'metadata.json')['ids']
    result = report(out,ids)
    rows = read(out/'results.json')
    assert rows['active_transitions']==len(ids)*1000 and rows['warmup_transitions']==len(ids)*25
    write(study/f'{label}.score.json',result)
    write(study/'status.json',dict(state=f'{label}_complete',clean=result['clean'],total=result['total']))
    return result


def test_live(study, prepared):
    payload = prepared/'live_library.npz'
    receipt = prepared/'live_inputs.json'
    assert sha(payload)==read(receipt)['payload_sha256']
    edge_audit = study/'rear_edge_baseline/edge_audit.npz'
    with np.load(edge_audit,allow_pickle=False) as z:
        early = z['values'][z['ticks']<=310]
        error = (early[:,:,0]-early[:,:,1])*1000
        valid = np.isfinite(error)
        assert valid.sum(0).min() >= 4, 'INSUFFICIENT_LIVE_EDGE_COVERAGE'
        absolute = np.abs(error[valid])
        assert np.percentile(absolute,90) <= 10., 'LIVE_EDGE_ERROR_EXCEEDS_STUDY_TOLERANCE'
    write(study/'edge_validation.json',dict(valid_frames=int(valid.sum()),frames=int(valid.size),
                 p90_absolute_error_mm=float(np.percentile(absolute,90)),
                 max_absolute_error_mm=float(absolute.max()),development_only=True,passed=True))
    library = PREPARED/'library.npz'
    source = HERE/'live_sitting_edge.py'
    extra = ['--adapt-library',str(payload)]
    inputs = [payload,receipt,edge_audit,study/'edge_validation.json']
    base = read(study/'baseline.score.json')
    candidate = stage(study,'candidate',study/'matched',library,worker=source,extra=extra,inputs=inputs)
    old = set(base['failed'])
    new = set(candidate['failed'])
    compare = dict(baseline_clean=base['clean'],candidate_clean=candidate['clean'],
                   matched_improved=len(old)>len(new) and new<=old,
                   repaired=sorted(old-new),regressions=sorted(new-old))
    write(study/'matched_comparison.json',compare)
    if not compare['matched_improved']:
        return
    retained = {}
    for name in ('development','test','fresh','positioned'):
        bank = Path(read(BASE/name/'manifest.json')['args']['bank'])
        retained[name] = stage(study,f'regression_{name}',bank,library,worker=source,extra=extra,inputs=inputs)
    verified = all(r['clean']==r['total'] for r in retained.values())
    protocol = study/'gate.json'
    write(protocol,dict(compare,development_regression_passed=verified,
                         retained={k:r['clean'] for k,r in retained.items()},
                         total=sum(r['total'] for r in retained.values()),
                         policy_source_sha256=sha(source),live_payload_sha256=sha(payload)))
    if verified:
        fresh_once(study,prepared)


def fresh_once(study, prepared):
    payload = prepared/'live_library.npz'
    receipt = prepared/'live_inputs.json'
    stage(study,'fresh_once',study/'reserved',PREPARED/'library.npz',worker=HERE/'live_sitting_edge.py',
          extra=['--adapt-library',str(payload)],
          inputs=[payload,receipt,study/'rear_edge_baseline/edge_audit.npz',study/'edge_validation.json'],
          protocol=study/'gate.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('study',type=Path)
    p.add_argument('--reserve',action='store_true')
    p.add_argument('--test-live',type=Path)
    p.add_argument('--fresh-once',type=Path)
    a = p.parse_args()
    path = a.study.resolve()
    if a.fresh_once:
        fresh_once(path,a.fresh_once.resolve())
        sys.exit(0)
    if a.test_live:
        test_live(path,a.test_live.resolve())
        sys.exit(0)
    if a.reserve:
        reserve(path)
    stage(path,'baseline',path/'matched',PREPARED/'library.npz',worker=HERE/'live_sitting_edge.py')
