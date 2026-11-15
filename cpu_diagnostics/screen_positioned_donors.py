"""Compare donor paths on the unresolved shifted seats before policy routing."""

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from repair_fresh_sitting import POSITIONED, checked_trace
from evaluate_fresh_sitting import CHECKPOINT_SHA, POSITIONED_REPAIR_SHA, REPAIR_RUN
from repair_sitting_touchdown import HOLD_TICK, STUDY, retarget, retime
from prepare_sitting_repairs import BodyModel, MODEL, encode, leg_fk, smoothstep
from run_repaired_sitting_ppo import EVAL_SEED, EVAL_WALL, HERE, LAUNCHER, MIN_DISK, ROOT, gpu, gpu_slot, read, report, sha, write
from common import OBS
from policy import Learner, command


class DiagnosticLearner(Learner):
    def prior(self,obs,details=False):
        if self.paths.device != obs.device:
            self.paths = self.paths.to(obs.device)
        tick = (obs[:,335]*1000).round().long().clamp(0,999)
        rows = torch.arange(len(obs),device=obs.device)
        latent = self.paths[rows,tick]
        if details:
            _, choice, distance = super().prior(obs,True)
            return latent,choice,distance
        return latent


def worker(paths, argv):
    import run_training
    assert '--mode' in argv and argv[argv.index('--mode')+1]=='evaluate'
    values = torch.as_tensor(np.load(paths,allow_pickle=False))
    assert values.shape[1:] == (1000,23) and torch.isfinite(values).all()
    DiagnosticLearner.paths = values
    run_training.Learner = DiagnosticLearner
    sys.argv = [sys.argv[0],*argv]
    run_training.main()


def prepare(library, plan, out, refinement=None):
    torch.set_num_threads(1)
    checkpoint = STUDY/'train_63111/final.pt'
    assert sha(checkpoint)==CHECKPOINT_SHA and sha(library)==POSITIONED_REPAIR_SHA
    inputs = {str(p):sha(p) for p in (MODEL,library,plan,Path(__file__),checkpoint,HERE/'repair_sitting_touchdown.py')}
    folders = [*(REPAIR_RUN/k for k in ('development','test','fresh')),POSITIONED/'evaluation']
    bundles = [checked_trace(p,inputs) for p in folders]
    rows = {r['id']:(results,g[i],z,i) for results,g,z in bundles for i,r in enumerate(results['episodes'])}
    old = Learner(np.zeros(OBS),np.ones(OBS),EVAL_SEED)
    old.load(checkpoint)
    with np.load(library,allow_pickle=False) as z:
        old.bind_sitting({**old.sitting,**dict(z)})
    initial = torch.from_numpy(np.stack([z['obs'][0,i] for _,_,z,i in rows.values()]))
    with torch.no_grad():
        priors = []
        for tick in range(1000):
            initial[:,335] = tick/1000
            priors.append(old.prior(initial).numpy())
    priors = dict(zip(rows,np.stack(priors,axis=1)))
    bank = POSITIONED/'bank'
    meta = read(bank/'metadata.json')
    model = BodyModel(MODEL)
    fk, legs = leg_fk(model,meta['joint_names'])
    limits = [np.asarray(meta[k]) for k in ('lower','upper')]
    lo, hi = [torch.from_numpy(x.astype(np.float32)) for x in limits]
    priorities = read(plan)
    if refinement:
        source_plan = read(refinement/'probe.json')
        source_file = refinement/'paths.npy'
        assert sha(source_file) == read(refinement/'freeze.json')['inputs'][str(source_file)]
        previous_paths = np.load(source_file,allow_pickle=False)
        for p in (source_file,refinement/'probe.json',refinement/'freeze.json'):
            inputs[str(p)] = sha(p)
    paths, geoms, receipts, indices, rejected = [], [], [], [], []
    for target,choices in priorities.items():
        _, target_geo, trace, i = rows[target]
        assert not rows[target][0]['episodes'][i]['clean']
        for choice in choices:
            if isinstance(choice,dict):
                donor, shift = choice.get('donor'), choice.get('shift',0.)
                pitch = choice.get('pitch',0.)
            else:
                donor, shift = choice
                pitch = 0.
            if donor is None:
                record = choice.get('probe')
                if record:
                    probe_row = next(j for j,p in enumerate(source_plan['probes']) if p['id']==record and p['target']==target)
                    base = previous_paths[probe_row]
                else:
                    base = priors[target]
                start = trace['previous'][0,i]
                desired = base.copy()
                if 'speed' in choice:
                    controls = np.repeat(start[None],1000,axis=0)
                    previous = torch.from_numpy(start[:23].copy())
                    for tick,row in enumerate(base):
                        previous = command(previous,torch.from_numpy(row),lo,hi)
                        controls[tick,:23] = previous.numpy()
                    latent, applied, goal, checks = retime(controls,start,choice['first_load_tick'],
                                                           choice['last_slip_tick'],limits,choice['speed'])
                    error = float(abs(applied-goal[:,:23]).max())
                    if error > .015:
                        rejected.append(dict(target=target,choice=choice,reason=f'command error {error}'))
                        continue
                    desired[checks['start_tick']:HOLD_TICK] = latent[checks['start_tick']:HOLD_TICK]
                else:
                    tick = np.arange(1000)
                    center = choice['peak_tick']
                    lead, tail = choice.get('lead_ticks',30), choice.get('tail_ticks',9)
                    pulse = smoothstep((tick-(center-lead))/24)*(1-smoothstep((tick-(center+tail))/24))
                    desired[:,[15,16]] += (choice['ankle_rad']*pulse[:,None]).astype(np.float32)
                checks = dict(method='touchdown_perturbation',variant=choice)
                donor = source_plan['probes'][probe_row]['donor'] if record else target
                shift = None
            else:
                results, donor_geo, source, k = rows[donor]
                assert results['episodes'][k]['clean']
                if shift is None:
                    desired = priors[donor]
                    checks = dict(method='replay')
                else:
                    pose = source['body'][:,k,0].copy()
                    pose[:,:3] -= np.asarray(results['origins'][k])
                    motion = dict(body=pose,q=source['q'][:,k],load=source['support_bw'][:,k])
                    waypoint = source['command'][:,k].astype(float)
                    base = torch.from_numpy(source['previous'][HOLD_TICK,k,:23].copy())
                    for tick in range(HOLD_TICK,1000):
                        base = command(base,torch.from_numpy(priors[donor][tick]),lo,hi)
                        waypoint[tick,:23] = base.numpy()
                    planning_surface = target_geo
                    if isinstance(choice,dict) and 'approach_height_offset_m' in choice:
                        center = target_geo['top_center'].copy()
                        center[2] += choice['approach_height_offset_m']
                        planning_surface = dict(target_geo,top_center=center)
                    try:
                        waypoint, checks = retarget(motion,donor_geo,planning_surface,limits,waypoint,model,fk,legs,
                                                    fore_aft_m=shift,pitch_adjust_rad=pitch,
                                                    pitch_window_ticks=choice.get('pitch_window_ticks') if isinstance(choice,dict) else None)
                        desired, applied = encode(waypoint,trace['previous'][0,i],*limits)
                        error = float(abs(applied-waypoint[:,:23]).max())
                        if error > .015:
                            t,j = np.unravel_index(np.abs(applied-waypoint[:,:23]).argmax(),applied.shape)
                            raise ValueError((error,int(t),meta['joint_names'][j],float(waypoint[t,j]),float(applied[t,j])))
                    except (AssertionError,ValueError) as exc:
                        rejected.append(dict(target=target,donor=donor,fore_aft_m=shift,pitch_rad=pitch,reason=str(exc)))
                        continue
                    checks.update(method='retarget',command_error_rad=error,pitch_rad=pitch,variant=choice)
            ids = len(paths)
            paths.append(desired)
            geoms.append(dict(target_geo,case=ids,surface_id=ids))
            indices.append(i)
            receipts.append(dict(id=f'probe/{ids+1:02d}',target=target,donor=donor,fore_aft_m=shift,**checks))
    assert 1 <= len(paths) <= 72
    assert {r['target'] for r in receipts} == set(priorities), rejected
    out.mkdir(exist_ok=False)
    new = out/'bank'
    new.mkdir()
    with np.load(bank/'bank.npz',allow_pickle=False) as z:
        arrays = {name:(z[name][:,indices] if name=='q_post' else z[name][indices]) for name in z.files}
    arrays['reference'] = np.arange(len(paths))
    np.savez(new/'bank.npz',**arrays)
    write(new/'geometry.json',geoms)
    write(new/'metadata.json',dict(meta,cases=len(paths),ids=[r['id'] for r in receipts],
                                   evaluation_split='diagnostic_donor_screen'))
    np.save(out/'paths.npy',np.stack(paths))
    write(out/'probe.json',dict(case_index_uses_direct_diagnostic_path=True,probes=receipts,
                                rejected_offline=rejected,inputs=inputs))
    return checkpoint


def screen(library,plan,out,refinement):
    checkpoint = prepare(library,plan,out,refinement)
    command_line = ['bash',str(LAUNCHER),str(Path(__file__).resolve()),'worker',str(out/'paths.npy'),
                    '--mode','evaluate','--bank',str(out/'bank'),'--output',str(out/'evaluation'),
                    '--checkpoint',str(checkpoint),'--seed',str(EVAL_SEED),'--device','cuda:0',
                    '--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
    study = read(out/'probe.json')
    frozen = {**study['inputs'],**{str(p):sha(p) for p in (LAUNCHER,out/'probe.json',out/'paths.npy',
             *(out/'bank').iterdir(),*(HERE/'trainer').glob('*.py'),
             *(ROOT/'scripts'/f'run_{n}.py' for n in ('supported_postures','supported_transition','resting_posture')),
             *(ROOT/'sitting_task/src/humanoid_sitting').rglob('*.py'))}}
    write(out/'freeze.json',dict(inputs=frozen,command=command_line,
                                 wall_s=EVAL_WALL,controls_per_trial=1000,warmup_controls=25))
    with gpu_slot(out/'status.json'):
        for name,digest in frozen.items():
            assert sha(name)==digest
        assert not gpu('--query-compute-apps=pid')
        assert int(gpu('--query-gpu=temperature.gpu')) < 85 and shutil.disk_usage(out).free > MIN_DISK
        write(out/'status.json',dict(state='diagnostic_screen',owner_pid=os.getpid()))
        start = time.monotonic()
        with (out/'simulation.log').open('x') as log:
            child = subprocess.Popen(command_line,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while child.poll() is None:
                    assert time.monotonic()-start < EVAL_WALL
                    assert int(gpu('--query-gpu=temperature.gpu')) < 85 and shutil.disk_usage(out).free > MIN_DISK
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
        assert child.returncode==0
    probes = study['probes']
    report(out/'evaluation',[r['id'] for r in probes])
    rows = read(out/'evaluation/results.json')['episodes']
    write(out/'comparison.json',[dict(r,clean=m['clean'],hold_s=m['final_hold_s'],slip_ticks=m['slip_ticks'],
                                     illegal_ticks=m['illegal_ticks'],flags=m['physical_failures']) for r,m in zip(probes,rows)])
    write(out/'status.json',dict(state='complete',probes=len(rows),clean=sum(r['clean'] for r in rows)))


if __name__ == '__main__':
    if sys.argv[1:2]==['worker']:
        worker(Path(sys.argv[2]),sys.argv[3:])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('library',type=Path)
        parser.add_argument('plan',type=Path)
        parser.add_argument('output',type=Path)
        parser.add_argument('--refine-from',type=Path)
        a = parser.parse_args()
        screen(a.library.resolve(),a.plan.resolve(),a.output.resolve(),a.refine_from.resolve() if a.refine_from else None)
