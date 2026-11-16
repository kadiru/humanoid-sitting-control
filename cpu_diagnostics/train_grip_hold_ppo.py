"""Validate the revised hold credit on saved traces and retrain in situ."""

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np
import torch

from compare_touchdown_ppo import BASELINE, verify_audit
from grip_hold_ppo import ENTRY_UTILIZATION, GRIP_LIMIT, HOLD_TICK, hold_credit, seat_state
from repair_live_touchdown import CHECKPOINT, HERE, MODEL, PRIOR, ROOT, read, sha, write
from run_repaired_sitting_ppo import GPU, LAUNCHER, MIN_DISK, gpu, gpu_slot
from validate_camera_touchdown import fallback_priors

from audit import weights
from common import OBS
from policy import Learner, command


SEED = 63111
OLD_COMPARISON = ROOT/'outputs/touchdown_feedback_ppo_comparison_20261115'


def saved_credit(folder,indices):
    fields = ('force_w','normal_w','friction','bodyweight_N','rest_patch','permitted')
    with np.load(folder/'trajectory.npz',allow_pickle=False) as trace:
        saved = {k:trace['physical_'+k][700:,indices].copy() for k in fields}
        q = trace['physical_joint_velocity'][HOLD_TICK:,indices]
        power = np.abs(trace['physical_torque'][HOLD_TICK:,indices]*q).sum(-1)
        good = trace['good'][HOLD_TICK:,indices]
    force,normal,weight = (saved[k] for k in ('force_w','normal_w','bodyweight_N'))
    fn = (force*normal).sum(-1)
    tangent = np.linalg.norm(force-fn[...,None]*normal,axis=-1)
    active = np.linalg.norm(force,axis=-1)>.01*weight[...,None]
    capacity = np.maximum(saved['friction']*np.maximum(fn,0),1e-7)
    grip = np.where(active&(tangent>1e-7),tangent/capacity,0.)[:,:,:5].max(-1)
    load = (fn.clip(min=0)*saved['rest_patch']*saved['permitted']).sum(-1)/weight
    production = {k:torch.as_tensor(v.reshape((-1,*v.shape[2:]))) for k,v in saved.items()}
    production_grip,production_load = seat_state(production)
    np.testing.assert_allclose(production_grip.numpy().reshape(grip.shape),grip,rtol=3e-6,atol=1e-6)
    np.testing.assert_allclose(production_load.numpy().reshape(load.shape),load,rtol=3e-6,atol=1e-6)
    eligible = good & (load[50:]>=.05)
    credit = .18*np.clip(1-grip[50:]/.90,0,1)*eligible
    effort = .12*.50/(.50+power)*eligible
    risk = .90*np.clip((grip[50:]-.65)/.25,0,2)**2
    actual = hold_credit(*map(torch.as_tensor,(grip[50:],load[50:],power,good)))
    for computed,expected in zip(actual,(credit,effort,risk),strict=True):
        np.testing.assert_allclose(computed.numpy(),expected,rtol=3e-6,atol=1e-6)
    discount = .999**np.arange(250)
    return (grip[:50].max(0),grip[50:].max(0),power.mean(0),
            (discount[:,None]*(credit+effort-risk)).sum(0))


def audit_saved_regressions():
    comparison = OLD_COMPARISON/'comparison.json'
    baseline = read(OLD_COMPARISON/'comparison_protocol.json')['baseline']
    inputs = {str(comparison):sha(comparison)}
    comparisons = []
    for label,rows in read(comparison)['cases'].items():
        if label=='untouched':
            continue
        indices = [i for i,r in enumerate(rows) if r['touchdown_clean'] and not r['ppo_clean']]
        if not indices:
            continue
        before,after = Path(baseline[label]),OLD_COMPARISON/label
        for folder in (before,after):
            audit,_ = verify_audit(folder)
            inputs.update({str(folder/name):digest for name,digest in audit['files'].items()})
        a,b = saved_credit(before,indices),saved_credit(after,indices)
        for j,index in enumerate(indices):
            assert a[0][j]<2 and a[0][j]>=ENTRY_UTILIZATION
            assert a[1][j]<=GRIP_LIMIT and b[1][j]>GRIP_LIMIT
            assert a[3][j]>b[3][j]
            comparisons.append(dict(id=rows[index]['id'],entry_peak=float(a[0][j]),
                 grip_peak_before_and_after=[float(a[1][j]),float(b[1][j])],
                 hold_power_before_and_after_w=[float(a[2][j]),float(b[2][j])],
                 hold_credit_before_and_after=[float(a[3][j]),float(b[3][j])]))
    assert len(comparisons)==25
    return dict(passed=True,known_regressions=25,fresh_used=False,cases=comparisons,inputs=inputs)


def prepare(out,bank_label,updates):
    assert bank_label in ('feedback_bank','test','candidate','development','positioned','target_bank')
    grade = audit_saved_regressions()
    gate = BASELINE/'ppo_gate.json'
    assert read(gate)['passed']
    baseline = Path(read(gate)['evidence'][bank_label])
    audit,rows = verify_audit(baseline)
    bank_path = Path(read(baseline/'manifest.json')['args']['bank'])
    metadata = read(bank_path/'metadata.json')
    assert all(row['clean'] for row in rows)
    ids = [row['id'] for row in rows]
    assert ids==metadata['ids']
    with np.load(baseline/'trajectory.npz',allow_pickle=False) as z:
        obs = z['obs']
        q = z['q']
        desired = fallback_priors(z,list(range(len(ids))))
        applied = z['command'][:HOLD_TICK,:,:23]
        good = z['good']
        load = z['support_bw']
    assert good[HOLD_TICK-1].all()
    phases = [[.02*(np.flatnonzero(load[:,i]>=.05)[0]+1),
               .02*(1000-np.cumprod(good[::-1,i]).sum()+1)] for i in range(len(ids))]
    with np.load(bank_path/'bank.npz',allow_pickle=False) as z:
        reset = {k:z[k] for k in z.files if k.startswith('reset_')}
    warm = torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    agent = Learner(warm['mean'],warm['scale'],SEED)
    for key in ('actor','critic'):
        getattr(agent,key).load_state_dict(warm[key])
    agent.bind_library(warm['library'])
    with np.load(PRIOR,allow_pickle=False) as z:
        sitting = {**warm['sitting'],**dict(z)}
    signature = obs[0,:,336:].copy()
    dist = np.square((signature[:,None]-signature)/np.asarray(sitting['signature_scale'])).mean(-1)
    np.fill_diagonal(dist,np.inf)
    assert dist.min()>0
    radius = np.minimum(dist.min(1)/16,5e-7).astype(np.float32)
    agent.bind_sitting(dict(sitting,positioned_signature=signature,positioned_desired=desired,
                           positioned_radius=radius))
    assert int(agent.sitting['hold_start_tick'])==HOLD_TICK
    limits = [torch.tensor(metadata[key]) for key in ('lower','upper')]
    previous = torch.from_numpy(reset['reset_command'][:,:23].copy())
    max_error = 0.
    with torch.no_grad():
        for tick in range(HOLD_TICK):
            sensor = torch.from_numpy(obs[tick].copy())
            assert not agent.residual_mask(sensor).any()
            replay = agent.prior(sensor)
            torch.testing.assert_close(replay,torch.from_numpy(desired[:,tick]),atol=0,rtol=0)
            previous = command(previous,replay,*limits)
            max_error = max(max_error,float((previous-torch.from_numpy(applied[tick])).abs().max()))
    assert max_error<=2e-7
    out.mkdir()
    reward_audit = out/'saved_reward_audit.json'
    write(reward_audit,grade)
    bank = out/'training_bank'
    bank.mkdir()
    np.savez_compressed(bank/'bank.npz',**reset,q_post=q,reference=np.arange(len(ids)),
                        admitted=np.ones(len(ids),bool),phase_times=np.asarray(phases,np.float32))
    (bank/'geometry.json').write_bytes((bank_path/'geometry.json').read_bytes())
    write(bank/'metadata.json',dict(metadata,positive_demonstrations=len(ids)))
    initial = out/'initial.pt'
    agent.save(initial)
    demos = out/'demonstrations.npz'
    np.savez_compressed(demos,obs=obs[HOLD_TICK:].reshape(-1,OBS))
    worker = HERE/'grip_hold_ppo.py'
    sources = [Path(__file__),worker,gate,CHECKPOINT,MODEL,PRIOR,initial,demos,reward_audit,LAUNCHER,
               *bank.iterdir(),*bank_path.iterdir(),*(HERE/'trainer').glob('*.py'),
               *(ROOT/'scripts'/f'run_{n}.py' for n in ('supported_postures','supported_transition','resting_posture')),
               *(ROOT/'sitting_task/src/humanoid_sitting').rglob('*.py')]
    inputs = {str(p):sha(p) for p in sources}
    inputs.update({str(baseline/name):digest for name,digest in audit['files'].items()})
    inputs.update(grade['inputs'])
    args = ['bash',str(LAUNCHER),str(worker),'train','--mode','train','--bank',str(bank),
            '--output',str(out/'training'),'--checkpoint',str(initial),'--demonstrations',str(demos),
            '--updates',str(updates),'--seed',str(SEED),'--device','cuda:0',
            '--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
    write(out/'freeze.json',dict(inputs=inputs,command=args,gpu=GPU,updates=updates,
         replay_max_error_rad=max_error,layout=bank_label,training_cases=len(ids),
         fresh_used_in_training=False,hold_start_tick=HOLD_TICK,wall_s=min(updates*200+300,7*3600)))


def train(out):
    freeze = read(out/'freeze.json')
    with gpu_slot(out/'status.json'):
        for path,digest in freeze['inputs'].items():
            assert sha(path)==digest,path
        assert int(gpu('--query-gpu=temperature.gpu'))<85 and shutil.disk_usage(out).free>MIN_DISK
        write(out/'status.json',dict(state='training',owner_pid=os.getpid()))
        start = time.monotonic()
        with (out/'training.log').open('x') as log:
            child = subprocess.Popen(freeze['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                while child.poll() is None:
                    assert time.monotonic()-start<freeze['wall_s']
                    assert int(gpu('--query-gpu=temperature.gpu'))<85 and shutil.disk_usage(out).free>MIN_DISK
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
    audit = weights(out/'training','final.pt')
    assert audit['updates']==freeze['updates']
    write(out/'status.json',dict(state='complete',**audit))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--bank',default='test')
    parser.add_argument('--updates',type=int,default=24)
    args = parser.parse_args()
    assert 1<=args.updates<=128
    output = args.out.resolve()
    torch.set_num_threads(1)
    prepare(output,args.bank,args.updates)
    train(output)
