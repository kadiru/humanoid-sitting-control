"""Collect simulated option outcomes and fit staged and shared sitting actors."""
import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from run import DISK_RESERVE, P, PREPARED, read, sha, stage, write
import planted_descent_study as scorer
from sitting_skills import ALIGN, REST, Direct, Options

BASE = P / 'planted_descent_01' / 'wire_brake'
STUDY = P / 'sitting_skills_01'
SOURCE = ('sitting_skills.py', 'sitting_skills_worker.py', 'sitting_skills_study.py')
HOLD_ROWS = (0, 1, 2, 9, 10, 11)
GPU = 'GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
NAMES = ('bank.npz', 'metadata.json', 'geometry.json')
PROBE_ACTIONS = ((0,0,0,0),(2,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,2))


def prepare():
    STUDY.mkdir()
    original = BASE/'bank'
    new = STUDY/'training_bank'
    shutil.copytree(original, new)
    i = np.arange(12)
    dx = np.where(i % 2 == 0, .0018, -.0018)
    dy = np.where(i % 3 == 1, .0020, -.0015)
    yaw = np.where(i % 2 == 0, .004, -.004)
    with np.load(original/'bank.npz', allow_pickle=False) as content:
        products = {k:content[k] for k in content.files}
    pose = products['reset_root_pose']
    pose[:,0] += dx
    pose[:,1] += dy
    pose[:,3:] = (Rotation.from_euler('z', yaw[:,None])*Rotation.from_quat(pose[:,3:])).as_quat()
    np.savez_compressed(new/'bank.npz', **products)
    write(STUDY/'split.json', dict(training=dict(x_m=dx.tolist(), y_m=dy.tolist(), yaw_rad=yaw.tolist(),
                                                bank_sha256=sha(new/'bank.npz')),
                                 evaluation=dict(bank_sha256=sha(original/'bank.npz'), cases=12),
                                 grade='nearby starts on the same four furniture assets'))


def probe_bank(bank, groups):
    src = STUDY/'training_bank'
    bank.mkdir()
    geom = read(src/'geometry.json')
    meta = read(src/'metadata.json')
    assert len(geom) == 12
    repeated = []
    for group in range(groups):
        for row, g in enumerate(geom):
            case = 12*group+row
            repeated.append(dict(g, case=case, surface_id=case, label=g['label']+f'/probe_{group}'))
    write(bank/'geometry.json', repeated)
    write(bank/'metadata.json', dict(meta, cases=len(repeated), ids=[g['label'] for g in repeated]))
    with np.load(src/'bank.npz', allow_pickle=False) as z:
        arrays = {key:np.concatenate([z[key]]*groups,axis=int(key=='q_post')) for key in z.files}
    arrays['reference'] = np.arange(len(repeated))
    np.savez_compressed(bank/'bank.npz', **arrays)


def rollout(name, mode, phases, choices, weights=None, bank_kind='train', guard_from=None):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU
    frozen = read(BASE/'evaluation.freeze.json')
    for filename, digest in frozen['inputs'].items():
        assert sha(filename) == digest, filename
    target = STUDY/name
    target.mkdir()
    bank = target/'bank'
    config = read(BASE/'step_config.json')
    if bank_kind == 'grouped':
        groups = len(PROBE_ACTIONS)
        probe_bank(bank,groups)
        config['templates_by_row'] *= groups
        plant = config['planted_descent']
        plant['rows'] = [12*k+i for k in range(groups) for i in plant['rows']]
        plant['treatments'] = {str(12*k+int(i)):v for k in range(groups) for i,v in plant['treatments'].items()}
        choices = [list(opt) for opt in PROBE_ACTIONS for _ in range(12)]
    else:
        shutil.copytree(STUDY/'training_bank' if bank_kind == 'train' else BASE/'bank', bank)
        choices = [choices]*12
    shutil.copyfile(BASE/'landing_profile.json', target/'landing_profile.json')
    write(target/'step_config.json',config)
    conf = dict(mode=mode, phases=phases, choices=choices,
                hold_rows=[i for i in range(len(choices)) if i%12 in HOLD_ROWS],
                seed=73281, split=bank_kind)
    extra = [STUDY/'split.json']
    if guard_from:
        assert bank_kind == 'matched' and mode == 'evaluate'
        check = STUDY/guard_from
        assert all(sha(check/'bank'/n)==sha(BASE/'bank'/n) for n in NAMES)
        trained = Path(read(check/'skills.json')['weights'])
        evidence = read(check/'evaluation.freeze.json')
        audit = read(check/'evaluation'/'independent_audit.json')
        assert audit['passed'] and audit['strict_predicates_exact'] and not audit['physical_tolerances_changed']
        assert sha(weights)==evidence['inputs'][str(trained)]
        incumbent, candidate = (read(run/'measures.json') for run in (BASE, check))
        conf['verified_rows'] = [
            row for row,(old,new) in enumerate(zip(incumbent,candidate,strict=True))
            if old['case']==new['case'] and new['supported'] and new['descent_feet_planted']
            and (row not in HOLD_ROWS or new['hold_mechanical_power_W']<=old['hold_mechanical_power_W']+.10)
        ]
        conf['guard_from'] = guard_from
        extra.extend([check/'measures.json',check/'evaluation'/'independent_audit.json', BASE/'measures.json'])
    if weights:
        weights = Path(weights).resolve(strict=True)
        conf['weights'] = str(weights)
        extra.append(weights)
    write(target/'skills.json', conf)
    write(target/'reservation.json', dict(files={str(f):sha(f) for f in bank.iterdir()}))
    cmd = frozen['command']
    stage(target,'evaluation',bank,PREPARED/'library.npz',worker=P/'sitting_skills_worker.py',
          extra=cmd[3:cmd.index('--mode')],disk_reserve=DISK_RESERVE,
          inputs=[*map(Path,frozen['inputs']), *(P/s for s in SOURCE), *extra, target/'skills.json',
                  target/'landing_profile.json',target/'step_config.json'])
    scorer.STUDY = STUDY
    scorer.score(name)
    assert (target/'evaluation'/'skill_decisions.npz').is_file()


def outcomes(folder):
    rows = read(folder/'measures.json')
    with np.load(folder/'evaluation'/'trajectory.npz') as z, np.load(folder/'evaluation'/'single_step.npz') as step:
        force = z['physical_force_w']
        bw = z['physical_bodyweight_N']
        seat = z['support_bw']
        good = z['good'] & z['learn']
        effort = np.square(z['physical_torque'][:,:,:23]/z['physical_torque_scale'][:,:,:23]).mean(-1)
        power = np.abs(z['physical_torque'][:,:,:23]*z['physical_joint_velocity'][:,:,:23]).sum(-1)
        floor = np.linalg.norm(force[:,:,5:],axis=-1) > .01*bw[:,:,None]
        slip = floor & (z['physical_slip_m_s'][:,:,5:] > .02)
        results = []
        for i,row in enumerate(rows):
            ok = row['supported'] and row['descent_feet_planted']
            reach = np.flatnonzero((seat[:,i]>=.05) & z['learn'][:,i])
            contact = int(reach[0]) if len(reach) else 750
            descent = slice(0,contact+1)
            handoff = slice(contact,min(contact+155,750))
            hold = slice(-250,None)
            settled = step['verified_floor'][-50:,i].all() and not step['slip'][:,i].any()
            transfer = -.60
            if handoff.start < handoff.stop:
                transfer = (float(np.mean(good[handoff,i])) - .70*slip[handoff,i].any()
                            + .6*float(np.minimum(seat[handoff,i],.45).mean()))
            local = [
                1.0*settled + .32*bool(len(reach)) - .10*row['step_mechanical_work_J']/50.,
                1.1*bool(len(reach)) - .55*slip[descent,i].any() - .0026*contact,
                transfer,
                1.4*float(good[hold,i].mean()) + .9*float(np.minimum(seat[hold,i],.60).mean())
                    - 1.1*float(effort[hold,i].mean())/.04 - .12*float(power[hold,i].mean()),
            ]
            results.append((ok,np.asarray(local), 6.*ok+sum(local)))
    return results


def fit(name, separate, runs, phases):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU
    torch.manual_seed(73281)
    result = {run: outcomes(STUDY/run) for run in runs}
    samples = {}
    for run in runs:
        choices = read(STUDY/run/'skills.json')['choices']
        with np.load(STUDY/run/'evaluation'/'skill_decisions.npz') as z:
            for i,p,action,feat,used in zip(*(z[k] for k in ('row','phase','action','features','enabled')),strict=True):
                if not used or p not in phases:
                    continue
                ok,local,total = result[run][i]
                reward = 6.*ok+local[p] if separate else total
                # A probe changes one decision at a time. Do not credit altered
                # downstream states to another stage's zero action.
                choice = choices[i] if isinstance(choices[0], list) else choices
                if any(a for j,a in enumerate(choice) if j!=p):
                    continue
                samples.setdefault((int(i)%12,int(p)),{}).setdefault(int(action),[]).append((float(reward),feat,ok))
    examples = []
    for (_,phase), options in sorted(samples.items()):
        if 0 not in options or len(options)<2:
            continue
        candidates = {a:float(np.mean([r[0] for r in entries])) for a,entries in options.items()
                      if all(r[2] for r in entries) or not all(r[2] for r in options[0])}
        if not candidates:
            continue
        preferred = max(candidates,key=candidates.get)
        if 0 in candidates and candidates[preferred]-candidates[0] < .055:
            preferred = 0
        examples.append((options[0][0][1], phase, preferred))
    assert examples and set(p for _,p,_ in examples) == set(phases)
    device = torch.device('cuda:0')
    model = Options(separate).to(device)
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    path = STUDY/(name+'_initial.pt')
    torch.save(dict(model=initial, separate=separate),path)
    opt = torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.005)
    x = torch.as_tensor(np.stack([e[0] for e in examples]), device=device)
    stage_tensor = torch.tensor([e[1] for e in examples],device=device)
    target = torch.tensor([e[2] for e in examples],device=device)
    history = []
    for _ in range(190):
        loss = torch.nn.functional.cross_entropy(model(x,stage_tensor),target,label_smoothing=.03)
        history.append(float(loss.detach()))
        opt.zero_grad()
        loss.backward()
        opt.step()
    learned = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    changed = float(sum((learned[k]-v).square().sum() for k,v in initial.items()).sqrt())
    with torch.no_grad():
        match = model(x,stage_tensor).argmax(1).eq(target).float().mean().item()
    assert changed > 0 and np.isfinite(history).all()
    destination = STUDY/(name+'.pt')
    torch.save(dict(model=learned,separate=separate),destination)
    write(STUDY/(name+'_fit.json'),dict(training_runs=runs,example_count=len(examples),epochs=len(history),
                                      initial_loss=history[0],final_loss=history[-1],fit_accuracy=match,
                                      weight_delta_l2=changed,checkpoint_sha256=sha(destination), phases=phases,
                                      targets=[dict(phase=p,action=a) for _,p,a in examples]))
    print(name, len(examples),'examples, weight change',round(changed,4),'fit',match,flush=True)


def fit_direct(runs):
    """Imitate the same independently passing full trajectories in one actor."""
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU
    torch.manual_seed(73281)
    data, commands, alignment = [], [], []
    for run in runs:
        folder = STUDY/run
        rows = read(folder/'measures.json')
        with np.load(folder/'evaluation'/'trajectory.npz') as z, np.load(
                folder/'evaluation'/'skill_decisions.npz') as decision:
            valid = [i for i,r in enumerate(rows) if r['supported'] and r['descent_feet_planted']]
            if not valid:
                continue
            data.append(z['obs'][::2,valid].reshape(-1,447))
            commands.append(z['desired'][::2,valid].reshape(-1,23))
            for row,phase,features,action,enabled in zip(*(decision[k] for k in (
                    'row','phase','features','action','enabled')),strict=True):
                if phase==ALIGN and enabled and row in valid:
                    alignment.append((features,action))
    assert data and alignment
    device = torch.device('cuda:0')
    x = torch.as_tensor(np.concatenate(data),dtype=torch.float32,device=device)
    y = torch.as_tensor(np.concatenate(commands),dtype=torch.float32,device=device)
    a = torch.as_tensor(np.stack([row[0] for row in alignment]),device=device)
    ay = torch.as_tensor([row[1] for row in alignment],device=device)
    model = Direct(x.mean(0),x.std(0).clamp_min(.035)).to(device)
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save(dict(model=initial),STUDY/'direct_initial.pt')
    opt = torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.0005)
    history = []
    for _ in range(1400):
        batch = torch.randint(len(x),(512,),device=device)
        pose = torch.nn.functional.smooth_l1_loss(model(x[batch]),y[batch],beta=.02)
        approach = torch.nn.functional.cross_entropy(model.align(a),ay)
        loss = pose+.04*approach
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    with torch.no_grad():
        pose_rms = (model(x)-y).square().mean().sqrt().item()
    learned = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    changed = float(sum((learned[k]-v).square().sum() for k,v in initial.items()).sqrt())
    assert changed > 0 and np.isfinite(history).all()
    target = STUDY/'direct.pt'
    torch.save(dict(model=learned), target)
    write(STUDY/'direct_fit.json',dict(training_runs=runs, observation_samples=len(x),
                                       alignment_samples=len(a), iterations=len(history),
                                       first_loss=history[0], final_loss=history[-1],
                                       training_joint_target_rms_rad=pose_rms,
                                       weight_delta_l2=changed, checkpoint_sha256=sha(target)))
    print('direct',len(x),'observations, training target RMS',pose_rms,flush=True)


def report(names):
    before=read(BASE/'measures.json')
    fields=[]
    for name in names:
        target=STUDY/name
        assert all(sha(target/'bank'/n)==sha(BASE/'bank'/n) for n in NAMES)
        rows=read(target/'measures.json')
        pair=[]
        for b,c in zip(before,rows,strict=True):
            assert b['case']==c['case']
            pair.append(dict(c, arm=name, original_hold_s=b['hold_s'],
                             original_hold_power_W=b['hold_mechanical_power_W'],
                             original_load_pct=b['hold_seat_normal_pct'],
                             original_torque_rms=b['hold_body_joint_RMS_Nm']))
        passed=[r['case'] for r in pair if r['supported'] and r['descent_feet_planted']]
        print(name,len(passed),'planted passes of 12',
              [r['case'] for r in pair if r['case'] not in passed],flush=True)
        fields.extend(pair)
    with (STUDY/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in fields for k in r)))
        w.writeheader()
        w.writerows(fields)


if __name__ == '__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('operation',choices=('prepare','rollout','fit','direct','report'))
    cli.add_argument('--name')
    cli.add_argument('--mode',choices=('probe','sample','evaluate','direct'),default='probe')
    cli.add_argument('--phases',nargs='+',type=int,default=[REST])
    cli.add_argument('--options',nargs=4,type=int,default=[0,0,0,0])
    cli.add_argument('--bank',choices=('train','grouped','matched'),default='train')
    cli.add_argument('--weights',type=Path)
    cli.add_argument('--guard-from')
    cli.add_argument('--shared', action='store_true')
    cli.add_argument('runs',nargs='*')
    args=cli.parse_intermixed_args()
    if args.operation=='prepare':
        prepare()
    if args.operation=='rollout':
        rollout(args.name,args.mode,args.phases,args.options,args.weights,args.bank,args.guard_from)
    if args.operation=='fit':
        fit(args.name,not args.shared,args.runs,args.phases)
    if args.operation=='direct':
        fit_direct(args.runs)
    if args.operation=='report':
        report(args.runs)
