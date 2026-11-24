"""Collect geometry-driven approaches, update a high-level actor with on-policy PPO and score the fixed suite."""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from furniture_scale_suite import SUITE, run_folder
from run import P, PREPARED, DISK_RESERVE, read, write, sha, stage
from scale_approach import LEG, Approach, edge_features

STUDY = P / 'scale_approach_ppo_01'
WORKER = P / 'scale_approach_worker.py'
UPDATES = 5
TEACHERS = (2, 1, 3, 4)


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def csvout(path, data):
    with path.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def training():
    return [r for r in read(SUITE/'cases.json') if r['split']=='train' and r['scale_pct'] in (88,64)]


def make_library(entries, dest):
    from stance23_planner import model,chains,legs,forward
    library = []
    by_run = {}
    for case,folder,i in entries:
        by_run.setdefault(folder,[]).append((case,i))
    for folder,group in by_run.items():
        ev = folder/'evaluation'
        result = read(ev/'results.json')
        digest = read(ev/'independent_audit.json')
        assert digest['passed'] and sha(ev/'trajectory.npz')==digest['files']['trajectory.npz']
        seen = read(ev/'seen_seat.json')
        with np.load(ev/'standing_scan.npz') as ss, np.load(ev/'standing_rims.npz') as rim, np.load(ev/'trajectory.npz') as z:
            scans = {k:ss[k] for k in ss.files if k in ('depth','intrinsics','q','gravity','mount')}
            rv = rim['values']
            desired,q,b,good = z['desired'],z['q'],z['body'],z['good']
            for case,i in group:
                assert good[-250:,i].all()
                direct = np.where(rv[:,i,6]==1,rv[:,i,0],np.nan)
                fr = [(scans['depth'][t,i],scans['intrinsics'][t,i],scans['q'][t,i],scans['gravity'][t,i],
                       scans['mount'][t]) for t in range(len(direct))]
                feat,kind = edge_features(seen['estimated_height_m'][i],direct,fr,model)
                if kind=='unavailable':
                    continue
                pose = b[:,i,result['body_names'].index('pelvis')]
                rot = Rotation.from_quat(pose[:,3:]).as_matrix()
                progress = np.clip((pose[0,2]-pose[:,2])/max(.035,pose[0,2]-np.median(pose[-250:,2])),0,1)
                ticks = np.r_[np.arange(0,1000,10),999]
                deriv = np.zeros((len(ticks),len(LEG),2))
                for k,t in enumerate(ticks):
                    for chain,ids in zip(chains,legs):
                        _,jac = forward(chain,q[t,i,ids])
                        shift = np.vstack((-rot[t].T[:,[0,2]],np.zeros((3,2))))*progress[t]
                        derivative = np.linalg.solve(jac.T@jac+.001*np.eye(6),jac.T@shift)
                        deriv[k,[LEG.index(j) for j in ids],:] = derivative
                response = np.stack([np.interp(np.arange(1000),ticks,v) for v in deriv.reshape(len(ticks),-1).T],-1)
                library.append((case,feat,desired[:,i][:,LEG],response.reshape(1000,len(LEG),2)))
    assert library
    names,features,paths,jac = zip(*library)
    np.savez_compressed(dest,cases=names,features=features,desired=np.asarray(paths,dtype='f'),response=np.asarray(jac,dtype='f'))


def base_demos():
    return [(r['case'],Path(r['trace']).parent.parent,int(r['trajectory_index']))
            for r in rows(SUITE/'supported_training_rollouts.csv') if r['split']=='train']


def bank_at(target,cases):
    assert len(set(c['case'] for c in cases))==len(cases)
    sources = {Path(c['bank']) for c in cases}
    if len(sources)==1:
        src = next(iter(sources))
        inds = [c['trajectory_index'] for c in cases]
        if inds == list(range(len(read(src/'geometry.json')))):
            shutil.copytree(src,target)
            return
    data = {}
    geom = {}
    for src in sources:
        with np.load(src/'bank.npz') as z:
            data[str(src)] = {k:z[k] for k in z.files}
        geom[str(src)] = read(src/'geometry.json')
    target.mkdir()
    meta = read(next(iter(sources))/'metadata.json')
    gs = [dict(geom[c['bank']][c['trajectory_index']],label='sit/'+c['case'],case=i,surface_id=i)
          for i,c in enumerate(cases)]
    write(target/'geometry.json',gs)
    write(target/'metadata.json',dict(meta,cases=len(gs),ids=[g['label'] for g in gs],evaluation_split='train'))
    bank = {}
    for k in next(iter(data.values())):
        axis = int(k=='q_post')
        bank[k] = (np.arange(len(gs)) if k=='reference' else np.stack(
            [np.take(data[c['bank']][k],c['trajectory_index'],axis=axis) for c in cases],axis=axis))
    np.savez_compressed(target/'bank.npz',**bank)


def rollout(name,cases,config):
    from publish_confirmed import assess
    folder = STUDY/name
    if config['mode']!='test':
        assert all(c['split']=='train' for c in cases)
    if (folder/'evaluation.score.json').exists():
        assert read(folder/'approach.json') == config
        assert read(folder/'case_ids.json') == [c['case'] for c in cases]
    else:
        folder.mkdir()
        bank = folder/'bank'
        bank_at(bank,cases)
        baseline = run_folder(100,0)
        frozen = read(baseline/'evaluation.freeze.json')
        for p,h in frozen['inputs'].items():
            assert sha(p)==h,p
        shutil.copyfile(baseline/'landing_profile.json',folder/'landing_profile.json')
        write(folder/'approach.json',config)
        write(folder/'case_ids.json',[c['case'] for c in cases])
        write(folder/'reservation.json',dict(files={str(f):sha(f) for f in bank.iterdir()}))
        extras = [*map(Path,frozen['inputs']),Path(__file__),P/'scale_approach.py',P/'pelvis_marker_worker.py',
                  folder/'landing_profile.json',folder/'approach.json',folder/'case_ids.json',Path(config['library']),
                  *[Path(c['mesh']) for c in cases]]
        if 'checkpoint' in config:
            extras.append(Path(config['checkpoint']))
        cmd = frozen['command']
        stage(folder,'evaluation',bank,PREPARED/'library.npz',worker=WORKER,extra=cmd[3:cmd.index('--mode')],
              inputs=extras,disk_reserve=DISK_RESERVE)
    audited,checks = assess(folder)
    write(folder/'witness_counts.json',checks)
    ev=folder/'evaluation'
    with np.load(ev/'contact_load_geometry.npz') as meshes, np.load(ev/'floor_slip_witness.npz') as feet:
        unresolved=np.nansum(1-meshes['observations'][25:,:,:,9],axis=(0,2))+np.nansum(1-feet['measurements'][25:,:,:,0],axis=(0,2))
    assert int(unresolved.sum()) == checks['unconfirmed_mesh_forces']+checks['unconfirmed_floor_forces']
    with np.load(folder/'evaluation/approach_choice.npz') as z:
        for i,(case,r) in enumerate(zip(cases,audited,strict=True)):
            assert r['asset']==case['asset']
            r.update(case=case['case'],split=case['split'],scale_pct=case['scale_pct'],
                     unconfirmed_force_samples=int(unresolved[i]),action=int(z['choice'][i]),eligible=bool(z['active'][i]),edge_source=str(z['edge_source'][i]))
    audited=[{k:(v.item() if isinstance(v,np.generic) else v) for k,v in r.items()} for r in audited]
    write(folder/'scores.json',audited)
    print(name,Counter(r['scale_pct'] for r in audited if r['supported_sit']),flush=True)
    return audited


def collect():
    STUDY.mkdir(exist_ok=True)
    if not (STUDY/'initial_library.npz').exists():
        make_library(base_demos(),STUDY/'initial_library.npz')
    cases=training()
    assert len(cases)==48
    for choice in TEACHERS:
        rollout(f'collect_{choice}',cases,dict(mode='collect',library=str(STUDY/'initial_library.npz'),
                                               choices=[choice]*len(cases)))


def reward(folder):
    rr = read(folder/'scores.json')
    with np.load(folder/'evaluation/trajectory.npz') as z:
        good = z['good'] & z['learn']
        held = good.sum(axis=0)*.02
        live = z['learn'].mean(axis=0)
    return np.array([3*float(r['supported_sit'])+.5*float(r['strict_clean'])+
                     .8*min(h/5,1)+.25*l-.4*float(r['reason']=='Forbidden contact')-
                     .3*float(r['reason']=='Touchdown impact') for r,h,l in zip(rr,held,live)],dtype='f')


def prepare_training():
    cases=training()
    source={c['case']:c for c in cases}
    best={}
    accepted=[]
    for a in TEACHERS:
        folder=STUDY/f'collect_{a}'
        for i,r in enumerate(read(folder/'scores.json')):
            assert r['case'] in source
            if r['supported_sit'] and r['eligible'] and r['unconfirmed_force_samples']==0:
                accepted.append(dict(case=r['case'],scale_pct=r['scale_pct'],approach=a,
                                     strict_clean=r['strict_clean'],hold_s=r['terminal_hold_s'],
                                     folder=str(folder),trajectory_index=i,trace_sha256=r['trajectory_sha256']))
                quality=(r['strict_clean'],r['terminal_hold_s'])
                if r['case'] not in best or quality>best[r['case']][0]:
                    best[r['case']]=(quality,folder,i)
    write(STUDY/'collected_demos.json',[dict(case=c,folder=str(f),index=i) for c,(_,f,i) in best.items()])
    if accepted:
        csvout(STUDY/'verified_collection.csv',accepted)
    make_library(base_demos()+[(c,f,i) for c,(_,f,i) in best.items()],STUDY/'train_library.npz')
    write(STUDY/'teacher_labels.json',[dict(folder=str(f),index=i) for _,f,i in best.values()])

def optimize(update):
    import torch
    from run_repaired_sitting_ppo import gpu_slot
    torch.manual_seed(8831)
    initial = STUDY/'initial.pt'
    if update == 0:
        with gpu_slot(STUDY/'optimizer_slot.json'):
            net=Approach().to('cuda:0')
            opt=torch.optim.Adam(net.parameters(),lr=.003)
            ft,labels=[],[]
            for label in read(STUDY/'teacher_labels.json'):
                folder,i=Path(label['folder']),label['index']
                with np.load(folder/'evaluation/approach_choice.npz') as z:
                    ft.append(z['features'][i]);labels.append(z['choice'][i])
            if ft:
                x=torch.as_tensor(np.asarray(ft),device='cuda',dtype=torch.float)
                y=torch.as_tensor(np.asarray(labels),device='cuda')
                for _ in range(30):
                    logits,_=net(x)
                    loss=torch.nn.functional.cross_entropy(logits,y,label_smoothing=.22)
                    opt.zero_grad();loss.backward();opt.step()
            torch.save(net.cpu().state_dict(),initial)
            del net,opt
            torch.cuda.empty_cache()
        return
    u = update - 1
    folder = STUDY/f'ppo_{update}'
    prev = initial if u == 0 else STUDY/f'policy_{u}.pt'
    with np.load(folder/'evaluation/approach_choice.npz') as z:
        batch={k:z[k][z['active']] for k in ('features','choice','old_logp','old_value')}
        ret=reward(folder)[z['active']]
    with gpu_slot(STUDY/'optimizer_slot.json'):
        net=Approach().cuda();net.load_state_dict(torch.load(prev,map_location='cuda',weights_only=True))
        starting=[p.detach().clone() for p in net.actor.parameters()]
        opt=torch.optim.Adam(net.parameters(),lr=.002)
        if u:
            opt.load_state_dict(torch.load(STUDY/f'optimizer_{u}.pt',map_location='cuda',weights_only=True))
        x=torch.as_tensor(batch['features'],device='cuda',dtype=torch.float)
        action=torch.as_tensor(batch['choice'],device='cuda')
        old=torch.as_tensor(batch['old_logp'],device='cuda',dtype=torch.float)
        returns=torch.as_tensor(ret,device='cuda')
        adv=returns-torch.as_tensor(batch['old_value'],device='cuda',dtype=torch.float)
        adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-5)
        for epoch in range(8):
            logits,val=net(x)
            dist=torch.distributions.Categorical(logits=logits)
            lp=dist.log_prob(action)
            if epoch==0:
                torch.testing.assert_close(lp.detach(),old,rtol=1e-5,atol=2e-5)
            ratio=(lp-old).exp()
            policy=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
            loss=policy+.35*(val-returns).square().mean()-.018*dist.entropy().mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),.5);opt.step()
        delta=max((p-s).abs().max().item() for p,s in zip(net.actor.parameters(),starting))
        assert np.isfinite(delta) and delta>0
        prev=STUDY/f'policy_{u+1}.pt'
        torch.save(opt.state_dict(),STUDY/f'optimizer_{update}.pt')
        torch.save(net.cpu().state_dict(),prev)
        with (STUDY/'updates.jsonl').open('a') as f:
            entry=dict(update=u+1,episodes=len(training()),actor_samples=len(ret),mean_return=float(ret.mean()),
                       max_actor_weight_change=delta,checkpoint_sha256=sha(prev))
            print(json.dumps(entry),file=f);print(entry,flush=True)
        del net,opt,x,returns,adv,action,old,logits,val,dist,loss,starting,policy,ratio,lp
        torch.cuda.empty_cache()


def train():
    if not (STUDY/'teacher_labels.json').exists():
        prepare_training()
    # Each optimizer releases its CUDA context before the next Isaac process
    # acquires the project's exclusive 4090 slot.
    if not (STUDY/'initial.pt').exists():
        subprocess.run([sys.executable, __file__, 'optimize', '0'],check=True)
    cases=training()
    prev=STUDY/'initial.pt'
    for update in range(1,UPDATES+1):
        cfg=dict(mode='sample',seed=7030+update,library=str(STUDY/'train_library.npz'),checkpoint=str(prev))
        rollout(f'ppo_{update}',cases,cfg)
        if not (STUDY/f'policy_{update}.pt').exists():
            subprocess.run([sys.executable,__file__,'optimize',str(update)],check=True)
        prev=STUDY/f'policy_{update}.pt'
    write(STUDY/'final.json',dict(checkpoint=str(prev),sha256=sha(prev),library=str(STUDY/'train_library.npz'),
                                   library_sha256=sha(STUDY/'train_library.npz'),updates=UPDATES))


def test():
    final=read(STUDY/'final.json')
    assert sha(final['checkpoint'])==final['sha256'] and sha(final['library'])==final['library_sha256']
    cases=read(SUITE/'cases.json')
    baseline={r['case']:r for r in rows(SUITE/'case_results.csv')}
    out=[]
    for size in (100,88,76,64):
        for half in (0,16):
            cc=[c for c in cases if c['bank']==str(run_folder(size,half)/'bank')]
            assert len(cc)==16
            rr=rollout(f'test_{size}_{half}',cc,dict(mode='test',checkpoint=final['checkpoint'],library=final['library']))
            for r in rr:
                b=baseline[r['case']]
                was=b['supported_sit']=='True'
                r.update(baseline_supported=was,baseline_strict=b['strict_clean']=='True',baseline_hold_s=b['terminal_hold_s'],
                         repaired=r['supported_sit'] and not was,regressed=was and not r['supported_sit'])
                out.append(r)
    assert len(out)==len({r['case'] for r in out})==128
    csvout(STUDY/'paired_results.csv',out)
    summary={}
    for group in ['train','held_out_scale','held_out_object','all']:
        rr=[r for r in out if group=='all' or r['split']==group]
        summary[group]=dict(total=len(rr),before=sum(r['baseline_supported'] for r in rr),
                            after=sum(r['supported_sit'] for r in rr),
                            strict_before=sum(r['baseline_strict'] for r in rr),
                            strict_after=sum(r['strict_clean'] for r in rr),
                            repairs=[r['case'] for r in rr if r['repaired']],
                            regressions=[r['case'] for r in rr if r['regressed']])
    write(STUDY/'paired_summary.json',summary)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('operation',choices=['collect','train','test','optimize'])
    ap.add_argument('update',nargs='?',type=int)
    args=ap.parse_args()
    if args.operation=='optimize':
        optimize(args.update)
    else:
        {'collect':collect,'train':train,'test':test}[args.operation]()
