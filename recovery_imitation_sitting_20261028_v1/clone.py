"""Bounded supervised fits using clean original and physically recovered trials."""
import argparse
import math
import time
from pathlib import Path
import numpy as np
import torch
from common import read, write, sha
from policy import Learner, teacher_targets


def original(bank):
    meta=read(bank/'metadata.json')
    with np.load(bank/'bank.npz',allow_pickle=False) as z:
        positive=np.flatnonzero(z['admitted']); assert len(positive)==64
        x=z['obs'][:,positive].reshape(-1,336).copy()
        target=teacher_targets(torch.from_numpy(z['previous'][:,positive,:23]),
            torch.from_numpy(z['command'][:,positive,:23]),torch.tensor(meta['lower']),torch.tensor(meta['upper'])).numpy()
        return x,target.reshape(-1,23),[meta['ids'][i] for i in positive]


def recovery(root):
    audit=read(root/'independent_audit.json'); assert audit['passed']
    rows=read(root/'results.json')['episodes']; assert len(rows)==72
    keep=np.flatnonzero(audit['clean']); assert len(keep)>0,'NO_CLEAN_RECOVERY_DEMONSTRATIONS'
    with np.load(root/'trajectory.npz',allow_pickle=False) as z:
        learner=z['recovery_learner'][:,keep]
        assert learner.any(0).all()
        prior=z['recovery_prior'][:,keep]
        desired=z['recovery_target'][:,keep]
        teacher=prior+np.clip(.1*(desired-prior),-.003,.003)
        changed=np.max(abs(z['command'][:,keep,:23]-teacher),axis=-1)>2e-7
        exercised=(changed&learner).any(0)
        keep=keep[exercised]; assert len(keep)>0,'NO_ACTUAL_RECOVERY_DISTURBANCE'
        x=z['recovery_obs'][:,keep].reshape(-1,336).copy()
        y=z['recovery_target'][:,keep].reshape(-1,23).copy()
    return x,y,dict(clean_cases=[rows[i]['id'] for i in keep],count=len(keep),
        trajectory_sha256=sha(root/'trajectory.npz'),full20s=True,autonomous_success=False,
        teacher_interventions=True,failed_cases_excluded=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--round',type=int,choices=[0,1,2],required=True)
    p.add_argument('--checkpoint');p.add_argument('--prior-demo');p.add_argument('--recovery');a=p.parse_args()
    started=time.monotonic();torch.set_num_threads(1);np.random.seed(a.seed)
    out=Path(a.output);out.mkdir(exist_ok=False);bank=Path(a.bank); provenance=[]
    if a.round==0:
        assert not a.checkpoint and not a.recovery and not a.prior_demo
        x,y,ids=original(bank);provenance=[dict(source='original_teacher',cases=ids,bank_sha256=sha(bank/'bank.npz'))]
    else:
        assert a.checkpoint and a.prior_demo and a.recovery
        with np.load(a.prior_demo,allow_pickle=False) as z:x=z['obs'].copy();y=z['target'].copy()
        rx,ry,info=recovery(Path(a.recovery));x=np.concatenate((x,rx));y=np.concatenate((y,ry));provenance=[info,dict(prior_demo_sha256=sha(a.prior_demo))]
    assert np.isfinite(x).all() and np.isfinite(y).all();x=x.astype('float32');y=y.astype('float32')
    np.savez_compressed(out/'demonstrations.npz',obs=x,target=y)
    agent=Learner(x.mean(0),x.std(0),a.seed,'cuda:0')
    if a.checkpoint:agent.load(a.checkpoint)
    else:agent.actor[-1].bias.data.copy_(torch.as_tensor(y.mean(0),device='cuda:0'))
    assert agent.updates==agent.actor_steps==agent.critic_steps==0
    features=agent.normalize(torch.as_tensor(x,device='cuda:0'));targets=torch.as_tensor(y,device='cuda:0')
    def errors():
        with torch.no_grad():
            sums=0.;maximum=0.;count=0
            for i in range(0,len(x),4096):
                err=agent.actor(features[i:i+4096])-targets[i:i+4096]
                sums+=float(err.double().square().sum());count+=err.numel();maximum=max(maximum,float(err.abs().max()))
            return dict(desired_rmse_rad=math.sqrt(sums/count),desired_max_error_rad=maximum)
    initial=errors();opt=torch.optim.Adam(agent.actor.parameters(),lr=3e-4)
    steps=12000 if a.round==0 else 6000
    for step in range(steps):
        lr=3e-5+(3e-4-3e-5)*.5*(1+math.cos(math.pi*step/steps))
        for group in opt.param_groups:group['lr']=lr
        ix=torch.randint(len(x),(2048,),device='cuda:0')
        loss=((agent.actor(features[ix])-targets[ix])/.03).square().mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),1.);opt.step()
        if (step+1)%1000==0:
            record=dict(step=step+1,loss=float(loss.detach()),seconds=time.monotonic()-started)
            write(out/'status.json',record);print(record,flush=True)
    final=errors();assert all(torch.isfinite(v).all() for v in agent.actor.parameters())
    agent.save(out/'initial.pt')
    write(out/'fit.json',dict(seed=a.seed,round=a.round,steps=steps,batch_size=2048,samples=len(x),
        initial=initial,final=final,source=provenance,seconds=time.monotonic()-started,
        measured_on_all_training_data=True,generalization_claimed=False,closed_loop_gate_pending=True,
        checkpoint_sha256=sha(out/'initial.pt'),demonstrations_sha256=sha(out/'demonstrations.npz')))
    assert final['desired_rmse_rad']<initial['desired_rmse_rad'],'SUPERVISED_FIT_DID_NOT_IMPROVE'

if __name__=='__main__':main()
