"""CPU behavior cloning from aligned, strictly clean development examples."""
import argparse
from pathlib import Path
import numpy as np
import torch
from common import read, write, sha
from policy import Learner


def main():
    p=argparse.ArgumentParser(); p.add_argument('--bank',required=True); p.add_argument('--output',required=True); p.add_argument('--seed',type=int,required=True)
    a=p.parse_args(); out=Path(a.output); out.mkdir(exist_ok=False); bank=Path(a.bank)
    torch.set_num_threads(1); np.random.seed(a.seed)
    with np.load(bank/'bank.npz') as z:
        positive=np.flatnonzero(z['admitted']); validation=positive[::5]; training=np.setdiff1d(positive,validation)
        assert len(training)>=4 and len(validation)>=1
        obs=z['obs']; targets=z['bc_target']
        x=obs[:,training].reshape(-1,336).astype('float32'); y=targets[:,training].reshape(-1,23).astype('float32')
        xv=obs[:,validation].reshape(-1,336).astype('float32'); yv=targets[:,validation].reshape(-1,23).astype('float32')
    agent=Learner(x.mean(0),x.std(0),a.seed)
    x=agent.normalize(torch.from_numpy(x)); xv=agent.normalize(torch.from_numpy(xv)); y=torch.from_numpy(y); yv=torch.from_numpy(yv)
    def error(features,target):
        with torch.no_grad():
            e=(agent.actor(features).tanh()-target).square()
            moving=target.abs()>.1
            return dict(mse=float(e.mean()),moving_mse=float(e[moving].mean()),moving_channels=int(moving.sum()))
    initial=error(xv,yv); opt=torch.optim.Adam(agent.actor.parameters(),lr=3e-4)
    # Fixed finite pretraining; no adaptive retry or tuning on validation.
    for step in range(2000):
        ix=torch.randint(len(x),(1024,)); pred=agent.actor(x[ix]).tanh()
        loss=((pred-y[ix]).square()*(1+4*(y[ix].abs()>.1))).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),1.); opt.step()
    final=error(xv,yv)
    passed=bool(final['moving_mse']<.75*initial['moving_mse'] and final['mse']<initial['mse'])
    agent.save(out/'initial.pt')  # PPO starts fresh Adam state after cloning.
    write(out/'pretrain_audit.json',dict(passed=passed,seed=a.seed,steps=2000,batch_size=1024,
          initial=initial,final=final,training_cases=training.tolist(),validation_cases=validation.tolist(),
          validation_is_development_only=True,positive_cases=positive.tolist(),fresh_holdout_used=False,
          bank_sha256=sha(bank/'bank.npz'),checkpoint_sha256=sha(out/'initial.pt')))
    assert passed,('CLONING_VALIDATION_FAILED',initial,final)


if __name__=='__main__':main()
