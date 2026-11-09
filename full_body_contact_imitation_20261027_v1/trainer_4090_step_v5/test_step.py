"""CPU optimizer contracts on saved demonstration observations; not PPO rollout evidence."""
import copy
from pathlib import Path
import numpy as np
import torch
from common import ROOT,write
from policy import Learner


def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    torch.set_num_threads(1);prior=ROOT/'outputs/full_body_contact_imitation_v4';reports=[]
    for task in ['sitting','leaning']:
        with np.load(prior/'dataset'/task/'bank.npz') as z:
            # Fixed late-stage recorded observations; no new physics and no
            # claim to reconstruct the unavailable failed stochastic rollout.
            obs=torch.tensor(z['obs'][750:814].reshape(-1,336))
            target=torch.tensor(z['bc_target'][750:814].reshape(-1,23))
        for seed in [64111,64117]:
            agent=Learner(np.zeros(336),np.ones(336),seed)
            agent.load(prior/f'bc_{task}_{seed}/initial.pt')
            action,logp,value,mean=agent.choose(obs)
            # Synthetic advantage toward the saved demonstration command.
            advantage=-(action.tanh()-target).square().mean(-1)
            before=copy.deepcopy(agent.actor.state_dict())
            metrics=agent.update(obs,action,logp,mean,advantage,value+advantage)
            with torch.no_grad():
                actual_kl=float(((agent.actor(agent.normalize(obs))-mean).square()/(2*.25**2)).sum(-1).mean())
            change=float(sum((agent.actor.state_dict()[k]-before[k]).double().square().sum() for k in before).sqrt())
            assert metrics['actor_steps']>0 and change>0 and actual_kl<=.02
            assert 4<=metrics['candidate_tests']<=32
            assert len(metrics['accepted_step_scales'])==metrics['actor_steps']
            reports.append(dict(task=task,seed=seed,actual_final_kl=actual_kl,actor_l2_change=change,**metrics))
            # An invalid distant reference must exhaust the finite candidates
            # and preserve both actor parameters and its existing Adam state.
            a,lp,v,mu=agent.choose(obs);state=copy.deepcopy(agent.actor.state_dict());opt=copy.deepcopy(agent.actor_opt.state_dict())
            rejected=agent.update(obs,a,lp,mu+100.,advantage,v+advantage)
            assert rejected['actor_steps']==0 and rejected['candidate_tests']==8
            assert equal(state,agent.actor.state_dict()) and equal(opt,agent.actor_opt.state_dict())
    result=dict(passed=True,classification='CPU contract fixtures with recorded inputs and synthetic advantages; zero simulator or PPO rollout evidence',
                reports=reports,rejection_restores_parameters_and_adam=True,physical_gates_unchanged=True,maximum_candidate_tests_per_update=32)
    write(Path(__file__).with_name('step_tests.json'),result);print(result)


if __name__=='__main__':main()
