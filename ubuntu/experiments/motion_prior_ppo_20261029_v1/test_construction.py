"""FrozenCPU checks for the new policy interface, recurrence and optimizer."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from common import read,write,sha
from policy import Learner,command

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(64701)
    prepared=Path(a.prepared);agent=Learner(np.zeros(447),np.ones(447),64111,'cpu');agent.load(prepared/'initial_64111.pt')
    with np.load(prepared/'demonstrations.npz') as z:x=torch.from_numpy(z['obs'].copy())
    agent.bind_demonstrations(x);ix=torch.linspace(0,len(x)-1,1024).long();obs=x[ix].clone()
    before=agent.choose(obs,False)[0];changed=obs.clone();changed[:,:333]+=.03
    # The nominal sequence must not follow drift in previous commands or pose.
    torch.testing.assert_close(agent.prior(obs),agent.prior(changed),rtol=0,atol=0)
    metrics=[]
    for i in range(16):
        action,logp,value,old_mean=agent.choose(obs)
        advantage=(action[:,0]-old_mean[:,0])/.0075
        target=value+advantage
        m=agent.update(obs,action,logp,old_mean,advantage,target);metrics.append(m)
        assert m['actor_steps']>0,('OPTIMIZER_FAILED_CONTROLLED_DESCENT',i,m)
        assert m['measured_kl']<=.02 and m['hard_imitation_ceiling'] is False
        assert all(set(c['rejected_by'])<=set(['kl','objective','nonfinite']) for c in m['candidate_tests'])
    assert not torch.equal(before,agent.choose(obs,False)[0])
    lower=torch.full((23,),-1.);upper=-lower;previous=torch.zeros(1024,23)
    huge=torch.linspace(-1e4,1e4,previous.numel()).reshape_as(previous)
    applied=command(previous,huge,lower,upper)
    assert bool((applied.abs()<=.003).all())
    torch.testing.assert_close(command(previous,previous,lower,upper),previous,atol=0,rtol=0)
    write(Path(a.output),dict(passed=True,construction_only=True,synthetic_optimizer_updates=16,
        accepted_actor_steps=sum(m['actor_steps'] for m in metrics),policy_changed_all16_updates=True,
        nominal_motion_independent_of_current_pose_and_prior_command=True,
        action_filter_rate_and_margin_preserved=True,no_physical_success_claim=True,
        source_hashes={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')},metrics=metrics))

if __name__=='__main__':main()
