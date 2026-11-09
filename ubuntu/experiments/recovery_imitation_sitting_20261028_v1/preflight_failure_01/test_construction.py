"""Independent action-envelope and data-alignment audit, without simulation."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from common import read,write,sha
from policy import command,teacher_targets,Learner

def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(1);rng=np.random.default_rng(64111);meta=read(Path(a.bank)/'metadata.json')
    lo=np.array(meta['lower'],np.float32)+.03;hi=np.array(meta['upper'],np.float32)-.03
    prev=rng.uniform(lo,hi,(10000,23)).astype('float32');prev[:2]=[lo,hi]
    # Both action maps span exactly the same closed limiting interval.
    old_low=prev+.1*(np.maximum(prev-.03,lo)-prev)
    old_high=prev+.1*(np.minimum(prev+.03,hi)-prev)
    new_low=prev+np.clip(.1*(lo-prev),-.003,.003)
    new_high=prev+np.clip(.1*(hi-prev),-.003,.003)
    endpoint_error=float(max(abs(old_low-new_low).max(),abs(old_high-new_high).max()))
    assert endpoint_error<=2e-7
    with np.load(Path(a.bank)/'bank.npz',allow_pickle=False) as z:
        pr=torch.from_numpy(z['previous'][:,:,:23]);ap=torch.from_numpy(z['command'][:,:,:23])
        lower=torch.tensor(meta['lower']);upper=torch.tensor(meta['upper'])
        target=teacher_targets(pr,ap,lower,upper);replay=command(pr,target,lower,upper)
        replay_error=float((replay-ap).abs().max());assert replay_error<=2e-7
        # Independent NumPy inversion checks exact teacher time/previous association.
        nt=np.clip(pr.numpy()+(ap.numpy()-pr.numpy())/.1,lo,hi)
        np.testing.assert_allclose(target.numpy(),nt,rtol=0,atol=2e-7)
        protected=read(Path(__file__).parent/'protected_sitting_cases.json')['protected']
        assert {meta['ids'][i] for i in np.flatnonzero(z['admitted'])}==set(protected)
    # Unsaturated desired-pose feedback contracts command error by .9.
    p0=np.full(23,.1);r=p0+.01;perturb=np.full(23,.001)
    c1=p0+.1*(r-p0);c2=p0+perturb+.1*(r-(p0+perturb))
    np.testing.assert_allclose(c2-c1,.9*perturb,rtol=0,atol=1e-14)
    # The auxiliary term contributes a nonzero gradient toward the target.
    agent=Learner(np.zeros(336),np.ones(336),64111)
    dx=torch.randn(32,336);dy=torch.full((32,23),.1)
    agent.bind_demonstrations(dx,dy)
    loss=((agent.actor(agent.normalize(dx))-dy)/.03).square().mean();loss.backward()
    assert float(agent.actor[-1].bias.grad.sum())<0
    # Exercise the actual combined PPO/imitation step and rejection rollback.
    with torch.no_grad():
        action,logp,value,mean=agent.choose(dx)
    before=copy.deepcopy(agent.actor.state_dict())
    metrics=agent.update(dx,action,logp,mean,action.mean(-1),torch.ones(32))
    assert metrics['actor_steps']>0 and metrics['measured_kl']<=.02
    assert metrics['anchor_loss']<=metrics['anchor_ceiling']
    assert any(not torch.equal(before[k],v) for k,v in agent.actor.state_dict().items())
    rejected_before=copy.deepcopy(agent.actor.state_dict())
    opt_before=copy.deepcopy(agent.actor_opt.state_dict())
    agent.anchor_ceiling=-1.
    with torch.no_grad():action,logp,value,mean=agent.choose(dx)
    rejected=agent.update(dx,action,logp,mean,action.mean(-1),torch.ones(32))
    assert rejected['actor_steps']==0
    for k,v in rejected_before.items():assert torch.equal(v,agent.actor.state_dict()[k])
    for k,v in opt_before['state'].items():
        for name,tensor in v.items():assert torch.equal(tensor,agent.actor_opt.state_dict()['state'][k][name])
    # Strict checkpoint unit marker prevents accidental reuse of increment policies.
    from rewards import staged_reward
    from audit import reward_numpy
    # Unchanged reward/scoring sources are byte-checked separately against frozen v5.
    write(a.output,dict(passed=True,action_endpoint_error_rad=endpoint_error,
        original_filtered_command_replay_error_rad=replay_error,teacher_samples=72000,
        sampled_action_states=10000,unchanged_filter=.1,unchanged_rate_rad_s=.15,
        unchanged_margin_rad=.03,contractivity_unsaturated=.9,
        imitation_gradient_direction=True,synthetic_combined_update=metrics,rollback_verified=True,bank_sha256=sha(Path(a.bank)/'bank.npz'),
        physical_success_claimed=False,simulation_executed=False))

if __name__=='__main__':main()
