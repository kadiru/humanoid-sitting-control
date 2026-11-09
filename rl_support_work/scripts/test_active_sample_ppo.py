"""Regression for actor momentum and contaminated inactive statistics."""
import argparse,copy,json
from pathlib import Path
from types import SimpleNamespace
import torch
from tensordict import TensorDict
from humanoid_sitting.active_sample_ppo import optimize_batch

class Model(torch.nn.Module):
    def __init__(self,actor=False):
        super().__init__();self.net=torch.nn.Linear(4,2 if actor else 1);self.actor=actor
        if actor:self.logstd=torch.nn.Parameter(torch.zeros(2)-1)
    def forward(self,obs,stochastic_output=False):
        y=self.net(obs['policy'])
        if self.actor:
            self.dist=torch.distributions.Normal(y,self.logstd.exp());return self.dist.sample() if stochastic_output else y
        return y
    def get_output_log_prob(self,a):return self.dist.log_prob(a).sum(-1)
    @property
    def output_entropy(self):return self.dist.entropy().sum(-1)
    @property
    def output_distribution_params(self):return self.dist.mean,self.dist.stddev
    def get_kl_divergence(self,old,new):return torch.distributions.kl_divergence(torch.distributions.Normal(*old),torch.distributions.Normal(*new)).sum(-1)

def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();torch.manual_seed(56801)
    actor=Model(True);critic=Model();alg=SimpleNamespace(actor=actor,critic=critic,clip_param=.1,value_loss_coef=1.,entropy_coef=0.,max_grad_norm=1.)
    alg.optimizer=torch.optim.Adam([*actor.parameters(),*critic.parameters()],lr=1e-4)
    obs=TensorDict({'policy':torch.randn(16,4),'learn':torch.ones(16,1)},batch_size=[16])
    with torch.no_grad():
        actions=actor(obs,True);oldlog=actor.get_output_log_prob(actions);params=tuple(x.clone() for x in actor.output_distribution_params);values=critic(obs).clone()
    batch=SimpleNamespace(observations=obs,actions=actions,old_actions_log_prob=oldlog[:,None],old_distribution_params=params,values=values,returns=torch.randn(16,1),advantages=torch.randn(16,1))
    for _ in range(3):optimize_batch(alg,batch)
    before=copy.deepcopy(actor.state_dict());astate=[copy.deepcopy(alg.optimizer.state[p]) for p in actor.parameters()];cprev=copy.deepcopy(critic.state_dict())
    for _ in range(3):optimize_batch(alg,batch,False)
    assert equal(before,actor.state_dict());assert all(equal(s,alg.optimizer.state[p]) for p,s in zip(actor.parameters(),astate));assert not equal(cprev,critic.state_dict())
    batch.observations['learn'].zero_();allbefore=copy.deepcopy(alg.optimizer.state_dict());ap=copy.deepcopy(actor.state_dict());cp=copy.deepcopy(critic.state_dict())
    assert optimize_batch(alg,batch) is None
    assert equal(allbefore,alg.optimizer.state_dict()) and equal(ap,actor.state_dict()) and equal(cp,critic.state_dict())
    batch.observations['learn'][:8]=1
    del actor.dist  # Discard cached non-leaf distribution tensors before cloning.
    clone=copy.deepcopy(alg);clean=copy.deepcopy(batch)
    # Inactive values are poisoned; active updates must remain bit-identical.
    batch.observations['policy'][8:]=1e6;batch.returns[8:]=-1e9;batch.advantages[8:]=1e9
    r1=optimize_batch(alg,batch);r2=optimize_batch(clone,clean)
    assert r1==r2 and equal(alg.actor.state_dict(),clone.actor.state_dict()) and equal(alg.critic.state_dict(),clone.critic.state_dict()) and equal(alg.optimizer.state_dict(),clone.optimizer.state_dict())
    result=dict(pass_all=True,disabled_actor_weights_and_adam_state_unchanged=True,critic_updates_while_actor_disabled=True,all_inactive_skips_optimizer=True,inactive_poison_excluded_from_losses_kl_and_gradients=True)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists();out.write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
