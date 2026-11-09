"""CPU-only validation on all saved72 cases plus synthetic optimizer checks."""
import copy,importlib.util,json,time
from pathlib import Path
import numpy as np
import torch
from policy import Learner,command
from common import OBS,ROOT,sha

torch.set_num_threads(1)
here=Path(__file__).resolve().parent
source=ROOT/'outputs/armchair_rest_ppo_20261105_v1'
prior=source/'prepared/initial.pt'
data=source/'learned_sitting72/trajectory.npz'
oldsource=ROOT/'experiments/armchair_rest_trainer_20261105_v1/policy.py'
spec=importlib.util.spec_from_file_location('unchanged_old_policy',oldsource)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
agent=Learner(np.zeros(OBS),np.ones(OBS),65011);agent.load(prior)
previous=old.Learner(np.zeros(OBS),np.ones(OBS),65011);previous.load(prior)
started=time.monotonic()
with np.load(data) as d:
    assert d['obs'].shape==(1000,72,OBS)
    observations=torch.tensor(d['obs'].reshape(-1,OBS))
    priorcommands=torch.tensor(d['previous'][:,:,:23].reshape(-1,23))
with np.load(source/'prepared/demonstrations.npz') as d:
    demonstrations=torch.tensor(d['obs'][::23])
agent.bind_demonstrations(demonstrations)
old_changed=0;maximum_old=0.;validated=0
for start in range(0,len(observations),512):
    obs=observations[start:start+512]
    baseline=agent.prior(obs)
    assert not agent.residual_mask(obs).bool().any()
    for seed in (65011,65017):
        torch.manual_seed(seed+start);sample,logp,_,_=agent.choose(obs,True)
        assert torch.equal(sample,baseline) and torch.equal(logp,torch.zeros_like(logp))
    torch.manual_seed(65011+start);bad=previous.choose(obs,True)[0]
    delta=(bad-baseline).abs();old_changed+=int((delta.max(1).values>0).sum())
    maximum_old=max(maximum_old,float(delta.max()));validated+=len(obs)
    assert time.monotonic()-started<240
active=demonstrations[agent.residual_mask(demonstrations)[:,0].bool()][:512]
assert len(active)>32
mix=torch.cat((observations[:256],active))
torch.manual_seed(66101)
a,lp,v,mu=agent.choose(mix)
mask=agent.residual_mask(mix)
expected=(torch.distributions.Normal(mu,agent.std).log_prob(a)*mask).sum(1)
torch.testing.assert_close(lp,expected,rtol=0,atol=0)
assert (a[256:]-mu[256:]).abs().sum()>0
protected_before=agent.choose(observations[::71],True)[0].clone()
before=copy.deepcopy(agent.actor.state_dict());critic_before=copy.deepcopy(agent.critic.state_dict())
# Artificial advantage correlated with sampled residual; checks optimizer plumbing only.
advantage=(a-mu)[:,0]/agent.std
stats=agent.update(mix,a,lp,mu,advantage,v+advantage)
assert stats['actor_steps']>0
assert any(not torch.equal(before[k],x) for k,x in agent.actor.state_dict().items())
assert any(not torch.equal(critic_before[k],x) for k,x in agent.critic.state_dict().items())
assert torch.equal(protected_before,agent.choose(observations[::71],True)[0])
# No actor step for a wholly deterministic batch; this must not trip NaN paths.
frozen=copy.deepcopy(agent.actor.state_dict())
pobs=observations[:128];pa,pl,pv,pm=agent.choose(pobs)
pstats=agent.update(pobs,pa,pl,pm,torch.zeros_like(pv),pv+1.)
assert pstats['actor_steps']==0
assert all(torch.equal(frozen[k],x) for k,x in agent.actor.state_dict().items())
assert pstats['active_actor_samples']==0 and np.isfinite(pstats['measured_kl'])
old_optimized_rejected=False
try:agent.load(source/'training_65011/final.pt')
except AssertionError as e:old_optimized_rejected=str(e)=='OLD_TRAINING_CANNOT_RESUME'
assert old_optimized_rejected
result=dict(passed=True,saved_cases=72,full_saved_control_observations=validated,
    protected_stochastic_samples_checked=2*validated,
    original_unprotected_noise_samples=old_changed,original_max_noise_rad=maximum_old,
    corrected_protected_action_difference_rad=0.,active_exploration_preserved=True,
    active_actor_optimizer_steps=stats['actor_steps'],synthetic_critic_steps=stats['critic_steps']+pstats['critic_steps'],
    mixed_mask_logprob_exact=True,zero_active_batch_actor_steps=0,old_optimized_checkpoint_rejected=True,
    physical_simulation_steps=0,physical_PPO_updates=0,
    limitation='Software distribution/optimizer checks only. Does not repair9new-skill regressions,8original sitting failures, or validate new furniture demonstrations.',
    seconds=time.monotonic()-started,
    inputs={str(p):sha(p) for p in (prior,data,oldsource,here/'policy.py')})
with (here/'validation.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
