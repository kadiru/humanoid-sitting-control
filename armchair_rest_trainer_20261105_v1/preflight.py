"""CPU validation of saved rewards, command recurrence, and protected policy routing."""
import argparse
from pathlib import Path
import numpy as np
import torch
from common import ROOT,OBS,read,write,sha
from rewards import staged_reward
from audit import reward_numpy
from policy import Learner

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    prep=read(a.prepared/'preparation.json');assert prep['passed']
    demo=ROOT/'outputs/armchair_rest_ppo_20261105_v1'
    roots=[demo/r['label'] for r in read(demo/'demo_selection.json')['results'] if (demo/r['label']/'trajectory.npz').exists()]
    roots += [ROOT/'outputs/motion_prior_ppo_20261029_v1/prior_full72',ROOT/'outputs/motion_prior_ppo_20261029_v1/eval_sitting_64111_256']
    # Existing final comparison path is resolved from its stored manifest if needed;
    # absence cannot silently remove the positive/negative saved-data checks.
    roots=[r for r in roots if (r/'trajectory.npz').exists()]
    rows=[];clean_values=[];failed_values=[];discount=.999**np.arange(1000)
    for root in roots:
        with np.load(root/'trajectory.npz') as z:d=dict(z)
        effort=np.mean((d['physical_torque']/d['physical_torque_scale'])**2,axis=-1)
        d['mean_qualifying_effort']=np.cumsum(effort*d['good'],axis=0)/np.maximum(np.cumsum(d['good'],axis=0),1)
        d.pop('previous_potential',None);target=reward_numpy(d)*d['learn'];previous=torch.zeros(target.shape[1]);actual=[]
        for t in range(1000):
            x={k:torch.from_numpy(v[t]) for k,v in d.items() if v.shape[:2]==target.shape}
            x.update(previous_potential=previous,horizon=torch.full((target.shape[1],),t==999))
            r,terms=staged_reward(x);previous=terms['potential'];actual.append(r.numpy()*d['learn'][t])
        actual=np.array(actual);np.testing.assert_allclose(actual,target,rtol=3e-6,atol=3e-6)
        returns=(target*discount[:,None]).sum(0);clean=d['clean_completion'][-1].astype(bool)
        clean_values.extend(returns[clean].tolist());failed_values.extend(returns[~clean].tolist())
        rows.append(dict(path=str(root),sha256=sha(root/'trajectory.npz'),clean=clean.tolist(),returns=returns.tolist(),max_replay_error=float(abs(actual-target).max())))
    assert clean_values and failed_values and min(clean_values)>max(failed_values),'SUCCESS_RANKING'
    agent=Learner(np.zeros(OBS),np.ones(OBS),65011);agent.load(a.prepared/'initial.pt')
    original=torch.load(prep['initial_source'],map_location='cpu',weights_only=False)
    with np.load(ROOT/'outputs/full_body_contact_imitation_v4/dataset/sitting/bank.npz') as z:obs=z['obs'][:,z['admitted']].copy()
    initial=obs[0,:,222:333];obs=np.concatenate((obs,np.broadcast_to(initial[None],(1000,64,111))),axis=-1).reshape(-1,447)
    # Finite phase grid covers every protected signature through the full20s clock.
    protected=torch.tensor(obs[::37]);baseline=agent.choose(protected,False)[0].detach().clone()
    with np.load(a.prepared/'demonstrations.npz') as z:newobs=torch.tensor(z['obs'][::23])
    agent.bind_demonstrations(newobs)
    action,logp,value,mean=agent.choose(newobs);noise=torch.randn_like(value)
    m=agent.update(newobs,action.detach(),logp.detach(),mean.detach(),noise,value.detach()+noise)
    assert m['actor_steps']>0
    torch.testing.assert_close(agent.choose(protected,False)[0],baseline,rtol=0,atol=0)
    for k,v in original['library'].items():assert torch.equal(agent.sitting[k],v)
    write(a.output,dict(passed=True,scope='CPU arithmetic and synthetic optimizer plumbing only; no physical PPO result',new_demonstrations=prep['new_demonstrations'],saved_trajectory_rewards=rows,all_clean_outrank_all_unclean=True,minimum_clean_return=min(clean_values),maximum_unclean_return=max(failed_values),protected_sitting_action_change=0,protected_samples=len(protected),synthetic_actor_steps=m['actor_steps'],imitation_is_soft=True,physical_criteria_unchanged=True,action_limits_unchanged=True))
if __name__=='__main__':main()
