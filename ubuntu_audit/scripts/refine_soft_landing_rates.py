"""Bounded episode-level REINFORCE refinement of two sensor-only governor rates.
This is not PPO or full actor refinement. Simulation safety outcomes form reward,
never controller inputs. Use training banks only.
"""
import argparse,json
from pathlib import Path
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--output',required=True);p.add_argument('--results');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
if a.prepare:
 torch.manual_seed(712);mu=torch.tensor([.96,1.02]);std=torch.tensor([.06,.06]);actions=mu+std*torch.randn(24,2)
 assert actions.min()>.7 and actions.max()<1.2
 (out/'sampled_parameters.json').write_text(json.dumps({'parameters':actions.tolist(),'mean':mu.tolist(),'std':std.tolist(),'seed':712},indent=2))
else:
 r=json.loads(Path(a.results).read_text());assert r['geometry_seed'] not in (24001,25001,26001)
 sample=json.loads((out/'sampled_parameters.json').read_text());actions=torch.tensor(sample['parameters']);mu=torch.tensor(sample['mean'],requires_grad=True);std=torch.tensor(sample['std']);assert len(actions)==len(r['outcomes'])
 reward=[]
 for o in r['outcomes']:
  # Every safety failure is penalized; reward keeps the original limits.
  failures=sum(o[k] for k in ['fall','unintended_contact','joint_torque_violation','excessive_descent','excessive_impact','pelvis_floor'])
  reward.append(2*float(o['success'])-4*failures-o['max_seat_impact_N']/632.527832-.5*o['max_descent_m_s'])
 rewards=torch.tensor(reward);advantages=(rewards-rewards.mean())/rewards.std().clamp_min(.01)
 logprob=torch.distributions.Normal(mu,std).log_prob(actions).sum(-1);loss=-(logprob*advantages).mean();loss.backward();grad=mu.grad.clone();change=(-.001*grad).clamp(-.025,.025);updated=(mu.detach()+change).clamp(.8,1.1)
 torch.save({'mean':updated,'initial_mean':mu.detach(),'std':std,'gradient':grad,'transitions':24000,'method':'one episode-level REINFORCE update; two governor rates only','student_weights_unchanged':True},out/'rates.pt')
 (out/'learned_parameters.json').write_text(json.dumps({'parameters':[updated.tolist()]},indent=2))
 (out/'training.json').write_text(json.dumps({'reward_mean':rewards.mean().item(),'reward':reward,'loss':loss.item(),'gradient':grad.tolist(),'initial_mean':mu.detach().tolist(),'learned_mean':updated.tolist(),'training_bank':r['geometry_seed'],'updates':1,'episodes':len(reward),'control_transitions':24000,'unchanged_safety_limits':True},indent=2));print(updated)
