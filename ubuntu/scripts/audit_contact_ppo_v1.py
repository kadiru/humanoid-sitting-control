"""Independent reward ranking, causal return and checkpoint replay checks."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from contact_reward_v1 import tick_reward,reward_to_go
from feedback_posture_ppo import FeedbackPPO,slew

def unit():
 n=1;k=7
 r=dict(force=np.zeros((n,k,3)),normal=np.tile([0.,0.,1.],(n,k,1)),weight=np.array([100.]),permitted=np.ones((n,k),bool),inside=np.ones((n,k),bool),rest_patch=np.array([[1,1,1,1,1,0,0]],bool),legal=np.ones(n,bool),wrong=np.zeros(n),failure=np.zeros(n,bool),alive=np.ones(n,bool),root_speed=np.zeros(n),gyro=np.zeros(n),good=np.zeros(n,bool),slip=np.zeros((n,k)))
 r['force'][0,0,2]=20
 good=float(tick_reward(r,10)[0]);assert abs(good-.04)<1e-12
 r['force'][0,0,2]=80;assert float(tick_reward(r,10)[0])==good
 r['force'][:]=0;assert float(tick_reward(r,10)[0])==0
 r['force'][0,0,2]=20;r['inside'][:]=False;assert float(tick_reward(r,10)[0])==0
 r['inside'][:]=True;r['wrong'][:]=20;assert float(tick_reward(r,10)[0])<0
 r['wrong'][:]=0;r['slip'][0,0]=.03;assert float(tick_reward(r,10)[0])<0
 r['slip'][:]=0;r['alive'][:]=False;assert float(tick_reward(r,10)[0])==0
 r['alive'][:]=True;assert float(tick_reward(r,7)[0])==0
 r['good'][:]=True;assert abs(float(tick_reward(r,10)[0])-.08)<1e-12
 x=torch.arange(30,dtype=torch.float64).reshape(10,3);base=x*.5;dense=x*.03;pen=torch.arange(3,dtype=torch.float64);ids=[0,2,9]
 got=reward_to_go(x,base,pen,ids,dense)
 manual=torch.stack([sum([-.25*(x[j]-base[j])/1350+dense[j] for j in range(i,10)])-pen for i in ids]);assert torch.allclose(got,manual,atol=1e-12,rtol=0)
 old=torch.zeros(24,3);new=slew(old,torch.full_like(old,100.));assert new.abs().max()<=.00040001
 print('AUDIT_UNIT_PASS 11 checks',flush=True)

def replay(folder):
 p=Path(folder);initial=torch.load(p/'initial.pt',map_location='cpu',weights_only=False)
 model=FeedbackPPO(initial['input_mean'],initial['input_scale'],63111,'cpu');model.load(p/'initial.pt')
 batches=torch.load(p/'decision_batches.pt',map_location='cpu',weights_only=False)
 for b in batches:
  expected=reward_to_go(b['torque_integrals'],b['baseline_integrals'],b['terminal_penalty'],b['decision_steps'],b['physical_cost'])
  assert torch.allclose(expected,b['reward'],atol=2e-5,rtol=1e-6)
  model.update(b['obs'].flatten(0,1),b['actions'].flatten(0,1),b['logp'].flatten(),b['old_mean'].flatten(0,1),b['reward'].flatten())
 final=torch.load(p/'final.pt',map_location='cpu',weights_only=False)
 errors={}
 for name in ['actor','critic']:
  errors[name]=max(float((v-final[name+'_state_dict'][k]).abs().max()) for k,v in getattr(model,name).state_dict().items())
  assert all(torch.isfinite(v).all() for v in final[name+'_state_dict'].values())
  assert any(not torch.equal(v,initial[name+'_state_dict'][k]) for k,v in final[name+'_state_dict'].items())
  assert errors[name]<2e-4,errors
 assert model.updates==final['updates']==initial['updates']+len(batches)
 (p/'cpu_replay_audit.json').write_text(json.dumps(dict(passed=True,max_parameter_error=errors,absolute_tolerance=2e-4,updates=model.updates),indent=2))
 print('CPU_REPLAY_PASS',errors,flush=True)
if __name__=='__main__':
 if len(sys.argv)==1:unit()
 else:replay(sys.argv[1])
