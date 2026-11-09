import os,time,json,hashlib,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';START=time.monotonic();H=Path(__file__).resolve().parent;R=Path('/home/kadir/humanoid-sitting')
sys.path.insert(1,str(R/'experiments/armchair_rest_ppo_20261105_v1'))
freeze=json.loads((H/'freeze.json').read_text())
for p,h in freeze['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
import numpy as np
import torch
torch.set_num_threads(1)
from rewards import staged_reward
from rest_effort import RestEffortWindow
keys=['alive','legal','good','phase','q_error','speed','gyro','support_bw','loaded_slip','wrong_force','peak_bw','normalized_command_delta','held_ticks','first_failure','horizon','clean_completion','stable_completion','torque_squared','previous_potential']
rows=[]
for filename in freeze['trajectories']:
 with np.load(filename) as z:d={k:z[k] for k in keys}
 count=d['torque_squared'].shape[1];window=RestEffortWindow(count,'cpu');means=[]
 for value in d['torque_squared']:means.append(window.update(torch.as_tensor(value)).numpy().copy())
 means=np.stack(means);independent=np.stack([np.mean(d['torque_squared'][max(0,t-249):t+1].astype(np.float64),axis=0) for t in range(1000)])
 np.testing.assert_allclose(means,independent,rtol=3e-6,atol=3e-6)
 x={k:torch.as_tensor(v) for k,v in d.items()};x['steady_torque_squared']=torch.as_tensor(means);reward,terms=staged_reward(x)
 safe=d['alive'].astype(bool)&d['legal'].astype(bool);good=safe&d['good'].astype(bool)
 phi=2*np.exp(-np.maximum(d['q_error'],0))*safe+2*(d['phase']>=1)*safe*np.clip(d['support_bw']/.2,0,1)*np.exp(-(d['speed']/.1)**2-(d['gyro']/.25)**2)+4*good*np.clip(d['held_ticks']/250,0,1)-np.clip((d['loaded_slip']/.02)**2,0,1)-np.clip((d['wrong_force']/10)**2,0,1)-np.clip(d['peak_bw']-1,0,1)**2-.1*np.clip((d['normalized_command_delta']**2).mean(-1),0,1)
 phi=np.where(d['first_failure']|d['horizon']|~d['alive'].astype(bool),0,phi)
 bonus=.2*d['clean_completion']/(1+independent/1500.)
 expected=.02-20*d['first_failure']+400*d['clean_completion']+20*d['stable_completion']+bonus+.999*phi-d['previous_potential']
 np.testing.assert_allclose(reward.numpy(),expected,rtol=3e-6,atol=3e-6)
 assert np.all(bonus[~d['clean_completion'].astype(bool)]==0)
 rows.append(dict(source=filename,clean_completion=d['clean_completion'][-1].tolist(),last_instant_proxy_Nm2=d['torque_squared'][-1].tolist(),final5s_proxy_Nm2=independent[-1].tolist(),old_terminal_bonus=(.2*d['clean_completion'][-1]/(1+d['torque_squared'][-1]/1500)).tolist(),new_terminal_bonus=bonus[-1].tolist(),maximum_window_error=float(abs(means-independent).max()),independent_reward_replay=True))
# A one-tick torque dip must not erase the preceding249ticks of effort.
w=RestEffortWindow(1,'cpu')
for i in range(249):w.update(torch.tensor([1500.]))
dip=float(w.update(torch.tensor([0.]))[0]);assert dip==1494.
reset=RestEffortWindow(1,'cpu');assert float(reset.update(torch.tensor([25.]))[0])==25.
bad_input=False
try:reset.update(torch.tensor([float('nan')]))
except ValueError:bad_input=True
assert bad_input
result=dict(passed=True,trajectories=rows,controls=sum(1000*len(r['clean_completion']) for r in rows),one_tick_dip_proxy_Nm2=dip,reset_history_isolated=True,nonfinite_rejected=True,original_success_action_and_reward_weights_unchanged=True,physical_success_claim=False,deployed=False,PPO_updates=0,seconds=time.monotonic()-START)
with (H/'validation.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({'passed':True,'trajectories':len(rows),'controls':result['controls'],'seconds':result['seconds']}),flush=True)
