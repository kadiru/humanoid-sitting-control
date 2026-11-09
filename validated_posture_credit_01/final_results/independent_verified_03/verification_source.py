"""Independent full-return PPO, budget, physical outcome and comparison audit."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import torch
from posture_credit_common import *
from posture_episode_ppo import PosturePPO,episode_return
from verify_seat_load_study import evaluation,starts,compare,load,equal
from verify_progressive_seat_study import quality

def contacts(path):
 z=np.load(path/'every_tick.npz');f=z['force'];n=z['normal'];fn=(f*n).sum(-1)
 ft=np.linalg.norm(f-fn[...,None]*n,axis=-1);active=np.linalg.norm(f,axis=-1)>.01*z['weight'][...,None]
 util=np.where(active,ft/(.6*np.maximum(fn,1e-7)),0).max(-1)
 rows=read(path/'results.json')['episodes'];assert len(z['t'])==1000
 assert (~z['no_slip']).sum(0).tolist()==[x['slip_ticks'] for x in rows]
 assert (~z['traction']).sum(0).tolist()==[x['traction_bad_ticks'] for x in rows]
 return dict(slip_cases=np.flatnonzero((~z['no_slip']).any(0)).tolist(),max_friction_utilization=float(util.max()),
   unwanted_contact_cases=np.flatnonzero((z['wrong']>10).any(0)).tolist(),physical_failure_cases=np.flatnonzero(z['failure'].any(0)).tolist(),
   margin_flags=[dict(case=i,ticks=int((~z['traction'][:,i]).sum()),times_s=z['t'][~z['traction'][:,i]].tolist(),peak_utilization=float(util[:,i].max())) for i in range(len(rows)) if (~z['traction'][:,i]).any()])

def prefix(reference,path):
 result=starts(reference,path);assert result['exact']
 a=np.load(reference/'trace.npz');b=np.load(path/'trace.npz');mask=a['t']<=11
 for k in ['q','body','command','reference','torque','geometry']:assert np.array_equal(a[k][mask],b[k][mask]),(path,k)
 for name in ['traction_startup.npz','episode_decisions.npz']:
  a=np.load(reference/name);b=np.load(path/name)
  keys=a.files if name=='traction_startup.npz' else ['observation']
  for k in keys:assert np.array_equal(a[k],b[k]),(path,name,k)
 return dict(**result,first11s_exact=True,actual_11s_actor_input_exact=True)

def replay_optimizer(folder,cal,updates=10):
 initial=load(folder/'initial.pt');final=load(folder/'final.pt');batches=load(folder/'decision_batches.pt');audit=read(folder/'learning_audit.json')
 assert len(batches)==len(audit)==updates
 policy=PosturePPO(cal['mean'],cal['scale'],1);policy.load(folder/'initial.pt')
 assert policy.actor[-1].weight.count_nonzero()==0 and policy.actor[-1].bias.count_nonzero()==0
 for i,(batch,row) in enumerate(zip(batches,audit)):
  obs=batch['obs'];action=batch['actions'];assert obs.shape==(18,414) and action.shape==(18,3)
  with torch.no_grad():
   mu=policy.distribution(obs).mean;logp=policy.distribution(obs).log_prob(action).sum(-1)
  assert torch.allclose(mu,batch['old_mean'],atol=2e-5,rtol=2e-5),(i,'behavior mean')
  assert torch.allclose(logp,batch['logp'],atol=1e-4,rtol=2e-5),(i,'behavior logp')
  replay=policy.update(obs,action,batch['logp'],batch['old_mean'],batch['reward'])
  saved=load(folder/f'model_{i+1:02d}.pt')
  assert replay['actor_steps']==row['optimizer']['actor_steps'] and replay['critic_steps']==8
  assert replay['max_accepted_kl']<=.020001 and row['optimizer']['max_accepted_kl']<=.02
  # Cross-device reductions can differ slightly; this explicit numerical audit
  # complements exact persisted optimizer and checkpoint identity checks.
  for role,model in [('actor_state_dict',policy.actor),('critic_state_dict',policy.critic)]:
   for k,t in model.state_dict().items():assert torch.allclose(t,saved[role][k],atol=5e-5,rtol=5e-4),(i,role,k,float(abs(t-saved[role][k]).max()))
  for role in ['actor_optimizer','critic_optimizer']:
   expected=sum(r['optimizer']['actor_steps'] if role=='actor_optimizer' else 8 for r in audit[:i+1])
   assert all(int(s['step'])==expected for s in saved[role]['state'].values())
 for role in ['actor_state_dict','critic_state_dict']:
  assert all(torch.isfinite(t).all() for t in final[role].values())
  assert any(not torch.equal(t,final[role][k]) for k,t in initial[role].items())
 assert equal(final,load(folder/f'model_{updates:02d}.pt'))
 return dict(passed=True,CPU_replay_all_updates=True,weight_atol=5e-5,weight_rtol=5e-4,actor_steps=sum(x['optimizer']['actor_steps'] for x in audit),critic_steps=8*updates,
   note='CPU replay allows documented floating-point tolerance; saved checkpoint, input calibration and optimizer-step identities are exact')

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 protocol=read(a.study/'protocol.json');assert sha(a.study/'protocol.json')==(a.study/'protocol.sha256').read_text().strip()
 check_hashes(protocol['hashes']);check_hashes(protocol['prior_hashes']);cal=read(protocol['development_only_calibration']);barrier=read(a.study/'holdout_barrier.json')
 out=dict(passed=False,source_hashes=len(protocol['hashes']),prior_hashes=len(protocol['prior_hashes']),training={},evaluations={},contacts={},startups={},comparisons={},caveats=protocol['caveats'])
 def source_check(path):
  manifest=read(path/'manifest.json')
  for name,h in manifest['sources'].items():assert protocol['hashes'][name]==h,(path,name)
  for name in ['run_posture_episode_ppo.py','posture_episode_ppo.py','run_traction_transfer_ppo.py','foot_wrench_observer.py','traction_clearance_planner_v2.py','traction_feedback.py']:
   assert sha(path/'source'/name)==protocol['hashes'][str(ROOT/'scripts'/name)],(path,name)
  assert read(path/'traction_config.json')==read(OLD/'selected_controller.json')
 for seed in SEEDS:
  folder=a.study/f'seed_{seed}/train';c=read(folder/'episode_contract.json');assert c['updates']==10 and c['episodes']==180 and c['active']==180000 and c['warmup']==4050 and c['transitions']==184050
  source_check(folder)
  assert c['initial_settle_excluded']==450 and c['span_rad']==.008 and c['decision_time_s']==11
  initial=load(folder/'initial.pt');smoke=load(DATA/f'preflight/smoke_{seed}/initial.pt');assert equal(initial,smoke)
  assert torch.equal(initial['input_mean'],torch.tensor(cal['mean'],dtype=torch.float32)) and torch.equal(initial['input_scale'],torch.tensor(cal['scale'],dtype=torch.float32))
  rows=read(folder/'results.json')['episodes'];audit=read(folder/'learning_audit.json');batches=load(folder/'decision_batches.pt')
  assert all(r['steps']==1000 and r['start_stage']==0 for r in rows)
  for i in range(10):
   rewards=episode_return(rows[i*18:(i+1)*18],cal['baseline_whole20_Nm2'])
   assert np.allclose(rewards,batches[i]['reward'].numpy(),atol=2e-6,rtol=1e-6)
   offsets=np.array(audit[i]['offsets_rad']);assert abs(offsets).max()<=.008
   assert np.allclose(offsets,.008*np.tanh(batches[i]['actions'].numpy()),atol=1e-8)
  opt=replay_optimizer(folder,cal)
  finished=read(a.study/f'training_finished_{seed}.json');assert finished==barrier['finals'][str(seed)] and finished['sha256']==sha(folder/'final.pt') and finished['timestamp']<=barrier['timestamp']
  out['training'][str(seed)]=dict(contract=c,optimizer=opt,stable_episodes=sum(r['success'] for r in rows),physical_failed_episodes=sum(any(r['failures'].values()) for r in rows),zero_unfinished_active_ticks=True,zero_pending_warmup=True,initial_equals_smoke_exact=True)
 for bank,count in [('development',18),('holdout',16)]:
  paths={k:Path(v) for k,v in protocol['development_baselines'].items()} if bank=='development' else {k:a.study/('holdout_'+k) for k in protocol['development_baselines']}
  for seed in SEEDS:paths[str(seed)]=a.study/(f'seed_{seed}/development_final' if bank=='development' else f'holdout_policy_{seed}')
  all_rows={};geoms={}
  for name,path in paths.items():
   cp=a.study/f'seed_{name}/train/final.pt' if name.isdigit() else OLD/f'ppo_01/seed_{name.split("_")[-1]}/train/final.pt' if name.startswith('previous') else None
   summary,rows,g=evaluation(path,count,cp);quality(summary,rows);out['evaluations'][bank+'_'+name]=summary
   if name.isdigit():source_check(path)
   out['startups'][bank+'_'+name]=starts(paths['reference'],path)
   if name.isdigit():out['startups'][bank+'_'+name]=prefix(paths['reference'],path)
   out['contacts'][bank+'_'+name]=contacts(path);all_rows[name]=rows;geoms[name]=g
   if name.isdigit():
    decisions=np.load(path/'episode_decisions.npz');actions=decisions['action'];assert actions.shape==(1,count,3)
    model=PosturePPO(cal['mean'],cal['scale'],1);model.load(cp)
    output=model.choose(torch.tensor(decisions['observation'][0]),False)[0].numpy()
    assert np.allclose(output,actions[0],atol=1e-5,rtol=1e-5)
   if bank=='holdout':assert (path/'manifest.json').stat().st_mtime>=barrier['timestamp']
  for seed in SEEDS:
   out['comparisons'][f'{bank}_{seed}']={b:compare(all_rows[str(seed)],all_rows[b],geoms[str(seed)]) for b in protocol['development_baselines']}
   out['comparisons'][f'{bank}_{seed}_strict']={b:compare([r for r in all_rows[str(seed)] if r['strict_transfer']],all_rows[b],geoms[str(seed)]) for b in protocol['development_baselines']}
 check_hashes(protocol['hashes']);check_hashes(protocol['prior_hashes']);out['passed']=True
 write=lambda p,x:p.write_text(json.dumps(x,indent=2))
 write(a.output/'verification.json',out);shutil.copy2(__file__,a.output/'verification_source.py')
 print(json.dumps(dict(passed=True,outcomes={k:{z:v[z] for z in ['successes','physical_failures','strict_successes']} for k,v in out['evaluations'].items()}),indent=2))
if __name__=='__main__':main()
