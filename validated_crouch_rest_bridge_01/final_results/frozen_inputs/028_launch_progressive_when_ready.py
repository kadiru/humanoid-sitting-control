"""One-shot preflight admission and launch; no agent or recurring scheduler."""
import hashlib,json,shutil,subprocess,time,traceback
from pathlib import Path
from run_progressive_seat_study import ROOT,DATA,SEEDS,read,sha,write,status

def main():
 state=DATA/'status.json'
 try:
  status(state,'awaiting_preflight')
  while True:
   statuses=[read(DATA/'preflight_01'/f'status_gpu{i}.json') if (DATA/'preflight_01'/f'status_gpu{i}.json').exists() else {} for i in [0,1]]
   assert not any(x.get('phase')=='failed' for x in statuses),statuses
   if all(x.get('phase')=='complete' for x in statuses):break
   time.sleep(10)
  admissions={}
  for label,path in [('near_reference5',DATA/'harness_near_01'),('near_reference4',DATA/'preflight_01/reference_stage4'),('learned_smoke',DATA/'preflight_01/learned_smoke')]:
   rows=read(path/'results.json')['episodes'];decision=dict(success=sum(x['success'] for x in rows),strict=sum(x['strict_transfer'] for x in rows),physical=sum(any(x['failures'].values()) for x in rows),count=len(rows));assert decision==dict(success=18,strict=18,physical=0,count=18),(label,decision);admissions[label]=decision
  for stage in [1,2,3,4,5]:assert read(DATA/'preflight_01'/f'reset_stage{stage}/reset_probe.json')['passed']
  prior=read(DATA/'prior_hashes.json')
  for p,h in prior.items():assert sha(p)==h,p
  smoke=read(DATA/'optimizer_smoke_01/verification.json');assert smoke['updates']==2 and smoke['transitions']==5760
  import torch
  initial=torch.load(DATA/'optimizer_smoke_01/initial.pt',weights_only=False,map_location='cpu');warm=torch.load(DATA/'optimizer_smoke_01/critic_warmup.pt',weights_only=False,map_location='cpu');final=torch.load(DATA/'optimizer_smoke_01/final.pt',weights_only=False,map_location='cpu')
  for role in ['actor','critic']:
   key=role+'_state_dict';assert all(torch.isfinite(v).all() for v in final[key].values());assert any(not torch.equal(initial[key][k],v) for k,v in final[key].items())
  assert all(torch.equal(initial['actor_state_dict'][k],v) for k,v in warm['actor_state_dict'].items())
  write(DATA/'admission.json',dict(passed=True,checks=admissions,reset_stages=[1,2,3,4,5],smoke=smoke,standing_reference=read(DATA/'harness_standing_01/results.json')['successes'],standing_reference_validated=False,scope='Validated near-seat transfer admits bounded progressive learning. The standing reference still has2 physical failures; not relabeled or claimed solved.'))
  study=DATA/'ppo_01';study.mkdir(exist_ok=False);frozen=study/'frozen';frozen.mkdir();paths=set(read(DATA/'harness_near_01/manifest.json')['sources'])
  paths.update(str(ROOT/x) for x in ['scripts/run_progressive_seat_study.py','scripts/run_near_seat_preflight.py','scripts/launch_progressive_when_ready.py','docs/NEAR_SEAT_CURRICULUM_PROTOCOL.md'])
  paths.update(str(DATA/x) for x in ['reset_library_01.npz','reservation.json','admission.json','prior_hashes.json'])
  paths.update(str(p) for p in (DATA/'reserved_bank').iterdir() if p.is_file())
  paths.update(str(p) for p in (DATA.parent/'seat_load_transfer_01/banks/development').iterdir() if p.is_file())
  hashes={p:sha(p) for p in sorted(paths)}
  for i,p in enumerate(sorted(paths)):shutil.copy2(p,frozen/f'{i:02d}_{Path(p).name}')
  protocol=dict(seeds=SEEDS,block_updates=[24,16,16,16,16,16,16,8],updates_per_seed=128,critic_only_updates=8,actor_enabled_updates=120,decisions_per_update=32,physics_per_decision=5,environments=18,transitions_per_seed=368640,total_transitions=737280,initial_settle_ticks_per_process=450,training_processes_per_seed=8,initial_settle_ticks_per_seed=3600,stage_order=[5,4,3,2,1,0],gate='All18 full-duration success, no physical or all-tick slip/friction/legal-contact violations, and each final5s seat vertical load>=.6BW; otherwise repeat same stage within fixed budget.',selection='Full-budget final only',hashes=hashes,prior_hashes=prior,frozen_unix=time.time(),separate_development_smoke_transitions=5760)
  write(study/'protocol.json',protocol)
  for gpu,seed in enumerate(SEEDS):
   session=f'near_seat_ppo_{seed}';command=f'cd {ROOT} && IsaacLab/.venv/bin/python scripts/run_progressive_seat_study.py --gpu {gpu} --seed {seed} > {DATA}/worker_{seed}.log 2>&1'
   subprocess.run(['tmux','new-session','-d','-s',session,command],check=True)
  status(state,'ppo_running',study=str(study),seeds=SEEDS,transitions_total=737280,source_hashes=len(hashes),prior_hashes=len(prior))
 except Exception as e:status(state,'admission_or_launch_failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
