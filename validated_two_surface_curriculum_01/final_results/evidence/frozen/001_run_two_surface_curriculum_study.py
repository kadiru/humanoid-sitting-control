"""Persistent staged experiment coordinator; it is not another coding agent."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/home/kadir/humanoid-sitting');DATA=Path('/data/humanoid-sitting-perception/two_surface_curriculum_01');OUT=DATA/'ppo_01'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def status(phase,**kw):write(DATA/'status.json',dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),study=str(OUT),**kw))
def command(label,mode,gpu,seed,stage='all',checkpoint=None):
 wrapper='run_supported_gpu0.sh' if gpu==0 else 'run_resting_gpu1.sh'
 cmd=['bash','scripts/'+wrapper,'../scripts/run_two_surface_path.py','--bank',str(DATA/'selected_path_01'),'--output',str(OUT/label),'--mode',mode,'--seed',str(seed),'--start-stage',str(stage),'--device','cuda:0',f'--kit_args=--/renderer/activeGpu={gpu},--/renderer/multiGpu/enabled=false']
 if mode=='train':cmd+=['--iterations','32','--num-envs','16']
 if checkpoint:cmd+=['--checkpoint',str(checkpoint)]
 return cmd
def run(label,cmd):
 write(OUT/(label+'.command.json'),cmd)
 with (OUT/(label+'.log')).open('w') as f:p=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 write(OUT/(label+'.exit.json'),dict(returncode=p.returncode))
 if p.returncode:raise RuntimeError(f'{label} execution failed: {p.returncode}')
def worker(seed,gpu):
 history=[];checkpoint=None;passed=None;updates=0;transitions=0;active=0;warmup=0;stop=False
 for stage in range(6,-1,-1):
  for block in [1,2]:
   label=f'train_{seed}_stage{stage}_block{block}'
   write(OUT/f'worker_{seed}.json',dict(phase='training',active=label,history=history,updates=updates,transitions=transitions))
   run(label,command(label,'train',gpu,seed,stage,checkpoint))
   v=read(OUT/label/'verification.json');assert v['updates']==32 and v['transitions']==32768 and v['active']+v['warmup']==32768
   updates+=v['updates'];transitions+=v['transitions'];active+=v['active'];warmup+=v['warmup'];checkpoint=OUT/label/'final.pt'
   ev=f'gate_{seed}_stage{stage}_block{block}'
   write(OUT/f'worker_{seed}.json',dict(phase='gate',active=ev,history=history,updates=updates,transitions=transitions))
   run(ev,command(ev,'evaluate',gpu,54901,stage,checkpoint))
   result=read(OUT/ev/'results.json');ok=result['count']==2 and result['successes']==2
   history.append(dict(stage=stage,block=block,train=label,evaluation=ev,successes=result['successes'],passed=ok,checkpoint=str(checkpoint),budget=v))
   write(OUT/f'worker_{seed}.json',dict(phase='stage_complete',history=history,updates=updates,transitions=transitions,active_transitions=active,warmup_transitions=warmup))
   if ok:passed=checkpoint;break
   if block==2:stop=True
  if stop:break
 write(OUT/f'worker_{seed}.json',dict(phase='complete',history=history,updates=updates,transitions=transitions,active_transitions=active,warmup_transitions=warmup,last_checkpoint=str(checkpoint),last_passing_checkpoint=str(passed) if passed else None,reached_standing=any(x['stage']==0 for x in history),passed_standing=any(x['stage']==0 and x['passed'] for x in history)))
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--gpu',type=int);a=p.parse_args()
 if a.worker:
  try:worker(a.worker,a.gpu)
  except Exception as e:
   write(OUT/f'worker_{a.worker}_error.json',dict(error=repr(e)));raise
  return
 assert read(DATA/'preflight_verification_01.json')['passed']
 OUT.mkdir(exist_ok=False);(OUT/'frozen').mkdir()
 paths=[ROOT/'scripts/run_two_surface_path.py',ROOT/'scripts/run_two_surface_curriculum_study.py',ROOT/'scripts/test_two_surface_path.py',ROOT/'sitting_task/src/humanoid_sitting/two_surface_path.py',ROOT/'docs/TWO_SURFACE_CURRICULUM_PROTOCOL.md',ROOT/'scripts/run_supported_transition.py',ROOT/'scripts/run_supported_postures.py',ROOT/'scripts/run_resting_posture.py',ROOT/'sitting_task/src/humanoid_sitting/supported_surface.py',ROOT/'sitting_task/src/humanoid_sitting/resting_objective.py',DATA/'preflight_verification_01.json']
 paths+=list((DATA/'selected_path_01').iterdir());paths+=list((ROOT/'sitting_task/src/humanoid_sitting/tasks').glob('*.py'))
 hashes={str(p):sha(p) for p in paths if p.is_file()}
 prior=read(Path('/data/humanoid-sitting-perception/effort_rest_01/ppo/protocol.json'));old=dict(prior['hashes'],**prior['prior_study_hashes'])
 for p,h in old.items():assert sha(Path(p))==h,p
 import shutil
 for i,(p,h) in enumerate(hashes.items()):shutil.copy2(p,OUT/'frozen'/f'{i:03d}_{Path(p).name}')
 write(OUT/'protocol.json',dict(seeds=[54911,54917],max_updates_per_seed=448,max_transitions_per_seed=458752,max_total_transitions=917504,block_updates=32,transitions_per_block=32768,stages=[6,5,4,3,2,1,0],max_blocks_per_stage=2,stage_gate='2/2 full20s support successes',hashes=hashes,prior_hashes=old,development_smoke_transitions=2048,final_evaluation='All14 known starting conditions; last checkpoint and last-passing if different'))
 workers=[]
 for seed,gpu in [(54911,0),(54917,1)]:
  f=(OUT/f'worker_{seed}.log').open('w');proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(seed),'--gpu',str(gpu)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);workers.append((seed,proc,f))
 status('training',pids={str(s):p.pid for s,p,f in workers})
 while any(p.poll() is None for s,p,f in workers):time.sleep(10)
 for s,p,f in workers:
  f.close()
  if p.returncode:status('execution_failed',seed=s,returncode=p.returncode);return 1
 for p,h in hashes.items():assert sha(Path(p))==h,p
 jobs=[('fixed_all','fixed',None),('reference_all','reference',None)]
 for seed in [54911,54917]:
  state=read(OUT/f'worker_{seed}.json');jobs.append((f'eval_{seed}_last','evaluate',state['last_checkpoint']))
  if state['last_passing_checkpoint'] and state['last_passing_checkpoint']!=state['last_checkpoint']:jobs.append((f'eval_{seed}_last_passing','evaluate',state['last_passing_checkpoint']))
 for label,mode,checkpoint in jobs:
  status('final_evaluation',active=label);run(label,command(label,mode,0,54901,'all',checkpoint))
 status('complete_pending_verification')
 return 0
if __name__=='__main__':sys.exit(main())
