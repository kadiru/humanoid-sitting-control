"""Persistent independent experiment workers; this script is not a coding agent."""
import argparse,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/home/kadir/humanoid-sitting');DATA=Path('/data/humanoid-sitting-perception/split_support_curriculum_02');STUDY=DATA/'ppo_01'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def stamp(phase,**kwargs):return dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),**kwargs)
def run(folder,label,cmd):
 write(folder/(label+'.command.json'),cmd)
 with (folder/(label+'.log')).open('w') as f:p=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 write(folder/(label+'.exit.json'),dict(returncode=p.returncode));assert p.returncode==0,(label,p.returncode)
def command(folder,surface,gpu,label,mode,seed,stage='all',checkpoint=None,profile=None):
 cmd=['bash','scripts/'+('run_supported_gpu0.sh' if gpu==0 else 'run_resting_gpu1.sh'),'../scripts/run_split_support.py','--bank','/data/humanoid-sitting-perception/two_surface_curriculum_01/selected_path_01','--output',str(folder/label),'--mode',mode,'--surface',surface,'--profiles',str(DATA/'split_profiles'/(profile or f'{surface}_selected.json')),'--seed',str(seed),'--start-stage',str(stage),'--num-envs','16' if mode=='train' else '28','--device','cuda:0',f'--kit_args=--/renderer/activeGpu={gpu},--/renderer/multiGpu/enabled=false']
 if mode=='train':cmd+=['--iterations','32']
 if checkpoint:cmd+=['--checkpoint',str(checkpoint)]
 return cmd

def freeze(folder,surface,gpu,seeds):
 check=DATA/f'{surface}_preflight_verified_01/verification.json';assert read(check)['passed']
 folder.mkdir(exist_ok=False);(folder/'frozen').mkdir()
 paths=[ROOT/'scripts'/n for n in ['run_split_support.py','run_split_curriculum_study.py','verify_split_support.py','run_supported_transition.py','run_supported_postures.py','run_resting_posture.py','run_supported_gpu0.sh','run_resting_gpu1.sh']]
 paths +=[ROOT/'docs/SPLIT_SUPPORT_PROTOCOL.md',DATA/'split_profiles'/f'{surface}_selected.json',DATA/'split_profiles/old_reference.json',check]
 paths +=list((Path('/data/humanoid-sitting-perception/two_surface_curriculum_01/selected_path_01')).iterdir())
 paths +=list((ROOT/'sitting_task/src/humanoid_sitting/tasks').glob('*.py'))
 paths +=[ROOT/'sitting_task/src/humanoid_sitting'/n for n in ['two_surface_path.py','supported_surface.py','resting_objective.py']]
 hashes={str(p):sha(p) for p in paths if p.is_file()};old=read(DATA/'prior_preservation_before.json')['hashes']
 for p,h in old.items():assert sha(p)==h,p
 for i,p in enumerate(hashes):shutil.copy2(p,folder/'frozen'/f'{i:03d}_{Path(p).name}')
 write(folder/'protocol.json',dict(surface=surface,gpu=gpu,seeds=seeds,stages=[4,2,0,0],updates_per_seed=128,transitions_per_seed=131072,total_transitions=262144,hashes=hashes,prior_hashes=old,limitations=['Two known rigid surfaces, seven known starts per surface, four placement repetitions; no unseen surfaces or perturbations.','Two independent training seeds per surface reuse all evaluation conditions.','Reference-assisted residual PPO; repaired reference success is not evidence that PPO learned the transition.','Torque proxies are not electrical energy or proof of a global optimum.','Effort comparisons use common full-duration support successes only.','Reference and fixed posture appended reference features differ; physical state and sensor histories must match exactly.','Slew validation uses10Hz exported traces and frozen50Hz controller bounds.']))

def worker(surface,gpu):
 folder=STUDY/surface;seeds=[55211,55217] if surface=='sitting' else [55311,55317]
 freeze(folder,surface,gpu,seeds)
 try:
  for seed in seeds:
   checkpoint=None;history=[];updates=0;transitions=0;active=0;warm=0
   for block,stage in enumerate([4,2,0,0]):
    label=f'train_{seed}_block{block}_stage{stage}';write(folder/'status.json',stamp('training',active=label,seed=seed,stage=stage));write(folder/f'seed_{seed}.json',stamp('training',active=label,history=history,updates=updates,transitions=transitions))
    run(folder,label,command(folder,surface,gpu,label,'train',seed,stage,checkpoint));v=read(folder/label/'verification.json');assert v['updates']==32 and v['transitions']==32768 and v['active']+v['warmup']==32768
    updates+=32;transitions+=32768;active+=v['active'];warm+=v['warmup'];checkpoint=folder/label/'final.pt';history.append(dict(stage=stage,block=block,train=label,budget=v,checkpoint=str(checkpoint)))
   write(folder/f'seed_{seed}.json',stamp('complete',history=history,updates=updates,transitions=transitions,active_transitions=active,warmup_transitions=warm,last_checkpoint=str(checkpoint)))
   label=f'policy_{seed}';write(folder/'status.json',stamp('evaluation',active=label));run(folder,label,command(folder,surface,gpu,label,'evaluate',55101,checkpoint=checkpoint))
  for label,mode,profile in [('fixed','fixed',None),('old_reference','reference','old_reference.json'),('reference','reference',None)]:
   write(folder/'status.json',stamp('baseline',active=label));run(folder,label,command(folder,surface,gpu,label,mode,55101,profile=profile))
  write(folder/'status.json',stamp('complete_pending_verification'))
  run(folder,'verification_01',[sys.executable,str(ROOT/'scripts/verify_split_support.py'),'--data',str(folder),'--output',str(folder/'verified_01')])
  write(folder/'status.json',stamp('verified_complete'))
 except Exception as e:
  write(folder/'status.json',stamp('execution_failed',error=repr(e)));raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--surface',choices=['sitting','leaning'],required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();STUDY.mkdir(exist_ok=True);worker(a.surface,a.gpu)
if __name__=='__main__':main()
