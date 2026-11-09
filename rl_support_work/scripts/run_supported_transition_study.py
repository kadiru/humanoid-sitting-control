"""Bounded two-seed transition study; persistent coordinator, never an agent."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,time,sys
ROOT=Path('/home/kadir/humanoid-sitting')
DATA=Path('/data/humanoid-sitting-perception/supported_surfaces_01')
OUT=DATA/'transition_ppo_01'

def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def status(phase,**kw):write(DATA/'status.json',dict(phase=phase,stage='standing_to_support_ppo',utc=datetime.now(timezone.utc).isoformat(),study=str(OUT),**kw))
def command(label,mode,bank,gpu,seed,checkpoint=None):
 wrapper='run_supported_gpu0.sh' if gpu==0 else 'run_resting_gpu1.sh'
 cmd=['bash','scripts/'+wrapper,'python','-u','../scripts/run_supported_transition.py','--output',str(OUT/label),'--bank',str(DATA/'transition_banks_03'/bank),'--mode',mode,'--seconds','20','--seed',str(seed),'--device','cuda:0',f'--kit_args=--/renderer/activeGpu={gpu},--/renderer/multiGpu/enabled=false']
 if mode=='train':cmd+=['--num-envs','26','--iterations','224']
 if checkpoint:cmd+=['--checkpoint',str(checkpoint)]
 return cmd

def main():
 OUT.mkdir(exist_ok=False)
 assert json.loads((DATA/'reset_isolation_03/reset_probe.json').read_text())['passed']
 smoke=json.loads((DATA/'optimizer_smoke_03/verification.json').read_text());assert smoke['updates']==2 and smoke['transitions']==1664
 standing=json.loads((DATA/'standing_fixed_02/results.json').read_text())['episodes'];assert len(standing)==29 and all(x['duration_s']==20 and not any(x['failures'].values()) for x in standing)
 files=[ROOT/'scripts/run_supported_transition.py',ROOT/'scripts/run_supported_postures.py',ROOT/'scripts/run_resting_posture.py',ROOT/'scripts/run_supported_gpu0.sh',ROOT/'scripts/run_resting_gpu1.sh',ROOT/'sitting_task/src/humanoid_sitting/supported_surface.py',ROOT/'sitting_task/src/humanoid_sitting/resting_objective.py',ROOT/'runs/g1_kinematics_geometry_04/model.json',ROOT/'docs/SUPPORTED_TRANSITION_PROTOCOL.md',Path(__file__)]
 files+=list((ROOT/'sitting_task/src/humanoid_sitting/tasks').glob('*.py'))
 files+=list((DATA/'transition_banks_03').rglob('*'))
 hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
 protocol=dict(seeds=[54211,54217],updates_per_seed=224,environments=26,steps_per_update=64,transitions_per_seed=372736,total_transitions=745472,evaluation_seconds=20,train_geometries=13,heldout_geometries=16,hashes=hashes,development_smoke_transitions=1664)
 write(OUT/'protocol.json',protocol)
 workers=[]
 for seed,gpu in [(54211,0),(54217,1)]:
  label=f'train_{seed}';cmd=command(label,'train','train',gpu,seed);write(OUT/(label+'.command.json'),cmd)
  f=(OUT/(label+'.log')).open('w');proc=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);workers.append((seed,proc,f))
 status('training',pids={str(s):p.pid for s,p,f in workers})
 while any(p.poll() is None for s,p,f in workers):
  write(OUT/'workers.json',{str(s):dict(pid=p.pid,returncode=p.poll()) for s,p,f in workers});time.sleep(10)
 codes={str(s):p.returncode for s,p,f in workers}
 for s,p,f in workers:f.close()
 write(OUT/'workers.json',codes)
 if any(codes.values()):status('failed',returncodes=codes);return 1
 for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 evaluations=[]
 for bank in ['train','holdout']:evaluations.append(('fixed_'+bank,'fixed',bank,None))
 for seed in [54211,54217]:
  for bank in ['train','holdout']:evaluations.append((f'eval_{seed}_{bank}','evaluate',bank,OUT/f'train_{seed}'/'final.pt'))
 for label,mode,bank,checkpoint in evaluations:
  status('evaluation',active=label)
  cmd=command(label,mode,bank,0,54301,checkpoint);write(OUT/(label+'.command.json'),cmd)
  with (OUT/(label+'.log')).open('w') as f:result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:status('failed',active=label,returncode=result.returncode);return result.returncode
 status('complete_pending_verification')
 return 0
if __name__=='__main__':sys.exit(main())
