"""Single finite human-authorized PPO queue; one simulation/optimizer child."""
import os,json,hashlib,time,subprocess,signal,shutil,fcntl,traceback
from pathlib import Path
import numpy as np
ROOT=Path('/home/kadir/humanoid-sitting');D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
OLD=D/'contact_learning_ppo_20261026_v2';OUT=D/'contact_reward_comparison_128_v4';PYTHON=ROOT/'IsaacLab/.venv/bin/python'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(r):return bool(r['success'] and not any(r['failures'].values()) and not r['slip_ticks'] and not r['illegal_ticks'])
protocol=read(OUT/'protocol.json');budget=protocol['budget'];started=time.monotonic()-protocol.get('charged_prior_seconds',0);completed=[];verified=[];active=None
lock=(D/'contact_ppo_scientific_owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'launch_receipt.json').exists(),'Single launch only'
write(OUT/'launch_receipt.json',dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),protocol_sha256=sha(OUT/'protocol.json'),freeze_sha256=sha(OUT/'freeze.json')))

def freeze():
 for p,h in read(OUT/'freeze.json').items():assert sha(p)==h,('SOURCE_CHANGED',p)

def resources(child=None):
 assert time.monotonic()-started<budget['wall_seconds'],'Whole-study wall cap'
 size=protocol.get('charged_prior_bytes',0)+sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
 assert size<budget['max_output_bytes'],'Output cap'
 assert shutil.disk_usage(OUT).free>budget['min_free_bytes'],'Free-space reserve'
 mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')}
 assert mem['MemAvailable']>budget['min_system_available_bytes'],'System memory reserve'
 if child:
  ps=subprocess.check_output(['ps','-eo','pgid=,rss='],text=True)
  rss=sum(int(line.split()[1])*1024 for line in ps.splitlines() if int(line.split()[0])==child.pid)
  assert rss<budget['max_child_rss_bytes'],('RSS cap',rss)
  gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
  assert int(gpu.strip())*1024**2<budget['max_gpu_reserved_bytes'],'GPU memory cap'
 return size

def state(label,p,t):
 write(OUT/'status.json',dict(state='running',phase=label,owner_pid=os.getpid(),child_pid=p.pid,completed=completed,
   first_updates_verified=verified,seconds=time.monotonic()-started,phase_seconds=time.monotonic()-t,
   complete=False,physical_repair_claimed=False,automation='none'))

def job(label,command,cap,training=False):
 global active
 freeze();resources();assert not (OUT/(label+'.log')).exists()
 write(OUT/(label+'.command.json'),[str(x) for x in command]);t=time.monotonic()
 env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
 with (OUT/(label+'.log')).open('w') as log:
  p=subprocess.Popen([str(x) for x in command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env);active=p
  state(label,p,t);checked=False
  try:
   while p.poll() is None:
    time.sleep(10);resources(p);assert time.monotonic()-t<cap,('Child wall cap',label)
    if training and not checked and (OUT/label/'weights_ready.json').exists():
     from audit_checkpoint_updates_v2 import audit
     checkpoint=read(OUT/label/'weights_ready.json')['checkpoint']
     audit(OUT/label,checkpoint);verified.append(dict(phase=label,checkpoint=checkpoint,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())));checked=True
    state(label,p,t)
  except BaseException:
   if p.poll() is None:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise
 write(OUT/(label+'.exit.json'),dict(returncode=p.returncode,seconds=time.monotonic()-t))
 active=None;assert p.returncode==0,(label,p.returncode);completed.append(label);freeze()

def cmd(label,arm,mode='zero',seed=63111,updates=1,checkpoint=None,rng=None,bank=None):
 c=read(OUT/'base_command.json')
 def put(key,value):c[c.index(key)+1]=str(value)
 put('--bank',bank or OUT/'training_bank');put('--output',OUT/label);c[2]='../scripts/run_contact_learning_ppo_v4.py'
 c+=['--landing-config',str(D/'normal_landing3s_guarded_09.json'),'--reward-arm',arm,'--feedback-mode',mode,'--feedback-start','6.0','--feedback-seed',str(seed),'--feedback-updates',str(updates),'--feedback-span','.004','--feedback-rate','.004','--repair-kind','original','--repair-config',str(D/'normal_landing3s_guarded_09.json'),'--feedback-calibration',str(OLD/'calibrate/feedback_calibration.json')]
 if checkpoint:c+=['--feedback-checkpoint',str(checkpoint)]
 if rng:c+=['--feedback-rng-state',str(rng)]
 return c

def report(arm,seed,stage):
 rows=[];cases=[];lost=[];repaired=[];common=[];strict=[]
 for bank in ['known_old','known_transfer','known_recent']:
  before=read(D/'current_retention_09'/bank/'candidate/results.json')['episodes']
  after=read(OUT/f'eval_{arm}_{seed}_{stage:03d}_{bank}'/'results.json')['episodes']
  assert len(before)==len(after)==24 and all(x['steps']==1000 for x in after)
  for a,b in zip(before,after):
   assert (a['case'],a['profile'],a['surface_id'])==(b['case'],b['profile'],b['surface_id'])
   key=[bank,b['case']]
   if clean(a) and not clean(b):lost.append(key)
   if not clean(a) and clean(b):repaired.append(key)
   if clean(a) and clean(b):
    common.append((a,b))
    if a['strict_transfer'] and b['strict_transfer']:strict.append((a,b))
   cases.append(dict(bank=bank,case=b['case'],profile=b['profile'],baseline_clean=clean(a),clean=clean(b),stable=b['success'],slip_ticks=b['slip_ticks'],illegal_ticks=b['illegal_ticks'],failures=b['failures'],final_hold_s=b['final_hold_s'],traction_bad_ticks=b['traction_bad_ticks'],whole37_torque2=b['metrics_whole_trial']['total_squared_torque_Nm2']))
  rows.extend(after)
 def effort(pairs):
  keys=['total_squared_torque_Nm2','total_absolute_torque_Nm','all_joint_rms_Nm','leg_squared_torque_Nm2','upper_squared_torque_Nm2']
  return dict(count=len(pairs),windows={window:{key:{label:float(np.mean([pair[i][window][key] for pair in pairs])) for i,label in enumerate(['baseline','candidate'])} for key in keys} for window in ['metrics_whole_trial','metrics_last10s','metrics_last5s']}) if pairs else dict(count=0,windows={})
 entry=dict(arm=arm,seed=seed,updates=stage,clean=sum(clean(x) for x in rows),stable=sum(x['success'] for x in rows),
  slipping=sum(x['slip_ticks']>0 for x in rows),illegal=sum(x['illegal_ticks']>0 for x in rows),physical_failures=sum(any(x['failures'].values()) for x in rows),
  lost_protected=lost,repaired_old_failures=repaired,all8_repaired_and64_retained=len(repaired)==8 and not lost,
  effort_common_complete_clean=effort(common),effort_common_complete_strict=effort(strict),effort_caveat='Approximate implicit-PD telemetry; not energy or hardware effort. Only common complete clean successes are compared.',cases=cases)
 name=OUT/f'comparison_{arm}_{seed}_{stage:03d}.json';write(name,entry)
 index=read(OUT/'development_comparison.json') if (OUT/'development_comparison.json').exists() else dict(results=[],promoted=False,fresh_holdout_opened=False,leaning_trained=False)
 index['results'].append(entry);write(OUT/'development_comparison.json',index)

try:
 freeze()
 assert read(OUT/'source_review.json')['passed']
 for name,h in read(OUT/'source_review.json')['new_sources'].items():assert sha(ROOT/'scripts'/name)==h,name
 assert '580.196.07' in Path('/proc/driver/nvidia/version').read_text()
 subprocess.run(['nvidia-smi','-L'],check=True,stdout=subprocess.DEVNULL)
 workers=subprocess.check_output(['ps','-eo','pid=,args='],text=True)
 for line in workers.splitlines():
  if any(x in line for x in ['run_contact_learning_ppo_v1.py','run_contact_learning_ppo_v4.py','run_supported_gpu0.sh','launch_contact_ppo_v2.py']):
   raise AssertionError(('OTHER_SCIENTIFIC_WORKER',line))
 job('identity_zero',cmd('identity_zero','revised',updates=2),budget['identity_child_seconds'])
 job('identity_audit',[PYTHON,ROOT/'scripts/audit_contact_comparison_v3.py','reset',OUT,OLD],budget['audit_child_seconds'])
 job('identity_reward_audit',[PYTHON,ROOT/'scripts/audit_contact_comparison_v3.py',OUT/'identity_zero'],budget['audit_child_seconds'])
 prior={}
 for stage,chunk in zip(protocol['milestones'],protocol['chunks']):
  for arm,seed in protocol['order']:
   key=(arm,seed);prev=prior.get(key);initial=prev/'final.pt' if prev else OLD/f'train_{seed}/initial.pt';rng=prev/'training_rng_state.pt' if prev else None
   label=f'train_{arm}_{seed}_{stage:03d}'
   job(label,cmd(label,arm,'train',seed,chunk,initial,rng),chunk*budget['training_child_seconds_per_update']+budget['training_child_overhead_seconds'],training=True)
   assert read(OUT/label/'feedback_contract.json')['final_updates']==stage
   job(label+'_audit',[PYTHON,ROOT/'scripts/audit_contact_comparison_v3.py',OUT/label],budget['audit_child_seconds'])
   prior[key]=OUT/label
   for bank in ['known_old','known_transfer','known_recent']:
    e=f'eval_{arm}_{seed}_{stage:03d}_{bank}'
    job(e,cmd(e,arm,'evaluate',seed,1,OUT/label/'final.pt',bank=D/'banks'/bank),budget['evaluation_child_seconds'])
    job(e+'_audit',[PYTHON,ROOT/'scripts/audit_contact_comparison_v3.py',OUT/e],budget['audit_child_seconds'])
   report(arm,seed,stage)
 freeze();size=resources()
 write(OUT/'status.json',dict(state='comparison_complete',completed=completed,first_updates_verified=verified,seconds=time.monotonic()-started,bytes=size,promotion=False,complete=True,results=str(OUT/'development_comparison.json')))
except BaseException as e:
 write(OUT/'status.json',dict(state='stopped',error=repr(e),traceback=traceback.format_exc(),completed=completed,first_updates_verified=verified,seconds=time.monotonic()-started,retry=False,complete=False))
 raise
