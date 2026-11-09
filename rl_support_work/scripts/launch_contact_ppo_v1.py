"""Single owner, finite queue; no schedules, retries, tuning, or promotion."""
import os,json,hashlib,time,subprocess,signal,shutil,fcntl,traceback
from pathlib import Path
import numpy as np
ROOT=Path('/home/kadir/humanoid-sitting');D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');OUT=D/'contact_learning_ppo_20261026_v1'
PYTHON=ROOT/'IsaacLab/.venv/bin/python'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(r):return r['success'] and not any(r['failures'].values()) and not r['slip_ticks'] and not r['illegal_ticks']
protocol=read(OUT/'protocol.json');budget=protocol['budget'];started=time.monotonic()
lock=(OUT/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'launch_receipt.json').exists(),'Single launch only'
write(OUT/'launch_receipt.json',dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),protocol_sha256=sha(OUT/'protocol.json'),freeze_sha256=sha(OUT/'freeze.json')))
active=None;completed=[]
def freeze():
 for p,h in read(OUT/'freeze.json').items():assert sha(p)==h,('SOURCE_CHANGED',p)
def resources():
 assert time.monotonic()-started<budget['wall_seconds'],'Whole-study wall cap'
 size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
 assert size<budget['max_output_bytes'],'Output cap'
 assert shutil.disk_usage(OUT).free>budget['min_free_bytes'],'Free-space reserve'
 return size

def job(label,cmd):
 global active
 freeze();resources();assert not (OUT/(label+'.log')).exists()
 write(OUT/(label+'.command.json'),[str(x) for x in cmd]);t=time.monotonic()
 with (OUT/(label+'.log')).open('w') as log:
  p=subprocess.Popen([str(x) for x in cmd],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  active=p
  write(OUT/'status.json',dict(phase=label,state='running',owner_pid=os.getpid(),child_pid=p.pid,completed=completed,seconds=time.monotonic()-started))
  try:
   while p.poll() is None:
    time.sleep(10);resources();assert time.monotonic()-t<budget['child_wall_seconds'],'Child wall cap'
  except BaseException:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise
 write(OUT/(label+'.exit.json'),dict(returncode=p.returncode,seconds=time.monotonic()-t))
 active=None;assert p.returncode==0,(label,p.returncode)
 completed.append(label);freeze()

def cmd(label,mode='zero',seed=63111,updates=1,checkpoint=None,bank=None,old=False):
 c=read(OUT/'base_command.json')
 def put(key,value):c[c.index(key)+1]=str(value)
 put('--bank',bank or OUT/'training_bank');put('--output',OUT/label)
 c[2]='../scripts/run_low_seat_current_timing_probe.py' if old else '../scripts/run_contact_learning_ppo_v1.py'
 c+=['--landing-config',str(D/'normal_landing3s_guarded_09.json')]
 if not old:
  c+=['--feedback-mode',mode,'--feedback-start','6.0','--feedback-seed',str(seed),'--feedback-updates',str(updates),'--feedback-span','.004','--feedback-rate','.004','--repair-kind','original','--repair-config',str(D/'normal_landing3s_guarded_09.json')]
  if mode!='calibrate':c+=['--feedback-calibration',str(OUT/'calibrate/feedback_calibration.json')]
  if checkpoint:c+=['--feedback-checkpoint',str(checkpoint)]
 return c

def identity():
 counts={}
 for name in ['startup.npz','traction_startup.npz','trace.npz']:
  a=np.load(OUT/'old_zero'/name);b=np.load(OUT/'calibrate'/name)
  assert set(a.files)==set(b.files),(name,a.files,b.files)
  for k in a.files:assert np.array_equal(a[k],b[k],equal_nan=True),('ZERO_IDENTITY',name,k)
  counts[name]=len(a.files)
 write(OUT/'zero_identity_audit.json',dict(passed=True,arrays=counts,kind='Independent existing best09 implementation versus new zero residual on identical training bank'))

def indexed():
 a=np.load(OUT/'calibrate/trace.npz');b=np.load(OUT/'indexed_zero/trace.npz');counts={}
 for k in a.files:
  if k=='t':continue
  n=len(a[k]);assert np.array_equal(a[k],b[k][-n:],equal_nan=True),('INDEXED_ZERO_IDENTITY',k)
  counts[k]=n
 rows=read(OUT/'indexed_zero/results.json')['episodes'];assert len(rows)==48 and all(x['steps']==1000 for x in rows)
 write(OUT/'indexed_identity_audit.json',dict(passed=True,arrays=counts,episodes=48))

def evaluation_report():
 reports={}
 for seed in protocol['seeds']:
  rows=[];lost=[];repaired=[]
  for bank in ['known_old','known_transfer','known_recent']:
   ref=read(D/'current_retention_09'/bank/'candidate/results.json')['episodes'];new=read(OUT/f'evaluate_{seed}_{bank}'/'results.json')['episodes']
   assert len(new)==len(ref)==24 and all(x['steps']==1000 for x in new)
   for a,b in zip(ref,new):
    assert (a['case'],a['profile'])==(b['case'],b['profile'])
    if clean(a) and not clean(b):lost.append([bank,b['case']])
    if not clean(a) and clean(b):repaired.append([bank,b['case']])
   rows.extend(new)
  reports[str(seed)]=dict(clean=sum(clean(x) for x in rows),stable=sum(x['success'] for x in rows),slipping=sum(x['slip_ticks']>0 for x in rows),illegal=sum(x['illegal_ticks']>0 for x in rows),physical_failures=sum(any(x['failures'].values()) for x in rows),lost_protected=lost,repaired_old_failures=repaired,mean_all37_torque2=float(np.mean([x['metrics_whole_trial']['total_squared_torque_Nm2'] for x in rows])),physical_repair_verified=len(repaired)==8 and not lost)
 write(OUT/'development_comparison.json',dict(seeds=reports,promoted=False,fresh_holdout_opened=False,leaning_trained=False))
try:
 job('unit_audit',[PYTHON,ROOT/'scripts/audit_contact_ppo_v1.py'])
 job('old_zero',cmd('old_zero',old=True))
 job('calibrate',cmd('calibrate','calibrate'));identity()
 job('indexed_zero',cmd('indexed_zero',updates=2));indexed()
 for seed in protocol['seeds']:
  label=f'smoke_{seed}';job(label,cmd(label,'train',seed,2))
  job(f'cpu_replay_smoke_{seed}',[PYTHON,ROOT/'scripts/audit_contact_ppo_v1.py',OUT/label])
 for seed in protocol['seeds']:
  label=f'train_{seed}';initial=OUT/f'smoke_{seed}'/'initial.pt'
  job(label,cmd(label,'train',seed,8,initial))
  assert sha(initial)==sha(OUT/label/'initial.pt'),'Main must start from smoke initial'
  job(f'cpu_replay_train_{seed}',[PYTHON,ROOT/'scripts/audit_contact_ppo_v1.py',OUT/label])
 for seed in protocol['seeds']:
  for bank in ['known_old','known_transfer','known_recent']:
   label=f'evaluate_{seed}_{bank}';job(label,cmd(label,'evaluate',seed,1,OUT/f'train_{seed}/final.pt',D/'banks'/bank))
 evaluation_report();freeze();size=resources()
 write(OUT/'status.json',dict(state='pilot_complete',completed=completed,seconds=time.monotonic()-started,bytes=size,physical_success=read(OUT/'development_comparison.json'),promotion=False))
except BaseException as e:
 write(OUT/'status.json',dict(state='stopped',error=repr(e),traceback=traceback.format_exc(),completed=completed,seconds=time.monotonic()-started,retry=False))
 raise
