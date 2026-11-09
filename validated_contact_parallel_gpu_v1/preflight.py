"""One bounded GPU1 reset/physics eligibility check; GPU0 queue untouched."""
import os,json,time,subprocess,signal,hashlib,fcntl,traceback
from pathlib import Path
ROOT=Path('/home/kadir/humanoid-sitting');D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
OUT=D/'contact_parallel_gpu_v1';OLD=D/'contact_learning_ppo_20261026_v2';MAIN=D/'contact_reward_comparison_128_v4'
PY=ROOT/'IsaacLab/.venv/bin/python'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock=(OUT/'transition.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'preflight_receipt.json').exists(),'One launch only'
started=time.monotonic();write(OUT/'preflight_receipt.json',dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
completed=[];active=None
def resources():
 assert time.monotonic()-started<1800
 assert __import__('shutil').disk_usage(OUT).free>5*1024**3
 size=sum(p.stat().st_size for root in [MAIN,OUT] for p in root.rglob('*') if p.is_file())
 assert size+read(MAIN/'protocol.json')['charged_prior_bytes']<12*1024**3
 assert int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<11264
def run(label,cmd,cap):
 global active
 assert not (OUT/(label+'.log')).exists();write(OUT/(label+'.command.json'),cmd)
 with (OUT/(label+'.log')).open('w') as log:
  p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active=p;t=time.monotonic()
  while p.poll() is None:
   resources();assert time.monotonic()-t<cap
   write(OUT/'preflight_status.json',dict(state='running',phase=label,owner_pid=os.getpid(),child_pid=p.pid,completed=completed,seconds=time.monotonic()-started))
   time.sleep(5)
 write(OUT/(label+'.exit.json'),dict(returncode=p.returncode,seconds=time.monotonic()-t));active=None
 assert p.returncode==0,(label,p.returncode)
 completed.append(label)
try:
 for name,h in read(OUT/'freeze.json').items():assert sha(name)==h,('SOURCE_CHANGED',name)
 original=read(MAIN/'identity_zero.command.json');cmd=original.copy()
 cmd[1]=str(ROOT/'scripts/run_supported_gpu1_parallel_v1.sh')
 cmd[cmd.index('--output')+1]=str(OUT/'identity_zero')
 i=next(i for i,v in enumerate(cmd) if v.startswith('--kit_args='));cmd[i]='--kit_args=--/renderer/activeGpu=1,--/renderer/multiGpu/enabled=false'
 assert [i for i,(a,b) in enumerate(zip(original,cmd)) if a!=b]==[1,original.index('--output')+1,i]
 run('identity_gpu1',cmd,1500)
 run('exact_reset_audit',[str(PY),str(ROOT/'scripts/audit_contact_comparison_v3.py'),'reset',str(OUT),str(OLD)],180)
 run('reward_geometry_audit',[str(PY),str(ROOT/'scripts/audit_contact_comparison_v3.py'),str(OUT/'identity_zero')],180)
 write(OUT/'preflight_status.json',dict(state='passed',completed=completed,seconds=time.monotonic()-started,optimizer_updates=0,physical_repair_claimed=False,tolerance_relaxed=False))
except BaseException as e:
 if active is not None and active.poll() is None:
  os.killpg(active.pid,signal.SIGTERM)
  try:active.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
 write(OUT/'preflight_status.json',dict(state='failed',error=repr(e),traceback=traceback.format_exc(),completed=completed,seconds=time.monotonic()-started,retry=False,gpu0_queue_untouched=True))
 raise
