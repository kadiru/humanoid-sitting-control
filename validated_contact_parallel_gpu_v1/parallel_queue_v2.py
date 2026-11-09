"""Finite two-GPU continuation; no training/source changes or duplicate jobs.

The legacy GPU0 owner finishes its current64 block and evaluations, then its
existing exclusive-log guard stops it before revised63117_064. A documented
reservation activates that guard. This owner then acquires its released lock.
"""
import os,json,time,subprocess,signal,hashlib,fcntl,threading,traceback,ast,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path('/home/kadir/humanoid-sitting');D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
MAIN=D/'contact_reward_comparison_128_v4';OUT=D/'contact_parallel_gpu_v1';PYTHON=ROOT/'IsaacLab/.venv/bin/python'
UUIDS=['GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4','GPU-cde23916-c5fa-f18e-a343-127f732cc2f6']
GUARD='train_revised_63117_064';MARK='AUTHORIZED_DUAL_GPU_HANDOFF_V2: reservation, no simulation was run for this log.\n'
LANES={0:[('revised',63117,64,32),('original',63117,128,64),('revised',63117,128,64)],1:[('revised',63111,128,64),('original',63111,128,64)]}
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source_functions():
 # Import only the frozen pure command/report functions, never the old launch.
 tree=ast.parse((ROOT/'scripts/launch_contact_comparison_v4.py').read_text())
 funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['cmd','report','clean']]
 assert {n.name for n in funcs}=={'cmd','report','clean'}
 import numpy as np
 ns=dict(ROOT=ROOT,D=D,OUT=OUT,OLD=D/'contact_learning_ppo_20261026_v2',PYTHON=PYTHON,read=read,write=write,np=np)
 exec(compile(ast.Module(body=funcs,type_ignores=[]),'frozen_command_report','exec'),ns)
 return ns
def command(label,arm,gpu,mode,seed,chunk,checkpoint,rng=None,bank=None):
 ns=source_functions();ns['OUT']=MAIN
 c=ns['cmd'](label,arm,mode,seed,chunk,checkpoint,rng,bank or MAIN/'training_bank')
 c[c.index('--output')+1]=str(OUT/label)
 if gpu==1:
  c[1]=str(ROOT/'scripts/run_supported_gpu1_parallel_v1.sh')
  i=next(i for i,v in enumerate(c) if v.startswith('--kit_args='));c[i]='--kit_args=--/renderer/activeGpu=1,--/renderer/multiGpu/enabled=false'
 return [str(v) for v in c]

def main():
 approval=read(OUT/'hardware_rounding_authorization_v2.json')
 assert approval['approved'] is True and approval['scope']=='recorded_hardware_rounding_only'
 assert read(OUT/'preflight_status.json')['state']=='failed' # Preserve the original exact-check result.
 review=read(OUT/'hardware_admission_review_v1.json')
 assert review['all_complete_episode_records_exact'] and review['independent_reward_geometry_audit_passed']
 for p,h in approval['recorded_artifact_bindings'].items():assert sha(p)==h,('UNAPPROVED_HARDWARE_EVIDENCE',p)
 transition=(OUT/'transition.lock').open('a');fcntl.flock(transition,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'parallel_receipt.json').exists(),'One finite launch only'
 protocol=read(OUT/'parallel_protocol_v2.json');budget=read(MAIN/'protocol.json')['budget']
 bindings=read(OUT/'parallel_freeze_v2.json')
 for p,h in bindings.items():assert sha(p)==h,('SOURCE_CHANGED',p)
 assert sum(x[3] for lane in LANES.values() for x in lane)==288
 prior=read(MAIN/'status.json')
 assert prior['state']=='running' and prior['owner_pid']==28553
 allowed=['train_original_63117_064','train_original_63117_064_audit']+[f'eval_original_63117_064_{b}'+suffix for b in ['known_old','known_transfer','known_recent'] for suffix in ['', '_audit']]
 assert prior['phase'] in allowed,('UNEXPECTED_HANDOFF_PHASE',prior['phase'])
 assert not (MAIN/GUARD).exists()
 guard=MAIN/(GUARD+'.log')
 with guard.open('x') as f:f.write(MARK)
 started=time.monotonic();deadline=protocol['absolute_deadline_unix']
 write(OUT/'parallel_receipt.json',dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),legacy_owner_pid=28553,legacy_phase=prior['phase'],guard=str(guard),guard_sha256=sha(guard),scope='Remaining288 updates; two GPU workers; no extra PPO updates',sources=bindings))
 mutex=threading.Lock();stop=threading.Event();active={};completed=[];verified=[];lane_states={0:'legacy_chunk_and_evaluations',1:'starting'};failure=[]
 def freeze():
  for p,h in bindings.items():assert sha(p)==h,('SOURCE_CHANGED',p)
 def resources():
  assert time.time()<deadline,'Original52h wall cap'
  size=read(MAIN/'protocol.json')['charged_prior_bytes']
  for root in [MAIN,OUT]:
   for p in root.rglob('*'):
    try:
     if p.is_file():size+=p.stat().st_size
    except FileNotFoundError:pass # Atomic status temporary file renamed.
  assert size<budget['max_output_bytes'],('Combined output cap',size)
  assert shutil.disk_usage(OUT).free>budget['min_free_bytes']
  mem=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
  assert mem>budget['min_system_available_bytes']
  ps=subprocess.check_output(['ps','-eo','rss=,args='],text=True)
  rss=sum(int(l.split()[0])*1024 for l in ps.splitlines() if 'run_contact_learning_ppo_v4.py' in l)
  assert rss<48*1024**3,('Aggregate simulator RSS cap',rss)
  vals=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
  for line in vals.splitlines():
   i,m=map(int,line.split(','));assert m<([23552,11264][i]),('GPU memory cap',i,m)
  return size
 def publish():
  with mutex:
   write(OUT/'parallel_status.json',dict(state='running',owner_pid=os.getpid(),lanes=lane_states.copy(),active={str(g):{'pid':p.pid,'label':label} for g,(p,label) in active.items()},completed=completed.copy(),first_updates_verified=verified.copy(),seconds=time.monotonic()-started,legacy_status=str(MAIN/'status.json'),complete=False,physical_repair_claimed=False,automation='none'))
 def gpu_binding(p,gpu):
  groups={int(l.split()[0]):int(l.split()[1]) for l in subprocess.check_output(['ps','-eo','pid=,pgid='],text=True).splitlines()}
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  found=[]
  for line in apps.splitlines():
   uuid,pid=[v.strip() for v in line.split(',')]
   if groups.get(int(pid))==p.pid:found.append(uuid)
  assert found and set(found)=={UUIDS[gpu]},('GPU_BINDING',gpu,found)
 def job(gpu,label,cmd,cap,training=False):
  if stop.is_set():raise RuntimeError('Other lane stopped')
  freeze();resources();assert not (OUT/(label+'.log')).exists()
  write(OUT/(label+'.command.json'),cmd);t=time.monotonic()
  env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
  with (OUT/(label+'.log')).open('w') as log:
   p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
   with mutex:active[gpu]=(p,label);lane_states[gpu]=label
   checked=False;publish()
   try:
    while p.poll() is None:
     if stop.wait(5):raise RuntimeError('Other lane stopped')
     resources();assert time.monotonic()-t<cap,('Child deadline',label)
     if training and not checked and (OUT/label/'weights_ready.json').exists():
      gpu_binding(p,gpu)
      from audit_checkpoint_updates_v2 import audit
      cp=read(OUT/label/'weights_ready.json')['checkpoint'];audit(OUT/label,cp)
      with mutex:verified.append(dict(gpu=gpu,phase=label,checkpoint=cp,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
      checked=True
     publish()
   except BaseException:
    if p.poll() is None:
     os.killpg(p.pid,signal.SIGTERM)
     try:p.wait(timeout=20)
     except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    raise
  with mutex:active.pop(gpu,None)
  write(OUT/(label+'.exit.json'),dict(returncode=p.returncode,seconds=time.monotonic()-t,gpu=gpu))
  assert p.returncode==0,(label,p.returncode)
  with mutex:completed.append(label)
  freeze()
 def report(arm,seed,stage,gpu):
  with mutex:
   source_functions()['report'](arm,seed,stage)
   old=read(MAIN/'development_comparison.json');new=read(OUT/'development_comparison.json')
   results=old['results']+new['results'];keys=[(r['arm'],r['seed'],r['updates']) for r in results];assert len(keys)==len(set(keys))
   write(OUT/'combined_development_comparison.json',dict(results=results,promoted=False,fresh_holdout_opened=False,leaning_trained=False,hardware_assignment=protocol['hardware_assignment'],caveat='Seed63111 moves from4090 to4070 after64 updates; recorded hardware-only rounding differences explicitly approved. Physical traces and all48 hardware-check episode records matched exactly. Both reward arms within each seed use matched hardware after the same boundary.'))
 def lane(gpu):
  old_lock=None
  try:
   if gpu==0:
    while True:
     if stop.wait(5):raise RuntimeError('Other lane stopped before handoff')
     s=read(MAIN/'status.json');resources()
     if s['state']=='stopped':
      assert guard.read_text()==MARK
      assert s['error']=='AssertionError()'
      assert s['completed'][-1]=='eval_original_63117_064_known_recent_audit'
      assert 'assert not (OUT/(label+\'.log\')).exists()' in s['traceback']
      assert not (MAIN/GUARD).exists()
      assert any(r['arm']=='original' and r['seed']==63117 and r['updates']==64 for r in read(MAIN/'development_comparison.json')['results'])
      old_lock=(D/'contact_ppo_scientific_owner.lock').open('a')
      try:fcntl.flock(old_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
      except BlockingIOError:old_lock.close();old_lock=None;continue
      write(OUT/'legacy_handoff.json',dict(expected_guard_stop=True,legacy_status=s,scientific_chunk_interrupted=False,original_gpu0_lock_acquired=True,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
      break
     assert s['state']=='running',s['state']
   for arm,seed,stage,chunk in LANES[gpu]:
    previous_stage=stage-chunk
    previous=OUT/f'train_{arm}_{seed}_{previous_stage:03d}'
    if not previous.exists():previous=MAIN/previous.name
    assert read(previous/'independent_audit.json')['passed']
    assert read(previous/'feedback_contract.json')['final_updates']==previous_stage
    label=f'train_{arm}_{seed}_{stage:03d}'
    write(OUT/(label+'.resume_binding.json'),{str(p):sha(p) for p in [previous/'final.pt',previous/'training_rng_state.pt',previous/'independent_audit.json']})
    job(gpu,label,command(label,arm,gpu,'train',seed,chunk,previous/'final.pt',previous/'training_rng_state.pt'),chunk*budget['training_child_seconds_per_update']+budget['training_child_overhead_seconds'],True)
    assert read(OUT/label/'feedback_contract.json')['final_updates']==stage
    job(gpu,label+'_audit',[str(PYTHON),str(ROOT/'scripts/audit_contact_comparison_v3.py'),str(OUT/label)],budget['audit_child_seconds'])
    for bank in ['known_old','known_transfer','known_recent']:
     e=f'eval_{arm}_{seed}_{stage:03d}_{bank}'
     job(gpu,e,command(e,arm,gpu,'evaluate',seed,1,OUT/label/'final.pt',bank=D/'banks'/bank),budget['evaluation_child_seconds'])
     job(gpu,e+'_audit',[str(PYTHON),str(ROOT/'scripts/audit_contact_comparison_v3.py'),str(OUT/e)],budget['audit_child_seconds'])
    report(arm,seed,stage,gpu)
   with mutex:lane_states[gpu]='complete'
   publish()
  except BaseException as e:
   with mutex:failure.append(dict(gpu=gpu,error=repr(e),traceback=traceback.format_exc()))
   stop.set();raise
  finally:
   if old_lock is not None:old_lock.close()
 try:
  with ThreadPoolExecutor(max_workers=2) as pool:
   futures=[pool.submit(lane,g) for g in [0,1]]
   for f in futures:f.result()
  freeze();size=resources()
  final=read(OUT/'combined_development_comparison.json')
  assert {(r['arm'],r['seed']) for r in final['results'] if r['updates']==128}=={('original',63111),('revised',63111),('original',63117),('revised',63117)}
  write(OUT/'parallel_status.json',dict(state='comparison_complete',completed=completed,first_updates_verified=verified,seconds=time.monotonic()-started,combined_bytes=size,complete=True,promotion=False,results=str(OUT/'combined_development_comparison.json')))
 except BaseException:
  stop.set();write(OUT/'parallel_status.json',dict(state='stopped',failures=failure,completed=completed,first_updates_verified=verified,seconds=time.monotonic()-started,complete=False,retry=False,legacy_gpu0_chunk_not_interrupted=True));raise

if __name__=='__main__':main()
