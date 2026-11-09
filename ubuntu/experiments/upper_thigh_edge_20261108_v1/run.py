"""Start the wall clock before interpreter startup; kill on45s, never resume."""
import json,os,subprocess,sys,time,resource,fcntl,shutil
from pathlib import Path
PARENT_START=time.monotonic()
HERE=Path(__file__).resolve().parent
target=sys.argv[1]
assert target=='stool_edge'
assert not (HERE/(target+'.process_receipt.json')).exists(), 'ONE_ATTEMPT_ONLY'
freeze=json.loads((HERE/'freeze.json').read_text());assert freeze['independent_source_review']['passed']
assert shutil.disk_usage(HERE).free>=20_000_000_000
owner=open('/home/kadir/humanoid-sitting/outputs/furniture_scientific_owner.lock','a')
fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
def limits():
    resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(45,45))
start=PARENT_START;env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
before=resource.getrusage(resource.RUSAGE_CHILDREN)
p=subprocess.Popen([sys.executable,str(HERE/'validate.py'),target],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True,preexec_fn=limits)
timed_out=False
try:output,_=p.communicate(timeout=max(.001,45-(time.monotonic()-start)))
except subprocess.TimeoutExpired:
    timed_out=True;os.killpg(p.pid,9);output,_=p.communicate()
elapsed=time.monotonic()-start;after=resource.getrusage(resource.RUSAGE_CHILDREN)
result=dict(target=target,wall_seconds_including_interpreter_startup=elapsed,timeout=timed_out,returncode=p.returncode,cpu_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime),child_peak_rss_kib=after.ru_maxrss,output=output,new_physics_steps=0)
with (HERE/(target+'.process_receipt.json')).open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
