"""Start the wall clock before interpreter startup; kill on45s, never resume."""
import json,os,subprocess,sys,time,resource
from pathlib import Path
HERE=Path(__file__).resolve().parent
target='demo_rear_back_center'
start=time.monotonic();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
before=resource.getrusage(resource.RUSAGE_CHILDREN)
p=subprocess.Popen([sys.executable,str(HERE/'validate_observed_clearance.py')],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
timed_out=False
try:output,_=p.communicate(timeout=45)
except subprocess.TimeoutExpired:
    timed_out=True;os.killpg(p.pid,9);output,_=p.communicate()
elapsed=time.monotonic()-start;after=resource.getrusage(resource.RUSAGE_CHILDREN)
result=dict(target=target,wall_seconds_including_interpreter_startup=elapsed,timeout=timed_out,returncode=p.returncode,cpu_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime),child_peak_rss_kib=after.ru_maxrss,output=output,new_physics_steps=0)
with (HERE/'observed_clearance_process_receipt.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
