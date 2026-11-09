"""One finite CPU asset-build invocation. Never retry or launch robot motion."""
import os,time,subprocess,sys,json,resource,signal
from pathlib import Path
START=time.monotonic();H=Path(__file__).resolve().parent
assert not (H/'scene_process_receipt.json').exists()
def limits():
    resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(45,45))
p=subprocess.Popen([sys.executable,str(H/'build_scenes.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),preexec_fn=limits,start_new_session=True)
timeout=False
try:output,_=p.communicate(timeout=max(.001,45-(time.monotonic()-START)))
except subprocess.TimeoutExpired:
    timeout=True;os.killpg(p.pid,signal.SIGKILL);output,_=p.communicate()
r=resource.getrusage(resource.RUSAGE_CHILDREN)
result=dict(returncode=p.returncode,timeout=timeout,wall_seconds=time.monotonic()-START,cpu_seconds=r.ru_utime+r.ru_stime,peak_rss_kib=r.ru_maxrss,output=output,physics_steps=0,PPO_updates=0)
with (H/'scene_process_receipt.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
