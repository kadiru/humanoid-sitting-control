"""Frozen independent actual-mesh association for two measured corner flanks."""
import os,time,signal,resource,sys,json,hashlib,traceback
from pathlib import Path
START=time.monotonic();os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def timeout(*a):raise TimeoutError('45_SECOND_TOTAL_CORNER_CAPTURE_AUDIT_CAP')
signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,45)
R=Path('/home/kadir/humanoid-sitting');H=Path(__file__).resolve().parent;O=R/'outputs'/H.name/'camera_and_wrench_capture'
sys.path[1:1]=[str(R/'sitting_task/src'),str(R/'experiments/rest_cycle_corner_20261109_v1'),str(R/'experiments/furniture_perception_repair_20261029_v1'),str(R/'experiments/rest_cycle_partial_support_20261109_v1')]
def write(name,x):(O/name).write_text(json.dumps(x,indent=2)+'\n')
def main():
    import numpy as np
    from corner_edges import detect
    from humanoid_sitting.shared_frame import level_rotation
    import audit_depth
    for p,h in json.loads((H/'audit_freeze.json').read_text())['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    manifest=json.loads((O/'manifest.json').read_text());case=manifest['case'];audit_depth.ASSETS=H/'assets'
    frames=[]
    for tick in [0,50]:
        source=O/f'frame_{tick}.npz';depthcheck=audit_depth.check(source,case)
        frames.append(dict(tick=tick,depth=depthcheck))
    result=dict(passed=all(r['depth']['geometry_pass'] and r['depth']['original_plane_gate'] and r['depth']['target_observable'] for r in frames),frames=frames,elapsed_seconds=time.monotonic()-START,physics_steps=0,sitting_controls=0,complete_cycle=False)
    write('capture_audit.json',result);print(json.dumps(result))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        write('audit_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)
