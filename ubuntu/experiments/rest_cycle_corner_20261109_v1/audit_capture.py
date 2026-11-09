"""Frozen independent actual-mesh association for two measured corner flanks."""
import os,time,signal,resource,sys,json,hashlib,traceback
from pathlib import Path
START=time.monotonic();os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def timeout(*a):raise TimeoutError('45_SECOND_TOTAL_CORNER_CAPTURE_AUDIT_CAP')
signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,45)
R=Path('/home/kadir/humanoid-sitting');H=Path(__file__).resolve().parent;O=R/'outputs'/H.name/'camera_and_wrench_capture'
sys.path[1:1]=[str(R/'sitting_task/src'),str(R/'experiments/furniture_perception_repair_20261029_v1'),str(R/'experiments/rest_cycle_partial_support_20261109_v1')]
def write(name,x):(O/name).write_text(json.dumps(x,indent=2)+'\n')
def main():
    import numpy as np
    from corner_edges import detect
    from humanoid_sitting.shared_frame import level_rotation
    import audit_depth
    for p,h in json.loads((H/'audit_freeze.json').read_text())['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    manifest=json.loads((O/'manifest.json').read_text());case=manifest['case'];audit_depth.ASSETS=H/'assets'
    T=np.asarray(json.loads((H.parent/'rest_cycle_partial_support_20261109_v1/camera_preflight.json').read_text())['camera_in_body'])
    frames=[]
    for tick in [0,50]:
        source=O/f'frame_{tick}.npz';depthcheck=audit_depth.check(source,case)
        with np.load(source) as f:z=dict(f)
        g=z['geometry'];level=level_rotation(z['gravity'].astype(float).copy())
        sensor,arrays=detect(z['depth'],z['K'],T,g[2:5],float(g[5]),level,z['selected_pixels'])
        write(f'corner_sensor_{tick}.json',sensor);np.savez_compressed(O/f'corner_arrays_{tick}.npz',**arrays)
        envelope=dict(passed=False,reason='corner_not_qualified')
        if sensor['passed']:
            with np.load(case['mesh_npz']) as f:fv=f['vertices']
            basis=z['root_rotation']@level.T;verts=(fv-z['root_position'])@basis
            lines=np.asarray(sensor['outward_lines']);signed=verts[:,:2]@lines[:,:2].T+lines[:,2]
            excess=signed.max(0);n=level@g[2:5]
            envelope=dict(passed=bool(np.max(abs(excess))<=.006 and (verts@n+g[5]).max()<=.003),
                actual_supporting_edge_offsets_m=excess.tolist(),actual_top_excess_m=float((verts@n+g[5]).max()),
                independent_mesh_only=True)
        frames.append(dict(tick=tick,depth=depthcheck,corner=sensor,envelope=envelope))
    result=dict(passed=all(r['depth']['geometry_pass'] and r['depth']['original_plane_gate'] and r['depth']['target_observable'] and r['corner']['passed'] and r['envelope']['passed'] for r in frames),
        frames=frames,elapsed_seconds=time.monotonic()-START,physics_steps=0,sitting_controls=0,complete_cycle=False)
    write('capture_audit.json',result);print(json.dumps(result))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        write('audit_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)
