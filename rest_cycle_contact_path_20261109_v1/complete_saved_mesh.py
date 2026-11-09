"""Complete an unsealed saved-path mesh check; never rehabilitate runtime gates.

No path construction, optimizer, command encoding, or completed foot proof is
replayed. All earlier process costs stay charged and motion stays unadmitted.
"""
import time
START=time.monotonic()
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import json,hashlib,resource,signal,sys,traceback
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main(freeze_path,output):
    output=Path(output);output.mkdir(exist_ok=False)
    def write(name,data):
        text=json.dumps(data,indent=2,allow_nan=False)+'\n'
        if len(text)>4*1024**2:raise ValueError('FINITE_OUTPUT_LIMIT')
        with (output/name).open('x') as f:f.write(text)
    def deadline(*_):raise TimeoutError('42_SECOND_MESH_WORK_CAP')
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    cpus=sorted(os.sched_getaffinity(0))
    if len(cpus)<9:raise ValueError('NINE_AVAILABLE_CPU_CORES_REQUIRED')
    os.sched_setaffinity(0,set(cpus[:9]))
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,max(.001,42-(time.monotonic()-START)))
    try:
        freeze=json.loads(Path(freeze_path).read_text())
        if freeze['decision']!='COMPLETE_ONLY_UNSEALED_MESH_CHECK_NO_RUNTIME_OR_MOTION_ADMISSION':raise ValueError('WRONG_SCOPE')
        for p,h in freeze['sha256'].items():
            if sha(p)!=h:raise ValueError(('INPUT_CHANGED',p))
        write('contract.json',freeze)
        project=Path(freeze['project']);paths={k:Path(p) for k,p in freeze['inputs'].items()}
        sys.path.insert(1,str(project/'sitting_task/src'))
        import numpy as np
        from humanoid_sitting.shared_frame import BodyModel,level_rotation
        from parallel_mesh import audit
        plan=json.loads(paths['plan'].read_text())
        with np.load(paths['frame']) as f:frame=dict(f)
        with np.load(paths['mesh']) as f:furniture=dict(f)
        shapes=json.loads(paths['colliders'].read_text())['shapes']
        allowed=json.loads(paths['target_region'].read_text())['triangle_indices']
        model=BodyModel(paths['model']);initial=level_rotation(np.asarray(frame['gravity'],float).copy())
        nodes=plan['nodes']+plan['return_nodes'][1:]
        report=audit(model,shapes,furniture,allowed,nodes,frame['root_rotation']@initial.T,
                     frame['root_position'],workers=8,max_wall_s=42-(time.monotonic()-START))
        write('mesh_audit.json',report)
        previous=sum(json.loads(paths[key].read_text())['wall_through_evidence_s'] for key in ('serial_accounting','partitioned_accounting'))
        write('result.json',dict(mesh_passed=report['passed'],sampled_poses=report['sampled_poses'],
            collision_count=report['forbidden_or_early_clearance_intersection_count'],
            wall_s=time.monotonic()-START,previous_attempt_wall_s=previous,
            cumulative_charged_wall_s=previous+time.monotonic()-START,
            end_to_end_runtime_passed=False,full_model_admitted=False,physical_motion_admitted=False,
            physical_success=False,path_reconstructed=False,completed_proofs_replayed=False,
            solver_calls=0,simulation_steps=0,PPO_updates=0,retry_authorized=False))
        print(json.dumps({k:v for k,v in report.items() if k not in ('forbidden_or_early_clearance_intersections',)},indent=2))
    except BaseException as e:
        write('failure.json',dict(error=repr(e),traceback=traceback.format_exc(),wall_s=time.monotonic()-START,
            full_model_admitted=False,physical_motion_admitted=False,physical_success=False,retry_authorized=False))
        print(type(e).__name__,str(e))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        write('process_accounting.json',dict(wall_through_evidence_s=time.monotonic()-START,total_cap_s=45,
            work_cap_s=42,coordinator_and_worker_CPUs=9,maximum_workers=8,parent_AS_GiB=8,worker_AS_GiB=2,
            aggregate_AS_cap_GiB=24,GPU_contexts=0,output_bytes=sum(p.stat().st_size for p in output.iterdir())))


if __name__=='__main__':main(*sys.argv[1:])
