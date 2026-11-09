"""One bounded read-only saved-path collision evaluation, with full accounting."""
import time
START=time.monotonic()
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import hashlib,json,resource,signal,sys,traceback
from pathlib import Path


def main(freeze_path,output):
    output=Path(output);output.mkdir(exist_ok=False)
    def write(name,data):
        payload=(json.dumps(data,indent=2,allow_nan=False)+'\n').encode()
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())+len(payload)>16*1024**2:
            raise ValueError('OUTPUT_LIMIT')
        with (output/name).open('xb') as f:f.write(payload)
    def deadline(*_):raise TimeoutError('42_SECOND_COMPILED_AUDIT_WORK_STOP')
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    cpus=sorted(os.sched_getaffinity(0))
    if len(cpus)<9:raise ValueError('NINE_CPU_SLOTS_REQUIRED')
    os.sched_setaffinity(0,set(cpus[:9]))
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,max(.001,42-(time.monotonic()-START)))
    try:
        contract=json.loads(Path(freeze_path).read_text())
        if contract['decision']!='ONE_COMPILED_SAVED_MESH_AUDIT_NO_MOTION':raise ValueError('WRONG_EXECUTION_SCOPE')
        for path,digest in contract['sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError(('FROZEN_INPUT_CHANGED',path))
        write('contract.json',contract)
        project=Path(contract['project']);sys.path.insert(1,str(project/'sitting_task/src'))
        import numpy as np
        from humanoid_sitting.shared_frame import BodyModel,level_rotation
        from parallel_audit import audit
        p={k:Path(v) for k,v in contract['inputs'].items()};plan=json.loads(p['plan'].read_text())
        with np.load(p['frame']) as z:frame=dict(z)
        with np.load(p['furniture_mesh']) as z:furniture=dict(z)
        shapes=json.loads(p['colliders'].read_text())['shapes'];allowed=json.loads(p['target_region'].read_text())['triangle_indices']
        model=BodyModel(p['model']);initial=level_rotation(np.asarray(frame['gravity'],float).copy())
        nodes=plan['nodes']+plan['return_nodes'][1:]
        if len(nodes)!=1281 or len(shapes)!=38 or len(furniture['faces'])!=12:raise ValueError('FROZEN_COUNTS_CHANGED')
        report,stats=audit(model,shapes,furniture,allowed,nodes,frame['root_rotation']@initial.T,frame['root_position'],
            library=contract['library'],workers=8,max_wall_s=42-(time.monotonic()-START),progress_directory=output/'progress')
        write('mesh_audit.json',report);write('kernel_accounting.json',stats)
        write('result.json',dict(mesh_passed=report['passed'],sampled_poses=report['sampled_poses'],collision_count=report['forbidden_or_early_clearance_intersection_count'],
            wall_s=time.monotonic()-START,historical_wall_s=contract['historical_wall_s'],
            cumulative_charged_wall_s=contract['historical_wall_s']+time.monotonic()-START,
            end_to_end_runtime_passed=False,full_model_admitted=False,physical_motion_admitted=False,physical_success=False,
            path_reconstructed=False,completed_proofs_replayed=False,solver_calls=0,simulation_steps=0,PPO_updates=0,retry_authorized=False))
        print(json.dumps(dict(mesh_passed=report['passed'],samples=report['sampled_poses'],collision_count=report['forbidden_or_early_clearance_intersection_count'],**stats)))
    except BaseException as e:
        write('failure.json',dict(error=repr(e),traceback=traceback.format_exc(),wall_s=time.monotonic()-START,
            full_model_admitted=False,physical_motion_admitted=False,physical_success=False,retry_authorized=False))
        print(type(e).__name__,str(e))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        parent=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
        write('process_accounting.json',dict(wall_through_evidence_s=time.monotonic()-START,total_cap_s=45,work_stop_s=42,
            CPU_slots=9,workers=8,parent_AS_GiB=8,worker_AS_GiB=2,aggregate_AS_GiB=24,
            parent_user_cpu_s=parent.ru_utime,parent_system_cpu_s=parent.ru_stime,child_user_cpu_s=children.ru_utime,child_system_cpu_s=children.ru_stime,
            parent_max_RSS_KiB=parent.ru_maxrss,largest_child_RSS_KiB=children.ru_maxrss,GPU_contexts=0,
            output_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())))


if __name__=='__main__':main(*sys.argv[1:])
