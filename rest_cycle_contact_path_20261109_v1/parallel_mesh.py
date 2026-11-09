"""Partition the original mesh audit without changing any geometric operation.

Each original segment retains its four original samples. Only the last shard
includes the final endpoint. Results are merged in original sample/body/face
order, including the exact original first-fifty collision convention.
"""
import multiprocessing as mp
from multiprocessing.connection import wait
import os,resource,time,traceback
from audit_mesh_clearance import audit as serial_audit


def _worker(send,index,cpu,args):
    try:
        os.sched_setaffinity(0,{cpu})
        resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
        value=serial_audit(*args[:-1],include_terminal=args[-1])
        send.send((index,value,None))
    except BaseException:
        send.send((index,None,traceback.format_exc()))
    finally:send.close()


def audit(model,shapes,furniture,allowed_faces,nodes,world_basis,initial_world_position,*,workers=8,max_wall_s=30.):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('EXPLICIT_CPU_ONLY_CONTEXT_REQUIRED')
    if not 1<=workers<=8 or max_wall_s<=1. or not 2<=len(nodes)<=1281:
        raise ValueError('FINITE_PARALLEL_AUDIT_BUDGET_REQUIRED')
    start=time.monotonic();stop=start+max_wall_s-1.
    ctx=mp.get_context('fork');cpus=sorted(os.sched_getaffinity(0))
    count=min(workers,len(cpus),len(nodes)-1);processes=[];readers=[];results={};offsets={}
    try:
        for i in range(count):
            left=i*(len(nodes)-1)//count;right=(i+1)*(len(nodes)-1)//count
            receive,send=ctx.Pipe(duplex=False)
            args=(model,shapes,furniture,allowed_faces,nodes[left:right+1],world_basis,initial_world_position,
                  i==count-1)
            process=ctx.Process(target=_worker,args=(send,i,cpus[i],args))
            process.start();send.close();processes.append(process);readers.append(receive);offsets[i]=4*left
        pending=set(readers)
        while pending:
            remaining=stop-time.monotonic()
            if remaining<=0:raise TimeoutError('PARALLEL_MESH_WORK_BUDGET')
            for pipe in wait(pending,timeout=min(remaining,.1)):
                try:index,value,error=pipe.recv()
                except EOFError as e:raise RuntimeError('MESH_WORKER_RETURN_MISSING') from e
                pending.remove(pipe)
                if error is not None:raise RuntimeError('MESH_WORKER_FAILED\n'+error)
                results[index]=value
        rows=[];samples=tests=collisions=expected=0
        for index in range(count):
            value=results[index];samples+=value['sampled_poses'];tests+=value['triangle_tests']
            collisions+=value['forbidden_or_early_clearance_intersection_count'];expected+=value['expected_target_approaches']
            for row in value['forbidden_or_early_clearance_intersections']:
                rows.append({**row,'sample':row['sample']+offsets[index]})
        result={**results[0], 'passed':collisions==0,'sampled_poses':samples,'triangle_tests':tests,
                'forbidden_or_early_clearance_intersections':rows[:50],
                'forbidden_or_early_clearance_intersection_count':collisions,'expected_target_approaches':expected}
        if samples!=4*(len(nodes)-1)+1:raise ValueError('INCOMPLETE_OR_DUPLICATED_SAMPLE_PARTITION')
        return result
    finally:
        for process in processes:
            if process.is_alive():process.terminate()
        cleanup_end=min(start+max_wall_s,time.monotonic()+.8)
        for process in processes:process.join(timeout=max(0.,cleanup_end-time.monotonic()))
        for process in processes:
            if process.is_alive():process.kill()
        for process in processes:process.join(timeout=max(0.,start+max_wall_s-time.monotonic()))
        for pipe in readers:pipe.close()
