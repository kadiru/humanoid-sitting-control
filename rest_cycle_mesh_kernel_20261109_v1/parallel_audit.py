"""Bounded CPU partition; each segment occurs once, with one terminal sample."""
import json,multiprocessing as mp,os,resource,time,traceback
from multiprocessing.connection import wait
from pathlib import Path
from mesh_audit import audit as compiled_audit


def _worker(send,index,cpu,args,library,wall,progress_directory):
    def progress(row):
        if progress_directory:
            path=Path(progress_directory)/f'worker_{index}.json';tmp=path.with_suffix('.tmp')
            tmp.write_text(json.dumps(row,allow_nan=False));os.replace(tmp,path)
    try:
        os.sched_setaffinity(0,{cpu});resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
        report,stats=compiled_audit(*args[:-1],include_terminal=args[-1],library=library,
                                   max_wall_s=wall,progress=progress)
        send.send((index,report,stats,None))
    except BaseException:send.send((index,None,None,traceback.format_exc()))
    finally:send.close()


def audit(model,shapes,furniture,allowed_faces,nodes,world_basis,initial_world_position,
          *,library,workers=8,max_wall_s=40.,progress_directory=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('EXPLICIT_CPU_ONLY_REQUIRED')
    if not 1<=workers<=8 or not 1<max_wall_s<=42 or not 2<=len(nodes)<=1281:
        raise ValueError('FINITE_PARALLEL_BUDGET_REQUIRED')
    if progress_directory:Path(progress_directory).mkdir(exist_ok=False)
    start=time.monotonic();stop=start+max_wall_s-1.;ctx=mp.get_context('fork')
    cpus=sorted(os.sched_getaffinity(0));count=min(workers,len(cpus),len(nodes)-1)
    processes=[];readers=[];results={};offsets={}
    try:
        for i in range(count):
            left=i*(len(nodes)-1)//count;right=(i+1)*(len(nodes)-1)//count
            receive,send=ctx.Pipe(duplex=False)
            args=(model,shapes,furniture,allowed_faces,nodes[left:right+1],world_basis,initial_world_position,i==count-1)
            process=ctx.Process(target=_worker,args=(send,i,cpus[i],args,library,max_wall_s-1.,progress_directory))
            process.start();send.close();processes.append(process);readers.append(receive);offsets[i]=4*left
        pending=set(readers)
        while pending:
            remaining=stop-time.monotonic()
            if remaining<=0:raise TimeoutError('COMPILED_PARTITION_WORK_BUDGET')
            for pipe in wait(pending,timeout=min(.1,remaining)):
                try:index,report,stats,error=pipe.recv()
                except EOFError as e:raise RuntimeError('WORKER_RETURN_MISSING') from e
                pending.remove(pipe)
                if error:raise RuntimeError('MESH_WORKER_FAILED\n'+error)
                results[index]=(report,stats)
        rows=[];samples=tests=collisions=expected=compiled=0
        for index in range(count):
            value,stats=results[index];samples+=value['sampled_poses'];tests+=value['triangle_tests']
            collisions+=value['forbidden_or_early_clearance_intersection_count'];expected+=value['expected_target_approaches']
            compiled+=stats['compiled']
            rows.extend({**row,'sample':row['sample']+offsets[index]} for row in value['forbidden_or_early_clearance_intersections'])
        if samples!=4*(len(nodes)-1)+1 or compiled!=tests:raise ValueError('INCOMPLETE_OR_DUPLICATED_AUDIT')
        return {**results[0][0], 'passed':collisions==0,'sampled_poses':samples,'triangle_tests':tests,
            'forbidden_or_early_clearance_intersections':rows[:50],
            'forbidden_or_early_clearance_intersection_count':collisions,'expected_target_approaches':expected},dict(
                workers=count,compiled=compiled,worker_wall_s=[results[i][1]['wall_s'] for i in range(count)],wall_s=time.monotonic()-start)
    finally:
        for p in processes:
            if p.is_alive():p.terminate()
        cleanup=min(start+max_wall_s,time.monotonic()+.8)
        for p in processes:p.join(timeout=max(0.,cleanup-time.monotonic()))
        for p in processes:
            if p.is_alive():p.kill()
        for p in processes:p.join(timeout=max(0.,start+max_wall_s-time.monotonic()))
        for pipe in readers:pipe.close()
