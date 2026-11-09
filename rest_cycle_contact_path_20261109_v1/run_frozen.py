"""One frozen, bounded CPU construction; never starts simulation or PPO."""
import time
START=time.monotonic()
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import json,hashlib,signal,resource,sys,traceback
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main(freeze_path,output):
    output=Path(output);output.mkdir(exist_ok=False)
    def write(name,value):
        payload=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
        if sum(p.stat().st_size for p in output.iterdir())+len(payload)>32*1024**2:
            raise ValueError('TOTAL_OUTPUT_CAP_32_MIB')
        with (output/name).open('xb') as f:f.write(payload)
    def deadline(*_):raise TimeoutError('42_SECOND_WORK_STOP_WITHIN_45_SECOND_TOTAL_CAP')
    signal.signal(signal.SIGALRM,deadline)
    signal.setitimer(signal.ITIMER_REAL,max(.001,42-(time.monotonic()-START)))
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{available[0]})
    builder=None;stage='verify_frozen_sources_and_inputs'
    try:
        freeze=json.loads(Path(freeze_path).read_text())
        if freeze['decision']!='ONE_CPU_CONSTRUCTION_NO_AUTOMATIC_DOWNSTREAM':raise ValueError('NO_LAUNCH_CONTRACT')
        for p,h in freeze['sha256'].items():
            if sha(p)!=h:raise ValueError(('FROZEN_INPUT_CHANGED',p))
        partitioned=freeze['resources'].get('mesh_backend')=='exact_partition'
        if partitioned:
            if len(available)<9:raise ValueError('NINE_AVAILABLE_CPU_CORES_REQUIRED')
            os.sched_setaffinity(0,set(available[:9]))
        paths={k:Path(v) for k,v in freeze['inputs'].items()}
        if not paths['review_log'].read_text().rstrip().endswith('OK'):raise ValueError('SYNTHETIC_REVIEW_FAILED')
        write('contract.json',freeze)
        import numpy as np
        from scipy.spatial.transform import Rotation
        project=Path(freeze['project']);exp=project/'experiments';source=Path(__file__).resolve().parent
        sys.path[1:1]=[str(exp/'upper_thigh_support_20261108_v4'),
            str(exp/'rest_cycle_rolling_perch_20261109_v2'),str(exp/'rest_cycle_measured_support_20261109_v2'),
            str(exp/'furniture_repair_20261106_v1'),str(project/'sitting_task/src')]
        from humanoid_sitting.shared_frame import BodyModel
        from builder import ContactPath
        from plan_contract import check_plan
        from interior_audit import audit as audit_interiors
        from corner_audit import audit as audit_complete
        stage='bind_accepted_physical_reference'
        results=json.loads(paths['reference_results'].read_text())
        row=results['episodes'][16];independent=json.loads(paths['reference_audit'].read_text())
        assert row['id']=='known_old/16' and row['clean'] and row['full_standing_start'] and row['final_hold_s']>=5
        assert independent['clean'][16]
        assert freeze['sha256'][str(paths['reference_trajectory'])]==independent['files']['trajectory.npz']
        assert freeze['sha256'][str(paths['reference_results'])]==independent['files']['results.json']
        with np.load(paths['reference_trajectory']) as z:
            support=z['support_bw'][:,16]
            starts=np.flatnonzero(np.convolve((support>=.05).astype(int),np.ones(25,dtype=int),'valid')==25)
            if not len(starts):raise ValueError('NO_QUALIFIED_CONTACT_ORIENTATION_IN_ACCEPTED_REFERENCE')
            contact_tick=int(starts[0]);body=z['body'][:,16,results['body_names'].index('pelvis')]
            contact_R=Rotation.from_quat(body[contact_tick,3:]).as_matrix()
            held_R=Rotation.from_quat(body[-1,3:]).as_matrix()
        geometry=json.loads(paths['reference_geometry'].read_text())[16]
        reference=dict(normal=geometry['normal'],contact_rotation=contact_R.tolist(),held_rotation=held_R.tolist(),
                       case='known_old/16',contact_tick=contact_tick,contact_association='first25consecutive samples >=0.05BW',
                       source_final_hold_s=row['final_hold_s'])
        stage='bind_current_physical_start_and_camera'
        cap=json.loads(paths['capture_audit'].read_text());region=json.loads(paths['region_audit'].read_text())
        binding=json.loads(paths['binding'].read_text())
        assert cap['passed'] and region['passed']
        assert binding['frame_sha256']==freeze['sha256'][str(paths['frame'])]
        with np.load(paths['frame']) as f:frame=dict(f)
        model=BodyModel(paths['model']);inertia=json.loads(paths['inertia'].read_text())
        shapes=json.loads(paths['colliders'].read_text())['shapes'];patch=json.loads(paths['patch'].read_text())
        stage='fresh_per_control_construction'
        builder=ContactPath(model,inertia,shapes,frame,binding['previous_command'],patch,reference)
        write('association.json',builder.association)
        plan=builder.construct();write('plan.json',plan)
        stage='independent_finite_command_and_timing_audit'
        check=check_plan(plan,builder.mechanics.lower,builder.mechanics.upper,builder.previous)
        write('command_contract.json',check)
        if not check['passed']:raise ValueError('COMMAND_CONTRACT_FAILED')
        stage='independent_all_quarter_control_audit'
        interiors=audit_interiors(model,plan['nodes'],builder.mechanics.foot_bodies,builder.mechanics.feet,
                                  builder.vertices,builder.n,builder.d)
        write('interior_audit.json',interiors)
        if not interiors['passed']:raise ValueError('INTERIOR_CONSTRAINT_FAILED')
        stage='independent_full_original_model_and_mesh_audit'
        with np.load(paths['furniture_mesh']) as f:furniture=dict(f)
        allowed=json.loads(paths['target_region'].read_text())['triangle_indices']
        mesh=None
        if partitioned:
            from parallel_mesh import audit as parallel_audit
            def mesh(*args):
                return parallel_audit(*args,workers=8,max_wall_s=42-(time.monotonic()-START))
        complete=audit_complete(builder,plan,shapes,furniture,allowed,frame,clearance_evaluator=mesh)
        write('model_audit.json',complete)
        if not complete['model_admitted']:raise ValueError('FULL_ORIGINAL_MODEL_AUDIT_FAILED')
        write('result.json',dict(model_admitted=True,physical_motion_admitted=False,physical_success=False,
            controls=len(plan['forward']['commands']),rest_load_bw=plan['rest_load'],
            rest_effort_ratio=plan['rest_model_effort_ratio'],closure_resources=plan['closure_resources'],
            wall_s=time.monotonic()-START,GPU_launches=0,simulation_steps=0,PPO_updates=0,
            observer_and_native_quiet_admission_outstanding=True,repeat_authorized=False))
        print('MODEL_AUDIT_PASSED; physical motion remains unadmitted')
    except BaseException as e:
        failure=dict(stage=stage,builder_stage=getattr(builder,'stage',None),error=repr(e),
            traceback=traceback.format_exc(),wall_s=time.monotonic()-START,
            completed_controls=max(0,len(builder.nodes)-1) if builder is not None else 0,
            closure_FK_evaluations=builder.closure.calls if builder is not None else 0,
            linear_system_solves=builder.closure.linear_solves if builder is not None else 0,
            terminal_optimizer_calls=0,GPU_launches=0,simulation_steps=0,PPO_updates=0,
            model_admitted=False,physical_success=False,physical_motion_admitted=False,repeat_authorized=False)
        write('failure.json',failure)
        print(json.dumps({k:v for k,v in failure.items() if k!='traceback'},indent=2))
    finally:
        if builder is not None:
            write('construction_trace.json',builder.trace)
            if not (output/'plan.json').exists():write('partial_nodes_not_admitted.json',builder.nodes)
        signal.setitimer(signal.ITIMER_REAL,0)
        total=time.monotonic()-START
        write('process_accounting.json',dict(wall_through_evidence_s=total,configured_total_cap_s=45,
            work_stop_s=42,cap_met=total<=45,CPU_cores=9 if locals().get('partitioned',False) else 1,
            parent_address_space_cap_GiB=8,worker_address_space_cap_GiB=2 if locals().get('partitioned',False) else 0,
            maximum_worker_processes=8 if locals().get('partitioned',False) else 0,
            GPU_contexts=0,output_bytes=sum(p.stat().st_size for p in output.iterdir()),
            numerical_return_is_not_physical_success=True))


if __name__=='__main__':main(sys.argv[1],sys.argv[2])
