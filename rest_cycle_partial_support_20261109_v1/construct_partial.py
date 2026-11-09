"""Cold partial support construction from the newly qualified camera capture."""
import os,time,signal,resource,sys,json,hashlib,traceback
from pathlib import Path
START=time.monotonic();os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def timeout(*a):raise TimeoutError('45_SECOND_TOTAL_PARTIAL_CONSTRUCTION_CAP')
signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,45)
R=Path('/home/kadir/humanoid-sitting');H=Path(__file__).resolve().parent
S=R/'outputs'/H.name/'camera_and_wrench_capture';O=H/'partial_construction'
sys.path[1:1]=[str(R/'experiments/upper_thigh_support_20261108_v4'),str(R/'experiments/furniture_repair_20261106_v1'),str(R/'scripts'),str(R/'sitting_task/src')]
def write(name,x):(O/name).write_text(json.dumps(x,indent=2)+'\n')
def main():
    O.mkdir(exist_ok=False)
    try:
        freeze=json.loads((H/'partial_freeze.json').read_text())
        for p,h in freeze['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
        import numpy as np
        from scipy.spatial.transform import Rotation
        from humanoid_sitting.shared_frame import BodyModel,level_rotation
        from partial_plan import PartialPlan
        from audit_seat_plan import audit as audit_geometry
        from audit_complete_path import audit as audit_commands
        from audit_mesh_clearance import audit as audit_clearance
        cap=json.loads((S/'capture_audit.json').read_text());edge=json.loads((S/'edge_sensor.json').read_text())
        assert cap['depth_passed'] and cap['edge_passed'] and cap['edge_envelope_passed']
        with np.load(S/'frame_0.npz') as f:z=dict(f)
        assert hashlib.sha256((S/'frame_0.npz').read_bytes()).hexdigest()==cap['source_frame_sha256']
        binding=json.loads((S/'current_snapshot_binding.json').read_text())
        model=BodyModel(R/'runs/g1_kinematics_geometry_04/model.json')
        inertia=json.loads(Path('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json').read_text())
        shapes=json.loads((R/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json').read_text())['shapes']
        plan=PartialPlan(model,inertia,shapes,z['q'],z['gravity'],z['geometry'],z['selected_points'],np.asarray(binding['previous_command']),front_upper=edge['front_x_interval_m'][1])
        result=plan.solve_partial();write('plan.json',result)
        geometry=audit_geometry(plan,dict(result,earliest_settled_seconds=result['nominal_latest_settled_s']))
        commands=audit_commands(plan,result,result)
        mesh=H.parent/'rest_cycle_high_support_20261108_v1/assets/scenes/high_support_table_050/mesh.npz'
        with np.load(mesh) as f:furniture=dict(f)
        allowed=json.loads(mesh.with_name('target_region.json').read_text())['triangle_indices']
        basis=z['root_rotation']@level_rotation(z['gravity'].astype(float).copy()).T
        clearance=audit_clearance(model,shapes,furniture,allowed,result['nodes']+result['landing'][1:],basis,z['root_position'])
        end=result['nodes'][-1];root=np.asarray(end['root']);x=np.zeros(30);x[:23]=end['q'][:23];x[23:26]=root[:3,3];x[26:29]=Rotation.from_matrix(plan.R0.T@root[:3,:3]).as_rotvec()
        work=plan.mechanics.independent_work_audit(x,.1)
        mechanical_min=1.;base_error=0.
        for node in result['nodes']+result['landing']:
            q=np.asarray(node['q']);root=np.asarray(node['root']);x[:23]=q[:23];x[23:26]=root[:3,3];x[26:29]=Rotation.from_matrix(plan.R0.T@root[:3,:3]).as_rotvec()
            for load in ([0.,.1] if node['phase']=='landing' else [0.]):
                s=plan.mechanics.state(x,load);base_error=max(base_error,float(abs(s['base_residual']).max()))
                for eq,point,F in zip(plan.mechanics.foot_polygons,s['foot_points'],s['foot_forces']):
                    mechanical_min=min(mechanical_min,float((-(eq[:,:2]@point[:2]+eq[:,2])-.01).min()),float(F[2]/plan.mechanics.W-.05),float((.5*F[2]-np.linalg.norm(F[:2]))/plan.mechanics.W))
        admitted=bool(geometry['passed'] and commands['passed'] and clearance['passed'] and work['passed'] and mechanical_min>=-1e-8 and base_error<=1e-8)
        report=dict(model_admitted=admitted,geometry=geometry,commands=commands,clearance=clearance,virtual_work=work,
            minimum_mechanical_margin=mechanical_min,maximum_base_wrench_residual=base_error,
            model_effort_ratio_at10percent_load=result['model_effort_ratio_at10percent_load'],
            actual_supported_load_validated=False,return_primitive_admitted=False,physical_success=False,
            new_physics_steps=0,PPO_updates=0,elapsed_seconds=time.monotonic()-START)
        write('audit.json',report);print(json.dumps({k:v for k,v in report.items() if not isinstance(v,dict)}))
    except BaseException as e:
        write('failure.json',dict(error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START,physical_success=False,new_physics_steps=0))
        if 'plan' in locals():write('stage_evidence.json',getattr(plan,'stage_evidence',{}))
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)
if __name__=='__main__':main()
