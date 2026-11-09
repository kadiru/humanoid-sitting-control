"""One45s,onecore corner-perch construction with exact geometric rolling."""
import os,time,signal,resource,sys,json,hashlib,traceback
from pathlib import Path
START=time.monotonic();os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def deadline(*a):raise TimeoutError('45_SECOND_TOTAL_ROLLING_PERCH_CONSTRUCTION_CAP')
signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,45)
R=Path('/home/kadir/humanoid-sitting');E=R/'experiments';H=Path(__file__).resolve().parent;O=H/'nominal_construction'
sys.path[1:1]=[str(E/'rest_cycle_corner_20261109_v1'),str(E/'rest_cycle_rolling_perch_20261109_v2'),str(E/'rest_cycle_witness_pivot_20261109_v1'),str(E/'upper_thigh_support_20261108_v4'),str(E/'furniture_repair_20261106_v1'),str(R/'sitting_task/src')]
def write(n,x):(O/n).write_text(json.dumps(x,indent=2)+'\n')
def main():
    O.mkdir(exist_ok=False);plan=None
    try:
        for p,h in json.loads((H/'nominal_freeze.json').read_text())['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
        import numpy as np
        from scipy.spatial.transform import Rotation
        from humanoid_sitting.shared_frame import BodyModel,transform
        from sweep_plan import CornerPerch
        from corner_audit import audit
        src=R/'outputs/armchair_rest_ppo_20261105_v1/learned_sitting72'
        outcome=json.loads((src/'results.json').read_text());row=outcome['episodes'][16]
        independent=json.loads((src/'independent_audit.json').read_text())
        assert row['id']=='known_old/16' and row['clean'] and row['final_hold_s']>=5
        assert hashlib.sha256((src/'trajectory.npz').read_bytes()).hexdigest()==independent['files']['trajectory.npz']
        assert independent['validated_clean'][16] if 'validated_clean' in independent else independent['clean'][16]
        geometry=json.loads((R/'outputs/full_body_contact_imitation_v4/dataset/sitting/geometry.json').read_text())[16]
        with np.load(src/'trajectory.npz') as z:
            q=z['q'][-1,16];p=z['body'][-1,16,outcome['body_names'].index('pelvis')].copy();p[:3]-=np.asarray(outcome['origins'][16])
        sr=transform(Rotation.from_quat(p[3:]).as_matrix(),p[:3]);normal=np.asarray(geometry['normal']);e=np.array([1.,0,0])-normal*normal[0];e/=np.linalg.norm(e)
        source=dict(q=q.tolist(),root=sr.tolist(),normal=normal.tolist(),front_projection=float(np.asarray(geometry['top_center'])@e+geometry['depth']/2))
        # Offline design hypothesis from explicit fixture geometry. This is NOT
        # a current-camera observation and cannot admit simulator motion.
        S=R/'outputs/rest_cycle_corner_20261109_v1/camera_and_wrench_capture'
        with np.load(S/'frame_0.npz') as f:frame=dict(f)
        previous=json.loads((S/'current_snapshot_binding.json').read_text())['previous_command']
        from humanoid_sitting.shared_frame import level_rotation
        from shapely.geometry import Polygon
        level=level_rotation(frame['gravity'].astype(float).copy());basis=frame['root_rotation']@level.T
        worldtop=np.array([[-.48,-.06,.5],[-.08,-.06,.5],[-.08,.06,.5],[-.48,.06,.5]])
        vertices=(worldtop-frame['root_position'])@basis;n=basis.T@np.array([0.,0.,1.]);n/=np.linalg.norm(n)
        d=float((frame['root_position']-worldtop[0])@np.array([0.,0.,1.]))
        e=np.array([1.,0.,0.])-n*n[0];e/=np.linalg.norm(e);b=np.cross(n,e)
        poly=Polygon(np.c_[vertices@e,vertices@b]).buffer(-.006,join_style=2)
        edge=dict(e=e.tolist(),b=b.tolist(),vertices_uv=list(map(list,poly.exterior.coords[:-1])),holes_uv=[],uncertainty_m=.006,source='KNOWN_FIXTURE_NOMINAL_DESIGN_NOT_CAMERA',front_x_conservative=float(vertices[:,0].max()-.006))
        frame['geometry']=frame['geometry'].copy();frame['geometry'][2:5]=level.T@n;frame['geometry'][5]=d
        frame['selected_points']=vertices@level
        write('input_scope.json',dict(current_narrow_fixture_camera_observation=False,uses_known_fixture_geometry_for_design=True,uses_saved_real_standing_proprioception=True,
          physics_launch_allowed=False,reason='4090 occupied by another workload; camera validation remains outstanding'))
        model=BodyModel(R/'runs/g1_kinematics_geometry_04/model.json')
        inertia=json.loads(Path('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json').read_text())
        shapes=json.loads((E/'furniture_perception_repair_20261029_v1/render_colliders.json').read_text())['shapes']
        plan=CornerPerch(model,inertia,shapes,frame,previous,source,edge)
        result=plan.construct();write('plan.json',result);write('stage_evidence.json',plan.evidence)
        mesh=E/'rest_cycle_narrow_support_20261109_v1/assets/scenes/narrow_support_050/mesh.npz'
        with np.load(mesh) as z:furniture=dict(z)
        allowed=json.loads(mesh.with_name('target_region.json').read_text())['triangle_indices']
        result_audit=audit(plan,result,shapes,furniture,allowed,frame);write('audit.json',result_audit)
        write('result.json',dict(nominal_model_passed=result_audit['model_admitted'],model_admitted=False,current_camera_validated=False,physical_success=False,new_physics_steps=0,PPO_updates=0,
            nominal_rest_settle_s=result['nominal_rest_settle_s'],nominal_return_settle_s=result['nominal_return_settle_s'],rest_load=result['rest_load'],model_effort_ratio=result['rest_model_effort_ratio'],elapsed_seconds=time.monotonic()-START))
    except BaseException as e:
        write('failure.json',dict(error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START,physical_success=False,new_physics_steps=0))
        if plan is not None:write('stage_evidence.json',plan.evidence)
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)
if __name__=='__main__':main()
