"""Fresh input association and independent checks of an accepted fixed path."""
import os,time,signal,resource,sys,json,hashlib,traceback
from pathlib import Path
START=time.monotonic();os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def timeout(*_):raise TimeoutError('45_SECOND_CURRENT_PATH_ADMISSION')
signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,45)
R=Path('/home/kadir/humanoid-sitting');E=R/'experiments';H=Path(__file__).resolve().parent;O=Path(sys.argv[1])
sys.path[1:1]=[str(E/n) for n in ('rest_cycle_measured_support_20261109_v2','rest_cycle_patch_topology_20261109_v1','rest_cycle_rolling_perch_20261109_v2','rest_cycle_witness_pivot_20261109_v1','upper_thigh_support_20261108_v4','furniture_repair_20261106_v1','furniture_perception_repair_20261029_v1')]+[str(R/'sitting_task/src')]

def main():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from humanoid_sitting.shared_frame import BodyModel,transform,level_rotation
    from observed_patch_v3 import build
    from sweep_plan import CornerPerch
    from corner_audit import audit
    import audit_depth
    read=lambda p:json.loads(p.read_text())
    for p,h in read(H/'calibration_freeze.json')['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    model_folder=E/'rest_cycle_measured_support_20261109_v2'
    assert read(model_folder/'construction/result.json')['model_admitted']
    manifest=read(O/'manifest.json');case=manifest['case']
    assets=E/'rest_cycle_narrow_support_20261109_v1/assets';audit_depth.ASSETS=assets
    depth=audit_depth.check(O/'frame_0.npz',case)
    with np.load(O/'frame_0.npz') as f:frame=dict(f)
    saved=R/'outputs/rest_cycle_narrow_support_20261109_v1/camera_and_wrench_capture'
    with np.load(saved/'frame_0.npz') as f:old=dict(f)
    current=read(O/'current_snapshot_binding.json');binding=read(saved/'current_snapshot_binding.json')
    assert hashlib.sha256((O/'frame_0.npz').read_bytes()).hexdigest()==current['frame_sha256']
    tests=dict(actual_depth=depth['geometry_pass'] and depth['original_plane_gate'] and depth['target_observable'],
        q_binding=bool(np.max(abs(frame['q']-old['q']))<=2e-5),
        commands_binding=bool(np.max(abs(np.array(current['previous_command'])-binding['previous_command']))<=2e-7))
    level=level_rotation(frame['gravity'].astype(float).copy())
    T=np.asarray(read(E/'rest_cycle_partial_support_20261109_v1/camera_preflight.json')['camera_in_body'])
    edge,mask=build(frame,T,level);n=level@frame['geometry'][2:5];n/=np.linalg.norm(n)
    e,b=np.asarray(edge['e']),np.asarray(edge['b']);uv=np.asarray(edge['vertices_uv'])
    edge['front_x_conservative']=float((uv[:,0,None]*e+uv[:,1,None]*b-float(frame['geometry'][5])*n)[:,0].max())
    src=R/'outputs/armchair_rest_ppo_20261105_v1/learned_sitting72';outcome=read(src/'results.json');ind=read(src/'independent_audit.json')
    assert outcome['episodes'][16]['id']=='known_old/16' and outcome['episodes'][16]['clean']
    assert hashlib.sha256((src/'trajectory.npz').read_bytes()).hexdigest()==ind['files']['trajectory.npz']
    geo=read(R/'outputs/full_body_contact_imitation_v4/dataset/sitting/geometry.json')[16]
    with np.load(src/'trajectory.npz') as f:
        q=f['q'][-1,16];p=f['body'][-1,16,outcome['body_names'].index('pelvis')].copy();p[:3]-=np.asarray(outcome['origins'][16])
    sn=np.asarray(geo['normal']);se=np.array([1.,0,0])-sn*sn[0];se/=np.linalg.norm(se)
    source=dict(q=q,root=transform(Rotation.from_quat(p[3:]).as_matrix(),p[:3]),normal=sn,
        front_projection=float(np.asarray(geo['top_center'])@se+geo['depth']/2))
    model=BodyModel(R/'runs/g1_kinematics_geometry_04/model.json')
    inertia=read(Path('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json'))
    shapes=read(E/'furniture_perception_repair_20261029_v1/render_colliders.json')['shapes']
    plan=CornerPerch(model,inertia,shapes,frame,current['previous_command'],source,edge)
    result=read(model_folder/'construction/plan.json')
    with np.load(case['mesh_npz']) as f:mesh=dict(f)
    allowed=read(Path(case['mesh_npz']).with_name('target_region.json'))['triangle_indices']
    mechanical=audit(plan,result,shapes,mesh,allowed,frame);tests['current_path']=mechanical['model_admitted']
    value=dict(passed=bool(all(tests.values())),tests=tests,depth=depth,mechanical=mechanical,
        frame_sha256=current['frame_sha256'],elapsed_seconds=time.monotonic()-START,
        no_optimizer_called=True,physical_success=False,scope='Current-input admission of bounded calibration; load-based controller not admitted')
    (O/'current_admission.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({k:value[k] for k in ('passed','tests','elapsed_seconds')}));assert value['passed']

if __name__=='__main__':
    try:main()
    except BaseException as e:
        (O/'admission_failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START),indent=2)+'\n');raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)
