"""Independent audit of fresh camera and shadow sensor readouts, not a controller."""
import os,time,signal,resource,sys,json,hashlib
from pathlib import Path
START=time.monotonic();os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def timeout(*a):raise TimeoutError('45_SECOND_TOTAL_CAPTURE_AUDIT_CAP')
signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,45)
R=Path('/home/kadir/humanoid-sitting');H=Path(__file__).resolve().parent
O=R/'outputs'/H.name/'camera_and_wrench_capture'
sys.path[:0]=[str(R/'sitting_task/src'),str(R/'experiments/furniture_perception_repair_20261029_v1')]
def write(name,x):(O/name).write_text(json.dumps(x,indent=2)+'\n')
def main():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from edge import detect
    from release import separation
    from humanoid_sitting.shared_frame import BodyModel,level_rotation,apply
    import audit_depth
    assert not (O/'capture_audit.json').exists()
    freeze=json.loads((H/'audit_freeze.json').read_text())
    for p,h in freeze['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    manifest=json.loads((O/'manifest.json').read_text());case=manifest['case']
    audit_depth.ASSETS=H.parent/'rest_cycle_high_support_20261108_v1/assets'
    source=O/'frame_0.npz'
    depthcheck=audit_depth.check(source,case);write('depth_audit.json',depthcheck)
    with np.load(source) as f:z=dict(f)
    g=z['geometry'];level=level_rotation(z['gravity'].astype(float).copy())
    T=np.asarray(json.loads((H/'camera_preflight.json').read_text())['camera_in_body'])
    sensor,arrays=detect(z['depth'],z['K'],T,g[2:5],float(g[5]),level,z['selected_pixels'])
    write('edge_sensor.json',sensor);np.savez_compressed(O/'edge_arrays.npz',**arrays)
    envelope=dict(passed=False,reason='edge_not_qualified');clear=dict(state='unknown',reason='edge_not_qualified')
    if sensor['passed']:
        with np.load(case['mesh_npz']) as f:fv=f['vertices']
        basis=z['root_rotation']@level.T;verts=(fv-z['root_position'])@basis
        lo,hi=sensor['front_x_interval_m'];n=level@g[2:5]
        envelope=dict(passed=bool(lo-.006<=verts[:,0].max()<=hi+.006 and (verts@n+g[5]).max()<=.003),
            actual_front_x_m=float(verts[:,0].max()),interval=[lo,hi],actual_top_excess_m=float((verts@n+g[5]).max()))
        if envelope['passed']:
            model=BodyModel(R/'runs/g1_kinematics_geometry_04/model.json');fk=model.fk(z['q'])
            shapes=json.loads((R/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json').read_text())['shapes']
            bybody={}
            for s in shapes:bybody.setdefault(s['body'],[]).extend((apply(fk[s['body']],s['vertices'])@level.T).tolist())
            clear=separation(vertices=bybody,plane_normal=n,plane_offset=float(g[5]),front_upper=hi,front_error=.006,
                normal_error=.003,plane_center=level@g[6:9],qualified_edge=True,geometry_age=float(g[12]))
    write('finite_surface_audit.json',envelope);write('release_standing.json',clear)
    with np.load(O/'shadow_wrench.npz') as f:w=dict(f)
    with np.load(O/'standing_calibration.npz') as f:trajectory=dict(f)
    names=json.loads((O/'results.json').read_text())['body_names']
    assert len(w['physics_step'])==200 and np.array_equal(w['physics_step'],np.arange(101,301))
    assert w['joint_wrench'].shape[-2:]==(len(names),6)
    # Verify native quaternion/order against the independently recorded body poses.
    pose_error=float(abs(w['link_transform'][3::4]-trajectory['body']).max())
    foot=[names.index(s+'_ankle_roll_link') for s in ['left','right']]
    Tfoot=w['link_transform'][:,0,foot]
    rotations=Rotation.from_quat(Tfoot[:,:,3:7].reshape(-1,4)).as_matrix().reshape(200,2,3,3)
    transmitted=np.einsum('tbij,tbj->tbi',rotations,w['joint_wrench'][:,0,foot,:3])
    mass=w['mass'][:,0,foot]
    acceleration=np.diff(w['velocity'][:,0,foot,:3],axis=0)/.005
    gravity=np.array([0.,0.,-9.81])
    predicted=mass[1:,:,None]*(acceleration-gravity)-transmitted[1:]
    truth=w['truth_contact_force'][1:,-2:]
    residual=predicted-truth
    weight=float(w['mass'][0].sum()*9.81)
    err=np.linalg.norm(residual,axis=-1)/weight
    tests=dict(native_pose_order=pose_error<2e-6,finite=np.isfinite(err).all(),
        mean_foot_force_error=float(err.mean())<=.005,p99_foot_force_error=float(np.quantile(err,.99))<=.01)
    ft=dict(passed=bool(all(tests.values())),tests={k:bool(v) for k,v in tests.items()},
        native_pose_max_error=pose_error,foot_force_mean_error_bw=float(err.mean()),foot_force_p99_error_bw=float(np.quantile(err,.99)),
        predicted_foot_vertical_mean_N=predicted[:,:,2].mean(0).tolist(),truth_foot_vertical_mean_N=truth[:,:,2].mean(0).tolist(),
        interpretation='Joint-transmitted force plus foot inertia/gravity, not motor torque. Standing-only sensor mapping check; no loaded validation.',
        loaded_validation=False,actor_sensor_admitted=False,physical_hardware_validated=False,
        official_reference='https://docs.isaacsim.omniverse.nvidia.com/latest/sensors/isaacsim_sensors_physics_articulation_force.html')
    write('shadow_wrench_audit.json',ft)
    final=dict(depth_passed=depthcheck['geometry_pass'] and depthcheck['original_plane_gate'] and depthcheck['target_observable'],
        edge_passed=sensor['passed'],edge_envelope_passed=envelope['passed'],release_standing=clear['state'],
        standing_joint_force_mapping_passed=ft['passed'],sitting_controls=0,complete_cycle=False,PPO_updates=0,
        elapsed_seconds=time.monotonic()-START,source_frame_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    write('capture_audit.json',final);print(json.dumps(final))
if __name__=='__main__':
    try:main()
    finally:signal.setitimer(signal.ITIMER_REAL,0)
