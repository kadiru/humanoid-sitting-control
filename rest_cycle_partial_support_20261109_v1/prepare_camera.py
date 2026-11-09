"""One geometric camera-mount construction, no simulation or pose search."""
import os,time,signal,resource,sys,json,hashlib
from pathlib import Path
START=time.monotonic();os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def timeout(*a):raise TimeoutError('45_SECOND_CAMERA_CONSTRUCTION_CAP')
signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,45)
R=Path('/home/kadir/humanoid-sitting');H=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'sitting_task/src'))
def main():
    out=H/'camera_preflight.json';assert not out.exists()
    import numpy as np
    import trimesh
    from humanoid_sitting.shared_frame import BodyModel,apply
    frozen=json.loads((H/'camera_freeze.json').read_text())
    for p,h in frozen['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    with np.load(frozen['frame']) as f:z=dict(f)
    model=BodyModel(R/'runs/g1_kinematics_geometry_04/model.json');fk=model.fk(z['q'])
    shapes=json.loads((R/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json').read_text())['shapes']
    vertices=[];faces=[];offset=0
    for s in shapes:
        v=apply(fk[s['body']],s['vertices']);vertices.extend(v);faces.extend(np.asarray(s['faces'])+offset);offset+=len(v)
    vertices=np.asarray(vertices);faces=np.asarray(faces)
    # Closed-form conservative rear envelope. Keep original sensor height/lateral offset.
    origin=np.array([min(-.07,float(vertices[:,0].min())-.006),0.,.38])
    assert -.07-origin[0]<=.15,'CAMERA_STANDOFF_EXCEEDS15CM_DESIGN_LIMIT'
    with np.load(frozen['fixture_mesh']) as f:fv=f['vertices'];ff=f['faces']
    region=json.loads(Path(frozen['fixture_mesh']).with_name('target_region.json').read_text())['triangle_indices']
    top=fv[ff[region]].reshape(-1,3)
    front=float(top[:,0].max());height=float(np.median(top[:,2]))
    y=np.r_[-.45,-.35,-.25,-.15,.15,.25,.35,.45]
    edgeworld=np.c_[np.repeat(front,len(y)),y,np.repeat(height,len(y))]
    edge=(edgeworld-z['root_position'])@z['root_rotation']
    rearworld=np.array([float(top[:,0].min()),0.,height])
    rear=(rearworld-z['root_position'])@z['root_rotation']
    dirs=np.vstack([edge-origin,rear-origin]);theta=np.arctan2(-dirs[:,2],-dirs[:,0])
    angle=float(.5*(theta.min()+theta.max()))
    T=np.eye(4);T[:3,3]=origin
    T[:3,:3]=np.array([[0.,np.sin(angle),-np.cos(angle)],[1.,0.,0.],[0.,-np.cos(angle),-np.sin(angle)]])
    camera=(edge-origin)@T[:3,:3];K=z['K'];uv=camera[:,:2]/camera[:,2,None]*[K[0,0],K[1,1]]+[K[0,2]-.5,K[1,2]-.5]
    inside=(camera[:,2]>.08)&(uv[:,0]>=2)&(uv[:,0]<=z['depth'].shape[1]-3)&(uv[:,1]>=2)&(uv[:,1]<=z['depth'].shape[0]-3)
    mesh=trimesh.Trimesh(vertices,faces,process=False)
    vectors=edge-origin;length=np.linalg.norm(vectors,axis=1);directions=vectors/length[:,None]
    hits,rays,face_ids=mesh.ray.intersects_location(np.broadcast_to(origin,directions.shape),directions,multiple_hits=False)
    blocking=[]
    for hit,ray in zip(hits,rays):
        distance=float((hit-origin)@directions[ray])
        if 0<=distance<length[ray]-.006:blocking.append(dict(ray=int(ray),distance_m=distance,target_distance_m=float(length[ray])))
    tests=dict(all_flank_rays_inside_image=bool(inside.all()),no_robot_ray_occlusions=not blocking,
       camera_behind_robot_envelope=bool(origin[0]<=vertices[:,0].min()-.006+1e-12),proper_rotation=bool(abs(np.linalg.det(T[:3,:3])-1)<1e-9))
    report=dict(passed=all(tests.values()),tests=tests,camera_in_body=T.tolist(),camera_position_m=origin.tolist(),
       rearward_mount_change_m=float(-.07-origin[0]),downward_angle_deg=float(np.rad2deg(angle)),
       visibility_points_world=edgeworld.tolist(),camera_pixels=uv.tolist(),blocked_rays=blocking,
       construction='Behind complete current robot collision envelope by6mm; pitch bisects extreme near/far tabletop ray angles. One candidate, no parameter sweep.',
       sensor_hardware_assumption_changed=True,physical_robot_mount_validated=False,
       actual_camera_capture_required=True,action_limits_unchanged=True,physics_steps=0,
       elapsed_seconds=time.monotonic()-START)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':
    try:main()
    finally:signal.setitimer(signal.ITIMER_REAL,0)
