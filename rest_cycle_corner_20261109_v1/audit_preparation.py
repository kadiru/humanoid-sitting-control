"""Rotate the unchanged50cm table to present one corner; preserve all geometry/materials."""
import os,time,json,hashlib,signal,resource,shutil,sys
from pathlib import Path
START=time.monotonic()
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def timeout(*args):raise TimeoutError('45_SECOND_PREPARATION_CAP')
signal.signal(signal.SIGALRM,timeout);signal.alarm(45)
R=Path('/home/kadir/humanoid-sitting');H=Path(__file__).resolve().parent

def main():
    import numpy as np
    from pxr import Usd,UsdGeom,UsdPhysics,Vt
    frozen=json.loads((H/'preparation_audit_freeze.json').read_text())
    for path,h in frozen['files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
    import numpy as np
    from pxr import Usd,UsdGeom,UsdPhysics
    old=R/'experiments/rest_cycle_high_support_20261108_v1/assets'
    folder=H/'assets/scenes/corner_support_table_050'
    case=json.loads((H/'assets/cases.json').read_text())[0]
    region=json.loads((folder/'target_region.json').read_text())
    with np.load(folder/'mesh.npz') as z:v=z['vertices'];faces=z['faces']
    with np.load(old/'scenes/high_support_table_050/mesh.npz') as z:source_vertices=z['vertices'];source_faces=z['faces']
    yaw=np.asarray(case['geometry_variant']['rigid_rotation']);translation=np.asarray(case['geometry_variant']['rigid_translation'])
    # Independent saved reset/mesh clearance and actualUSD round-trip.
    sys.path[1:1]=[str(R/'sitting_task/src'),str(R/'experiments/furniture_repair_20261106_v1')]
    from humanoid_sitting.shared_frame import BodyModel
    from audit_mesh_clearance import audit
    model=BodyModel(R/'runs/g1_kinematics_geometry_04/model.json')
    shapes=json.loads((R/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json').read_text())['shapes']
    frame=R/'outputs/rest_cycle_partial_support_20261109_v1/camera_and_wrench_capture/frame_0.npz'
    with np.load(frame) as z:
        node=dict(q=z['q'].tolist(),root=np.eye(4).tolist(),phase='approach')
        clearance=audit(model,shapes,dict(vertices=v,faces=faces),region['triangle_indices'],[node,node],z['root_rotation'],z['root_position'])
    reread=Usd.Stage.Open(str(folder/'furniture.usda'));mesh=UsdGeom.Mesh(reread.GetPrimAtPath('/Furniture/Mesh'))
    points=np.asarray(mesh.GetPointsAttr().Get());topology=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
    tests=dict(rigid_roundtrip=bool(np.max(abs((v-translation)@yaw-source_vertices))<1e-12),visual_collision_vertices_match=bool(np.max(abs(points-v))<1e-7),topology_exact=bool(np.array_equal(topology,faces)),
        collision_enabled=bool(UsdPhysics.CollisionAPI(mesh.GetPrim()).GetCollisionEnabledAttr().Get()),
        exact_triangles=UsdPhysics.MeshCollisionAPI(mesh.GetPrim()).GetApproximationAttr().Get()=='none',
        anchored=bool(UsdPhysics.RigidBodyAPI(reread.GetDefaultPrim()).GetKinematicEnabledAttr().Get()),
        height=abs(v[:,2].max()-.5)<1e-12,grounded=abs(v[:,2].min())<1e-7,standing_clearance=clearance['passed'])
    mat=UsdPhysics.MaterialAPI(reread.GetPrimAtPath('/Furniture/Material'))
    tests['materials_unchanged']=abs(mat.GetStaticFrictionAttr().Get()-.8)<1e-7 and abs(mat.GetDynamicFrictionAttr().Get()-.6)<1e-7 and mat.GetRestitutionAttr().Get()==0
    tests={k:bool(v) for k,v in tests.items()};tests['source_topology_unchanged']=bool(np.array_equal(source_faces,faces))
    result=dict(passed=all(tests.values()),tests=tests,clearance=clearance,case=case,elapsed_seconds=time.monotonic()-START,
        current_camera_validated=False,physical_support_validated=False,physical_success=False,physics_steps=0,PPO_updates=0)
    (H/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['case','clearance']}))
    assert result['passed'],'HIGH_SUPPORT_FIXTURE_REJECTED'

if __name__=='__main__':main()
