"""Create a separate50cm test table; preserve the original geometry/materials."""
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
    frozen=json.loads((H/'preparation_freeze.json').read_text())
    for path,h in frozen['files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
    old=R/'experiments/furniture_examples_20261105_v1'
    source=old/'scenes/modern_coffee_table_01_top';assets=H/'assets'
    folder=assets/'scenes/high_support_table_050';folder.mkdir(parents=True,exist_ok=False)
    with np.load(source/'mesh.npz') as z:v=z['vertices'].copy();faces=z['faces'].copy()
    scale=.50/float(v[:,2].max());v[:,2]*=scale;v[:,0]+=.10
    shutil.copy2(source/'furniture.usda',folder/'furniture.usda')
    stage=Usd.Stage.Open(str(folder/'furniture.usda'));g=UsdGeom.Mesh(stage.GetPrimAtPath('/Furniture/Mesh'))
    g.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
    g.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([v.min(0),v.max(0)]).astype(np.float32)))
    stage.GetRootLayer().Save()
    np.savez_compressed(folder/'mesh.npz',vertices=v,faces=faces)
    region=json.loads((source/'target_region.json').read_text());region['center'][0]+=.10;region['center'][2]*=scale
    (folder/'target_region.json').write_text(json.dumps(region,indent=2)+'\n')
    case=next(c for c in json.loads((old/'cases.json').read_text()) if c['id']=='modern_coffee_table_01_top')
    case.update(id='high_support_table_050',title='50cm support table — separate test variant',
        mesh_npz=str(folder/'mesh.npz'),mesh_usd=str(folder/'furniture.usda'),bounds=[v.min(0).tolist(),v.max(0).tolist()],
        target_center=region['center'],target_normal=region['mean_normal'],
        geometry_variant=dict(original_height_m=.39,height_m=.50,z_scale=scale,x_shift_m=.10),contact_scoring_validated=False)
    (assets/'cases.json').write_text(json.dumps([case],indent=2)+'\n')
    variant=next(v for v in json.loads((old/'protocol.json').read_text())['variants'] if v['id']=='sit_modern_coffee_table_01_center')
    variant.update(id='sit_high_support_050_center',case=case['id'])
    (assets/'protocol.json').write_text(json.dumps(dict(variants=[variant]),indent=2)+'\n')
    bank=assets/'banks'/variant['id'];bank.mkdir(parents=True)
    shutil.copy2(old/'banks/sit_modern_coffee_table_01_center/reset_candidates.npz',bank/'reset_candidates.npz')
    # Independent saved reset/mesh clearance and actualUSD round-trip.
    sys.path[1:1]=[str(R/'sitting_task/src'),str(R/'experiments/furniture_repair_20261106_v1')]
    from humanoid_sitting.shared_frame import BodyModel
    from audit_mesh_clearance import audit
    model=BodyModel(R/'runs/g1_kinematics_geometry_04/model.json')
    shapes=json.loads((R/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json').read_text())['shapes']
    frame=R/'outputs/rest_cycle_wide_camera_20261108_v1/sit_coffee_wide/frame_0.npz'
    with np.load(frame) as z:
        node=dict(q=z['q'].tolist(),root=np.eye(4).tolist(),phase='approach')
        clearance=audit(model,shapes,dict(vertices=v,faces=faces),region['triangle_indices'],[node,node],z['root_rotation'],z['root_position'])
    reread=Usd.Stage.Open(str(folder/'furniture.usda'));mesh=UsdGeom.Mesh(reread.GetPrimAtPath('/Furniture/Mesh'))
    points=np.asarray(mesh.GetPointsAttr().Get());topology=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
    tests=dict(visual_collision_vertices_match=bool(np.max(abs(points-v))<1e-7),topology_exact=bool(np.array_equal(topology,faces)),
        collision_enabled=bool(UsdPhysics.CollisionAPI(mesh.GetPrim()).GetCollisionEnabledAttr().Get()),
        exact_triangles=UsdPhysics.MeshCollisionAPI(mesh.GetPrim()).GetApproximationAttr().Get()=='none',
        anchored=bool(UsdPhysics.RigidBodyAPI(reread.GetDefaultPrim()).GetKinematicEnabledAttr().Get()),
        height=abs(v[:,2].max()-.5)<1e-12,grounded=abs(v[:,2].min())<1e-7,standing_clearance=clearance['passed'])
    mat=UsdPhysics.MaterialAPI(reread.GetPrimAtPath('/Furniture/Material'))
    tests['materials_unchanged']=abs(mat.GetStaticFrictionAttr().Get()-.8)<1e-7 and abs(mat.GetDynamicFrictionAttr().Get()-.6)<1e-7 and mat.GetRestitutionAttr().Get()==0
    result=dict(passed=all(tests.values()),tests=tests,clearance=clearance,case=case,elapsed_seconds=time.monotonic()-START,
        current_camera_validated=False,physical_support_validated=False,physical_success=False,physics_steps=0,PPO_updates=0)
    (H/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['case','clearance']}))
    assert result['passed'],'HIGH_SUPPORT_FIXTURE_REJECTED'

if __name__=='__main__':main()
