"""Explicit new anchored narrow support fixture; no old mesh is modified."""
import os,time,json,hashlib,signal,resource,shutil,sys
from pathlib import Path
START=time.monotonic();os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def timeout(*a):raise TimeoutError('45_SECOND_PREPARATION_CAP')
signal.signal(signal.SIGALRM,timeout);signal.alarm(45)
R=Path('/home/kadir/humanoid-sitting');H=Path(__file__).resolve().parent
def main():
    import numpy as np
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Vt,Gf
    for p,h in json.loads((H/'preparation_freeze.json').read_text())['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    folder=H/'assets/scenes/narrow_support_050';folder.mkdir(parents=True,exist_ok=False)
    v=np.array([[-.48,-.06,0],[-.08,-.06,0],[-.08,.06,0],[-.48,.06,0],[-.48,-.06,.5],[-.08,-.06,.5],[-.08,.06,.5],[-.48,.06,.5]])
    faces=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    np.savez_compressed(folder/'mesh.npz',vertices=v,faces=faces)
    stage=Usd.Stage.CreateNew(str(folder/'furniture.usda'));UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
    root=UsdGeom.Xform.Define(stage,'/Furniture');stage.SetDefaultPrim(root.GetPrim())
    rigid=UsdPhysics.RigidBodyAPI.Apply(root.GetPrim());rigid.CreateRigidBodyEnabledAttr(True);rigid.CreateKinematicEnabledAttr(True)
    mesh=UsdGeom.Mesh.Define(stage,'/Furniture/Mesh');mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist());mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([v.min(0),v.max(0)]).astype(np.float32)));mesh.CreateDisplayColorAttr([(.57,.39,.22)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True);UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
    mat=UsdShade.Material.Define(stage,'/Furniture/Material');phys=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys.CreateStaticFrictionAttr(.8);phys.CreateDynamicFrictionAttr(.6);phys.CreateRestitutionAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat,UsdShade.Tokens.weakerThanDescendants,'physics');stage.GetRootLayer().Save()
    old=R/'experiments/rest_cycle_high_support_20261108_v1/assets';case=json.loads((old/'cases.json').read_text())[0]
    region=dict(center=[-.28,0,.5],mean_normal=[0,0,1],triangle_indices=[2,3],source='analytic new anchored fixture; distinct from furniture asset')
    (folder/'target_region.json').write_text(json.dumps(region,indent=2)+'\n')
    case.update(id='narrow_support_050',asset='analytic_narrow_pedestal',title='Dedicated12cm narrow support test fixture',mesh_npz=str(folder/'mesh.npz'),mesh_usd=str(folder/'furniture.usda'),
        bounds=[v.min(0).tolist(),v.max(0).tolist()],target_center=region['center'],target_normal=region['mean_normal'],geometry_variant=dict(width_m=.12,depth_m=.40,height_m=.50,anchored=True,old_assets_modified=False),contact_scoring_validated=False)
    (H/'assets/cases.json').write_text(json.dumps([case],indent=2)+'\n')
    variant=json.loads((old/'protocol.json').read_text())['variants'][0];variant.update(id='sit_narrow_support_050_center',case=case['id'])
    (H/'assets/protocol.json').write_text(json.dumps(dict(variants=[variant]),indent=2)+'\n')
    bank=H/'assets/banks'/variant['id'];bank.mkdir(parents=True);shutil.copy2(old/'banks/sit_high_support_050_center/reset_candidates.npz',bank/'reset_candidates.npz')
    sys.path[1:1]=[str(R/'sitting_task/src'),str(R/'experiments/furniture_repair_20261106_v1')]
    from humanoid_sitting.shared_frame import BodyModel
    from audit_mesh_clearance import audit
    model=BodyModel(R/'runs/g1_kinematics_geometry_04/model.json');shapes=json.loads((R/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json').read_text())['shapes']
    with np.load(R/'outputs/rest_cycle_partial_support_20261109_v1/camera_and_wrench_capture/frame_0.npz') as z:
        node=dict(q=z['q'].tolist(),root=np.eye(4).tolist(),phase='approach')
        clearance=audit(model,shapes,dict(vertices=v,faces=faces),[2,3],[node,node],z['root_rotation'],z['root_position'])
    reread=Usd.Stage.Open(str(folder/'furniture.usda'));m=UsdGeom.Mesh(reread.GetPrimAtPath('/Furniture/Mesh'));actual=np.asarray(m.GetPointsAttr().Get())
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1);unique,counts=np.unique(edges,axis=0,return_counts=True)
    tests=dict(closed_manifold=bool(np.all(counts==2)),exact_topology=bool(np.array_equal(np.asarray(m.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),faces)),
        visual_collision_match=bool(abs(actual-v).max()<1e-7),standing_clearance=bool(clearance['passed']),collision_enabled=bool(UsdPhysics.CollisionAPI(m.GetPrim()).GetCollisionEnabledAttr().Get()),
        exact_triangles=bool(UsdPhysics.MeshCollisionAPI(m.GetPrim()).GetApproximationAttr().Get()=='none'),anchored=bool(UsdPhysics.RigidBodyAPI(reread.GetDefaultPrim()).GetKinematicEnabledAttr().Get()))
    report=dict(passed=all(tests.values()),tests=tests,clearance=clearance,case=case,elapsed_seconds=time.monotonic()-START,physical_support_validated=False,new_physics_steps=0)
    (H/'preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['case','clearance']}));assert report['passed']
if __name__=='__main__':main()
