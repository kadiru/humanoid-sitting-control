"""Build and independently compare visual/collision assets for the200 layouts.

Static asset preparation only. No robot reset, simulation, optimizer or PPO.
"""
import os,time,json,hashlib,resource
os.environ['CUDA_VISIBLE_DEVICES']=''
START=time.monotonic()
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,Gf
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

freeze=read(HERE/'scene_freeze.json')
for p,d in freeze['files'].items():assert sha(p)==d,p
protocol=read(HERE/'protocol.json');directory=HERE/'scenes';directory.mkdir(exist_ok=False)
checks=[];cases=[];newfiles={}
for v in protocol['variants']:
    if time.monotonic()-START>44:raise TimeoutError('STATIC_ASSET_PREPARATION_CAP')
    folder=directory/v['id'];folder.mkdir()
    with np.load(v['base_mesh_npz']) as z:base={k:z[k].copy() for k in z.files}
    M=np.asarray(v['furniture_transform'],float);vertices=base['vertices']@M[:3,:3].T+M[:3,3]
    assert np.linalg.det(M[:3,:3])>0.999999999 and np.max(abs(M[:3,:3].T@M[:3,:3]-np.eye(3)))<1e-12
    assert np.array_equal(vertices[:,2],base['vertices'][:,2])
    mesh=dict(base,vertices=vertices);np.savez_compressed(folder/'mesh.npz',**mesh)
    stage=Usd.Stage.CreateNew(str(folder/'furniture.usda'))
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
    root=UsdGeom.Xform.Define(stage,'/Furniture');stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().GetReferences().AddReference(v['base_mesh_usd'])
    root.ClearXformOpOrder();root.AddTransformOp().Set(Gf.Matrix4d(M.T.tolist()))
    stage.GetRootLayer().Save()
    # Reopen and independently obtain COMPOSED USD coordinates and indices.
    loaded=Usd.Stage.Open(str(folder/'furniture.usda'));shape=UsdGeom.Mesh(loaded.GetPrimAtPath('/Furniture/Mesh'))
    P=np.asarray(shape.GetPointsAttr().Get(),float)
    A=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(shape.GetPrim()))
    world=P@A[:3,:3]+A[3,:3]
    np.testing.assert_allclose(world,vertices,rtol=1e-7,atol=1e-7)
    np.testing.assert_array_equal(np.asarray(shape.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),base['faces'])
    assert UsdPhysics.CollisionAPI(shape.GetPrim()).GetCollisionEnabledAttr().Get()
    assert UsdPhysics.MeshCollisionAPI(shape.GetPrim()).GetApproximationAttr().Get()=='none'
    assert UsdPhysics.RigidBodyAPI(loaded.GetDefaultPrim()).GetKinematicEnabledAttr().Get()
    src=Usd.Stage.Open(v['base_mesh_usd'])
    for getter in ('GetStaticFrictionAttr','GetDynamicFrictionAttr','GetRestitutionAttr'):
        newmat=UsdPhysics.MaterialAPI(loaded.GetPrimAtPath('/Furniture/Material'))
        oldmat=UsdPhysics.MaterialAPI(src.GetPrimAtPath('/Furniture/Material'))
        assert getattr(newmat,getter)().Get()==getattr(oldmat,getter)().Get()
    pending=v['surface_intent'].endswith('pending_hip_height_feasibility')
    if pending:
        region=dict(triangle_indices=[],status='UNSELECTED_HIP_LEAN_RIM; never reuse sitting top labels as a leaning patch',actor_truth_input=False)
    else:
        region=read(Path(v['base_mesh_npz']).with_name('target_region.json'))
        if 'center' in region:region['center']=(M[:3,:3]@np.asarray(region['center'])+M[:3,3]).tolist()
        if 'mean_normal' in region:region['mean_normal']=(M[:3,:3]@np.asarray(region['mean_normal'])).tolist()
        region['status']='Inherited proposed face IDs, rigidly transformed; fresh sensor/contact audit required'
        region['actor_truth_input']=False
    write(folder/'target_region.json',region)
    case=dict(id=v['id'],asset=v['asset'],title=v['title'],task=v['task'],
              mesh_usd=str(folder/'furniture.usda'),mesh_npz=str(folder/'mesh.npz'),
              target_region=str(folder/'target_region.json'),target_region_selected=not pending,
              robot_lateral_offset_m=v['original_robot_lateral_offset_m'],
              intent=v['support_intent'],bounds=[vertices.min(0).tolist(),vertices.max(0).tolist()],
              physics_launch_ready=False,original_criteria_unchanged=True)
    cases.append(case)
    checks.append(dict(id=v['id'],visual_collision_same_mesh_and_transform=True,
                       maximum_visual_evaluator_vertex_difference_m=float(np.max(abs(world-vertices))),
                       faces=len(base['faces']),native_height_preserved=True,friction_unchanged=True,
                       target_region_selected=not pending,robot_clearance_checked=False,physical_support_tested=False))
    for p in folder.iterdir():newfiles[str(p.relative_to(HERE))]=sha(p)
    if len(checks)%50==0:print('Prepared and compared',len(checks),'layouts',flush=True)
assert len(cases)==200
write(HERE/'prepared_cases.json',cases)
write(HERE/'scene_checks.json',dict(passed=True,scope='Static visual/collider/evaluator correspondence only',
    layouts=200,proposed_target_regions=160,hip_rim_regions_unselected=40,checks=checks,
    new_physics_steps=0,new_demonstrations=0,physics_launch_ready=False,
    wall_seconds=time.monotonic()-START,cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime))
write(HERE/'scene_manifest.json',newfiles)
print(json.dumps(dict(layouts=200,new_physics_steps=0,admitted_demonstrations=0,wall_seconds=time.monotonic()-START)),flush=True)
