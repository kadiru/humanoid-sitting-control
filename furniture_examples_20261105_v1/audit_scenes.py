"""Independent CPU checks of prepared mesh topology, standing poses and coverage."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics
ROOT=Path('/home/kadir/humanoid-sitting');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'sitting_task/src'))
from humanoid_sitting.shared_frame import BodyModel,apply,transform
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    protocol=read(HERE/'protocol.json');vs=protocol['variants'];cases={c['id']:c for c in read(HERE/'cases.json')}
    assert len(vs)==22 and len({v['id'] for v in vs})==22
    assert sum(v['task']=='sitting' for v in vs)==10 and sum(v['task']=='leaning' for v in vs)==12
    assert len({v['asset'] for v in vs})==5
    model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json')
    inventory=read(ROOT/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json')['shapes']
    with np.load(ROOT/'outputs/full_body_contact_imitation_v4/dataset/leaning/bank.npz') as z:lean={k:z[k][0].copy() for k in ['reset_q','reset_root_pose']}
    checks=[];files={};meshes={}
    for ident,c in cases.items():
        p=HERE/'scenes'/ident
        with np.load(p/'mesh.npz') as z:vertices=z['vertices'];faces=z['faces']
        assert np.isfinite(vertices).all() and vertices.shape[1]==3 and faces.shape[1]==3
        assert vertices[:,2].min()>-1e-7 and vertices[:,0].max()<0
        stage=Usd.Stage.Open(str(p/'furniture.usda'));g=UsdGeom.Mesh(stage.GetPrimAtPath('/Furniture/Mesh'))
        np.testing.assert_allclose(np.asarray(g.GetPointsAttr().Get()),vertices,rtol=1e-7,atol=1e-7)
        np.testing.assert_array_equal(np.asarray(g.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),faces)
        assert UsdPhysics.CollisionAPI(g.GetPrim()).GetCollisionEnabledAttr().Get()
        assert UsdPhysics.MeshCollisionAPI(g.GetPrim()).GetApproximationAttr().Get()=='none'
        assert UsdPhysics.RigidBodyAPI(stage.GetDefaultPrim()).GetKinematicEnabledAttr().Get()
        region=read(p/'target_region.json')['triangle_indices'];assert region and min(region)>=0 and max(region)<len(faces)
        for name in ['mesh.npz','furniture.usda','target_region.json']:files[str(p/name)]=sha(p/name)
        meshes[ident]=(vertices,faces)
    for v in vs:
        if v['task']=='sitting':
            with np.load(HERE/'banks'/v['id']/'reset_candidates.npz') as z:q=z['q'][0,0];pose=z['root_pose'][0,0]
        else:
            q=lean['reset_q'];pose=lean['reset_root_pose'].copy();pose[:2]+=[v['x'],v['y']]
            pose[3:7]=(Rotation.from_euler('z',v['yaw_deg'],degrees=True)*Rotation.from_quat(pose[3:7])).as_quat()
        assert np.isfinite(q).all() and np.isfinite(pose).all()
        root=transform(Rotation.from_quat(pose[3:7]).as_matrix(),pose[:3]);fk=model.fk(q)
        vertices,faces=meshes[v['case']];tri=vertices[faces];lo,hi=tri.min(1),tri.max(1)
        overlaps=0;robot=[]
        for s in inventory:
            pts=apply(root@fk[s['body']],np.array(s['vertices']));robot.append((s['body'],pts,s['faces']))
            overlaps+=int(((hi>=pts.min(0)).all(1)&(lo<=pts.max(0)).all(1)).sum())
        checks.append(dict(id=v['id'],task=v['task'],asset=v['asset'],initial_triangle_AABB_overlaps=overlaps,standing_root=pose.tolist(),static_pass=overlaps==0))
        np.savez_compressed(HERE/'banks'/(v['id']+'_preview.npz'),q=q,root_pose=pose)
    result=dict(passed=all(x['static_pass'] for x in checks),counts=protocol['counts'],checks=checks,files=files,
        classification='STATIC_PREPARATION_ONLY',simulator_started=False,physical_support_verified=False,
        positive_demonstrations_admitted=0,limitations=['Triangle-AABB separation is conservative startup screening only.','Target regions are proposals until actual contacts are audited.','Camera recognition and full20s holds remain untested.'])
    (HERE/'static_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['checks','files']}),flush=True)

if __name__=='__main__':main()
