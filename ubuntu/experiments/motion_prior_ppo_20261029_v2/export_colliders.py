"""Read-only CPU extraction from the same cached robot USD, including instances."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import trimesh
from pxr import Usd,UsdGeom,UsdPhysics

def main():
    p=argparse.ArgumentParser();p.add_argument('--usd',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    stage=Usd.Stage.Open(str(a.usd));root=stage.GetDefaultPrim();cache=UsdGeom.XformCache();shapes=[]
    wanted=['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link']
    for body in wanted:
        b=stage.GetPrimAtPath(str(root.GetPath())+'/'+body);assert b.IsValid()
        for prim in Usd.PrimRange(b,Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            assert prim.IsA(UsdGeom.Mesh),('UNSUPPORTED_BODY_COLLIDER',str(prim.GetPath()))
            mesh=UsdGeom.Mesh(prim);v=np.asarray(mesh.GetPointsAttr().Get(),float);counts=mesh.GetFaceVertexCountsAttr().Get();indices=mesh.GetFaceVertexIndicesAttr().Get();f=[];start=0
            for n in counts:
                face=indices[start:start+n];start+=n;f.extend([[face[0],face[j],face[j+1]] for j in range(1,n-1)])
            transform=np.asarray(cache.ComputeRelativeTransform(prim,b)[0]);v=(np.c_[v,np.ones(len(v))]@transform)[:,:3]
            approx=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            assert approx=='convexHull',('UNSUPPORTED_APPROXIMATION',body,approx)
            hull=trimesh.convex.convex_hull(v)
            shapes.append(dict(body=body,path=str(prim.GetPath()),approximation=approx,vertices=hull.vertices.tolist(),faces=hull.faces.tolist(),instance_proxy=prim.IsInstanceProxy()))
    assert {s['body'] for s in shapes}==set(wanted)
    layers={layer.realPath:hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest() for layer in stage.GetUsedLayers() if layer.realPath and Path(layer.realPath).is_file()}
    result=dict(shapes=shapes,usd=str(a.usd),source_layers=layers,scope='Authored collider hulls from the exact cached G1 USD, traversing instance proxies. Controlled replicas; not a bit-exact PhysX cooking certification.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print({s['body']:len(s['vertices']) for s in shapes})

if __name__=='__main__':main()
