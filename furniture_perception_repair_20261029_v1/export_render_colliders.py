"""CPU rendering geometry from exact authored colliders, including instances."""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics
from paths import BODY,HERE,read,write,sha

def main():
    source=read(BODY/'body_colliders.json');stage=Usd.Stage.Open(source['usd']);root=stage.GetDefaultPrim();cache=UsdGeom.XformCache();shapes=[]
    for body in root.GetChildren():
        if not body.HasAPI(UsdPhysics.RigidBodyAPI):continue
        for prim in Usd.PrimRange(body,Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():continue
            transform=np.asarray(cache.ComputeRelativeTransform(prim,body)[0]);kind=prim.GetTypeName()
            if kind=='Mesh':
                g=UsdGeom.Mesh(prim);v=np.asarray(g.GetPointsAttr().Get(),float);indices=g.GetFaceVertexIndicesAttr().Get();f=[];start=0
                for n in g.GetFaceVertexCountsAttr().Get():
                    poly=indices[start:start+n];start+=n;f.extend([[poly[0],poly[j],poly[j+1]] for j in range(1,n-1)])
                approximation=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get();assert approximation=='convexHull'
                mesh=trimesh.convex.convex_hull(v)
            elif kind=='Cylinder':
                g=UsdGeom.Cylinder(prim);mesh=trimesh.creation.cylinder(radius=g.GetRadiusAttr().Get(),height=g.GetHeightAttr().Get(),sections=32)
                axis=str(g.GetAxisAttr().Get());rotation=Rotation.align_vectors(np.eye(3)[['X','Y','Z'].index(axis)][None],np.array([[0,0,1]]))[0]
                mesh.vertices=rotation.apply(mesh.vertices);approximation='analyticCylinder_display32sides'
            elif kind=='Cube':
                size=UsdGeom.Cube(prim).GetSizeAttr().Get();mesh=trimesh.creation.box(extents=[size]*3);approximation='analyticCube'
            else:raise ValueError(('UNSUPPORTED_COLLIDER',kind,str(prim.GetPath())))
            v=(np.c_[mesh.vertices,np.ones(len(mesh.vertices))]@transform)[:,:3]
            shapes.append(dict(body=body.GetName(),path=str(prim.GetPath()),vertices=v.tolist(),faces=mesh.faces.tolist(),approximation=approximation))
    assert len(shapes)>=38 and {x['body'] for x in shapes}>={'pelvis','torso_link','left_ankle_roll_link','right_ankle_roll_link'}
    write(HERE/'render_colliders.json',dict(shapes=shapes,source_layers=source['source_layers'],usd=source['usd'],
        scope='Authored collision hulls. Cylinders displayed with32sides. Not a bit-exact PhysX cooking certification.'))
    print('Exported',len(shapes),'complete robot collider shapes onCPU.')
if __name__=='__main__':main()
