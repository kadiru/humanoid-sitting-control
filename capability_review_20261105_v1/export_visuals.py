"""CPU-only visual mesh export; never creates a simulator or physics scene."""
from pathlib import Path
import json, hashlib
import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics

ROOT = Path('/home/kadir/humanoid-sitting')
OUT = ROOT/'outputs/capability_review_20261105_v1'
OUT.mkdir(exist_ok=True)
source = json.loads((ROOT/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json').read_text())
stage = Usd.Stage.Open(source['usd'])
cache = UsdGeom.XformCache()
arrays, records = {}, []
for body in stage.GetDefaultPrim().GetChildren():
    if not body.HasAPI(UsdPhysics.RigidBodyAPI):
        continue
    meshes = []
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() != 'Mesh' or '/visuals/' not in str(prim.GetPath()):
            continue
        g = UsdGeom.Mesh(prim)
        if g.ComputeVisibility() == 'invisible':
            continue
        v = np.array(g.GetPointsAttr().Get(), dtype=float)
        inds = np.asarray(g.GetFaceVertexIndicesAttr().Get())
        faces, start = [], 0
        for n in g.GetFaceVertexCountsAttr().Get():
            poly = inds[start:start+n]; start += n
            faces.extend([[poly[0], poly[j], poly[j+1]] for j in range(1,n-1)])
        transform = np.asarray(cache.ComputeRelativeTransform(prim,body)[0])
        v = (np.c_[v,np.ones(len(v))] @ transform)[:,:3]
        m = trimesh.Trimesh(v, faces, process=True)
        count = len(m.faces)
        limit = 3500 if body.GetName() in ['head_link','torso_link','pelvis'] else 1200
        if count > limit:
            m = m.simplify_quadric_decimation(face_count=limit)
        meshes.append(m)
        records.append(dict(body=body.GetName(),path=str(prim.GetPath()),original_triangles=count,display_triangles=len(m.faces)))
    if meshes:
        m = trimesh.util.concatenate(meshes)
        arrays[body.GetName()+'_v'] = np.asarray(m.vertices,np.float32)
        arrays[body.GetName()+'_f'] = np.asarray(m.faces,np.int32)
assert len(arrays)//2 >= 40
np.savez_compressed(OUT/'visuals.npz',**arrays)
(OUT/'visuals_manifest.json').write_text(json.dumps(dict(source_usd=source['usd'],source_sha256=hashlib.sha256(Path(source['usd']).read_bytes()).hexdigest(),meshes=records,scope='Display-only decimation of authored visual meshes. Recorded robot poses and all physical scores unchanged.'),indent=2))
print('Exported',len(arrays)//2,'bodies',sum(len(v) for k,v in arrays.items() if k.endswith('_f')),'triangles',flush=True)
