"""Original NumPy distances and scalar clipping, with edge loops in C++.

All samples, phase labels, broad-phase candidates, transforms, margins, and
contact classifications are retained. No GPU imports or changed tolerances.
"""
import importlib.util
from pathlib import Path
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp
from humanoid_sitting.shared_frame import apply


class Prepared:
    def __init__(self,module,equations):
        self.equations=np.array(equations,dtype=np.float64,order='C',copy=True)
        if self.equations.ndim!=2 or self.equations.shape[1]!=4 or not np.isfinite(self.equations).all():
            raise ValueError('FINITE_PLANES_REQUIRED')
        self.equations.flags.writeable=False
        self.capsule=module.prepare(self.equations)


class Clipper:
    def __init__(self,library):
        spec=importlib.util.spec_from_file_location('numpy_clip',str(Path(library).resolve()))
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.compiled=0

    def prepare(self,equations):return Prepared(self.module,equations)

    def polygon(self,triangle,equations,expansion=.006):
        triangle=np.ascontiguousarray(triangle,dtype=np.float64)
        if triangle.shape!=(3,3):raise ValueError('TRIANGLE_REQUIRED')
        if not np.isfinite(triangle).all() or not np.isfinite(expansion):raise ValueError('FINITE_GEOMETRY_REQUIRED')
        prepared=equations if isinstance(equations,Prepared) else self.prepare(equations)
        out=self.module.clip(triangle,prepared.capsule,expansion)
        self.compiled+=1
        if not np.isfinite(out).all():raise ValueError('NONFINITE_CLIPPED_POLYGON')
        return out

    def intersects(self,triangle,equations,expansion=.006):
        return bool(len(self.polygon(triangle,equations,expansion)))


def audit(model,shapes,furniture,allowed_faces,nodes,world_basis,initial_world_position,
          *,library,max_wall_s=40.,progress=None,include_terminal=True):
    start=time.monotonic();clipper=Clipper(library)
    if not 0<max_wall_s<=42 or not 2<=len(nodes)<=1281:
        raise ValueError('FINITE_AUDIT_BUDGET_REQUIRED')
    triangles=furniture['vertices'][furniture['faces']]
    if triangles.shape[0]>100000 or not np.isfinite(triangles).all():raise ValueError('FINITE_MESH_REQUIRED')
    tlo=triangles.min(1);thi=triangles.max(1)
    prepared=[]
    for shape in shapes:
        vertices=np.asarray(shape['vertices']);hull=ConvexHull(vertices)
        prepared.append((shape['body'],vertices[hull.vertices],clipper.prepare(hull.equations)))
    if len(prepared)>128:raise ValueError('FINITE_COLLIDER_COUNT_REQUIRED')
    allowed=set(allowed_faces);rest={'pelvis','left_hip_yaw_link','right_hip_yaw_link'}
    # Keep all original interpolation operations and the exact original order.
    sampled=[]
    for a,b in zip(nodes,nodes[1:]):
        ra=np.asarray(a['root']);rb=np.asarray(b['root'])
        rotations=Slerp([0,1],Rotation.from_matrix(np.stack([ra[:3,:3],rb[:3,:3]])))
        for u in (0.,.25,.5,.75):
            rt=np.eye(4);rt[:3,:3]=rotations(u).as_matrix();rt[:3,3]=(1-u)*ra[:3,3]+u*rb[:3,3]
            sampled.append((np.asarray(a['q'])*(1-u)+np.asarray(b['q'])*u,rt,a['phase']))
    if include_terminal:
        sampled.append((np.asarray(nodes[-1]['q']),np.asarray(nodes[-1]['root']),nodes[-1]['phase']))
    collisions=[];collision_count=expected=tests=0
    for sample,(q,root,phase) in enumerate(sampled):
        if time.monotonic()-start>max_wall_s:raise TimeoutError('COMPILED_MESH_WORK_BUDGET')
        fk=model.fk(q);rt=np.eye(4);rt[:3,:3]=world_basis@root[:3,:3]
        rt[:3,3]=initial_world_position+world_basis@root[:3,3]
        for body,vertices,equations in prepared:
            pose=rt@fk[body];world=apply(pose,vertices);expansion=.006
            lo=world.min(0)-expansion;hi=world.max(0)+expansion
            candidates=np.flatnonzero(np.all(thi>=lo,axis=1)&np.all(tlo<=hi,axis=1))
            for face in candidates:
                tests+=1;local=(triangles[face]-pose[:3,3])@pose[:3,:3]
                if not clipper.intersects(local,equations,expansion):continue
                row=dict(sample=sample,body=body,face=int(face),phase=phase)
                if body in rest and phase=='landing' and face in allowed:expected+=1
                else:
                    collision_count+=1
                    if len(collisions)<50:collisions.append(row)
        if progress is not None and (sample%128==0 or sample==len(sampled)-1):
            progress(dict(samples_completed=sample+1,samples_total=len(sampled),triangle_tests=tests,
                compiled=clipper.compiled,collision_count=collision_count,
                first_collisions=collisions,elapsed_s=time.monotonic()-start,complete=False))
    report=dict(passed=collision_count==0,sampled_poses=len(sampled),triangle_tests=tests,
        forbidden_or_early_clearance_intersections=collisions,
        forbidden_or_early_clearance_intersection_count=collision_count,
        expected_target_approaches=expected,clearance_buffer_m=.006,
        scope='Evaluation-only mesh geometry at sampled kinematic poses. Continuous dynamics, actual contact membership, load and slip still require unchanged simulation audit.',
        simulation_success=False)
    return report,dict(compiled=clipper.compiled,wall_s=time.monotonic()-start)
