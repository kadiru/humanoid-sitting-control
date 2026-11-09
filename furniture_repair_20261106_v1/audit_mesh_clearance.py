"""Evaluation-only triangle/convex collision check; never actor input."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp
from humanoid_sitting.shared_frame import apply

def clip_triangle(triangle, equations, expansion):
    polygon=np.asarray(triangle)
    for normal in equations:
        if not len(polygon):return polygon
        distance=polygon@normal[:3]+normal[3]-expansion
        if np.all(distance<=0):continue
        if np.all(distance>0):return np.empty((0,3))
        out=[]
        for i,p in enumerate(polygon):
            j=(i+1)%len(polygon);q=polygon[j];a=distance[i];b=distance[j]
            if a<=0:out.append(p)
            if (a<=0)!=(b<=0):out.append(p+a/(a-b)*(q-p))
        polygon=np.asarray(out)
    return polygon

def audit(model, shapes, furniture, allowed_faces, nodes, world_basis, initial_world_position):
    triangles=furniture['vertices'][furniture['faces']]
    tlo=triangles.min(1);thi=triangles.max(1)
    prepared=[]
    for shape in shapes:
        vertices=np.asarray(shape['vertices']);hull=ConvexHull(vertices)
        prepared.append((shape['body'],vertices[hull.vertices],hull.equations))
    allowed=set(allowed_faces);rest={'pelvis','left_hip_yaw_link','right_hip_yaw_link'}
    sampled=[]
    for a,b in zip(nodes,nodes[1:]):
        ra=np.asarray(a['root']);rb=np.asarray(b['root'])
        rotations=Slerp([0,1],Rotation.from_matrix(np.stack([ra[:3,:3],rb[:3,:3]])))
        for u in (0.,.25,.5,.75):
            rt=np.eye(4);rt[:3,:3]=rotations(u).as_matrix();rt[:3,3]=(1-u)*ra[:3,3]+u*rb[:3,3]
            sampled.append((np.asarray(a['q'])*(1-u)+np.asarray(b['q'])*u,rt,a['phase']))
    sampled.append((np.asarray(nodes[-1]['q']),np.asarray(nodes[-1]['root']),nodes[-1]['phase']))
    collisions=[];expected=[];tests=0
    for sample,(q,root,phase) in enumerate(sampled):
        fk=model.fk(q);rt=np.eye(4);rt[:3,:3]=world_basis@root[:3,:3]
        rt[:3,3]=initial_world_position+world_basis@root[:3,3]
        for body,vertices,equations in prepared:
            pose=rt@fk[body];world=apply(pose,vertices);expansion=.006
            lo=world.min(0)-expansion;hi=world.max(0)+expansion
            candidates=np.flatnonzero(np.all(thi>=lo,axis=1)&np.all(tlo<=hi,axis=1))
            for face in candidates:
                tests+=1;local=(triangles[face]-pose[:3,3])@pose[:3,:3]
                polygon=clip_triangle(local,equations,expansion)
                if not len(polygon):continue
                row=dict(sample=sample,body=body,face=int(face),phase=phase)
                if body in rest and phase=='landing' and face in allowed:
                    expected.append(row)
                else:collisions.append(row)
    return dict(passed=not collisions,sampled_poses=len(sampled),triangle_tests=tests,
                forbidden_or_early_clearance_intersections=collisions[:50],
                forbidden_or_early_clearance_intersection_count=len(collisions),
                expected_target_approaches=len(expected),clearance_buffer_m=.006,
                scope='Evaluation-only mesh geometry at sampled kinematic poses. Continuous dynamics, actual contact membership, load and slip still require unchanged simulation audit.',
                simulation_success=False)
