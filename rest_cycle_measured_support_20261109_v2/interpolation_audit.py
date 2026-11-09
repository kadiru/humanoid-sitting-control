"""Check the geometric interpolation between admitted path knots.

The existing clearance audit uses linear joint/translation interpolation and
Slerp rotation. Endpoint foot/plane checks do not certify that interpolation.
This reader checks every scheduled control sample, without solving a new pose.
It does not claim that interpolation equals the actual dynamic trajectory.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def audit_interpolation(model, nodes, scheduled, feet, anchors, pelvis_vertices, normal, offset):
    commands=np.asarray(scheduled['commands'],float)
    if commands.ndim!=2 or commands.shape[1]!=37 or not 1<=len(commands)<=1000:
        raise ValueError('FINITE_ORIGINAL_CONTROL_GRID_REQUIRED')
    if not np.isfinite(commands).all():raise ValueError('NONFINITE_COMMAND_GRID')
    vertices=np.asarray(pelvis_vertices,float);normal=np.asarray(normal,float)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices) or
        not np.isfinite(vertices).all() or normal.shape!=(3,) or not np.isfinite(normal).all() or
        abs(np.linalg.norm(normal)-1)>1e-8 or not np.isfinite(offset)):
        raise ValueError('FINITE_GEOMETRY_REQUIRED')
    if len(feet)!=2 or len(anchors)!=2:raise ValueError('TWO_FIXED_FOOT_ANCHORS_REQUIRED')
    cursor=0;rows=[];maximum_foot=0.;minimum_gap=float('inf');first=None;endpoint_maximum=0.
    for segment,(a,b) in enumerate(zip(nodes,nodes[1:])):
        matches=np.flatnonzero(np.max(abs(commands[cursor:]-b['command']),axis=1)<1e-12)
        if not len(matches):raise ValueError('ORDERED_COMMAND_NODE_ASSOCIATION_MISSING')
        count=int(matches[0])+1
        ra,rb=np.asarray(a['root'],float),np.asarray(b['root'],float)
        qa,qb=np.asarray(a['q'],float),np.asarray(b['q'],float)
        rotations=Slerp([0,1],Rotation.from_matrix(np.stack([ra[:3,:3],rb[:3,:3]])))
        for j in range(1,count+1):
            u=j/count;q=(1-u)*qa+u*qb
            root=np.eye(4);root[:3,:3]=rotations(u).as_matrix();root[:3,3]=(1-u)*ra[:3,3]+u*rb[:3,3]
            fk=model.fk(q);errors=[]
            for body,anchor in zip(feet,anchors):
                pose=root@fk[body]
                errors.extend(pose[:3,3]-anchor[:3,3])
                errors.extend(Rotation.from_matrix(anchor[:3,:3].T@pose[:3,:3]).as_rotvec())
            error=float(np.max(abs(np.asarray(errors))))
            gap=float(((vertices@root[:3,:3].T+root[:3,3])@normal+offset).min())
            if not np.isfinite([error,gap]).all():raise ValueError('NONFINITE_INTERPOLATED_STATE')
            maximum_foot=max(maximum_foot,error);minimum_gap=min(minimum_gap,gap)
            if j==count:endpoint_maximum=max(endpoint_maximum,error)
            row=dict(command_index=cursor+j-1,segment=segment,fraction=u,
                     foot_error=error,pelvis_plane_gap_m=gap,endpoint=j==count)
            rows.append(row)
            if first is None and (error>2e-5 or gap < -1e-6):first=row
        cursor+=count
    if cursor!=len(commands):raise ValueError('UNASSOCIATED_COMMAND_TAIL')
    return dict(passed=first is None,control_samples=len(rows),
        maximum_foot_error=maximum_foot,maximum_endpoint_foot_error=endpoint_maximum,
        minimum_pelvis_plane_gap_m=minimum_gap,first_violation=first,
        foot_tolerance=2e-5,plane_tolerance_m=1e-6,physical_success=False,
        interpretation='Checks the declared sampled geometric interpolation; does not identify the actual dynamic state or a cause of recorded slip.',
        rows=rows)
