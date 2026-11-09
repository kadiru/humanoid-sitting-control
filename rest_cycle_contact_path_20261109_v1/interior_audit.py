"""Independent quarter-control geometry checks using original FK and limits."""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp


def audit(model,nodes,feet,anchors,vertices,normal,offset):
    if not 2<=len(nodes)<=645:raise ValueError('FINITE_PATH_REQUIRED')
    vertices=np.asarray(vertices,float);normal=np.asarray(normal,float)
    rows=[]
    for index,(a,b) in enumerate(zip(nodes,nodes[1:])):
        ra,rb=np.asarray(a['root']),np.asarray(b['root'])
        rotations=Slerp([0,1],Rotation.from_matrix(np.stack([ra[:3,:3],rb[:3,:3]])))
        for u in (.25,.5,.75,1.):
            q=(1-u)*np.asarray(a['q'])+u*np.asarray(b['q'])
            root=np.eye(4);root[:3,:3]=rotations(u).as_matrix();root[:3,3]=(1-u)*ra[:3,3]+u*rb[:3,3]
            fk=model.fk(q);errors=[]
            for body,anchor in zip(feet,anchors):
                pose=root@fk[body];errors.extend(pose[:3,3]-anchor[:3,3])
                errors.extend(Rotation.from_matrix(anchor[:3,:3].T@pose[:3,:3]).as_rotvec())
            foot=float(max(abs(np.asarray(errors))))
            gap=float(((vertices@root[:3,:3].T+root[:3,3])@normal+offset).min())
            if not np.isfinite([foot,gap]).all():raise ValueError('NONFINITE_INTERIOR')
            rows.append(dict(control=index,fraction=u,foot_error=foot,plane_gap_m=gap))
    failures=[r for r in rows if r['foot_error']>2e-5 or r['plane_gap_m'] < -1e-6]
    return dict(passed=not failures,samples=len(rows),maximum_foot_error=max(r['foot_error'] for r in rows),
                minimum_plane_gap_m=min(r['plane_gap_m'] for r in rows),first_failure=failures[0] if failures else None,
                failure_count=len(failures),rows=rows,physical_success=False,
                scope='Declared geometric interpolation at four physics fractions per control; not predicted dynamic tracking.')
