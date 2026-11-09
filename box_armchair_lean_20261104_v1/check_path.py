"""Independent validation against the actual bounded chair face, offline only."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from plan import ROOT,SHAPES,INERTIA,LeanPlan
from humanoid_sitting.shared_frame import BodyModel,apply
from inspect_plan import p,nodes,t,cap,folder
case=json.loads((folder/'manifest.json').read_text())['case']
scene=Path(case['mesh_npz']).parent
z=np.load(scene/'mesh.npz')
patch=z['vertices'][z['faces'][json.loads((scene/'target_region.json').read_text())['triangle_indices']]]
n=np.array(case['target_normal']);center=np.array(case['target_center'])
up=np.array([0.,0.,1.])-n*n[2];up/=np.linalg.norm(up);right=np.cross(up,n)
hull=ConvexHull(np.stack([patch@right,patch@up],-1).reshape(-1,2));eq=hull.equations
# Resolve arbitrary world yaw and translation for this external check only.
qworld=cap['root_rotation']@p.level.T
margin=[];other=[];nongap=[];cop=[]
for rec in nodes:
    root,f=p.pose(np.array(rec['q']))
    cop.append(min(rec['cop']-p.foot_bounds[0],p.foot_bounds[1]-rec['cop']))
    for b,vertices in p.mesh.items():
        points=apply(root@f[b],vertices)@qworld.T+cap['root_position']
        gap=(points-center)@n
        uv=np.stack([points@right,points@up],-1)
        inset=-(uv@eq[:,:2].T+eq[:,2]).max(-1)
        near=gap<.005
        if b in ('torso_link','logo_link'):
            if near.any():
                margin.append(inset[near].min())
        else:
            # Expanded patch catches collisions approaching the upper seam/side.
            possible=inset>-.015
            if possible.any():other.append((gap[possible].min(),b))
cmd=np.array([r['cmd'] for r in nodes]);bounds=np.min([cmd-p.lower,p.upper-cmd])
print('Independent checks:',min(margin,default=None),min(other),min(cop),bounds)
