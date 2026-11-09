import sys,json
from pathlib import Path
R=Path('/home/kadir/humanoid-sitting')
sys.path[:0]=[str(Path(__file__).resolve().parent),str(R/'experiments/furniture_perception_repair_20261029_v1'),str(R/'sitting_task/src')]
import numpy as np
from types import SimpleNamespace
from placement import render,bind_target
K=np.array([[8.,0.,4.],[0.,8.,4.],[0.,0.,1.]])
T=np.eye(4);R0=np.eye(3)
V=np.array([[-.8,-.8,2.],[.8,-.8,2.],[0.,.8,2.]])
# Far labelled target, then nearer non-target blocker on identical rays.
vertices=np.vstack([V,V*.75]);faces=np.array([[0,1,2],[3,4,5]])
depth,tag=render(vertices,faces,np.array([1,0]),K,T,R0,8,8)
assert np.any(np.isfinite(depth))
assert np.max(abs(depth[np.isfinite(depth)]-1.5))<1e-12
assert np.all(tag[np.isfinite(depth)]==0)
assert np.all(tag[~np.isfinite(depth)]==-1)
depth2,tag2=render(vertices,faces[::-1],np.array([0,1]),K,T,R0,8,8)
assert np.array_equal(tag,tag2) and np.array_equal(depth,depth2,equal_nan=True)
camera=SimpleNamespace(valid=np.ones((7,7,2),bool),ground=np.zeros((7,7,2),bool),N=np.broadcast_to([0.,0.,1.],(7,7,2,3)).copy())
tags=np.ones((8,8),int);tags[0,0]=-1;tags[1,1]=0
camera.ground[3,3,:]=True;camera.N[4,4,:,2]=.5
bind_target(camera,tags)
for r,c,s in np.argwhere(camera.target):
    corners=[(r,c),(r,c+1),(r+1,c)] if s==0 else [(r,c+1),(r+1,c+1),(r+1,c)]
    assert all(tags[a,b]==1 for a,b in corners)
assert not camera.target[3,3].any() and not camera.target[4,4].any()
result=dict(passed=True,tests=['perspective_depth','nearest_surface_labels','order_independence','unknown_not_filled','all_three_target_vertices','ground_and_steep_rejected'],physics_steps=0,actual_robot_data_used=False)
with Path(__file__).with_name('placement_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
