"""Saved model/trajectory optical coverage, no simulation or command generation."""
import sys,json,math
from pathlib import Path
import numpy as np
ROOT=Path('/home/kadir/humanoid-sitting');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'sitting_task/src'))
from humanoid_sitting.shared_frame import level_rotation,BodyModel
from scipy.spatial import ConvexHull
model=BodyModel(ROOT/"runs/g1_kinematics_geometry_04/model.json")
shapes=json.loads((ROOT/"experiments/furniture_perception_repair_20261029_v1/render_colliders.json").read_text())["shapes"]
hulls=[(sh["body"],ConvexHull(sh["vertices"]).equations) for sh in shapes]
def clear_rays(points,q,cam):
 fk=model.fk(q);clear=np.ones(len(points),bool)
 for body,eq in hulls:
  t=fk[body];o=(cam-t[:3,3])@t[:3,:3];v=(points-cam)@t[:3,:3]
  slopes=v@eq[:,:3].T;dist=o@eq[:,:3].T+eq[:,3]
  low=np.zeros(len(points));high=np.full(len(points),1-1e-6);possible=np.ones(len(points),bool)
  for j,b in enumerate(dist):
   a=slopes[:,j];pos=a>1e-12;neg=a < -1e-12
   high[pos]=np.minimum(high[pos],-b/a[pos]);low[neg]=np.maximum(low[neg],-b/a[neg]);possible[(~pos)&(~neg)&(b>0)]=False
  clear &= ~(possible&(low<=high)&(high*np.linalg.norm(points-cam,axis=1)>.05))
 return clear
rows=[]
for folder in sorted((ROOT/'outputs/furniture_camera_coverage_20261106_v1/calibration_demos').iterdir()):
 z=dict(np.load(folder/'frame_0.npz'));R0=level_rotation(z['gravity'].astype(float));basis=z['root_rotation']@R0.T
 # Exactly the initially admitted selected patch; never extend its contact boundary.
 P=z['selected_points']@R0.T;K=z['K'];cam=np.array([-.07,0,.38])
 plan=json.loads((folder/'plan.json').read_text());complete=json.loads((folder/'complete.json').read_text())
 nodes=plan['nodes']+complete['landing'];states=[];intervals=[]
 half=np.arctan((42-.5)/K[1,1])
 for i,n in enumerate(nodes):
  rt=np.array(n['root']);rel=(P-rt[:3,3])@rt[:3,:3]-cam
  elevation=np.arctan2(-rel[:,2],-rel[:,0]);radius=np.hypot(rel[:,0],rel[:,2]);side=np.abs(rel[:,1])
  # In the retained vertical band zcam>=radius*cos(half); horizontal remains within58px.
  good=(radius*np.cos(half)>K[0,0]*side/58)
  pixel_eligible=int(good.sum());good &= clear_rays(rel+cam,np.array(n["q"]),cam)
  e=elevation[good];states.append(dict(node=i,eligible=len(e),pixel_eligible=pixel_eligible,elevation_degrees=np.rad2deg([e.min(),e.max()]).tolist() if len(e) else None))
  if len(e)<30: intervals.append([]);continue
  # Exact event boundaries for >=30 points inside the vertical band.
  edges=sorted(set(np.r_[e-half,e+half]));valid=[]
  for a,b in zip(edges,edges[1:]):
   mid=(a+b)/2
   if np.count_nonzero(abs(e-mid)<=half)>=30:
    if valid and a<=valid[-1][1]+1e-12:valid[-1]=(valid[-1][0],b)
    else:valid.append((a,b))
  intervals.append(valid)
 common=[(0.,math.pi/2)]
 for acceptable in intervals:
  common=[(max(a,c),min(b,d)) for a,b in common for c,d in acceptable if max(a,c)<min(b,d)]
  if not common:break
 if common:
  a,b=max(common,key=lambda pair:pair[1]-pair[0]);angle=(a+b)/2
 else:angle=None
 rows.append(dict(case=folder.name,states=states,common_axis_intervals_deg=np.rad2deg(common).tolist(),chosen_axis_deg=None if angle is None else math.degrees(angle),
  source='All frozen planned approach+landing poses and original observed patch',occlusion_test_pending=False,unexpanded_collision_hull_proxy_not_visual_certificate=True,physical_validation=False))
(HERE/'camera_path_coverage_nearclip.json').write_text(json.dumps(dict(rows=rows,physics_steps=0),indent=2));print(json.dumps([{k:v for k,v in r.items() if k!='states'} for r in rows]))
