"""Forward kinematics from the loaded USD joint frames; no assumed leg dimensions."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class G1Model:
 def __init__(self,path):
  self.data=json.loads(Path(path).read_text());self.names=self.data['joint_names'];self.joints=self.data['joints'];self.mesh={c['body']:np.array(c['points']) for c in self.data['collisions']}
  self.frames={j['name']:(self.frame(j['p0'],j['q0']),np.linalg.inv(self.frame(j['p1'],j['q1']))) for j in self.joints}
 @staticmethod
 def frame(p,q):
  T=np.eye(4);T[:3,:3]=Rotation.from_quat(q).as_matrix();T[:3,3]=p;return T
 def fk(self,root,q):
  transforms={'pelvis':self.frame(root[:3],root[3:])};pending=self.joints.copy();qs=dict(zip(self.names,q))
  for _ in range(len(pending)+1):
   remaining=[]
   for j in pending:
    if not j['body0'] or j['body0'][0] not in transforms:remaining.append(j);continue
    A,B=self.frames[j['name']];R=np.eye(4)
    if j['name'] in qs:
     axis=np.zeros(3);axis['XYZ'.index(j['axis'])]=qs[j['name']];R[:3,:3]=Rotation.from_rotvec(axis).as_matrix()
    transforms[j['body1'][0]]=transforms[j['body0'][0]]@A@R@B
   if len(remaining)==len(pending):break
   pending=remaining
   if not pending:break
  return transforms
 def vertices(self,T,body):
  return self.mesh[body]@T[body][:3,:3].T+T[body][:3,3]

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--index',type=int,default=0);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
 model=G1Model('../runs/g1_kinematics_geometry_03/model.json');run=Path(a.run);r=json.loads((run/'results.json').read_text());d=np.load(run/'rollouts.npz');dt=float(d['dt']);i=a.index
 rows=[];geometry=[]
 for k in range(len(d['joint_pos'])):
  T=model.fk(d['root_pose'][k,i],d['joint_pos'][k,i]);verts=model.vertices(T,'left_knee_link');rows.append([k*dt,*T['left_knee_link'][:3,3],verts[:,0].min(),verts[:,0].max(),verts[:,2].min(),verts[:,2].max()]);
  if k%25==0:geometry.append((k,verts))
 arr=np.array(rows);np.save(out/'knee_geometry.npy',arr)
 if 'foot_positions' in d:
  error=[]
  for k in range(len(d['joint_pos'])-1):
   T=model.fk(d['root_pose'][k,i],d['joint_pos'][k,i]);error.append(np.linalg.norm(T['left_ankle_roll_link'][:3,3]-d['foot_positions'][k+1,i,0]))
  print('FK measured foot error max/mean',max(error),np.mean(error))
 fig,axs=plt.subplots(2,1,figsize=(10,10))
 for k,verts in geometry:
  if 2<k*dt<7:
   xz=verts[:,[0,2]];h=ConvexHull(xz);points=xz[h.vertices];points=np.vstack((points,points[0]));axs[0].plot(points[:,0],points[:,1],alpha=.4,label=f'{k*dt:.1f}s')
 axs[0].add_patch(plt.Rectangle((-.65,.2),.5,.1,color='steelblue',alpha=.5));axs[0].set_aspect('equal');axs[0].set_xlabel('World X (m)');axs[0].set_ylabel('World Z (m)');axs[0].set_title('Actual lower-leg collider versus original seat');axs[0].legend(ncol=3,fontsize=7)
 axs[1].plot(arr[:,0],arr[:,4],label='Shin minimum X');axs[1].plot(arr[:,0],arr[:,5],label='Shin maximum X');axs[1].axhline(-.15,color='steelblue',label='Seat front edge');axs[1].plot(arr[:,0],d['root_pose'][:,i,0],label='Pelvis X');axs[1].legend();axs[1].set_xlabel('Time (s)');axs[1].grid(alpha=.3);fig.tight_layout();fig.savefig(out/'clearance.png',dpi=150)
 summary={'model':'loaded USD joint frames and collider vertices','collider_body':'left_knee_link','local_collider_bounds':[model.mesh['left_knee_link'].min(axis=0).tolist(),model.mesh['left_knee_link'].max(axis=0).tolist()],'seat_front_x':-.15,'seat_z_range':[.2,.3]};(out/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
