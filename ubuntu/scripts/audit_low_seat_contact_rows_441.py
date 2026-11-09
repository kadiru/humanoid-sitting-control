"""Independent saved-primal reconstruction of all27708 node/window rows440."""
from pathlib import Path
import json,numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def audit_rows(a,records,initial,pf):
 ix=list(range(1351,1451))+list(range(1452,1467));assert len(ix)==115
 moving=[0]*16
 from humanoid_sitting.shared_frame import BodyModel
 m=BodyModel(D/'cooked_model_50/model.json');legs=[m.names.index(s+'_'+v+'_joint') for s in ['left','right'] for v in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']];moving=legs+[m.names.index(s+'_'+v+'_joint') for s in ['left','right'] for v in ['shoulder_pitch','elbow_pitch']]
 C=a['command'][ix][:,moving];tau=a['torque'][ix];Q=a['q'][ix];root=a['root'][ix];com=a['com'][ix];f=np.array([records[i]['phase'] for i in ix]);p=np.array([records[i]['orientation_phase'] for i in ix]);gap=np.array([records[i]['pelvis_gap_m'] for i in ix]);dt=a['node_dt'];W=np.zeros((115,8))
 for k,i in enumerate(ix):
  for j in range(8):
   dest=pf['mapping'][k,j]
   if dest>=0:W[k,j]=a['weights'][i,dest]
 X=np.zeros((114,23));source=json.loads((D/'clearance_lift_path_330/results.json').read_text())['dense']['path']
 for i in range(114):
  k=i+1;X[i,:2]=root[k,:2];X[i,2:4]=[f[k],p[k]];X[i,4:16]=Q[k,legs]
  if k<104:X[i,21:23]=(com[k,:2]-np.array(source[k]['com'][:2]))/.1
  X[i,16:21]=initial['null'][i].T@(W[k]-initial['base_weights'][i]-initial['wcom'][i]@X[i,21:23])
 dep=[0,1]+list(range(4,16));lo=initial['command_lo'];hi=initial['command_hi'];eff=initial['effort'];rates=initial['rates'];raw=[]
 for k in range(1,115):raw.extend(np.r_[C[k]-lo,hi-C[k],eff-tau[k],eff+tau[k],W[k],1-W[k]])
 for i in range(103):raw.extend(np.r_[X[i,dep]-initial['lower'][i,dep],initial['upper'][i,dep]-X[i,dep]])
 for k in range(104):raw.extend([f[k+1]-f[k],p[k+1]-p[k]])
 for sign in [-1,1]:
  for k in range(114):raw.extend(rates*dt[k]+sign*(C[k+1]-C[k]))
 for sign in [-1,1]:
  for k in range(114):raw.extend(((hi-C[k+1]) if sign==-1 else (C[k+1]-lo))*dt[k]+sign*.18*(C[k+1]-C[k]))
 raw.extend(gap[1:99]-.012);raw.extend(.012-gap[100:105]);raw.extend(gap[1:105]+1e-8);raw.extend(gap[65:104]-gap[66:105]);raw.extend(.006*dt[99:104]-gap[99:104]+gap[100:105]);raw.append(0.)
 for H,k in zip(initial['support_h'],initial['support_k']):raw.append(-H[:2]@com[k+1,:2]-H[2]-.005)
 assert len(raw)==20284
 ids=np.r_[np.arange(100),99,np.arange(100,115)];times=np.r_[0,np.cumsum(np.r_[dt[:99],.5,dt[99:]])];targets=[]
 for t in times:
  for shift in [0.,.02]:
   t1=min(t+shift,times[-1]);now=np.array([np.interp(t1,times,C[ids,j]) for j in range(16)]);prev=np.array([np.interp(t1-.02,times,C[ids,j]) for j in range(16)]);targets.append(10*now-9*prev)
 target=np.array(targets);raw=np.r_[raw,np.c_[hi-target,target-lo].ravel()]
 keep=np.load(D/'lattice_path_430/final.npz')['keep'].astype(bool);tighten=np.zeros(27708);eps=8*np.finfo(np.float32).eps
 for k in range(114):
  for j in range(64,80):
   row=k*80+j
   if keep[row]:tighten[row]=eps
 tighten[12004:12212]=eps
 for start in [12212,14036]:
  for k in range(114):tighten[start+k*16:start+(k+1)*16]=dt[k]/.02*4*initial['ulp']
 for start in [15860,17684]:
  for k in range(114):tighten[start+k*16:start+(k+1)*16]=dt[k]*32*initial['ulp']
 tighten[19760:20284]=eps;tighten[20284:]=np.tile(np.r_[32*initial['ulp'],32*initial['ulp']],232)
 V=np.c_[X[:,2:4],X[:,21:23],X[:,16:21]].ravel();shape=initial['shape_free'];bounds=initial['bounds'][shape];coordinate=np.r_[V[shape]-bounds[:,0],bounds[:,1]-V[shape]]
 errors=dict(original=float(abs(raw-a['node_raw']).max()),prior=float(abs(raw-tighten-a['node_prior']).max()),coordinates=float(abs(X-a['node_X']).max()),coordinate_bounds=float(abs(coordinate-a['node_coordinate_bounds']).max()),hover=abs((gap[99]-.012)/.012-float(a['node_hover'][0])))
 assert max(errors.values())<2e-12,errors
 return dict(rows=27708,coordinate_bound_rows=len(coordinate),maximum_errors=errors)
