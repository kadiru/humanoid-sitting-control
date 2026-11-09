"""Frozen local ordered-drive/contact surrogate; zero physics, no controller admission."""
from pathlib import Path
import datetime, hashlib, json
import numpy as np
from scipy.spatial.transform import Rotation
from low_seat_contact_dynamics_290 import ContactDynamics, skew

D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
O=D/'ordered_contact_screen_317'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()

def matrices(model,q,R):
 fk=model.m.fk(q);axes=[];origins=[]
 for name in model.m.names:
  j=model.joints[name];T=fk[j['body0'][0]]@model.m.frames[name][0]
  axes.append(R@T[:3,'XYZ'.index(j['axis'])]);origins.append(R@T[:3,3])
 def jac(name,p):
  V=np.zeros((3,43));W=V.copy();V[:,:3]=np.eye(3);V[:,3:6]=-skew(p);W[:,3:6]=np.eye(3)
  for j in model.ancestors[name]:V[:,j+6]=np.cross(axes[j],p-origins[j]);W[:,j+6]=axes[j]
  return V,W
 M=np.diag(model.arm);J=[];IW=[]
 for mass,name,com,I in zip(model.inertia['mass'],model.inertia['body_names'],model.inertia['local_com'],model.inertia['inertia_body']):
  T=fk[name];p=R@(T[:3,:3]@com+T[:3,3]);Rb=R@T[:3,:3];Iw=Rb@I@Rb.T;V,W=jac(name,p)
  M+=mass*V.T@V+W.T@Iw@W;J.append((V,W));IW.append(Iw)
 C=[]
 for side in ['left','right']:
  name=side+'_ankle_roll_link';v=model.m.vertices[name];sole=v[abs(v[:,2]-v[:,2].min())<1e-10].mean(0);T=fk[name]
  C.extend(jac(name,R@(T[:3,:3]@sole+T[:3,3])))
 return M,np.concatenate(C),J,IW

def stage_model(M,C,f,v0,error,kp,kd,limits,contact_first):
 # Explicit surrogate: eight authored position stages, four velocity stages.
 # Force-drive coefficients follow pinned public source, NOT installed GPU replay.
 inv=np.linalg.solve(M,np.eye(43));W=inv@C.T;G=C@W
 def contact(v):return v-W@np.linalg.solve(G,C@v)
 h=.005/8;v=v0.copy();x=np.zeros(43);imp=np.zeros(37)
 for _ in range(8):
  v+=h*(inv@f)
  if contact_first:v=contact(v)
  for j in range(37):
   k=j+6;a=h*(h*kp[j]+kd[j]);r=inv[k,k]
   impulse=(h*kp[j]*(error[j]-x[k])-a*v[k])/(1+a*r)
   impulse=np.clip(impulse,-h*limits[j],h*limits[j]);imp[j]+=impulse;v+=inv[:,k]*impulse
  if not contact_first:v=contact(v)
  x+=h*v
 position_v=v.copy()
 for _ in range(4):v=contact(v) # Drive force updates frozen; bilateral contact reprojected.
 return x,v,position_v,imp/.005

def main():
 assert not O.exists();scope=read(D/'ordered_contact_revision_317.json')
 assert read(D/'incoming_audit_v2_305/independent_audit.json')['diagnostic_checks_passed']
 P=D/'incoming_diagnostic_v2_303';s=dict(np.load(P/'substep_states.npz'));k=dict(np.load(P/'implicit_launches.npz'))
 inert=read(D/'contact_dynamics_screen_v3_290/inertial_model.json');meta=read(P/'actuator_metadata.json');props=dict(np.load(P/'backend_properties.npz'))
 model=ContactDynamics(D/'cooked_model_50/model.json',inert,meta)
 coverage=dict(np.load(D/'ankle_impulse_296/arrays.npz'))['coverage']
 files=[Path(__file__).resolve(),Path('scripts/low_seat_contact_dynamics_290.py').resolve(),D/'ordered_contact_revision_317.json',P/'substep_states.npz',P/'implicit_launches.npz',P/'backend_properties.npz',P/'actuator_metadata.json',D/'ankle_impulse_296/arrays.npz',D/'contact_dynamics_screen_v3_290/inertial_model.json',D/'ordered_drive_source_316/source_manifest.json',D/'cooked_model_50/model.json']
 hashes={str(p):sha(p) for p in files};names=meta['joint_names'];pb=meta['body_names'].index('pelvis');com=np.array(inert['local_com'][pb])
 arrays={z:[] for z in ['index','case','available','q','R','v0','target','actual_q_delta','actual_v1','M','C','force','kp','kd','limits','pred_x','pred_v','position_v','drive_effort','observed_contact_velocity']}
 for i in range(0,1000,20):
  for c in range(24):
   q=s['q'][i,c].astype(float);R=Rotation.from_quat(s['link_transforms'][i,c,pb,3:]).as_matrix();omega=s['link_velocities'][i,c,pb,3:].astype(float)
   vb=s['link_velocities'][i,c,pb,:3]-np.cross(omega,R@com);v0=np.r_[vb,omega,s['qd'][i,c].astype(float)]
   R1=Rotation.from_quat(s['link_transforms'][i+1,c,pb,3:]).as_matrix();w1=s['link_velocities'][i+1,c,pb,3:].astype(float)
   v1=np.r_[s['link_velocities'][i+1,c,pb,:3]-np.cross(w1,R1@com),w1,s['qd'][i+1,c].astype(float)]
   M,C,J,IW=matrices(model,q,R);eps=1e-6
   _,_,jp,_=matrices(model,q+eps*v0[6:],Rotation.from_rotvec(eps*omega).as_matrix()@R)
   _,_,jm,_=matrices(model,q-eps*v0[6:],Rotation.from_rotvec(-eps*omega).as_matrix()@R)
   force=np.zeros(43)
   for mass,(V,W),I,(Vp,Wp),(Vm,Wm) in zip(inert['mass'],J,IW,jp,jm):
    a=(Vp-Vm)@v0/(2*eps);alpha=(Wp-Wm)@v0/(2*eps);w=W@v0
    force+=V.T@(mass*(np.array([0.,0.,-9.81])-a))-W.T@(I@alpha+np.cross(w,I@w))
   kp=k['kp'][i,c].astype(float);kd=k['kd'][i,c].astype(float);target=k['target_q'][i,c].astype(float);limits=props['dof_max_forces'][0,c].astype(float)
   assert np.array_equal(target,s['dof_position_targets'][i+1,c]) and not np.any(k['target_qd'][i,c])
   avail=bool(coverage[i,c]);pred=[stage_model(M,C,force,v0,target-q,kp,kd,limits,o) for o in [False,True]] if avail else [(np.full(43,np.nan),np.full(43,np.nan),np.full(43,np.nan),np.full(37,np.nan))]*2
   values=[i,c,avail,q,R,v0,target,s['q'][i+1,c]-q,v1,M,C,force,kp,kd,limits,np.array([p[0] for p in pred]),np.array([p[1] for p in pred]),np.array([p[2] for p in pred]),np.array([p[3] for p in pred]),C@v0]
   for key,value in zip(arrays,values):arrays[key].append(value)
  if i%200==0:print('processed',i,flush=True)
 arrays={k:np.array(v) for k,v in arrays.items()};rows=[]
 for c in range(24):
  for lo,hi in [(8.,11.),(11.,13.)]:
   use=(arrays['case']==c)&(s['t'][arrays['index']]>=lo-1e-12)&(s['t'][arrays['index']]<hi-1e-12);valid=use&arrays['available']
   row=dict(case=c,window=[lo,hi],samples=int(use.sum()),available=int(valid.sum()),unavailable=int((use&~arrays['available']).sum()),orders=[])
   for o in range(2):
    if not valid.any():row['orders'].append(None);continue
    dv=arrays['pred_v'][valid,o]-arrays['actual_v1'][valid];dq=arrays['pred_x'][valid,o,6:]-arrays['actual_q_delta'][valid]
    row['orders'].append(dict(order=['drive_then_contact','contact_then_drive'][o],joint_velocity_rms=float(np.sqrt(np.mean(dv[:,6:]**2))),joint_position_increment_rms=float(np.sqrt(np.mean(dq**2))),roll_velocity_rms=[float(np.sqrt(np.mean(dv[:,6+j]**2))) for j in model.roll],predicted_roll_median=np.median(arrays['pred_v'][valid,o][:,6+np.array(model.roll)],axis=0).tolist(),actual_roll_median=np.median(arrays['actual_v1'][valid][:,6+np.array(model.roll)],axis=0).tolist(),modeled_sole_velocity_max=float(np.max(abs(np.einsum('nij,nj->ni',arrays['C'][valid],arrays['pred_v'][valid,o]))))))
   row['recorded_sole_angular_rms']=[float(np.sqrt(np.mean(arrays['observed_contact_velocity'][use,sl]**2))) for sl in [slice(3,6),slice(9,12)]]
   rows.append(row)
 assert all(sha(p)==h for p,h in hashes.items())
 O.mkdir();np.savez_compressed(O/'arrays.npz',**arrays)
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=scope,rows=rows,records=1200,available=int(arrays['available'].sum()),unavailable=int((~arrays['available']).sum()),source_sha256=hashes,physics_ticks=0,optimizer_updates=0,physical_admission=False,repair_complete=False,model_validated_as_PhysX=False)
 (O/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2))
 print(json.dumps({k:result[k] for k in ['records','available','unavailable','model_validated_as_PhysX']}),flush=True)
if __name__=='__main__':main()
