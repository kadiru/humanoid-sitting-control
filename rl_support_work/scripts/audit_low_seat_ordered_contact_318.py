"""Independent ordered surrogate arithmetic; never a physical admission."""
from pathlib import Path
import datetime,json,hashlib
import numpy as np
from scipy.linalg import lu_factor,lu_solve
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'ordered_contact_screen_317';O=D/'ordered_contact_audit_318'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def S(v):
 x,y,z=v;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
def recursive(m,inert,q,R,arm):
 fk=m.fk(q);V={'pelvis':np.c_[np.eye(3),np.zeros((3,40))]};W={'pelvis':np.c_[np.zeros((3,3)),np.eye(3),np.zeros((3,37))]};pending=list(m.joints)
 while pending:
  progress=False
  for j in pending[:]:
   b0,b1=j['body0'][0],j['body1'][0]
   if b0 not in V:continue
   d=R@(fk[b1][:3,3]-fk[b0][:3,3]);v=V[b0]-S(d)@W[b0];w=W[b0].copy()
   if j['name'] in m.names:
    ix=m.names.index(j['name'])+6;T=fk[b0]@m.frames[j['name']][0];axis=R@T[:3,'XYZ'.index(j['axis'])]
    v[:,ix]+=np.cross(axis,R@(fk[b1][:3,3]-T[:3,3]));w[:,ix]+=axis
   V[b1]=v;W[b1]=w;pending.remove(j);progress=True
  assert progress
 J=[];IW=[];M=np.diag(arm)
 for name,mass,com,I in zip(inert['body_names'],inert['mass'],inert['local_com'],inert['inertia_body']):
  Rb=R@fk[name][:3,:3];v=V[name]-S(Rb@com)@W[name];w=W[name];iw=Rb@I@Rb.T
  J.append((v,w));IW.append(iw);M+=mass*v.T@v+w.T@iw@w
 C=[]
 for side in ['left','right']:
  n=side+'_ankle_roll_link';vs=m.vertices[n];p=vs[abs(vs[:,2]-vs[:,2].min())<1e-10].mean(0)
  C.extend([V[n]-S(R@fk[n][:3,:3]@p)@W[n],W[n]])
 return M,np.concatenate(C),J,IW
def replay(M,C,force,v0,error,kp,kd,limits,first):
 mass=lu_factor(M);columns=lu_solve(mass,np.eye(43));A=np.block([[M,-C.T],[C,np.zeros((12,12))]]);saddle=lu_factor(A)
 def project(v):return lu_solve(saddle,np.r_[M@v,np.zeros(12)])[:43]
 h=.000625;v=v0.copy();x=np.zeros(43);total=np.zeros(37)
 for _ in range(8):
  v+=h*lu_solve(mass,force)
  if first:v=project(v)
  for j in range(37):
   z=j+6;r=columns[z,z]
   # Solve impulse = h*(Kp*(error-x-h*(v+r*impulse))-Kd*(v+r*impulse)).
   rhs=h*(kp[j]*(error[j]-x[z]-h*v[z])-kd[j]*v[z]);coef=1+h*(h*kp[j]+kd[j])*r
   impulse=max(-h*limits[j],min(h*limits[j],rhs/coef));total[j]+=impulse;v+=columns[:,z]*impulse
  if not first:v=project(v)
  x+=h*v
 pv=v.copy()
 for _ in range(4):v=project(v)
 return x,v,pv,total/.005
def main():
 assert not O.exists();r=read(P/'results.json');a=dict(np.load(P/'arrays.npz'));s=dict(np.load(D/'incoming_diagnostic_v2_303/substep_states.npz'));k=dict(np.load(D/'incoming_diagnostic_v2_303/implicit_launches.npz'))
 inert=read(D/'contact_dynamics_screen_v3_290/inertial_model.json');meta=read(D/'incoming_diagnostic_v2_303/actuator_metadata.json');m=BodyModel(D/'cooked_model_50/model.json');arm=np.r_[np.zeros(6),meta['armature'][0]]
 coverage=dict(np.load(D/'ankle_impulse_296/arrays.npz'))['coverage'];pb=meta['body_names'].index('pelvis');com=np.array(inert['local_com'][pb]);errors=[];bias_errors=[]
 assert all(sha(p)==h for p,h in r['source_sha256'].items())
 for n,(i,c) in enumerate(zip(a['index'],a['case'])):
  assert np.array_equal(a['q'][n],s['q'][i,c]);assert a['available'][n]==coverage[i,c];assert np.array_equal(a['target'][n],k['target_q'][i,c])
  assert np.array_equal(a['actual_q_delta'][n],s['q'][i+1,c].astype(float)-s['q'][i,c].astype(float))
  R=Rotation.from_quat(s['link_transforms'][i,c,pb,3:]).as_matrix();om=s['link_velocities'][i,c,pb,3:].astype(float)
  v=np.r_[s['link_velocities'][i,c,pb,:3]-np.cross(om,R@com),om,s['qd'][i,c].astype(float)]
  assert np.array_equal(v,a['v0'][n]);M,C,J,IW=recursive(m,inert,a['q'][n],R,arm)
  err=[float(abs(M-a['M'][n]).max()),float(abs(C-a['C'][n]).max())];assert max(err)<1e-10
  eps=1e-6
  _,_,jp,_=recursive(m,inert,a['q'][n]+eps*v[6:],Rotation.from_rotvec(eps*om).as_matrix()@R,arm)
  _,_,jm,_=recursive(m,inert,a['q'][n]-eps*v[6:],Rotation.from_rotvec(-eps*om).as_matrix()@R,arm)
  f=np.zeros(43)
  for mass,(V,W),I,(Vp,Wp),(Vm,Wm) in zip(inert['mass'],J,IW,jp,jm):
   accel=(Vp-Vm)@v/(2*eps);alpha=(Wp-Wm)@v/(2*eps);w=W@v
   f+=V.T@(mass*(np.array([0.,0.,-9.81])-accel))-W.T@(I@alpha+np.cross(w,I@w))
  be=float(abs(f-a['force'][n]).max());assert be<1e-6;bias_errors.append(be)
  if a['available'][n]:
   for order in range(2):
    pred=replay(M,C,a['force'][n],v,a['target'][n]-a['q'][n],a['kp'][n],a['kd'][n],a['limits'][n],bool(order))
    for value,key in zip(pred,['pred_x','pred_v','position_v','drive_effort']):
     e=float(abs(value-a[key][n,order]).max());assert e<1e-8,(n,key,e);err.append(e)
  else:
   assert all(np.isnan(a[key][n]).all() for key in ['pred_x','pred_v','position_v','drive_effort'])
  errors.append(err[:2]+[max(err[2:],default=0.)])
  if n%300==0:print('audited',n,flush=True)
 # Check every reported case/window metric using source selections and scalar sums.
 checks=0
 for row in r['rows']:
  c=row['case'];lo,hi=row['window'];use=(a['case']==c)&(s['t'][a['index']]>=lo-1e-12)&(s['t'][a['index']]<hi-1e-12);valid=use&a['available']
  assert row['available']==int(valid.sum()) and row['samples']==int(use.sum())
  for order,rec in enumerate(row['orders']):
   if not valid.any():assert rec is None;continue
   dv=(a['pred_v'][valid,order,6:]-a['actual_v1'][valid,6:]).ravel();dq=(a['pred_x'][valid,order,6:]-a['actual_q_delta'][valid]).ravel()
   for val,key in [(dv,'joint_velocity_rms'),(dq,'joint_position_increment_rms')]:
    assert abs((sum(float(x)*float(x) for x in val)/len(val))**.5-rec[key])<1e-12;checks+=1
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,records=len(a['index']),available=int(a['available'].sum()),unavailable=int((~a['available']).sum()),maximum_mass_error=float(np.max(errors,axis=0)[0]),maximum_contact_error=float(np.max(errors,axis=0)[1]),maximum_replay_error=float(np.max(errors,axis=0)[2]),maximum_bias_error=max(bias_errors),arithmetic_bounds=dict(mass_contact=1e-10,finite_difference_bias=1e-6,ordered_replay=1e-8),summary_checks=checks,scope='All1200 source selections and recursive mass/contact/bias calculations; all available two-order trajectories independently replayed by direct saddle projection and scalar implicit equation. Audits surrogate arithmetic, NOT installed GPU/internal stages or physical success.',physical_admission=False,model_validated_as_PhysX=False,repair_complete=False,physics_ticks=0,source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),P/'results.json',P/'arrays.npz']})
 O.mkdir();(O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({'independent_audit.json':sha(O/'independent_audit.json')},indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
