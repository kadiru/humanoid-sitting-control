"""Independent recursive-twist inertia and saddle-system audit of offline290."""
from pathlib import Path
import datetime,json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdPhysics
from humanoid_sitting.shared_frame import BodyModel,level_rotation,apply
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'contact_dynamics_screen_v3_290';O=D/'contact_dynamics_audit_v2_291'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def crossmat(v):
 return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0.]])
def recurrence(m,inert,q,g):
 R=level_rotation(g);fk=m.fk(q);vj={'pelvis':np.c_[np.eye(3),np.zeros((3,40))]};wj={'pelvis':np.c_[np.zeros((3,3)),np.eye(3),np.zeros((3,37))]};pending=list(m.joints)
 while pending:
  progress=False
  for j in pending[:]:
   b0,b1=j['body0'][0],j['body1'][0]
   if b0 not in vj:continue
   d=R@(fk[b1][:3,3]-fk[b0][:3,3]);v=vj[b0]-crossmat(d)@wj[b0];w=wj[b0].copy()
   if j['name'] in m.names:
    k=m.names.index(j['name'])+6;frame=fk[b0]@m.frames[j['name']][0];axis=R@frame[:3,'XYZ'.index(j['axis'])];v[:,k]+=np.cross(axis,R@(fk[b1][:3,3]-frame[:3,3]));w[:,k]+=axis
   vj[b1]=v;wj[b1]=w;pending.remove(j);progress=True
  assert progress
 J=[];IW=[]
 for n,com,I in zip(inert['body_names'],inert['local_com'],inert['inertia_body']):
  Rb=R@fk[n][:3,:3];J.append((vj[n]-crossmat(Rb@com)@wj[n],wj[n]));IW.append(Rb@I@Rb.T)
 C=[]
 for s in ['left','right']:
  n=s+'_ankle_roll_link';verts=m.vertices[n];bottom=verts[abs(verts[:,2]-verts[:,2].min())<1e-10].mean(0);C.extend([vj[n]-crossmat(R@fk[n][:3,:3]@bottom)@wj[n],wj[n]])
 return J,IW,np.concatenate(C)
def main():
 O.mkdir(exist_ok=False);r=read(P/'results.json');a=np.load(P/'arrays.npz');inert=read(P/'inertial_model.json');act=read(D/'drive_diagnostic_246/actuator_metadata.json');m=BodyModel(D/'cooked_model_50/model.json');assert len(r['rows'])==3800
 assert all(sha(p)==h for p,h in r['source_sha256'].items());base=read(D/'drive_diagnostic_246/static_inertial_model.json');assert inert['mass']==base['mass'] and inert['local_com']==base['local_com']
 st=Usd.Stage.Open(str(D/'asset_contact_47/g1.usd'));mass_mismatches=[]
 for i,n in enumerate(inert['body_names']):
  api=UsdPhysics.MassAPI(st.GetPrimAtPath('/g1/'+n));d=np.array(api.GetDiagonalInertiaAttr().Get(),dtype=float);q=api.GetPrincipalAxesAttr().Get();z=np.r_[float(q.GetReal()),np.array(q.GetImaginary())].astype(float)
  if not np.any(z):I=np.diag(d);assert np.all(d==d[0])
  else:
   w,x,y,z=z/np.linalg.norm(z);R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]);I=(R*d)@R.T
  assert abs(I-inert['inertia_body'][i]).max()<1e-15
  eq=float(api.GetMassAttr().Get())==base['mass'][i];assert eq==inert['authored_properties'][i]['mass_exact']
  if not eq:mass_mismatches.append(n)
 assert len(mass_mismatches)==6 and not inert['authored_mass_exact_frozen']
 raw={}
 for run,folder in [(234,'centered_probe_234'),(273,'roll_feedback_probe_273')]:raw[run]={(o['case'],o['tick']):o for o in read(D/folder/'candidate/touchdown_raw_176.json')}
 K=np.diag(np.r_[np.zeros(6),act['stiffness'][0]]);B=np.diag(np.r_[np.zeros(6),act['damping'][0]]);arm=np.r_[np.zeros(6),act['armature'][0]];U=np.zeros((43,2));U[np.array(r['roll_ids'])+6,np.arange(2)]=1;h=.005;errors=[];fd_errors=[];fd_records=0
 for k,row in enumerate(r['rows']):
  obs=raw[row['run']][row['case'],row['tick']]
  for key,src in [('q','q'),('gravity','gravity'),('raw_qd','qd'),('gyro','gyro')]:assert np.array_equal(a[key][k],obs[src])
  J,IW,C=recurrence(m,inert,a['q'][k],a['gravity'][k]);M=np.diag(arm)
  for mass,(V,W),I in zip(inert['mass'],J,IW):M+=mass*V.T@V+W.T@I@W
  E=M+h*B+h*h*K;A=np.block([[E,-C.T],[C,np.zeros((12,12))]]);x=np.zeros((43,2));v=x.copy();maxres=0
  for _ in range(4):
   rhs=np.r_[M@v+h*K@(U-x),np.zeros((12,2))];s=np.linalg.solve(A,rhs);maxres=max(maxres,float(abs(A@s-rhs).max()));v=s[:43];x+=h*v
  e=np.array([abs(M-a['M'][k]).max(),abs(C-a['C'][k]).max(),abs(v-a['dv'][k]).max(),abs(x-a['dq'][k]).max(),maxres,abs(C@v).max()]);assert np.all(e<1e-9),(k,e);errors.append(e)
  bound=.00075*abs(v[6+np.array(r['roll_ids'])]).sum(1);assert abs(bound-row['roll_dv_box_bound']).max()<1e-12
  #16 declared pose fixtures, every37 joint columns and all44 bodies.
  if row['tick'] in [100,450,600,800]:
   R=level_rotation(a['gravity'][k]);eps=1e-6;err=0.
   for j in range(37):
    qp=a['q'][k].copy();qm=qp.copy();qp[j]+=eps;qm[j]-=eps;fp=m.fk(qp);fm=m.fk(qm)
    for b,n in enumerate(inert['body_names']):
     V=R@(apply(fp[n],inert['local_com'][b])-apply(fm[n],inert['local_com'][b]))/(2*eps);W=R@Rotation.from_matrix(fp[n][:3,:3]@fm[n][:3,:3].T).as_rotvec()/(2*eps)
     err=max(err,float(abs(V-J[b][0][:,j+6]).max()),float(abs(W-J[b][1][:,j+6]).max()))
   assert err<2e-8;fd_errors.append(err);fd_records+=1
  if k%1000==0:print('audited',k,flush=True)
 assert fd_records==16
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,records=3800,body_tensors=44,authored_mass_exact_frozen=False,mass_mismatch_bodies=mass_mismatches,maximum_errors=dict(zip(['mass_matrix','contact_matrix','velocity_response','position_response','saddle_residual','contact_response'],np.max(errors,axis=0).tolist())),finite_difference_records=fd_records,finite_difference_joint_body_columns=16*37*44,maximum_finite_difference_error=max(fd_errors),physics_ticks=0,optimizer_updates=0,physical_admission=False,repair_complete=False,scope='Independent recursive body twists/energy assembly, direct55x55 saddle solve vs Schur elimination for all3800 records;16 stratified FK finite-difference fixtures. Confirms specified approximate model arithmetic, not actual PhysX response or mass equality.',source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),P/'results.json',P/'arrays.npz',P/'inertial_model.json']})
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));np.savez_compressed(O/'errors.npz',errors=errors,finite_difference_errors=fd_errors);(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
