"""Independent finite-difference full6DoF wrench and sensitivity audit.
Imports no proposed observer. Finite changes of foot pose and gravitational
potential reproduce rows of the saved wrench map and gravity torque.
"""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,level_rotation,apply
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'spatial_wrench_diagnosis_255';O=D/'spatial_wrench_audit_256'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);r=read(P/'results.json');x=np.load(P/'matrices.npz');assert len(r['rows'])==2400
 assert all(sha(p)==h for p,h in r['source_sha256'].items())
 m=BodyModel(D/'cooked_model_50/model.json');it=read(D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json');names=m.names;eps=1e-6;errs=[];solve=[];sensitivity=[];synthetic=[]
 for k,row in enumerate(r['rows']):
  side=row['side'];ids=[names.index(side+'_'+j+'_joint') for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']];q=x['q'][k];R=level_rotation(x['gravity'][k]);foot=side+'_ankle_roll_link';Af=[];gf=[]
  for j in ids:
   qp=q.copy();qm=q.copy();qp[j]+=eps;qm[j]-=eps;fp=m.fk(qp);fm=m.fk(qm)
   v=R@(fp[foot][:3,3]-fm[foot][:3,3])/(2*eps);w=R@Rotation.from_matrix(fp[foot][:3,:3]@fm[foot][:3,:3].T).as_rotvec()/(2*eps);Af.append(np.r_[v,w])
   # Use all bodies independently rather than observer descendant selection.
   dg=sum(mass*9.81*(R@(apply(fp[b],com)-apply(fm[b],com)))[2]/(2*eps) for b,mass,com in zip(it['body_names'],it['mass'],it['local_com']));gf.append(dg)
  Af=np.array(Af);gf=np.array(gf);errs.append([float(abs(Af-x['A'][k]).max()),float(abs(gf-x['gravity_torque'][k]).max())])
  assert errs[-1][0]<2e-8 and errs[-1][1]<2e-6,(k,errs[-1])
  # Independent least squares uses saved exact map, then unit sign probes
  # achieve the componentwise infinity-norm sensitivity bound.
  A=x['A'][k];rhs=x['gravity_torque'][k]-x['torque'][k,ids];w=np.linalg.lstsq(A,rhs,rcond=None)[0];solve.append(float(abs(w-x['wrench'][k]).max()));assert solve[-1]<1e-8
  inv=np.linalg.solve(A,np.eye(6));sens=np.sum(abs(inv),axis=1);sensitivity.append(float(abs(sens-row['per_1Nm_each_joint_linf_wrench_sensitivity']).max()));assert sensitivity[-1]<1e-10
  for i in [1,2,3]:
   dw=np.linalg.solve(A,np.sign(inv[i]));assert abs(abs(dw[i])-sens[i])<1e-10
  # Known synthetic wrench checks all six axes without using any truth force.
  known=np.array([3.,-2.,100.,.5,-.8,.1]);tau=x['gravity_torque'][k]-A@known;recon=np.linalg.solve(A,x['gravity_torque'][k]-tau);synthetic.append(float(abs(recon-known).max()));assert synthetic[-1]<1e-10
  assert np.linalg.matrix_rank(A)==row['rank']==6
  if k%200==0:print('audited',k,flush=True)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,rows=len(errs),finite_difference_step_rad=eps,finite_difference_J_max=float(np.max(errs,axis=0)[0]),finite_difference_gravity_max_Nm=float(np.max(errs,axis=0)[1]),independent_lstsq_max=float(max(solve)),sensitivity_max=float(max(sensitivity)),synthetic_wrench_max=float(max(synthetic)),physics_ticks=0,optimizer_updates=0,physical_admission=False,interpretation='Arithmetic/observability checks only. Quasi-static wrench remains limited by approximate PD torque, ignored dynamics/other contacts. No actual contact moment labels available, no feedback admission or threshold change.',source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),P/'results.json',P/'matrices.npz']})
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));np.savez_compressed(O/'finite_difference_errors.npz',errors=errs);(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
