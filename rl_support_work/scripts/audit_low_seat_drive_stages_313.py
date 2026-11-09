"""Independent raw/frame/constitutive-identity replay; no controller or physics."""
from pathlib import Path
import hashlib,json,datetime,math
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'incoming_diagnostic_v2_303';S=D/'drive_stage_diagnosis_312';O=D/'drive_stage_audit_313'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def rotate(q,v):
 q=np.array(q,dtype=float);q=q/np.linalg.norm(q,axis=-1,keepdims=True)
 u=q[...,:3];a=2*np.cross(u,v)
 return v+q[...,3,None]*a+np.cross(u,a)
def main():
 assert not O.exists()
 result=read(S/'results.json');assert result['records']==888000 and not result['physical_admission']
 assert all(sha(p)==h for p,h in result['source_sha256'].items())
 z=dict(np.load(P/'substep_states.npz'));k=dict(np.load(P/'implicit_launches.npz'));p=dict(np.load(P/'backend_properties.npz'));saved=dict(np.load(S/'arrays.npz'));meta=read(P/'actuator_metadata.json');model=read(D/'cooked_model_50/model.json');joints={j['name']:j for j in model['joints']};names=meta['joint_names'];bn=meta['body_names']
 t=z['t'].astype(float);dt=t[1:]-t[:-1];T=k['target_q'].astype(float);VT=k['target_qd'].astype(float);Q=z['q'].astype(float);V=z['qd'].astype(float);KP=p['dof_stiffnesses'][0].astype(float);KD=p['dof_dampings'][0].astype(float);A=p['dof_armatures'][0].astype(float)
 n=0;maximum=0.;errors={}
 def ck(name,a,b,tol=1e-10):
  nonlocal n,maximum
  a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(name,a.shape,b.shape);assert np.isfinite(a).all() and np.isfinite(b).all(),name
  e=float(np.max(abs(a-b)));assert e<=tol,(name,e,tol);maximum=max(maximum,e);errors[name]=max(errors.get(name,0),e);n+=1
 assert np.array_equal(k['substep_before']+1,z['substep'][1:])
 assert np.array_equal(k['target_q'],z['dof_position_targets'][1:]) and np.array_equal(k['target_qd'],z['dof_velocity_targets'][1:])
 assert all(np.array_equal(p[key],np.broadcast_to(p[key][0],p[key].shape)) for key in ['dof_stiffnesses','dof_dampings','dof_armatures','dof_max_forces','dof_friction_properties'])
 incoming=np.empty((1000,24,37));margins=np.empty_like(incoming)
 for ji,name in enumerate(names):
  j=joints[name];bi=bn.index(j['body1'][0]);ax=np.eye(3)['XYZ'.index(j['axis'])]
  # Independent world-frame rotation/dot rather than selecting local component.
  quat=z['link_transforms'][1:,:,bi,3:];localmoment=z['link_incoming_joint_force'][1:,:,bi,3:].astype(float)
  wm=rotate(quat,rotate(j['q1'],localmoment));wa=rotate(quat,rotate(j['q1'],ax))
  incoming[:,:,ji]=np.einsum('tci,tci->tc',wm,wa)
  lower,upper=[float(v)*math.pi/180 for v in j['limits']]
  margins[:,:,ji]=np.minimum(Q[1:,:,ji]-lower,upper-Q[1:,:,ji])
 FD=np.diff(Q,axis=0)/dt[:,None,None];mid=.5*Q[:-1]+.5*Q[1:];spring=(T-mid)*KP
 req=VT+(spring-incoming)/KD;arm=A*np.diff(V,axis=0)/dt[:,None,None]
 calc=dict(t0=t[:-1],t1=t[1:],target=T,q_start=Q[:-1],q_end=Q[1:],raw_start=V[:-1],raw_end=V[1:],position_difference_rate=FD,transmitted_axial=incoming,armature_delta_velocity_term=arm,spring_mid=spring,raw_end_PD=(T-Q[1:])*KP+(VT-V[1:])*KD,start_PD=(T-Q[:-1])*KP+(VT-V[:-1])*KD,pose_difference_PD=(T-mid)*KP+(VT-FD)*KD,required_conditional_velocity=req,required_with_armature_term=req-arm/KD,limit_margin=margins)
 for key,value in calc.items():ck(key,saved[key],value)
 # Verify the defining equations independently, including the separate armature variant.
 ck('conditional_identity',spring+(VT-saved['required_conditional_velocity'])*KD,incoming)
 ck('conditional_armature_identity',spring+(VT-saved['required_with_armature_term'])*KD,incoming+arm)
 cov=dict(np.load(D/'ankle_impulse_296/arrays.npz'))['coverage'];assert np.array_equal(cov,saved['coverage'])
 def st(a):
  a=sorted(float(v) for v in a);m=len(a);return {'median':(a[(m-1)//2]+a[m//2])/2,'rms':math.sqrt(math.fsum(v*v for v in a)/m),'max_abs':max(map(abs,a))}
 columns=['raw_end','position_difference_rate','transmitted_axial','spring_mid','required_conditional_velocity','required_with_armature_term','armature_delta_velocity_term']
 calc.update(raw_end_PD_residual=calc['raw_end_PD']-incoming,start_PD_residual=calc['start_PD']-incoming,pose_difference_PD_residual=calc['pose_difference_PD']-incoming,required_minus_raw=req-V[1:],required_minus_position_difference=req-FD)
 for row in result['rows']:
  c=row['case'];ji=names.index(row['joint']);lo,hi=row['window'];use=(t[:-1]>=lo-1e-12)&(t[1:]<=hi+1e-12)
  assert row['samples']==sum(use) and row['observed_floor_only_samples']==sum(use&cov[:,c])
  for key in columns+['raw_end_PD_residual','start_PD_residual','pose_difference_PD_residual','required_minus_raw','required_minus_position_difference']:
   other=st(calc[key][use,c,ji])
   for stat,value in other.items():ck('statistics',row[key][stat],value)
  ck('limit_margin',row['minimum_authored_limit_margin_rad'],min(margins[use,c,ji]))
  lim=p['dof_max_forces'][0,c,ji]
  assert row['incoming_over_configured_effort_count']==sum(abs(incoming[use,c,ji])>lim)
  assert row['raw_PD_over_configured_effort_count']==sum(abs(calc['raw_end_PD'][use,c,ji])>lim)
 assert len(result['rows'])==1776
 assert all(sha(p)==h for p,h in result['source_sha256'].items())
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,records=888000,case_joint_windows=1776,joints=37,checks=n,max_error=maximum,error_groups=errors,source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),S/'results.json',S/'arrays.npz']},physics_ticks=0,physical_admission=False,repair_complete=False,limitations=result['limitations'])
 O.mkdir();(O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2));print(json.dumps({k:out[k] for k in ['passed','records','case_joint_windows','checks','max_error','physical_admission']}),flush=True)
if __name__=='__main__':main()
