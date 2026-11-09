"""Independent component/quaternion replay of incoming-wrench frame and statistics."""
from pathlib import Path
import datetime,json,hashlib
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'incoming_diagnostic_v2_303';A=D/'incoming_diagnosis_306';O=D/'incoming_diagnosis_audit_306'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rotate(q,v):
 q=np.asarray(q,float);q=q/np.linalg.norm(q,axis=-1,keepdims=True);u=q[...,:3];w=q[...,3,None]
 return v+2*np.cross(u,np.cross(u,v)+w*v)
def main():
 O.mkdir(exist_ok=False);r=read(A/'results.json');assert all(sha(p)==h for p,h in r['source_sha256'].items())
 assert read(D/'incoming_audit_v2_305/independent_audit.json')['exact_prefix_replay']
 a=dict(np.load(A/'arrays.npz'));s=dict(np.load(P/'substep_states.npz'));b=dict(np.load(D/'ankle_impulse_296/arrays.npz'));meta=read(P/'actuator_metadata.json');g=read(D/'cooked_model_50/model.json');jm={j['name']:j for j in g['joints']};maximum=0.;checks=0
 def check(x,y):
  nonlocal maximum,checks
  assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all();e=float(abs(x-y).max());assert e<1e-10,e;maximum=max(maximum,e);checks+=1
 for side in ['left','right']:
  j=jm[side+'_ankle_roll_joint'];bi=meta['body_names'].index(j['body1'][0]);x=s['link_transforms'][:,:,bi].astype(float);w=s['link_incoming_joint_force'][:,:,bi].astype(float)
  force=rotate(x[...,3:],rotate(np.array(j['q1']),w[...,:3]));moment=rotate(x[...,3:],rotate(np.array(j['q1']),w[...,3:]));joint=x[...,:3]+rotate(x[...,3:],np.broadcast_to(np.array(j['p1']),x[...,:3].shape));mid=.5*b[side+'_joint'][1:]+.5*b[side+'_joint'][:-1]
  # Direct component expansion of the translation cross product.
  d=joint[1:]-mid;f=force[1:];shift=moment[1:]+np.stack([d[...,1]*f[...,2]-d[...,2]*f[...,1],d[...,2]*f[...,0]-d[...,0]*f[...,2],d[...,0]*f[...,1]-d[...,1]*f[...,0]],-1)
  axis=b[side+'_axis_mid'];axial=axis[...,0]*shift[...,0]+axis[...,1]*shift[...,1]+axis[...,2]*shift[...,2]
  values=dict(incoming_force=force,incoming_moment=moment,child_joint_origin=joint,incoming_moment_mid=shift,incoming_axial=axial,boundary_axial=b[side+'_axial'],telemetry=b[side+'_telemetry'],force_residual=force[1:]-b[side+'_boundary_force'],moment_residual=shift-b[side+'_boundary_moment'])
  for k,v in values.items():check(v,a[side+'_'+k])
  statvals=dict(incoming_axial_Nm=axial,inferred_boundary_Nm=b[side+'_axial'],approximate_PD_Nm=b[side+'_telemetry'],incoming_minus_boundary_Nm=axial-b[side+'_axial'],PD_minus_incoming_Nm=b[side+'_telemetry']-axial,force_residual_norm_N=np.sqrt(np.sum(values['force_residual']**2,axis=-1)),moment_residual_norm_Nm=np.sqrt(np.sum(values['moment_residual']**2,axis=-1)))
  for row in [z for z in r['rows'] if z['side']==side]:
   c=row['case'];lo,hi=row['window'];use=(s['t'][:-1]>=lo-1e-12)&(s['t'][1:]<=hi+1e-12)&b['coverage'][:,c];assert int(use.sum())==row['available_intervals']
   for k,v in statvals.items():
    if not use.any():assert row[k] is None;continue
    z=np.sort(v[use,c]);n=len(z);med=.5*(z[(n-1)//2]+z[n//2]);got=np.array([med,np.sqrt(sum(float(y*y) for y in z)/n),max(abs(z))]);want=np.array([row[k][x] for x in ['median','RMS','maximum_absolute']]);check(got,want)
 prop=dict(np.load(P/'backend_properties.npz'))
 for ji,row in enumerate(r['properties']):
  assert row['joint']==meta['joint_names'][ji]
  for k,n in [('stiffness','dof_stiffnesses'),('damping','dof_dampings'),('armature','dof_armatures'),('effort_limit','dof_max_forces')]:assert row[k]==float(prop[n][0,0,ji])
  assert np.array_equal(row['friction'],prop['dof_friction_properties'][0,0,ji])
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,foot_case_intervals=48000,case_windows=len(r['rows']),all37_properties_checked=True,maximum_arithmetic_error=maximum,checks=checks,tolerance=1e-10,physical_admission=False,repair_complete=False,scope='Independent manual quaternion rotation, translation cross components, projection and sorted-median/scalar-sum statistics. Arithmetic only; not isolated motor effort or physical repair.',source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),A/'results.json',A/'arrays.npz',P/'substep_states.npz',P/'backend_properties.npz']})
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
