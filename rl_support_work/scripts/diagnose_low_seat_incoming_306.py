"""Recorded transmitted incoming joint wrench versus audited foot impulse; no control."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'incoming_diagnostic_303';O=D/'incoming_diagnosis_306'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def stats(x):return dict(median=float(np.median(x)),RMS=float(np.sqrt(np.mean(x*x))),maximum_absolute=float(abs(x).max()))
def main():
 audit=read(D/'incoming_audit_305/independent_audit.json');assert audit['diagnostic_checks_passed'] and audit['exact_prefix_replay']
 assert read(D/'ankle_impulse_audit_297/independent_audit.json')['passed'];assert not O.exists()
 s=dict(np.load(P/'substep_states.npz'));old=dict(np.load(D/'contact_wrench_diagnostic_259/substep_states.npz'));assert all(np.array_equal(s[k],old[k]) for k in ['t','q','qd','command','torque','link_transforms','link_velocities'])
 b=dict(np.load(D/'ankle_impulse_296/arrays.npz'));meta=read(P/'actuator_metadata.json');g=read(D/'cooked_model_50/model.json');jm={j['name']:j for j in g['joints']};props=dict(np.load(P/'backend_properties.npz'));X=s['link_transforms'].astype(float);W=s['link_incoming_joint_force'].astype(float)
 files=[Path(__file__).resolve(),P/'substep_states.npz',P/'actuator_metadata.json',P/'backend_properties.npz',P/'backend_ordering.json',P/'live_stage_settings.json',D/'incoming_audit_305/independent_audit.json',D/'ankle_impulse_296/arrays.npz',D/'ankle_impulse_audit_297/independent_audit.json',D/'cooked_model_50/model.json'];hashes={str(p):sha(p) for p in files};O.mkdir();arr=dict(t=s['t'][1:],coverage=b['coverage']);rows=[];properties=[]
 for ji,name in enumerate(meta['joint_names']):
  properties.append(dict(joint=name,stiffness=float(props['dof_stiffnesses'][0,0,ji]),damping=float(props['dof_dampings'][0,0,ji]),armature=float(props['dof_armatures'][0,0,ji]),effort_limit=float(props['dof_max_forces'][0,0,ji]),friction=props['dof_friction_properties'][0,0,ji].tolist()))
 for side in ['left','right']:
  j=jm[side+'_ankle_roll_joint'];bi=meta['body_names'].index(j['body1'][0]);R=Rotation.from_quat(X[:,:,bi,3:].reshape(-1,4)).as_matrix().reshape(1001,24,3,3)
  # Documented CHILD JOINT frame. q1 rotates that frame into child link;
  # recorded link quaternion rotates into world. No fitted sign/axis/time lag.
  RJ=R@Rotation.from_quat(j['q1']).as_matrix();force=np.einsum('tcij,tcj->tci',RJ,W[:,:,bi,:3]);moment=np.einsum('tcij,tcj->tci',RJ,W[:,:,bi,3:]);joint=X[:,:,bi,:3]+np.einsum('tcij,j->tci',R,j['p1']);mid=(b[side+'_joint'][1:]+b[side+'_joint'][:-1])/2
  shifted=moment[1:]+np.cross(joint[1:]-mid,force[1:]);axial=np.sum(shifted*b[side+'_axis_mid'],axis=-1);required=b[side+'_axial'];telemetry=b[side+'_telemetry'];df=force[1:]-b[side+'_boundary_force'];dm=shifted-b[side+'_boundary_moment']
  for k,v in dict(incoming_force=force,incoming_moment=moment,child_joint_origin=joint,incoming_moment_mid=shifted,incoming_axial=axial,boundary_axial=required,telemetry=telemetry,force_residual=df,moment_residual=dm).items():arr[side+'_'+k]=v
  for c in range(24):
   for low,high in [(8.,11.),(11.,13.)]:
    use=(s['t'][:-1]>=low-1e-12)&(s['t'][1:]<=high+1e-12);ok=use&b['coverage'][:,c]
    row=dict(case=c,side=side,window=[low,high],intervals=int(use.sum()),available_intervals=int(ok.sum()))
    for n,v in [('incoming_axial_Nm',axial),('inferred_boundary_Nm',required),('approximate_PD_Nm',telemetry),('incoming_minus_boundary_Nm',axial-required),('PD_minus_incoming_Nm',telemetry-axial),('force_residual_norm_N',np.linalg.norm(df,axis=-1)),('moment_residual_norm_Nm',np.linalg.norm(dm,axis=-1))]:row[n]=stats(v[ok,c]) if ok.any() else None
    rows.append(row)
 limits=['Incoming joint wrench is documented in child joint frame and describes total transmitted force/couple, not isolated motor drive effort. It is evaluation data only.', 'Post-step k applies to interval(k-1,k]; recorded child-joint point is shifted to the audited midpoint origin and projected on its mean axis. No fitted sign, offset, time shift or smoothing.', 'Audited296 boundary inference retains finite5ms impulse, authored inertia/frozen COM and configured gravity/armature limitations. New source states must exactly equal259 before reusing it.', 'All24cases/bothfeet included; closed summaries retain294 observed floor-only coverage, with missing external-moment intervals unavailable. No sensor substitution, controller/criteria/material change or effort-result rewrite.', 'Backend gain/friction readbacks are compared with configured buffers separately. Equality does not establish the internal solver drive law, effective iterations, or continuous motion.', 'No physical repair, full20s success, new controller, PPO or fresh holdout is admitted.']
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,foot_case_intervals=48000,raw_states_exactly_match259=True,rows=rows,properties=properties,backend_checks=audit['incoming_checks'],limitations=limits,source_sha256=hashes,physics_ticks=0,optimizer_updates=0,physical_admission=False,repair_complete=False)
 assert all(sha(p)==h for p,h in hashes.items());np.savez_compressed(O/'arrays.npz',**arr);(O/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2));print(json.dumps([r for r in rows if r['case'] in [0,1] and r['window']==[11.,13.]]),flush=True)
if __name__=='__main__':main()
