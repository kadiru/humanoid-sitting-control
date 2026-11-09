"""Offline constitutive compatibility, not inferred internal solver state or control."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
P=D/'incoming_diagnostic_v2_303';O=D/'drive_stage_diagnosis_312'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def stats(a):return dict(median=float(np.median(a)),rms=float(np.sqrt(np.mean(a*a))),max_abs=float(np.max(abs(a))))
def main():
 assert not O.exists()
 assert read(D/'incoming_audit_v2_305/independent_audit.json')['diagnostic_checks_passed']
 assert read(D/'incoming_diagnosis_audit_306/independent_audit.json')['passed']
 files=[Path(__file__).resolve(),P/'substep_states.npz',P/'implicit_launches.npz',P/'backend_properties.npz',P/'actuator_metadata.json',P/'live_stage_settings.json',D/'cooked_model_50/model.json',D/'incoming_audit_v2_305/independent_audit.json',D/'incoming_diagnosis_audit_306/independent_audit.json',D/'ankle_impulse_296/arrays.npz',D/'drive_iteration_context_310/context.json']
 hashes={str(p):sha(p) for p in files}
 s=dict(np.load(P/'substep_states.npz'));k=dict(np.load(P/'implicit_launches.npz'));p=dict(np.load(P/'backend_properties.npz'));meta=read(P/'actuator_metadata.json');g=read(D/'cooked_model_50/model.json');joints={j['name']:j for j in g['joints']}
 names=meta['joint_names'];bodies=meta['body_names'];t=s['t'].astype(float);dt=np.diff(t)
 assert len(t)==1001 and np.max(abs(dt-.005))<1e-12
 assert np.array_equal(k['substep_before']+1,s['substep'][1:])
 assert np.array_equal(k['target_q'],s['dof_position_targets'][1:])
 assert np.array_equal(k['target_qd'],s['dof_velocity_targets'][1:])
 assert np.count_nonzero(k['target_qd'])==0 and np.count_nonzero(k['target_effort'])==0
 assert np.array_equal(k['kp'],np.broadcast_to(p['dof_stiffnesses'][0],k['kp'].shape))
 assert np.array_equal(k['kd'],np.broadcast_to(p['dof_dampings'][0],k['kd'].shape))
 kp=k['kp'].astype(float);kd=k['kd'].astype(float);assert np.all(kd>0)
 target=k['target_q'].astype(float);vtarget=k['target_qd'].astype(float)
 q0=s['q'][:-1].astype(float);q1=s['q'][1:].astype(float);v0=s['qd'][:-1].astype(float);v1=s['qd'][1:].astype(float)
 qmid=(q0+q1)/2;fd=(q1-q0)/dt[:,None,None]
 # All 37 scalar projections taken at CHILD JOINT origin/frame; no moment shift.
 # The free revolute axis is the same local axis on either side of its rotation.
 child=np.array([bodies.index(joints[n]['body1'][0]) for n in names])
 axis=np.array(['XYZ'.index(joints[n]['axis']) for n in names])
 raww=s['link_incoming_joint_force'][1:].astype(float)
 incoming=np.stack([raww[:,:,b,3+a] for b,a in zip(child,axis)],axis=-1)
 arm=p['dof_armatures'][0].astype(float)
 arm_term=(v1-v0)/dt[:,None,None]*arm
 spring=kp*(target-qmid)
 raw_pd=kp*(target-q1)+kd*(vtarget-v1)
 fd_pd=spring+kd*(vtarget-fd)
 start_pd=kp*(target-q0)+kd*(vtarget-v0)
 # A formal algebraic variable conditional on a force-drive constitutive identity.
 # Incoming wrench is total transmitted moment, not isolated drive torque.
 required=vtarget+(spring-incoming)/kd
 required_arm=vtarget+(spring-incoming-arm_term)/kd
 limits=np.array([joints[n]['limits'] for n in names])*np.pi/180
 limit_margin=np.minimum(q1-limits[:,0],limits[:,1]-q1)
 force_limits=p['dof_max_forces'][0].astype(float)
 b=dict(np.load(D/'ankle_impulse_296/arrays.npz'));coverage=b['coverage']
 arrays=dict(t0=t[:-1],t1=t[1:],target=target,q_start=q0,q_end=q1,raw_start=v0,raw_end=v1,position_difference_rate=fd,transmitted_axial=incoming,armature_delta_velocity_term=arm_term,spring_mid=spring,raw_end_PD=raw_pd,start_PD=start_pd,pose_difference_PD=fd_pd,required_conditional_velocity=required,required_with_armature_term=required_arm,limit_margin=limit_margin,coverage=coverage)
 rows=[]
 for c in range(24):
  for j,n in enumerate(names):
   for low,high in [(8.,11.),(11.,13.)]:
    use=(t[:-1]>=low-1e-12)&(t[1:]<=high+1e-12)
    row=dict(case=c,joint=n,window=[low,high],samples=int(use.sum()),observed_floor_only_samples=int(np.sum(use&coverage[:,c])))
    for key in ['raw_end','position_difference_rate','transmitted_axial','spring_mid','required_conditional_velocity','required_with_armature_term','armature_delta_velocity_term']:
     row[key]=stats(arrays[key][use,c,j])
    for key,val in [('raw_end_PD_residual',raw_pd-incoming),('start_PD_residual',start_pd-incoming),('pose_difference_PD_residual',fd_pd-incoming),('required_minus_raw',required-v1),('required_minus_position_difference',required-fd)]:
     row[key]=stats(val[use,c,j])
    row.update(minimum_authored_limit_margin_rad=float(limit_margin[use,c,j].min()),incoming_over_configured_effort_count=int(np.sum(abs(incoming[use,c,j])>force_limits[c,j])),raw_PD_over_configured_effort_count=int(np.sum(abs(raw_pd[use,c,j])>force_limits[c,j])))
    rows.append(row)
 # Three quantities are compared without interpreting either alternate as runtime velocity.
 selected=[r for r in rows if r['case'] in [0,1] and 'ankle_roll' in r['joint'] and r['window']==[11.,13.]]
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,records=int(incoming.size),joint_count=37,case_count=24,intervals=1000,rows=rows,selected=selected,source_sha256=hashes,alignment='Fixed prelaunch targets and post-step k over interval(k-1,k]; midpoint position for descriptive interval expression. No fitted lag, sign, gain or offset.',formulation=dict(raw_end='Kp*(target-q_end)+Kd*(target_velocity-raw_qd_end)',pose_difference='Kp*(target-(q_start+q_end)/2)+Kd*(target_velocity-(q_end-q_start)/dt)',conditional_velocity='target_velocity+[Kp*(target-q_mid)-incoming_axial]/Kd',armature_variant='Conditional velocity additionally subtracts configured_armature*(raw_qd_end-raw_qd_start)/dt / Kd; reported separately, not asserted effective rotor dynamics.'),limitations=['Backend incoming axial moment includes transmitted joint effects and is NOT isolated drive effort. The conditional velocity is an algebraic unknown under an unverified force-drive identity, NOT an observed solver-stage velocity.','Position differences are offline diagnostics only and NEVER replace runtime raw qd or any quiet/support criterion. No alternate quantity is admitted as a controller input.','All24 cases and37 joints retained, including incomplete external-wrench intervals. This joint-local comparison does not require complete external force closure; coverage labels remain descriptive.','Authored force drive and limit metadata are not independently captured effective backend drive-type/internal island iteration counts. Backend Kp/Kd/armature/limits were independently verified in305.','Official TGS documentation distinguishes accumulated position-iteration drive force from velocity iterations. Neither raw endpoint nor finite-difference PD is thereby certified as the actual drive law.','No forward prediction, system identification, force-fit, controller change, physical trajectory, PPO, holdout or physical admission. Existing290 ideal-contact/backward-Euler response is not validated by this comparison.'],physics_ticks=0,optimizer_updates=0,physical_admission=False,repair_complete=False)
 assert all(sha(p)==h for p,h in hashes.items())
 O.mkdir();np.savez_compressed(O/'arrays.npz',**arrays);(O/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False))
 (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2))
 print(json.dumps(dict(records=out['records'],selected=selected)),flush=True)
if __name__=='__main__':main()
