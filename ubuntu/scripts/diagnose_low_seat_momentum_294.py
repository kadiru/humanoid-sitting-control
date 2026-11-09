"""Offline external impulse / whole-body momentum consistency, no control use."""
from pathlib import Path
import datetime, hashlib, json
import numpy as np
from scipy.spatial.transform import Rotation
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
R=Path('/home/kadir/humanoid-sitting'); P=D/'contact_wrench_diagnostic_259'; O=D/'momentum_balance_294'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not O.exists()
 sources=[P/x for x in ['substep_states.npz','every_tick.npz','contact_body_names.json','static_inertial_model.json','actuator_metadata.json','diagnostic_protocol.json','live_stage_settings.json','contact_recorder_metadata.json']]
 sources += [D/'contact_wrench_audit_260/contact_wrenches.json',D/'contact_wrench_audit_260/independent_audit.json',D/'contact_dynamics_screen_v3_290/inertial_model.json',Path(__file__).resolve()]
 sources += [R/x for x in ['scripts/low_seat_contact_duration_36.py','scripts/run_supported_postures.py','scripts/run_resting_posture.py','sitting_task/src/humanoid_sitting/tasks/full_transition.py','sitting_task/src/humanoid_sitting/tasks/config.py','IsaacLab/source/isaaclab/isaaclab/sim/simulation_cfg.py','IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py','IsaacLab/source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py']]
 hashes={str(p):sha(p) for p in sources}
 scope=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=hashes,scope=__doc__,physics_ticks=0,optimizer_updates=0,controller_installed=False,
  alignment='Post-step contact query at k uses dt=.005. Its returned force is assigned to preceding interval(k-1,k]. No fitted time shift. First query at8s is excluded from impulses.',
  windows_steps=[1,4,20,200],gravity=[0.,0.,-9.81],gravity_source='Installed configuration default, not captured live in259. G1 source linear/angular damping0; no effective-live assertion.',
  availability='All non-foot normal history maxima exactly zero and wrong exactly zero for every enclosing20ms sample. Source history4, lazy_sensor_updateFalse. This is observed normal-contact coverage, not proof against cancelling resultants or unrecorded external forces. Where false, non-foot moments are unavailable.',
  inertia='290 authored rotational tensors plus frozen mass/COM; six authored/frozen mass differences retained. Not certified effective live mass matrix; no fitted inertia or armature correction.')
 assert not (D/'momentum_revision_294.json').exists();(D/'momentum_revision_294.json').write_text(json.dumps(scope,indent=2));O.mkdir()
 assert read(D/'contact_wrench_audit_260/independent_audit.json')['exact_prefix_replay']
 s=dict(np.load(P/'substep_states.npz'));e=dict(np.load(P/'every_tick.npz'));i=read(D/'contact_dynamics_screen_v3_290/inertial_model.json');a=read(P/'actuator_metadata.json');frozen=read(P/'static_inertial_model.json')
 assert i['body_names']==a['body_names']==frozen['body_names']
 for k in ['mass','local_com']:assert np.array_equal(i[k],frozen[k])
 t=s['t'];assert np.array_equal(s['substep'],np.arange(1600,2601));assert np.max(abs(np.diff(t)-.005))<2e-15
 mass=np.array(i['mass']);com_local=np.array(i['local_com']);Ib=np.array(i['inertia_body']);X=s['link_transforms'].astype(float);V=s['link_velocities'].astype(float)
 rot=Rotation.from_quat(X[...,3:].reshape(-1,4)).as_matrix().reshape(1001,24,44,3,3)
 com=X[...,:3]+np.einsum('tcbij,bj->tcbi',rot,com_local);origin=X[0,:,a['body_names'].index('pelvis'),:3].copy()
 pbody=V[...,:3]*mass[None,None,:,None];Ptotal=pbody.sum(2)
 # World rotational momentum R I_body R^T omega; orbital momentum about fixed O.
 wlocal=np.einsum('tcbji,tcbj->tcbi',rot,V[...,3:]);spin=np.einsum('tcbij,bjk,tcbk->tcbi',rot,Ib,wlocal)
 H=(spin+np.cross(com-origin[None,:,None,:],pbody)).sum(2)
 gravity=np.array([0.,0.,-9.81]);gforce=mass.sum()*gravity
 gmoment=np.cross(com-origin[None,:,None,:],mass[None,None,:,None]*gravity).sum(2)
 F=np.zeros((1001,24,3));T=np.zeros_like(F);seen=set()
 for r in read(D/'contact_wrench_audit_260/contact_wrenches.json'):
  k=r['substep']-1600;side=r['side'];key=(k,side);assert key not in seen;seen.add(key)
  ff=np.array(r['force_world_N']);mm=np.array(r['moment_about_link_world_Nm']);b=a['body_names'].index(side+'_ankle_roll_link')
  F[k]+=ff;T[k]+=mm+np.cross(X[k,:,b,:3]-origin,ff)
 assert len(seen)==2002
 # External impulses use recorded point-force returns; gravity angular impulse
 # uses specified trapezoidal COM integration. No fitted alignment/offset.
 impulseF=.005*(F[1:]+gforce);impulseT=.005*(T[1:]+.5*(gmoment[:-1]+gmoment[1:]))
 # Quantify endpoint-vs-trapezoidal gravity sampling as a separate assumption.
 gravity_sampling=.005*.5*(gmoment[1:]-gmoment[:-1])
 names=read(P/'contact_body_names.json');nonfoot=[j for j,n in enumerate(names) if n not in ['left_ankle_roll_link','right_ankle_roll_link']];assert len(nonfoot)==42
 nonfootmax=e['unexpected_body_normal'][:,:,nonfoot].max(-1)
 ti=(s['substep'][1:]+3)//4-1;assert np.allclose(e['t'][ti],np.ceil(s['substep'][1:]/4)*.02,atol=1e-12,rtol=0)
 coverage=(nonfootmax[ti]==0)&(e['wrong'][ti]==0)
 arrays=dict(t=t,P=Ptotal,H=H,origin=origin,foot_force=F,foot_moment=T,gravity_moment=gmoment,impulse_force=impulseF,impulse_moment=impulseT,gravity_sampling_difference=gravity_sampling,coverage=coverage,history_tick_index=ti,nonfoot_max=nonfootmax[ti])
 summaries=[]
 for width in [1,4,20,200]:
  starts=np.arange(0,1000,width);ends=starts+width;duration=width*.005
  impulse_f=impulseF.reshape(-1,width,24,3).sum(1);impulse_m=impulseT.reshape(-1,width,24,3).sum(1)
  dp=Ptotal[ends]-Ptotal[starts];dh=H[ends]-H[starts]
  rf=(dp-impulse_f)/duration;rt=(dh-impulse_m)/duration;available=coverage.reshape(-1,width,24).all(1)
  for k,v in dict(starts=starts,ends=ends,delta_P=dp,delta_H=dh,impulse_F=impulse_f,impulse_T=impulse_m,residual_F=rf,residual_T=rt,available=available).items():arrays[f'w{width}_{k}']=v
  for c in range(24):
   for low,high in [(8.,11.),(11.,13.)]:
    use=(t[starts]>=low-1e-12)&(t[ends]<=high+1e-12);ok=use&available[:,c]
    row=dict(case=c,window=[low,high],interval_ms=duration*1000,intervals=int(use.sum()),available_intervals=int(ok.sum()),incomplete_intervals=int((use&~available[:,c]).sum()))
    row.update(force_residual_mean_N=rf[ok,c].mean(0).tolist() if ok.any() else None,force_residual_RMS_N=float(np.sqrt(np.mean(np.sum(rf[ok,c]**2,axis=-1)))) if ok.any() else None,moment_residual_mean_Nm=rt[ok,c].mean(0).tolist() if ok.any() else None,moment_residual_RMS_Nm=float(np.sqrt(np.mean(np.sum(rt[ok,c]**2,axis=-1)))) if ok.any() else None)
    summaries.append(row)
 limits=[
  'This is a saved-data external impulse consistency check, not a dynamics certificate, new controller, simulation or physical repair.',
  'Contact truth and derived momentum differences are evaluation only. No runtime observation or raw qd is replaced.',
  'External-wrench availability is based on zero saved non-foot normal-force history and zero wrong resultant. Net-force cancellation and unrecorded forces are not independently ruled out. Incomplete intervals have no closed-balance statistic.',
  'Authored rotational tensors and preserved mass/COM are not effective live inertia. Six authored finger masses differ by one float32 step; frozen masses are retained. Generalized armature has no independently specified physical rotor/spatial momentum model here.',
  'Gravity and zero body damping follow installed configuration, not live259 capture. No gravity, tensor, alignment or truth offset is fitted.',
  'Fixed force alignment assumes post-step returned contact force over the preceding5ms. Within-step point timing, solver impulses and solver-internal velocity/pose differences are not resolved.',
  'Internal joint efforts cancel in the chosen physical whole-body model; approximate implicit-PD telemetry is not inserted as an external wrench. Constraint/drive implementation effects remain unmodeled.',
  'No residual threshold is used to admit a controller or relax physical/quiet criteria. Small integrated residuals would not validate local constrained-foot response290; large residuals alone do not identify a cause.'
 ]
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=scope,body_count=44,case_count=24,state_samples=1001,substep_intervals=1000,case_intervals=24000,observed_floor_only_case_intervals=int(coverage.sum()),incomplete_external_wrench_case_intervals=int((~coverage).sum()),total_mass_kg=float(mass.sum()),summaries=summaries,limitations=limits,source_sha256=hashes,physics_ticks=0,optimizer_updates=0,physical_admission=False,repair_complete=False)
 assert all(sha(p)==h for p,h in hashes.items());np.savez_compressed(O/'arrays.npz',**arrays);(O/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2))
 print(json.dumps({k:out[k] for k in ['case_intervals','observed_floor_only_case_intervals','incomplete_external_wrench_case_intervals','total_mass_kg']}),flush=True)
 print(json.dumps([r for r in summaries if r['case'] in [0,1] and r['window']==[11.,13.]]),flush=True)
if __name__=='__main__':main()
