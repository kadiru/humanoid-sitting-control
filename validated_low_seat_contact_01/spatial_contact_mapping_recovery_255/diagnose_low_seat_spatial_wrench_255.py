"""All24 saved diagnostic246 cases,8–12.9s; zero new physics or runtime edits."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,level_rotation
from low_seat_spatial_wrench_255 import SpatialWrench
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'drive_diagnostic_246';O=D/'spatial_wrench_diagnosis_255'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);assert read(D/'drive_audit_247/independent_audit.json')['exact_prefix_replay']
 m=BodyModel(D/'cooked_model_50/model.json');ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';obs=SpatialWrench(m,read(ip));tr=np.load(P/'trace.npz');sen=np.load(P/'actual_sensor_frames.npz')['sensors'];tk=np.load(P/'every_tick.npz');meta=read(P/'actuator_metadata.json');assert m.names==meta['joint_names'];bn=meta['body_names'];root=bn.index('pelvis')
 raw={(r['case'],r['tick']):r for r in read(P/'touchdown_raw_176.json')};ids=np.flatnonzero((tr['t']>=8-1e-9)&(tr['t']<13-1e-9));assert len(ids)==50
 rows=[];As=[];gs=[];ws=[];qsave=[];gravsave=[];ta=[];normal=[];rawchecks=0
 for k in ids:
  t=float(tr['t'][k]);si=round(t*10);ti=round(t/.02)-1;assert abs(tk['t'][ti]-t)<1e-9
  for c in range(24):
   q=tr['q'][k,c].astype(float);g=sen[si,c,339:342].astype(float);tau=tr['torque'][k,c].astype(float)
   if (c,round(t/.02)) in raw:
    r=raw[c,round(t/.02)];assert np.array_equal(q,np.array(r['q'],np.float32).astype(float));assert np.array_equal(g,np.array(r['gravity'],np.float32).astype(float));rawchecks+=1
   estimates=obs.estimate(q,tau,g);world_to_level=level_rotation(g)@Rotation.from_quat(tr['body'][k,c,root,3:]).as_matrix().T
   # Truth force ONLY evaluates estimates; never used to construct the model.
   truth=(world_to_level@tk['force'][ti,c,5:7].T).T
   failed=bool(np.any(tk['failure'][:ti+1,c]));slipped=bool(np.any(~tk['no_slip'][:ti+1,c]));leg_contact=bool(np.any(tk['unexpected_body_normal'][ti,c,[bn.index(s+'_'+j+'_link') for s in ['left','right'] for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch']]]>10))
   for j,side in enumerate(['left','right']):
    e=estimates[j];w=e['wrench'];inv=e['inverse'];assert w is not None
    foot=side+'_ankle_roll_link';foot_qd=sen[si,c,293:316]*10
    row=dict(case=c,t=t,side=side,rank=e['rank'],condition_raw=e['condition'],condition_scaled_0p1m=e['condition_0p1m'],wrench_Fxyz_N_Mxyz_Nm=w.tolist(),truth_force_level_N=truth[j].tolist(),force_error_N=(w[:3]-truth[j]).tolist(),per_1Nm_each_joint_linf_wrench_sensitivity=np.sum(abs(inv),axis=1).tolist(),prior_physical_failure=failed,prior_slip=slipped,other_leg_contact=leg_contact,actual_roll_moment_available=False)
    rows.append(row);As.append(e['A']);gs.append(e['gravity_torque']);ws.append(w);qsave.append(q);gravsave.append(g);ta.append(tau);normal.append(truth[j])
 # Two declared windows, all cases; never remove failures from inventory.
 summaries=[]
 for c in range(24):
  for side in ['left','right']:
   rr=[x for x in rows if x['case']==c and x['side']==side and x['t']>=11-1e-9];w=np.array([x['wrench_Fxyz_N_Mxyz_Nm'] for x in rr]);err=np.array([x['force_error_N'] for x in rr]);sens=np.array([x['per_1Nm_each_joint_linf_wrench_sensitivity'] for x in rr])
   summaries.append(dict(case=c,side=side,samples=len(rr),condition_scaled_max=max(x['condition_scaled_0p1m'] for x in rr),wrench_median=np.median(w,axis=0).tolist(),force_error_rms_N=np.sqrt(np.mean(err**2,axis=0)).tolist(),sensitivity_max=sens.max(0).tolist(),prior_failure_samples=sum(x['prior_physical_failure'] for x in rr),prior_slip_samples=sum(x['prior_slip'] for x in rr)))
 sources=[Path(__file__).resolve(),Path('scripts/low_seat_spatial_wrench_255.py').resolve(),D/'cooked_model_50/model.json',ip,P/'trace.npz',P/'actual_sensor_frames.npz',P/'every_tick.npz',P/'touchdown_raw_176.json',P/'actuator_metadata.json']
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,rows=rows,summaries=summaries,raw_q_gravity_exact_crosschecks=rawchecks,window=[8,12.9],times=50,case_count=24,side_count=2,moment_origin='ankle_roll_link frame origin; gravity-aligned coordinates',sensitivity_definition='Worst-case absolute output change if EACH of six joint torques changes by at most1Nm, linear fixed-q model.1Nm is a unit sensitivity, NOT an established telemetry error bound or confidence interval.',conditioning_definition='Raw mixed-unit matrix and dimensionless-scaled matrix A diag(1,1,1,0.1m,0.1m,0.1m); no condition threshold/admission imposed.',limitations='Quasi-static estimate from approximate implicit-PD telemetry. Dynamics/inertia/other leg contacts not modeled. Truth resultant foot force evaluates accuracy only. Full contact moment is not saved and is unavailable; contact-position cross net-force is NOT assumed exact roll wrench. No runtime controller or physical/quiet/material/solver criterion changed.',physics_ticks=0,optimizer_updates=0,physical_admission=False,source_sha256={str(p):sha(p) for p in sources})
 (O/'results.json').write_text(json.dumps(out,indent=2));np.savez_compressed(O/'matrices.npz',A=As,gravity_torque=gs,wrench=ws,q=qsave,gravity=gravsave,torque=ta,truth_force=normal)
 (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(dict(rows=len(rows),raw_checks=rawchecks,selected=[s for s in summaries if s['case'] in [0,1]])),flush=True)
if __name__=='__main__':main()
