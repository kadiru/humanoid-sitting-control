"""Compare offline spatial estimate to individually summed simulator contacts.
Contact truth remains evaluation-only; no physical/control criterion change.
"""
from pathlib import Path
import json,datetime,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import level_rotation
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'contact_wrench_diagnostic_259';A=D/'contact_wrench_audit_260';O=D/'contact_wrench_diagnosis_261'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);audit=read(A/'independent_audit.json');assert audit['exact_prefix_replay'] and audit['contact_return_checks']['passed']
 old=read(D/'spatial_wrench_diagnosis_v2_255/results.json');orig=np.load(D/'drive_diagnostic_246/trace.npz');tr=np.load(P/'trace.npz');sensor=np.load(P/'actual_sensor_frames.npz')['sensors'];names=read(P/'actuator_metadata.json')['body_names'];root=names.index('pelvis');w={(x['substep'],x['side']):x for x in read(A/'contact_wrenches.json')};assert np.array_equal(orig['q'],tr['q']) and np.array_equal(orig['torque'],tr['torque'])
 rows=[]
 for r in old['rows']:
  t=r['t'];c=r['case'];side=r['side'];sub=round(t/.005);k=round(t*10)-1;si=k+1;key=(sub,side)
  if key not in w:
   rows.append(dict(case=c,t=t,side=side,available=False));continue
  W=w[key];rot=level_rotation(sensor[si,c,339:342])@Rotation.from_quat(tr['body'][k,c,root,3:]).as_matrix().T
  truth=np.r_[rot@np.array(W['force_world_N'][c]),rot@np.array(W['moment_about_link_world_Nm'][c])];est=np.array(r['wrench_Fxyz_N_Mxyz_Nm']);fn=rot@np.array(W['normal_force_world_N'][c]);mn=rot@np.array(W['normal_moment_world_Nm'][c]);ff=rot@np.array(W['friction_force_world_N'][c]);mf=rot@np.array(W['friction_moment_world_Nm'][c])
  # The saved mean contact point is not a force-weighted CoP. Report its
  # wrench discrepancy independently; never substitute it as ground truth.
  ti=round(t/.02)-1
  rows.append(dict(case=c,t=t,side=side,available=True,estimated=est.tolist(),contact_wrench=truth.tolist(),error=(est-truth).tolist(),normal_force=fn.tolist(),normal_moment=mn.tolist(),friction_force=ff.tolist(),friction_moment=mf.tolist(),prior_physical_failure=r['prior_physical_failure'],prior_slip=r['prior_slip'],other_leg_contact=r['other_leg_contact']))
 summaries=[]
 for c in range(24):
  for side in ['left','right']:
   rs=[r for r in rows if r['case']==c and r['side']==side and r['t']>=11-1e-9 and r['available']]
   if rs:
    e=np.array([r['error'] for r in rs]);truth=np.array([r['contact_wrench'] for r in rs]);est=np.array([r['estimated'] for r in rs]);summaries.append(dict(case=c,side=side,samples=len(rs),estimated_median=np.median(est,0).tolist(),truth_median=np.median(truth,0).tolist(),error_rms=np.sqrt(np.mean(e*e,0)).tolist(),error_max_abs=abs(e).max(0).tolist(),normal_moment_median=np.median([r['normal_moment'] for r in rs],0).tolist(),friction_moment_median=np.median([r['friction_moment'] for r in rs],0).tolist()))
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,rows=rows,summaries=summaries,available=sum(r['available'] for r in rows),unavailable=sum(not r['available'] for r in rows),zero_extra_physics=True,physical_admission=False,limitations='Individually summed simulator contact forces/moments are diagnostic labels. Quasi-static estimator still uses approximate implicit-PD effort, ignores dynamic generalized forces and other contacts. Agreement or disagreement does not admit a controller, waive raw-velocity settling or prove a single cause.',source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),A/'independent_audit.json',A/'contact_wrenches.json',D/'spatial_wrench_diagnosis_v2_255/results.json',P/'trace.npz',P/'actual_sensor_frames.npz']})
 (O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({'results.json':sha(O/'results.json')},indent=2));print(json.dumps(dict(available=out['available'],unavailable=out['unavailable'],selected=[s for s in summaries if s['case'] in [0,1]])),flush=True)
if __name__=='__main__':main()
