"""Recorded actual414 joint-velocity channels in all clean/failed cases.
Separate normalized-channel arithmetic from physical velocity interpretation.
No resimulation, replacement of raw qd, or change to the settling contract.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from audit_low_seat_contact_probe_v2 import clean
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'centered_input_crosscheck_240'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False)
 source=Path('scripts/run_supported_transition.py');assert 'd.joint_vel.torch[:,self.ids]/10.' in source.read_text()
 names=read(D/'private_diagnostic_198/actuator_metadata.json')['joint_names'];plan=read(D/'centered_runtime_paths_231/case_0.json')['plan'];ids=plan['moving_ids'];assert max(ids)<23
 sources=[Path(__file__),source,D/'private_diagnostic_198/actuator_metadata.json',D/'centered_runtime_paths_231/case_0.json'];rows=[];checks=[]
 for run in ['normal_landing3s_06','bias_probe_210','centered_probe_234']:
  P=D/run/'candidate';a=np.load(P/'sensor_inputs.npz')['sensors'][0];res=read(P/'results.json')['episodes'];assert a.shape==(200,24,414)
  sources += [P/'sensor_inputs.npz',P/'results.json']
  if (P/'touchdown_raw_176.json').exists():
   sources.append(P/'touchdown_raw_176.json')
   for r in read(P/'touchdown_raw_176.json'):
    if r['tick']%5 or r['tick']>=1000:continue
    raw=np.array(r['qd'],np.float32)[:23];observed=a[r['tick']//5,r['case'],293:316];err=float(np.max(abs(raw/np.float32(10.)-observed)));assert err<=3e-7
    checks.append(dict(run=run,case=r['case'],tick=r['tick'],normalized_max_error=err))
  for c in range(24):
   for lo,hi in [(11.,14.9),(15.,19.9)]:
    steps=np.arange(round(lo*10),round(hi*10)+1);qd=a[steps,c,293:316]*np.float32(10.);maximum=abs(qd[:,ids]).max(axis=1);joint_medians=np.median(abs(qd),axis=0);j=ids[int(np.argmax(joint_medians[ids]))]
    rows.append(dict(run=run,case=c,window_s=[lo,hi],clean_full_trial=clean(res[c]),samples=len(steps),maximum_moving_raw_speed_median_rad_s=float(np.median(maximum)),minimum_maximum_moving_raw_speed_rad_s=float(min(maximum)),sampled_speed_condition_passes=int((maximum<=.05).sum()),dominant_median_joint=names[j],dominant_abs_raw_speed_median_rad_s=float(joint_medians[j]),ankle_roll_median_rad_s=np.median(qd[:,[19,20]],axis=0).tolist()))
 groups=[]
 for run in ['normal_landing3s_06','bias_probe_210','centered_probe_234']:
  for window in [[11.,14.9],[15.,19.9]]:
   r=[x for x in rows if x['run']==run and x['window_s']==window and x['clean_full_trial']]
   groups.append(dict(run=run,window_s=window,clean_cases=len(r),cases_with_no_sample_satisfying_speed_condition=sum(x['sampled_speed_condition_passes']==0 for x in r),sampled_speed_condition_passes=sum(x['sampled_speed_condition_passes'] for x in r),total_samples=sum(x['samples'] for x in r)))
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,rows=rows,clean_groups=groups,raw_crosschecks=checks,max_normalized_error=max(x['normalized_max_error'] for x in checks),scope=__doc__,interpretation='The actor records23 normalized current joint velocities at0.1s. Multiplication by10 is a decoded channel value, not bit-exact recovery of every raw float. Available selected20ms raw samples crosscheck normalization. Fourteen finger velocities are unavailable in414 and omitted. Sampled low-speed counts do not certify25 consecutive20ms quiet samples. Clean full-trial labels use unchanged physical criteria; comparison with the additional controller settling guard does not waive that guard or prove a velocity sensor/solver defect.',physics_ticks=0,optimizer_updates=0,physical_admission=False,source_sha256={str(p.resolve()):sha(p) for p in sources})
 (O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(dict(groups=groups,crosschecks=len(checks),max_normalized_error=out['max_normalized_error'])),flush=True)
if __name__=='__main__':main()
