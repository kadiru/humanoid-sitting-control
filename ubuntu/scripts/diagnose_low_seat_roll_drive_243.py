"""Recorded roll-drive decomposition; no simulation or controller change.

End-state algebra is explicitly NOT a reconstruction of solver motor effort.
All three runs/all24 cases and the verified private198/all37 joints are included.
"""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_low_seat_contact_probe_v2 import clean
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'roll_drive_diagnosis_243'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(x):return dict(min=float(np.min(x)),max=float(np.max(x)),median=float(np.median(x)),mean=float(np.mean(x)))
def main():
 O.mkdir(exist_ok=False)
 meta=read(D/'private_diagnostic_198/actuator_metadata.json');names=meta['joint_names'];kp=np.array(meta['stiffness']);kd=np.array(meta['damping']);rows=[];sources=[Path(__file__).resolve(),D/'private_diagnostic_198/actuator_metadata.json'];fig,ax=plt.subplots(1,3,figsize=(13,4))
 for run in ['normal_landing3s_06','bias_probe_210','centered_probe_234']:
  P=D/run/'candidate';tr=np.load(P/'trace.npz');a=np.load(P/'sensor_inputs.npz')['sensors'][0];episodes=read(P/'results.json')['episodes'];sources += [P/'trace.npz',P/'sensor_inputs.npz',P/'results.json']
  for c in range(24):
   for lo,hi in [(11.,14.9),(15.,19.9)]:
    k=np.arange(round(lo*10),round(hi*10)+1);q=tr['q'][k-1,c,:23];cmd=tr['command'][k-1,c,:23];v=a[k,c,293:316]*np.float32(10.);tau=tr['torque'][k-1,c,:23]
    assert np.array_equal(q,a[k,c,270:293])
    for j in range(23):
     p=kp[c,j]*(cmd[:,j]-q[:,j]);d=-kd[c,j]*v[:,j];r=dict(run=run,case=c,window_s=[lo,hi],joint=j,joint_name=names[j],clean_full_trial=clean(episodes[c]),samples=len(k),q_rad=stats(q[:,j]),command_error_rad=stats(cmd[:,j]-q[:,j]),raw_velocity_rad_s=stats(v[:,j]),position_term_Nm=stats(p),velocity_term_Nm=stats(d),approximate_telemetry_Nm=stats(tau[:,j]),end_state_algebra_residual_Nm=stats(p+d-tau[:,j]))
     rows.append(r)
     if j in [19,20] and c in [0,12,14] and lo==11:
      ax[[0,12,14].index(c)].scatter(float(np.median(p)),float(np.median(d)),label=f'{run.split("_")[0]} {"L" if j==19 else "R"}',s=24)
 private=np.load(D/'private_diagnostic_198/substep_states.npz');sources += [D/'private_diagnostic_198/substep_states.npz',D/'private_audit_v2_199/independent_audit.json'];assert read(sources[-1])['exact_prefix_replay']
 mask=private['t']>=11;private_rows=[]
 for c in range(24):
  for j in range(37):
   q=private['q'][mask,c,j].astype(float);v=private['qd'][mask,c,j].astype(float);cmd=private['command'][mask,c,j].astype(float);tau=private['torque'][mask,c,j].astype(float)
   private_rows.append(dict(case=c,joint=j,joint_name=names[j],q_range_rad=float(np.ptp(q)),raw_velocity_rad_s=stats(v),command_error_rad=stats(cmd-q),position_term_Nm=stats(kp[c,j]*(cmd-q)),velocity_term_Nm=stats(-kd[c,j]*v),approximate_telemetry_Nm=stats(tau),velocity_integral_rad=float(np.trapezoid(v,private['t'][mask])),net_q_change_rad=float(q[-1]-q[0])))
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,rows=rows,verified_private_rows=private_rows,assumptions='Algebra assumes zero velocity/extra-effort target from configured position-action source. Actual live drive targets and kernel input timing have not yet been captured. End-state terms do not reconstruct applied solver effort.',interpretation='Centered234 selected ankle-roll position error is small while negative velocity-dependent telemetry is substantial. Similar positive raw roll velocities appear in clean cases. This association does not identify a causal mechanism, validate new feedback, or waive the raw-speed guard. Direct link velocities and live drive inputs/settings are the next missing diagnostic evidence.',physics_ticks=0,optimizer_updates=0,physical_admission=False,source_sha256={str(p):sha(p) for p in sources})
 (O/'results.json').write_text(json.dumps(out,indent=2))
 for i,c in enumerate([0,12,14]):ax[i].set_title(f'Case {c}: roll terms,11–14.9s');ax[i].set_xlabel('Kp(command-q),Nm');ax[i].set_ylabel('-Kd raw qd,Nm');ax[i].grid(alpha=.2);ax[i].legend(fontsize=6)
 fig.suptitle('Recorded end-state algebra — approximate PD telemetry, not solver motor effort');fig.tight_layout();fig.savefig(O/'drive_terms.png',dpi=160);plt.close(fig)
 (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
 print(json.dumps(dict(records=len(rows),private_records=len(private_rows),physics_ticks=0)))
if __name__=='__main__':main()
