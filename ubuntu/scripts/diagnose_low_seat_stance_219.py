"""Saved-record stance and joint-velocity diagnosis. Zero simulation or controller change."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'stance_diagnosis_219'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);meta=read(D/'private_diagnostic_198/actuator_metadata.json');names=meta['joint_names'];rows=[];rawrows=[];sources=[Path(__file__).resolve(),D/'private_diagnostic_198/actuator_metadata.json']
 runs=['normal_landing3s_06','touchdown_probe_178','bias_probe_210']
 for run in runs:
  P=D/run/'candidate';tr=np.load(P/'trace.npz');tk=np.load(P/'every_tick.npz');res=read(P/'results.json')['episodes'];sources += [P/x for x in ['trace.npz','every_tick.npz','results.json']]
  raw=read(P/'touchdown_raw_176.json') if (P/'touchdown_raw_176.json').exists() else []
  plans=read(P/'fixed_plans_132.json') if (P/'fixed_plans_132.json').exists() else []
  sources += [P/x for x in ['touchdown_raw_176.json','fixed_plans_132.json'] if (P/x).exists()]
  for c in range(24):
   for start,end in [(11.,14.9),(15.,20.)]:
    ix=(tr['t']>=start-1e-9)&(tr['t']<=end+1e-9);it=(tk['t']>=start-1e-9)&(tk['t']<=end+1e-9);force=tk['force'][it,c,5:7].astype(float);vertical=force[:,:,2]/tk['weight'][it,c,None];q=tr['q'][ix,c].astype(float);cmd=tr['command'][ix,c].astype(float);tau=tr['torque'][ix,c].astype(float)
    rec=[x for x in raw if x['case']==c and start-1e-9<=x['tick']*.02<=end+1e-9];joints=[]
    for j,n in enumerate(names):
     z=dict(joint=n,q_median_rad=float(np.median(q[:,j])),q_range_rad=float(np.ptp(q[:,j])),command_minus_q_median_rad=float(np.median(cmd[:,j]-q[:,j])),approximate_torque_median_Nm=float(np.median(tau[:,j])))
     if rec:
      vel=np.array([x['qd'][j] for x in rec]);pos=np.array([x['q'][j] for x in rec]);z.update(raw_qd_median_rad_s=float(np.median(vel)),raw_qd_abs_median_rad_s=float(np.median(abs(vel))),raw_qd_abs_min_rad_s=float(min(abs(vel))),raw_qd_sign_changes=int(np.sum(vel[1:]*vel[:-1]<0)),raw_qd_samples=len(vel),raw_q_range_rad=float(np.ptp(pos)),sampled_position_difference_speed_median_rad_s=float(np.median(abs(np.diff(pos))/.02)))
     else:z.update(raw_qd_median_rad_s=None,raw_qd_samples=0)
     joints.append(z)
    feet=[]
    for f in [0,1]:
     b=meta['body_names'].index(['left','right'][f]+'_ankle_roll_link');rot=Rotation.from_quat(tr['body'][ix,c,b,3:]);rv=rot.as_euler('xyz',degrees=True)
     feet.append(dict(side=['left','right'][f],normal_load_median_bw=float(np.median(vertical[:,f])),normal_load_range_bw=[float(vertical[:,f].min()),float(vertical[:,f].max())],tangent_force_median_N=float(np.median(np.linalg.norm(force[:,f,:2],axis=1))),max_loaded_slip_speed_m_s=float(tk['slip'][it,c,5+f].max()),pose_euler_median_deg=np.median(rv,axis=0).tolist(),pose_euler_range_deg=np.ptp(rv,axis=0).tolist()))
    endpoint=None
    for p in plans:
     if p['case']==c:
      r=p['plan']['path'][-1]['root'];endpoint=dict(root_position_m=r[:3],root_rpy_deg=Rotation.from_quat(r[3:]).as_euler('xyz',degrees=True).tolist())
    rows.append(dict(run=run,case=c,window=[start,end],stable=res[c]['success'],final_hold_s=res[c]['final_hold_s'],slip_ticks=res[c]['slip_ticks'],physical_flags=res[c]['failures'],feet=feet,joints=joints,raw_settling_available=bool(rec),endpoint=endpoint))
  print(run,'complete',flush=True)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),physics_ticks=0,optimizer_updates=0,holdout_used=False,physical_admission=False,scope=__doc__,rows=rows,source_sha256={str(p):sha(p) for p in sources},interpretation='Selected paths intentionally translate the pelvis laterally by +/-50mm and produce asymmetric foot loading. Persistent ankle-roll qd dominates the quiet guard. These are associations, not proof that centering will fix dynamics; qd is never replaced by finite differences. Force/body labels are offline diagnostics only. Implicit-PD effort is approximate. Raw every20ms settling input exists only for selected four rows of178/210; unavailable entries remain null.')
 (O/'results.json').write_text(json.dumps(out,indent=2))
 fig,ax=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
 for col,c in enumerate([0,6]):
  for k,run in enumerate(runs):
   a=next(z for z in rows if z['run']==run and z['case']==c and z['window']==[11.,14.9]);ax[0,col].bar(np.array([0,1])+(k-1)*.23,[z['normal_load_median_bw'] for z in a['feet']],.23,label=run)
   if a['raw_settling_available']:
    js=[next(j for j in a['joints'] if j['joint']==s+'_ankle_roll_joint') for s in ['left','right']];ax[1,col].bar(np.array([0,1])+(k-1.5)*.28,[j['raw_qd_abs_median_rad_s'] for j in js],.28,label=run)
  ax[0,col].set_title(f'Case {c}, 11–14.9 s');ax[0,col].set_ylabel('Median actual floor load / BW');ax[1,col].set_ylabel('Median |raw ankle-roll qd| rad/s');ax[1,col].axhline(.05,color='k',ls='--',label='Unchanged settling guard')
  for a in ax[:,col]:a.set_xticks([0,1],['Left','Right']);a.legend(fontsize=6)
 fig.savefig(O/'stance_and_velocity.png',dpi=180);plt.close(fig)
 (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':main()
