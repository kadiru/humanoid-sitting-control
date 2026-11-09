"""Read-only falsification of early-unloading and command-tracking hypotheses."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');OUT=D/'command_tracking_diagnosis_30'
read=lambda p:json.loads(Path(p).read_text())
def main():
 OUT.mkdir(exist_ok=False);models=read(Path('runs/g1_kinematics_geometry_04/model.json'))
 from humanoid_sitting.shared_frame import BodyModel
 m=BodyModel('runs/g1_kinematics_geometry_04/model.json');legs=[m.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']];rows=[];hashes={}
 for name in ['normal_landing3s_06','joint_path_17']:
  p=D/name/'candidate';t=np.load(p/'trace.npz');ticks=np.load(p/'every_tick.npz');load=read(p/'load_feedback_audit.json');paths=np.load(p/'last_rolling_paths.npz')
  for case in range(24):
   bad=np.flatnonzero(ticks['wrong'][:,case]>10.);event=float(ticks['t'][bad[0]]) if len(bad) else None
   stop=event if event is not None else 15.;start=float(paths['start'][case])*.02;duration=7. if name=='joint_path_17' and case in [0,1,6,7,10,11] else 5.
   sampled=(t['t']>=max(start,stop-2.))&(t['t']<=stop)
   rates=np.diff(t['command'][:,case,legs],axis=0)/np.diff(t['t'])[:,None];rate_mask=sampled[1:]
   records=[]
   for at in np.flatnonzero(sampled):
    #Trace q/body are post-step; commanded reference was assembled pre-step.
    pretime=float(t['t'][at])-.02;u=np.clip((pretime-start)/duration*24,0,24);lo=min(int(u),23);f=u-lo;desiredq=(1-f)*paths['q'][case,lo]+f*paths['q'][case,lo+1]
    rec=dict(t=float(t['t'][at]),max_reference_to_applied_command_rad=float(np.max(abs(t['reference'][at,case,legs]-t['command'][at,case,legs]))),max_q_to_pre_step_nominal_q_rad=float(np.max(abs(t['q'][at,case,legs]-desiredq[legs]))))
    records.append(rec)
   before=[dict(t=float(x['tick'])*.02,**state) for x in load if max(start,stop-2.)<=float(x['tick'])*.02<stop for state in x['states'] if state['case']==case]
   maxfrac=max((x['applied_fraction'] for x in before),default=None);last=before[-1] if before else None
   rows.append(dict(controller=name,case=case,first_forbidden_gt10N_s=event,window_start_s=max(start,stop-2.),window_end_s=stop,sampled_intervals=int(rate_mask.sum()),sampled_any_leg_near_0p15_fraction=float(np.mean(np.max(abs(rates[rate_mask]),axis=1)>=.149)) if rate_mask.any() else None,max_sampled_reference_command_gap_rad=max((x['max_reference_to_applied_command_rad'] for x in records),default=None),load_feedback_max_fraction_before_event=maxfrac,last_load_record_before_event=last,samples=records))
  for f in ['trace.npz','every_tick.npz','last_rolling_paths.npz','load_feedback_audit.json']:hashes[str(p/f)]=hashlib.sha256((p/f).read_bytes()).hexdigest()
 hashes[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='All48 recorded episodes. No physics/controller/threshold changes. Rate data sampled every0.1s, not every0.02s; near-cap fractions are sample evidence, not exact tick saturation. q-to-nominal compares post-step q with pre-step nominal target and is descriptive; q-command offset alone includes static load compensation and is not tracking error.',conclusion='New joint17 housing contacts on cases0/1/6/7/10/11 were preceded by zero sensor-based load reduction, contradicting early unloading as their explanation. Simultaneous command lag and near-cap motion motivate a rate-feasible approach with tracking reserve, but do not establish sole causation.',simulations=0,optimizer_updates=0,holdout_used=False,source_input_hashes=hashes,rows=rows)
 (OUT/'results.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps([{k:v for k,v in r.items() if k not in ['samples','last_load_record_before_event']} for r in rows if r['first_forbidden_gt10N_s'] is not None],indent=2))
if __name__=='__main__':main()
