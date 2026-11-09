"""Recorded measured-settling and contact diagnosis; no new simulation."""
from pathlib import Path
import datetime,hashlib,json,ast
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel
from low_seat_fixed_foot_plan_126 import measured
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'touchdown_probe_178/candidate';O=D/'touchdown_diagnosis_184'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def first(t,mask):
 ids=np.flatnonzero(mask);return float(t[ids[0]]) if len(ids) else None

def main():
 O.mkdir(exist_ok=False);r=read(P/'results.json')['episodes'];geo=read(P/'geometry.json');tr=dict(np.load(P/'trace.npz'));tk=dict(np.load(P/'every_tick.npz'));raw=read(P/'touchdown_raw_176.json');guards=read(P/'fixed_guard_records_132.json');bal=read(P/'balance_feedback_161.json');plans=read(P/'fixed_plans_132.json')
 sensor=np.load(P/'sensor_inputs.npz')['sensors'];model=BodyModel(D/'cooked_model_50/model.json');cal=read(D.parent/'contact_effort_repair_01/plane_calibration.json')
 contact_names=read(P/'contact_body_names.json');contact_names=contact_names if isinstance(contact_names,list) else contact_names['body_names']
 tree=ast.parse(Path('scripts/run_supported_postures.py').read_text());contacts=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='CONTACTS' for x in n.targets));allowed={body for _,body,_ in contacts}
 byguard={(x['case'],x['tick']):x for x in guards};bybal={(x['case'],x['tick']):x for x in bal};planmap={x['case']:x['plan'] for x in plans}
 force=np.maximum(np.sum(tk['force']*tk['normal'],axis=-1),0);legal_load=np.sum(force*tk['rest_patch']*tk['permitted'],axis=-1)/tk['weight']
 rows=[];series=[]
 for row in r:
  i=row['case'];g=geo[row['geometry_index']];events={k:first(tk['t'],mask) for k,mask in [('slip',~tk['no_slip'][:,i]),('forbidden_contact',tk['wrong'][:,i]>10),('illegal_patch',~tk['legal'][:,i]),('physical_failure',tk['failure'][:,i])]}
  observed=[]
  for label,t in events.items():
   if t is None:continue
   # Latest saved pose/geometry at or before event, never fabricated exact-event pose.
   k=int(np.flatnonzero(tr['t']<=t+1e-10)[-1]);v=tr['geometry'][k,i]
   mh=ms=None;step=round(float(tr['t'][k])*10)
   if bool(v[11]) and step<200:
    _,_,_,mh,ms,_,_=measured(model,tr['q'][k,i],sensor[0,step,i,339:342],v,np.array([tr['estimated_root_x'][k,i],0.,0.]),cal)
   observed.append(dict(measured_height_cm=None if mh is None else mh*100,height_error_mm=None if mh is None else (mh-g['height'])*1000,measured_slope_deg=ms,slope_error_deg=None if ms is None else ms-g['pitch_deg'],event=label,time_s=t,saved_sample_s=float(tr['t'][k]),geometry_available=bool(v[11]),geometry_age_s=float(v[12])))
  bodies=[]
  if events['forbidden_contact'] is not None:
   k=int(np.flatnonzero(tk['wrong'][:,i]>10)[0]);forces=tk['unexpected_body_normal'][k,i]
   bodies=[dict(body=n,force_N=float(forces[j])) for j,n in enumerate(contact_names) if n not in allowed and forces[j]>10]
  q=sorted([x for x in raw if x['case']==i],key=lambda x:x['tick']);hold=[];released=[];below=[]
  for x in q:
   t=x['tick']*.02;a=byguard[(i,x['tick'])];b=bybal[(i,x['tick'])];cp=a['command_path'];s=x['settling'];mg=a['measured_guard']
   if cp['hold']>0:hold.append(t)
   if cp['released']:released.append(t)
   if not s['quiet'] and abs(cp['start']-cp['limit'])<1e-10:below.append(t)
   series.append(dict(case=i,t=t,position_before=x['position'],position_after=cp['end'],released=cp['released'],hold_ticks=cp['hold'],quiet=s['quiet'],pitch_error_deg=None if s['pitch_error_rad'] is None else np.rad2deg(s['pitch_error_rad']),gyro_y_rad_s=s['gyro_y_rad_s'],joint_speed_rad_s=s['joint_speed_rad_s'],guard_progress_allowed=a['progress_allowed'],geometry_available=mg['available'],geometry_age_s=float(x['geometry'][12]),estimated_gap_m=mg.get('pelvis_gap_m'),estimated_housing_m=mg.get('housing_clearance_m'),balance_offset_rad=b['feedback']['offset_after'],requested_seat_bw=cp.get('demanded_normal_bw',float(np.interp(cp['end'],np.r_[0,np.cumsum(np.max(abs(np.diff(np.array([z['load_checks'][0]['command'] for z in planmap[i]['path']]),axis=0)),axis=1))],[z['assumed_seat_normal_fraction'] for z in planmap[i]['path']])))))
  flags=np.flatnonzero(~tk['no_slip'][:,i]);late=np.flatnonzero((tk['t']>=15)&(legal_load[:,i]<.05))
  rows.append(dict(case=i,height_cm=g['height']*100,slope_deg=g['pitch_deg'],stable=row['success'],final_hold_s=row['final_hold_s'],slip_ticks=row['slip_ticks'],illegal_ticks=row['illegal_ticks'],warning_ticks=row['traction_bad_ticks'],physical_flags=row['failures'],events=events,geometry_before_events=observed,forbidden_bodies_at_first_event=bodies,selected=bool(q),settle_release_s=min(released) if released else None,first_quiet_count_s=min(hold) if hold else None,max_consecutive_quiet_ticks=max([byguard[(i,x['tick'])]['command_path']['hold'] for x in q],default=None),final_released=bool(q and byguard[(i,q[-1]['tick'])]['command_path']['released']),max_requested_seat_bw=max([z['requested_seat_bw'] for z in series if z['case']==i],default=None),last_subthreshold_legal_load_s=float(tk['t'][late[-1]]) if len(late) else None,peak_pelvis_speed_during_slips=float(tk['slip'][flags,i,0].max()) if len(flags) else None,peak_floor_speed_during_slips=float(tk['slip'][flags,i,5:7].max()) if len(flags) else None))
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,rows=rows,series=series,limitations='Truth contacts/loads are diagnosis only. Current measured guard and settling inputs are runtime. Saved0.1s poses bracket20ms labels. No single-cause or force-realization proof. A failed settling condition is not waived; absence of contact at a hover is not successful sitting.',physics_ticks=0,optimizer_updates=0,source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),P/'results.json',P/'touchdown_raw_176.json',P/'fixed_guard_records_132.json',P/'balance_feedback_161.json',P/'every_tick.npz',P/'trace.npz',P/'contact_body_names.json']})
 (O/'results.json').write_text(json.dumps(out,indent=2));np.savez_compressed(O/'support_load.npz',t=tk['t'],legal_seat_normal_bw=legal_load);(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
