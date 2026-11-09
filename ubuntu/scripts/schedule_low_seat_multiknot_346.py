"""Command-only inherited-filter/hover/normal-descent check on FIXED344 path.
Three synthetic quiet/load modes; no simulation/current sensors/control installation.
This is not a dynamic prediction or a minimum-duration proof. No parent composition yet.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from humanoid_sitting.shared_frame import BodyModel
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'multiknot_schedule_346'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);source=D/'multiknot_path_344/results.json';audit=D/'multiknot_audit_345/independent_audit.json';assert read(audit)['passed'];data=read(source);z=data['dense'];assert z['model_geometry_passed'] and z['model_command_passed'] and z['command_schedule_passed'];path=z['path'];Q=np.array([v['command'] for v in z['command_nodes']]);loads=np.array([v['assumed_seat_normal_fraction'] for v in z['command_nodes']]);gaps=np.array([p['pelvis_gap_m'] for p in path]);gaps[104:]=0.;assert np.all(np.diff(gaps)<=1e-10) and np.all(loads[:105]==0)
 m=BodyModel(D/'cooked_model_50/model.json');names=m.names;ids=[names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]+[names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['shoulder_pitch','elbow_pitch']];limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'};lo=np.array([limits[names[j]][0]+.03 for j in ids],np.float32);hi=np.array([limits[names[j]][1]-.03 for j in ids],np.float32);cap=np.array([.00225 if names[j].endswith(('ankle_pitch_joint','ankle_roll_joint')) else .003 for j in ids],np.float32);alpha=np.float32(.1);arc=np.r_[0,np.cumsum(np.max(abs(np.diff(Q,axis=0)),axis=1))];assert np.all(np.diff(arc)>0);hover=float(np.interp(.012,gaps[:105][::-1],arc[:105][::-1]));contact=float(arc[104]);max_ticks=1500
 # Optimistic continuous lower bounds under SAME polyline, before filter quantization.
 split=np.sort(np.r_[arc,hover]);grid=np.array([np.interp(split,arc,Q[:,j]) for j in range(37)]).T;gg=np.interp(split,arc,gaps);rates=cap.astype(float)/.02;segment=np.max(abs(np.diff(grid[:,ids],axis=0))/rates,axis=1);normal=np.where(split[:-1]>=hover-1e-12,np.maximum(gg[:-1]-gg[1:],0)/.006,0);necessary=float(np.maximum(segment,normal).sum()+.5)
 def at(s):
  k=max(0,min(int(np.searchsorted(arc,s,side='right')-1),len(Q)-2));f=(s-arc[k])/(arc[k+1]-arc[k]);return (Q[k]+f*(Q[k+1]-Q[k])).astype(np.float32)
 rows=[]
 for mode in ['synthetic_quiet_and_load','no_quiet','no_load']:
  s=0.;cmd=Q[0].astype(np.float32);applied=[cmd.copy()];positions=[s];targets=[];desired=[cmd.copy()];records=[];quiet=0;released=False;stalled=None
  for tick in range(max_ticks):
   if s==arc[-1]:break
   prev=cmd.copy();start=s;allowed=True;load=.1 if mode!='no_load' else 0.;limit=float(np.interp(load,loads[104:],arc[104:]));hold=False
   if not released:
    if abs(s-hover)<1e-10:
     if mode!='no_quiet' and quiet<25:quiet+=1;allowed=False;hold=True
     elif mode=='no_quiet':quiet=0;allowed=False;hold=True
     else:released=True
    if not released:limit=min(limit,hover)
   gap=float(np.interp(s,arc,gaps));goal=prev.copy();target=prev.copy()
   if allowed and s<limit:
    def valid(candidate):
     goal=at(candidate);raw=prev[ids]+(goal[ids]-prev[ids])/alpha;normal_ok=not released or gap-float(np.interp(candidate,arc,gaps))<=.006*.02+1e-14
     return bool(normal_ok and np.all(raw>=lo) and np.all(raw<=hi) and np.all(abs(alpha*(raw-prev[ids]))<=cap)),goal,raw
    edge=int(np.searchsorted(arc,s,side='right'));safe=s
    while edge<len(arc):
     proposed=min(float(arc[edge]),limit)
     if valid(proposed)[0]:
      safe=proposed
      if proposed==limit:break
      edge+=1
     else:
      lower,upper=safe,proposed
      for _ in range(45):
       mid=(lower+upper)/2
       if valid(mid)[0]:lower=mid
       else:upper=mid
      safe=lower;break
    if safe<=s+1e-12:stalled=dict(tick=tick,position=s,limit=limit);break
    okay,goal,raw=valid(safe);assert okay;target[ids]=raw;cmd[ids]=prev[ids]+np.clip(alpha*(np.clip(raw,lo,hi)-prev[ids]),-cap,cap);s=safe
   else:cmd=prev.copy()
   targets.append(target);applied.append(cmd.copy());desired.append(goal);positions.append(s);records.append([start,s,gap,float(np.interp(s,arc,gaps)),quiet,int(released),int(hold),load,float(np.interp(s,arc,loads))])
  arrays=dict(applied=np.array(applied),target=np.array(targets),desired=np.array(desired),position=np.array(positions),records=np.array(records),moving_ids=np.array(ids),lo=lo,hi=hi,cap=cap,Q=Q,arc=arc,gaps=gaps,loads=loads);np.savez_compressed(O/(mode+'.npz'),**arrays);complete=bool(s==arc[-1]);duration=len(targets)*.02;phase_counts={p:0 for p in ['shift','swing','approach','load']}
  for pos in positions[1:]:phase_counts[path[min(int(np.searchsorted(arc,pos,side='left')),114)]['phase']]+=1
  row=dict(mode=mode,complete=complete,command_ticks=len(targets),maximum_ticks=max_ticks,duration_s=duration,time_gate=bool(complete and duration<=13.5),stalled=stalled,synthetic_quiet_count=quiet,hover_released=released,final_position=s,hover=hover,contact=contact,final_gap_m=float(np.interp(s,arc,gaps)),maximum_assumed_load_bw=float(np.max(arrays['records'][:,8])) if records else 0.,endpoint_exact=bool(np.array_equal(cmd,Q[-1].astype(np.float32))),max_applied_vs_polyline_rad=float(abs(arrays['applied']-arrays['desired']).max()),phase_time_s={k:v*.02 for k,v in phase_counts.items()},physical_admission=False);rows.append(row);print(json.dumps(row),flush=True)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,source_sha256={str(p.resolve()):sha(p) for p in [source,audit,Path(__file__),D/'cooked_model_50/model.json']},rows=rows,necessary_continuous_reserved_rate_plus_hover_normal_time_s=necessary,nominal_ankle_pitch_roll_rate=.1125,other_nominal_rate=.15,feedback_rate_reserve=.0375,synthetic_hover_s=.5,normal_descent_m_s=.006,allowance_s=13.5,material_criteria_runtime_unchanged=True,parent_composition_checked=False,current_sensor_runtime=False,physics_ticks=0,optimizer_updates=0,physical_admission=False,limitations='Only nominal command arithmetic on fixed344polyline. Current25sample rawqd/pitch/gyro/freshgeometry/load guards are not replaced by fixtures. No measured support, lift authorization, parent/feedback composition, bothplacement or dynamic certificate; static modeled foot loads are assumptions.')
 (O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':main()
