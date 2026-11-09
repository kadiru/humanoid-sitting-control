"""Independent CPU/GPU float32 clamp/filter/increment and synthetic guard replay346.
No physics or evidence of actual raw-speed settling/support. Parent remains unchecked.
"""
from pathlib import Path
import json,datetime,hashlib
import numpy as np
import torch
from humanoid_sitting.shared_frame import BodyModel
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'multiknot_schedule_audit_347'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);src=D/'multiknot_schedule_346';data=read(src/'results.json');z=read(D/'multiknot_path_344/results.json')['dense'];m=BodyModel(D/'cooked_model_50/model.json');lim={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'};Q=np.array([v['command'] for v in z['command_nodes']]);arc=np.r_[0,np.cumsum(np.max(abs(np.diff(Q,axis=0)),axis=1))];gaps=np.array([e['pelvis_gap_m'] for e in z['path']]);gaps[104:]=0;loads=np.array([v['assumed_seat_normal_fraction'] for v in z['command_nodes']]);hover=float(np.interp(.012,gaps[:105][::-1],arc[:105][::-1]));rows=[];torch.set_num_threads(2)
 for p,h in data['source_sha256'].items():assert sha(p)==h,p
 for row in data['rows']:
  a=dict(np.load(src/(row['mode']+'.npz')));ids=a['moving_ids'];fixed=np.array([j for j in range(37) if j not in ids]);assert np.array_equal(a['Q'],Q) and np.array_equal(a['arc'],arc) and np.array_equal(a['gaps'],gaps) and np.array_equal(a['loads'],loads);n=len(a['target']);assert n==row['command_ticks'];assert a['applied'].shape==(n+1,37);assert np.array_equal(a['applied'][0],Q[0].astype(np.float32));assert np.array_equal(a['applied'][:,fixed],np.broadcast_to(Q[0,fixed].astype(np.float32),(n+1,len(fixed))))
  lo=np.array([lim[m.names[j]][0]+.03 for j in range(23)],np.float32);hi=np.array([lim[m.names[j]][1]-.03 for j in range(23)],np.float32);outs=[];clipped=[]
  for dev in ['cpu','cuda:0']:
   cmd=torch.tensor(a['applied'][0],device=dev);low=torch.tensor(lo,device=dev);high=torch.tensor(hi,device=dev);out=[cmd.cpu().numpy().copy()];nc=0
   for tar in a['target']:
    desired=torch.tensor(tar[:23],device=dev);nc+=int(((desired<low)|(desired>high)).sum());cmd[:23]+=(.1*(desired.clamp(min=low,max=high)-cmd[:23])).clamp(-.003,.003);out.append(cmd.cpu().numpy().copy())
   outs.append(np.array(out));clipped.append(nc)
  assert not any(clipped);assert np.array_equal(outs[0],outs[1]);assert np.array_equal(outs[0],a['applied']);before=a['applied'][:-1];after=a['applied'][1:];assert np.all(after<=before+np.float32(.003)) and np.all(after>=before-np.float32(.003));assert abs(after-before).max()<=.003+3e-7
  count=0;released=False;maxnormal=0.;holdticks=0;maxpoly=0.
  for k,rec in enumerate(a['records']):
   start,end,g0,g1,quiet,rel,hold,load,demand=rec;assert start==a['position'][k] and end==a['position'][k+1];assert abs(g0-np.interp(start,arc,gaps))<1e-14 and abs(g1-np.interp(end,arc,gaps))<1e-14
   if not released and abs(start-hover)<1e-10:
    if row['mode']=='no_quiet':count=0
    elif count<25:count+=1
    else:released=True
   assert count==int(quiet) and released==bool(rel)
   if hold:assert start==end and abs(start-hover)<1e-10;holdticks+=1
   if not released:assert end<=hover+1e-10
   if released:maxnormal=max(maxnormal,(g0-g1)/.02);assert (g0-g1)/.02<=.006+1e-10
   assert abs(demand-np.interp(end,arc,loads))<1e-14;assert demand<=load+1e-12
   if row['mode']=='no_load':assert load==0 and demand==0 and end<=arc[104]+1e-10
   if end>start:
    index=max(0,min(np.searchsorted(arc,end,side='right')-1,113));f=(end-arc[index])/(arc[index+1]-arc[index]);goal=(Q[index]+f*(Q[index+1]-Q[index])).astype(np.float32);assert np.array_equal(goal,a['desired'][k+1]);maxpoly=max(maxpoly,float(abs(after[k]-goal).max()));raw=before[k,ids]+(goal[ids]-before[k,ids])/np.float32(.1);assert np.array_equal(raw,a['target'][k,ids]);assert np.all(abs(np.float32(.1)*(raw-before[k,ids]))<=a['cap'])
   else:assert np.array_equal(a['target'][k],before[k])
  complete=bool(a['position'][-1]==arc[-1]);assert row['complete']==complete and row['time_gate']==bool(complete and n*.02<=13.5);assert row['duration_s']==n*.02;assert row['endpoint_exact']==bool(np.array_equal(a['applied'][-1],Q[-1].astype(np.float32)));assert row['synthetic_quiet_count']==count and row['hover_released']==released
  if row['mode']=='no_quiet':assert not released and not complete
  if row['mode']=='no_load':assert not complete
  rows.append(dict(mode=row['mode'],ticks=n,cpu_gpu_exact=True,exact_saved_filter_replay=True,clipped_targets=0,all14fingers_fixed=True,maximum_step_rad=float(abs(after-before).max()),maximum_sampled_normal_speed_m_s=maxnormal,synthetic_hold_ticks=holdticks,hover_guard_reached=bool(holdticks),quiet_load_fixture_exercised=bool(holdticks),complete=complete,duration_s=n*.02,time_gate=row['time_gate'],endpoint_exact=row['endpoint_exact'],max_applied_vs_polyline_rad=maxpoly,physical_admission=False))
 ids=a['moving_ids'];rates=np.array([.1125 if m.names[j].endswith(('ankle_pitch_joint','ankle_roll_joint')) else .15 for j in ids],np.float32).astype(float)
 # Reconstruct each split segment with scalar interpolation and sum max joint/normal durations.
 edges=np.sort(np.r_[arc,hover]);needed=.5
 for start,end in zip(edges,edges[1:]):
  ca=np.array([np.interp(start,arc,Q[:,j]) for j in ids]);cb=np.array([np.interp(end,arc,Q[:,j]) for j in ids]);rate=a['cap'].astype(float)/.02;dt=max(abs(cb-ca)/rate);normal=max(np.interp(start,arc,gaps)-np.interp(end,arc,gaps),0)/.006 if start>=hover-1e-12 else 0;needed+=max(dt,normal)
 assert abs(needed-data['necessary_continuous_reserved_rate_plus_hover_normal_time_s'])<1e-12
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,rows=rows,necessary_time_s=needed,physical_admission=False,physics_ticks=0,optimizer_updates=0,parent_composition_checked=False,all_guard_modes_reached_hover=all(r['hover_guard_reached'] for r in rows),guard_coverage_caveat='Stalled prefixes do not exercise hover release or load gating; do not report those guards as tested when no hover was reached.',source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),src/'results.json',src/'SHA256.json',D/'multiknot_path_344/results.json']})
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
