"""Saved344/346 timing and float32 inverse-target diagnosis; zero physics.
Conditional continuous-polyline timing is not a universal discrete/dynamic lower bound.
A one-joint rounding counterexample is not an executable full-path repair.
"""
from pathlib import Path
import json,datetime,hashlib
import numpy as np
import torch
from humanoid_sitting.shared_frame import BodyModel
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'multiknot_timing_diagnosis_v2_348'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);src=D/'multiknot_schedule_346';sched=read(src/'results.json');assert read(D/'multiknot_schedule_audit_347/independent_audit.json')['passed'];data=read(D/'multiknot_path_344/results.json');a=dict(np.load(src/'synthetic_quiet_and_load.npz'));m=BodyModel(D/'cooked_model_50/model.json');Q=a['Q'];arc=a['arc'];gaps=a['gaps'];ids=a['moving_ids'];rates=a['cap'].astype(float)/.02;hover=sched['rows'][0]['hover'];edges=np.sort(np.r_[arc,hover]);phase={p:dict(global015_s=0.,reserved_s=0.,normal_extra_s=0.,hover_s=.5 if p=='approach' else 0.) for p in ['shift','swing','approach','load']};dominance={m.names[j]:0. for j in ids};segments=[]
 for lo,hi in zip(edges,edges[1:]):
  qa=np.array([np.interp(lo,arc,Q[:,j]) for j in ids]);qb=np.array([np.interp(hi,arc,Q[:,j]) for j in ids]);v=abs(qb-qa);dt0=float(max(v)/.15);dt=float(max(v/rates));g0=np.interp(lo,arc,gaps);g1=np.interp(hi,arc,gaps);dn=float(max(g0-g1,0)/.006 if lo>=hover-1e-12 else 0.);k=min(int(np.searchsorted(arc,hi,side='left')),114);p=data['dense']['path'][k]['phase'];dominance[m.names[ids[int(np.argmax(v/rates))]]]+=dt;phase[p]['global015_s']+=dt0;phase[p]['reserved_s']+=dt;phase[p]['normal_extra_s']+=max(dn-dt,0);segments.append(dict(start=float(lo),end=float(hi),end_knot=k,phase=p,global015_s=dt0,reserved_s=dt,normal_s=dn,normal_extra_s=max(dn-dt,0)))
 total=sum(v['reserved_s']+v['normal_extra_s']+v['hover_s'] for v in phase.values());assert abs(total-sched['necessary_continuous_reserved_rate_plus_hover_normal_time_s'])<1e-12
 prev=a['applied'][-1];pos=float(a['position'][-1]);edge=int(np.searchsorted(arc,pos,side='right'));goal=Q[edge].astype(np.float32);raw=prev[ids]+(goal[ids]-prev[ids])/np.float32(.1);bad=[]
 for i,j in enumerate(ids):
  if a['lo'][i]<=raw[i]<=a['hi'][i]:continue
  direction=np.float32(-np.inf if goal[j]<prev[j] else np.inf);nextq=np.nextafter(prev[j],direction);inverse=prev[j]+(nextq-prev[j])/np.float32(.1);extreme=a['lo'][i] if direction<0 else a['hi'][i];forward=prev[j]+np.clip(np.float32(.1)*(extreme-prev[j]),-np.float32(.003),np.float32(.003));outs=[]
  for dev in ['cpu','cuda:0']:
   before=torch.tensor(prev[j],device=dev);tar=torch.tensor(extreme,device=dev);low=torch.tensor(a['lo'][i],device=dev);high=torch.tensor(a['hi'][i],device=dev);out=before+(.1*(tar.clamp(min=low,max=high)-before)).clamp(-.003,.003);outs.append(float(out.cpu()))
  assert outs[0]==outs[1]==float(forward)
  bad.append(dict(joint=m.names[j],current=float(prev[j]),next_knot_command=float(goal[j]),desired_lo=float(a['lo'][i]),desired_hi=float(a['hi'][i]),raw_for_next_knot=float(raw[i]),next_representable_q=float(nextq),inverse_raw_for_one_ulp=float(inverse),inverse_raw_within_clamp=bool(a['lo'][i]<=inverse<=a['hi'][i]),legal_extreme_raw=float(extreme),forward_filtered_from_legal_extreme=float(forward),forward_moves_one_ulp=bool(forward==nextq),cpu_gpu_scalar_exact=True))
 assert bad and any(not b['inverse_raw_within_clamp'] and b['forward_moves_one_ulp'] for b in bad)
 identical={};permissions={}
 for r in sched['rows']:
  v=dict(np.load(src/(r['mode']+'.npz')));identical[r['mode']]=all(np.array_equal(v[k],a[k]) for k in ['applied','target','position']);assert np.array_equal(np.delete(v['records'],7,axis=1),np.delete(a['records'],7,axis=1));wanted=0. if r['mode']=='no_load' else .1;assert np.all(v['records'][:,7]==wanted);permissions[r['mode']]=wanted
 assert all(identical.values());assert all(not r['hover_released'] and r['synthetic_quiet_count']==0 for r in sched['rows'])

 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,phase_time=phase,continuous_reserved_normal_hover_time_s=total,dominant_joint_duration_s=dominance,segments=segments,stall=dict(tick=sched['rows'][0]['command_ticks'],time_s=sched['rows'][0]['duration_s'],position=pos,next_knot=edge,gap_m=float(np.interp(pos,arc,gaps)),hover_not_reached=True,limiting_inverse_raw=bad),all_three_command_prefixes_exact=identical,requested_load_permissions_bw=permissions,nonpermission_records_exact=True,preserved_recovery='multiknot_metadata_recovery_348',interpretation='346 inverse-target construction stalls before hover. Its algebraic inverse float32 test is sufficient but not necessary: a legal clamp endpoint yields one ULP forward movement here even though inverse raw for that same output is outside clamp. This does NOT certify multi-joint/polyline progress or a corrected schedule and does not change any filter/limit. Separately the fixed continuous-polyline rate/normal/0.5s-hover model requires17.269704s under its stated assumptions, already over13.5; not a universal discrete/physical lower bound. No hover/quiet/load guard was exercised.',physics_ticks=0,optimizer_updates=0,runtime_changed=False,physical_admission=False,source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),src/'results.json',src/'SHA256.json',D/'multiknot_schedule_audit_347/independent_audit.json',D/'multiknot_path_344/results.json']})
 (O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['segments','source_sha256']}),flush=True)
if __name__=='__main__':main()
