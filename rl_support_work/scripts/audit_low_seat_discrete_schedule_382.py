"""Independent ALL saved command ticks CPU/GPU filter/rate/bounds and synthetic permission audit.
No real current sensors, physical support, dynamic execution or parent composition tested.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
import torch
from humanoid_sitting.shared_frame import BodyModel
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'discrete_schedule_audit_382'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);r=read(D/'discrete_schedule_381/results.json');model=BodyModel(D/'cooked_model_50/model.json');limits={j['name']:np.deg2rad(j['limits']) for j in model.joints if j['type']=='PhysicsRevoluteJoint'};lo=np.array([limits[model.names[j]][0]+.03 for j in range(23)],np.float32);hi=np.array([limits[model.names[j]][1]-.03 for j in range(23)],np.float32);rows=[]
 for p,h in r['source_sha256'].items():assert sha(p)==h,p
 for rr in r['rows']:
  a=dict(np.load(D/'discrete_schedule_381'/(rr['mode']+'.npz')));ids=a['moving'];fixed=[j for j in range(37) if j not in ids];A=a['applied'];n=len(a['target']);assert n==rr['command_ticks'];assert np.array_equal(A[0],a['C'][0].astype(np.float32));assert np.array_equal(A[:,fixed],np.broadcast_to(A[0,fixed],(len(A),len(fixed))));outcomes=[];clipped=0
  for device in ['cpu','cuda:0']:
   cmd=torch.tensor(A[0],device=device);low=torch.tensor(lo,device=device);high=torch.tensor(hi,device=device);history=[cmd.cpu().numpy().copy()]
   for t in a['target']:
    desired=torch.tensor(t[:23],device=device);clipped+=int(((desired<low)|(desired>high)).sum());cmd[:23]+=(.1*(desired.clamp(min=low,max=high)-cmd[:23])).clamp(-.003,.003);history.append(cmd.cpu().numpy().copy())
   outcomes.append(np.array(history))
  assert clipped==0 and np.array_equal(outcomes[0],outcomes[1]) and np.array_equal(outcomes[0],A)
  increment=abs(np.diff(A.astype(float),axis=0))[:,ids];assert np.all(increment<=a['rates']*.02);errors=abs(A[1:,ids].astype(float)-a['goal'][:,ids].astype(float));assert np.all(errors<=a['ulp'])
  minmargin=min(min(float(u[j])-float(limits[model.names[j]][0]),float(limits[model.names[j]][1])-float(u[j])) for u in a['target'] for j in ids);assert minmargin>=.03
  quiet=0;released=False;holds=0;prevgap=float(a['gaps'][0]);normal=[];replaymax=0.;loadmax=0.;maxgapbeforehover=-1.
  for k,rec in enumerate(a['records']):
   knot,f,hold,qn,rel=rec;knot=int(knot)
   if hold:
    assert np.array_equal(a['target'][k],A[k]) and np.array_equal(A[k+1],A[k]);gap=float(a['gaps'][knot]);holds+=1
    if knot==99 and rr['mode']!='no_quiet':quiet+=1
   else:
    wanted=(a['C'][knot-1]+float(f)*(a['C'][knot]-a['C'][knot-1])).astype(np.float32);assert np.array_equal(wanted,a['goal'][k]);gap=float(a['gaps'][knot-1]+f*(a['gaps'][knot]-a['gaps'][knot-1]));expectedf=round(f*int(a['segment_ticks'][knot-1]))/int(a['segment_ticks'][knot-1]);replaymax=max(replaymax,abs(expectedf-f))
   released=quiet>=25;assert quiet==int(qn) and released==bool(rel)
   if knot>=100 and not hold:assert released;normal.append((prevgap-gap)/.02)
   if not released:assert knot<=99
   if rr['mode']=='no_load':assert knot<=104
   load=0 if rr['mode']=='no_load' else float(a['assumed_loads'][knot]) if hold else float(a['assumed_loads'][knot-1]+f*(a['assumed_loads'][knot]-a['assumed_loads'][knot-1]));loadmax=max(loadmax,load);prevgap=gap
  assert replaymax<1e-14;assert quiet==rr['quiet_samples'];assert n*.02==rr['duration_s']
  if rr['mode']=='no_quiet':assert quiet==0 and not released and not rr['complete']
  if rr['mode']=='no_load':assert not rr['complete'] and loadmax==0
  maxnormal=max(normal,default=0.);rows.append(dict(mode=rr['mode'],ticks=n,cpu_gpu_exact=True,actual_saved_filter_exact=True,all14_fingers_fixed=True,clipped_targets=0,reserved_rate_passed=True,minimum_desired_physical_margin_rad=minmargin,maximum_applied_step_rad=float(increment.max()) if n else 0,maximum_goal_departure_rad=float(errors.max()) if n else 0,quiet_hold_ticks=holds,quiet_samples=quiet,synthetic_hover_reached=bool(holds),released=released,maximum_nominal_normal_speed_m_s=maxnormal,strict_nominal_normal_limit_passed=bool(maxnormal<=.006),duration_s=n*.02,complete=rr['complete'],physical_admission=False))
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,rows=rows,arithmetic_records=sum(x['ticks'] for x in rows),all_geometry_or_physics_not_certified=True,current_input_runtime_tested=False,physical_admission=False,physics_ticks=0,policy_optimizer_updates=0,source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),D/'discrete_schedule_381/results.json',D/'discrete_schedule_381/SHA256.json']});(O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
