"""Conditional exact-forward-filter construction for351, only after every static/NLP gate passes.
No simulation, saved-sensor substitution, current runtime or physical admission.
"""
from pathlib import Path
import numpy as np,json,hashlib,datetime
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'guarded_path_schedule_353'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ordered(x):
 b=int(np.float32(x).view(np.uint32));return b^0x80000000 if b<0x80000000 else (~b)&0xffffffff
def decode(i):
 b=i^0x80000000 if i>=0x80000000 else (~i)&0xffffffff
 return np.array(b,np.uint32).view(np.float32)[()]
def step(prev,u,lo,hi,cap):return np.float32(prev+np.clip(np.float32(np.float32(.1)*np.float32(np.clip(u,lo,hi)-prev)),-cap,cap))
def inverse_forward(prev,goal,lo,hi,cap):
 # Search the FINITE legal target lattice using the actual forward expression.
 # Endpoints and equality are exact float32; no inverse-only criterion or tolerance waiver.
 a=ordered(lo);b=ordered(hi)
 if goal<step(prev,lo,lo,hi,cap) or goal>step(prev,hi,lo,hi,cap):return None
 while a<b:
  mid=(a+b)//2
  if step(prev,decode(mid),lo,hi,cap)<goal:a=mid+1
  else:b=mid
 u=decode(a)
 return u if step(prev,u,lo,hi,cap)==goal else None

def main():
 r=read(D/'guarded_path_351/results.json');a=read(D/'guarded_path_audit_352/independent_audit.json');z=r['dense'];assert a['passed'];assert z['model_geometry_passed'] and z['model_command_passed'] and z['guarded_timing_constraints_passed'];assert r['optimizer']['numerical_constraints_passed'];O.mkdir(exist_ok=False)
 meta=read(D/'cooked_model_50/model.json');from humanoid_sitting.shared_frame import BodyModel
 m=BodyModel(D/'cooked_model_50/model.json');moving=[m.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]+[m.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['shoulder_pitch','elbow_pitch']]
 ini=np.load(D/'guarded_path_351/initial.npz');lo=ini['desired_lo'];hi=ini['desired_hi'];cap=(ini['rates']*.02).astype(np.float32);C=np.array([v['command'] for v in z['command_nodes']],np.float32);ni=np.ceil(np.array(z['segment_durations_s'])/.02).astype(int);assert np.all(ni>=1);gaps=np.array([v['pelvis_gap_m'] for v in z['path']]);load=np.array([v['assumed_seat_normal_fraction'] for v in z['command_nodes']]);rows=[]
 # Synthetic quiet samples are labeled fixtures and never represent measured qd/support.
 for mode in ['synthetic_quiet_and_load','no_quiet','no_load']:
  prev=C[0].copy();applied=[prev.copy()];desired=[];goals=[];records=[];failure=None;quiet=0;released=False;finished=False
  for k in range(1,115):
   if k==100:
    hold_ticks=25 if mode!='no_quiet' else max(0,950-len(desired))
    for j in range(hold_ticks):
     u=prev[moving].copy();goal=prev.copy();new=prev.copy()
     for p,idx in enumerate(moving):new[idx]=step(prev[idx],u[p],lo[p],hi[p],cap[p])
     assert np.array_equal(new,prev)
     if mode!='no_quiet':quiet+=1
     released=quiet>=25;desired.append(u);goals.append(goal);applied.append(new.copy());records.append([k-1,1.,gaps[k-1],0.,quiet,released,True]);prev=new
    if not released:break
   if k==105 and mode=='no_load':break
   for j in range(1,ni[k-1]+1):
    f=j/int(ni[k-1]);goal=(C[k-1].astype(float)+f*(C[k].astype(float)-C[k-1].astype(float))).astype(np.float32);u=[]
    for p,idx in enumerate(moving):
     v=inverse_forward(prev[idx],goal[idx],lo[p],hi[p],cap[p])
     if v is None:failure=dict(knot=k,segment_tick=j,joint=m.names[idx],previous=float(prev[idx]),goal=float(goal[idx]),legal_forward_lower=float(step(prev[idx],lo[p],lo[p],hi[p],cap[p])),legal_forward_upper=float(step(prev[idx],hi[p],lo[p],hi[p],cap[p])));break
     u.append(v)
    if failure:break
    u=np.array(u,np.float32);new=prev.copy()
    for p,idx in enumerate(moving):new[idx]=step(prev[idx],u[p],lo[p],hi[p],cap[p])
    assert np.array_equal(new,goal);gap=gaps[k-1]+f*(gaps[k]-gaps[k-1]);ld=load[k-1]+f*(load[k]-load[k-1]);desired.append(u);goals.append(goal);applied.append(new.copy());records.append([k,f,gap,ld,quiet,released,False]);prev=new
   if failure:break
   finished=k==114
  row=dict(mode=mode,command_ticks=len(desired),duration_s=.02*len(desired),complete=finished,time_gate=bool(finished and .02*len(desired)<=13.5),lattice_failure=failure,quiet_samples=quiet,released=released,quiet_inputs_synthetic=True,no_measured_support=True,physical_admission=False)
  rows.append(row);np.savez_compressed(O/(mode+'.npz'),applied=applied,desired=desired,goals=goals,records=records,moving=moving,lo=lo,hi=hi,cap=cap)
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,segment_ticks=ni.tolist(),rounding_rule='Each segment independently rounded upward to20ms;25extra quiet fixture samples. No fitted/model-predicted velocity.',source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),D/'guarded_path_351/results.json',D/'guarded_path_audit_352/independent_audit.json']},physics_ticks=0,physical_admission=False)
 (O/'results.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
