"""Conditional ALL675-tick nominal command construction for separately optimized379.
No physics or actual raw-speed/support evidence. Synthetic quiet/load permissions only.
Uses actual float32 global filter; reports nearest-achievable output errors explicitly.
Dense nominal geometry/current-input/parent/balance/bias/both-placement gates remain.
"""
from pathlib import Path
import numpy as np,json,hashlib,datetime
from humanoid_sitting.shared_frame import BodyModel
from low_seat_forward_lattice_377 import target_for_goal,forward
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'discrete_schedule_381'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 r=read(D/'discrete_path_379/results.json');audit=read(D/'discrete_path_audit_380/independent_audit.json');z=r['dense'];assert audit['passed'];assert z['model_geometry_passed'] and z['model_command_passed'] and r['optimizer']['numerical_constraints_passed'];assert r['discrete']['integer_duration_s']==13.5
 fin=dict(np.load(D/'discrete_path_379/final.npz'));assert fin['slack']==0;assert np.diff(np.vstack([[0.,0.],fin['x'][:1026].reshape(114,9)[:103,:2],[1.,1.]]),axis=0).min()>=0
 O.mkdir(exist_ok=False);fixture=dict(np.load(D/'discrete_preparation_audit_378/filter_fixtures.npz'));lo=fixture['lo'];hi=fixture['hi'];ulp=fixture['ulp'];ids=fixture['moving'];model=BodyModel(D/'cooked_model_50/model.json');C=np.array([e['command'] for e in z['command_nodes']]);ticks=np.array(r['discrete']['segment_ticks'],int);assert sum(ticks)==650 and sum(ticks[99:104])==101;rates=np.array([.1125 if model.names[j].endswith(('ankle_pitch_joint','ankle_roll_joint')) else .15 for j in ids]);gaps=np.array([e['pelvis_gap_m'] for e in z['path']]);loads=np.array([e['assumed_seat_normal_fraction'] for e in z['command_nodes']]);rows=[]
 for mode in ['synthetic_quiet_and_load','no_quiet','no_load']:
  prev=C[0].astype(np.float32);applied=[prev.copy()];desired=[];goals=[];records=[];failure=None;quiet=0;released=False;complete=False
  def tick(goal,k,f,hold):
   nonlocal prev,failure
   tar=prev.copy();out=prev.copy()
   for col,j in enumerate(ids):
    result=target_for_goal(prev[j],goal[j],lo[col],hi[col])
    if result is None:
     failure=dict(reason='goal outside legal forward image',knot=int(k),fraction=float(f),joint=model.names[j],previous=float(prev[j]),goal=float(goal[j]));return False
    tar[j],out[j]=result
    if abs(float(out[j])-float(goal[j]))>float(ulp[col]):
     failure=dict(reason='nearest-output departure exceeds declared oneULP bound',knot=int(k),fraction=float(f),joint=model.names[j]);return False
    if abs(float(out[j])-float(prev[j]))>rates[col]*.02:
     failure=dict(reason='actual reserved increment exceeded',knot=int(k),fraction=float(f),joint=model.names[j]);return False
   desired.append(tar);goals.append(goal);applied.append(out.copy());records.append([k,f,hold,quiet,released]);prev=out;return True
  for k in range(1,115):
   if k==100:
    n=25 if mode!='no_quiet' else 675-len(desired)
    for _ in range(n):
     if mode!='no_quiet':quiet+=1
     released=quiet>=25
     if not tick(prev.copy(),99,1.,True):break
    if failure or not released:break
   if k==105 and mode=='no_load':
    while len(desired)<675:
     if not tick(prev.copy(),104,1.,True):break
    break
   for j in range(1,ticks[k-1]+1):
    f=j/int(ticks[k-1]);goal=(C[k-1]+f*(C[k]-C[k-1])).astype(np.float32)
    if not tick(goal,k,f,False):break
   if failure:break
   complete=k==114
  A=np.array(applied);G=np.array(goals);maxerr=float(abs(A[1:].astype(float)-G.astype(float)).max()) if len(G) else None
  outcome=dict(mode=mode,command_ticks=len(desired),duration_s=len(desired)*.02,complete=complete,integer_time_gate=bool(complete and len(desired)<=675),quiet_samples=quiet,released=released,failure=failure,maximum_goal_departure_rad=maxerr,exact_goal_records=int(np.all(A[1:]==G,axis=1).sum()) if len(G) else 0,synthetic_permissions=True,actual_current_inputs=False,physical_admission=False);rows.append(outcome)
  np.savez_compressed(O/(mode+'.npz'),applied=A,target=np.array(desired),goal=G,records=np.array(records),moving=ids,lo=lo,hi=hi,rates=rates,ulp=ulp,C=C,segment_ticks=ticks,gaps=gaps,assumed_loads=loads)
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,rows=rows,all_actual_filter_records_built=True,actual_physical_ticks=0,physical_admission=False,repair_complete=False,source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),Path('scripts/low_seat_forward_lattice_377.py'),D/'discrete_path_379/results.json',D/'discrete_path_audit_380/independent_audit.json',D/'discrete_preparation_audit_378/filter_fixtures.npz']});(O/'results.json').write_text(json.dumps(r,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
