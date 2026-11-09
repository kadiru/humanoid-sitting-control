"""Localize saved351 constraint violations by units and knot; no optimization or physics."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'guarded_path_failure_354'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 r=read(D/'guarded_path_351/results.json');a=read(D/'guarded_path_audit_352/independent_audit.json');assert a['passed'] and not r['optimizer']['numerical_constraints_passed'];O.mkdir(exist_ok=False)
 z=r['dense'];v=np.load(D/'guarded_path_351/final.npz');eq=v['eq'];iq=v['ineq'];X=v['x'][:114*21].reshape(114,21);dt=v['x'][114*21:];assert eq.shape==(1443,) and iq.shape==(16876,)
 meta=read(D/'cooked_model_50/model.json');from humanoid_sitting.shared_frame import BodyModel
 m=BodyModel(D/'cooked_model_50/model.json');moving=[m.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]+[m.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['shoulder_pitch','elbow_pitch']]
 groups=[];records=[]
 def group(name,arr,unit,kind='ineq',start=1,labels=None):
  arr=np.asarray(arr);arr=arr[:,None] if arr.ndim==1 else arr;violation=abs(arr) if kind=='eq' else np.maximum(-arr,0);idx=np.unravel_index(np.argmax(violation),arr.shape);groups.append(dict(group=name,units=unit,records=arr.size,violations_over_existing_1e_8_arithmetic_bound=int((violation>1e-8).sum()),maximum_violation=float(violation[idx]),worst_knot=int(start+idx[0]),component=labels[idx[1]] if labels else int(idx[1]),original_criterion='equality zero' if kind=='eq' else 'inequality nonnegative'))
  for i,j in zip(*np.where(violation>1e-8)):records.append(dict(group=name,knot=int(start+i),component=labels[j] if labels else int(j),value=float(arr[i,j]),units=unit))
 E=eq[:1442].reshape(103,14)
 for name,sl,unit in [('left_foot_position',slice(0,3),'m'),('left_orientation_equation',slice(3,6),'0.2*sin(angle)'),('right_foot_position',slice(6,9),'m'),('right_orientation_equation',slice(9,12),'0.2*sin(angle)'),('COMxy',slice(12,14),'m')]:group(name,E[:,sl],unit,'eq')
 group('hover_gap',eq[-1:],'m','eq',99)
 L=iq[:9120].reshape(114,80);jl=[m.names[j] for j in moving]
 for name,sl,unit,lab in [('command_lower',slice(0,16),'rad',jl),('command_upper',slice(16,32),'rad',jl),('effort_positive',slice(32,48),'Nm',jl),('effort_negative',slice(48,64),'Nm',jl),('corner_nonnegative',slice(64,72),'BW',None),('corner_upper',slice(72,80),'BW',None)]:group(name,L[:,sl],unit,labels=lab)
 off=9120;group('phase_monotonic',iq[off:off+208].reshape(104,2),'fraction',labels=['descent','orientation']);off+=208
 for name,unit in [('rate_increasing','rad per segment'),('rate_decreasing','rad per segment'),('desired_headroom_upper','rad*s'),('desired_headroom_lower','rad*s')]:group(name,iq[off:off+1824].reshape(114,16),unit,labels=jl);off+=1824
 for name,count,start in [('before_hover_gap',98,1),('after_hover_gap',5,100),('nonnegative_gap',104,1),('approach_gap_monotonic',39,66),('normal_descent',5,100)]:group(name,iq[off:off+count],'m',start=start);off+=count
 group('total_time',iq[off:],'s',start=114);assert off+1==len(iq)
 # Explicitly distinguish an infeasible objective from a feasible schedule or lower bound.
 raw=[];cmd=np.array([n['command'] for n in z['command_nodes']])[:,moving];ini=np.load(D/'guarded_path_351/initial.npz');enddesired=cmd[1:]+.18*np.diff(cmd,axis=0)/dt[:,None]
 for k,j in zip(*np.where((enddesired<ini['command_lo'])|(enddesired>ini['command_hi']))):raw.append(dict(knot=int(k+1),joint=jl[j],desired=float(enddesired[k,j]),lower=float(ini['command_lo'][j]),upper=float(ini['command_hi'][j]),duration_s=float(dt[k])))
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,groups=groups,violating_records=records,linear_ramp_target_violations=raw,objective_is_not_feasible_schedule=True,objective_is_not_lower_bound=True,global_infeasibility_not_established=True,source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),D/'guarded_path_351/final.npz',D/'guarded_path_351/results.json',D/'guarded_path_audit_352/independent_audit.json']},physics_ticks=0,policy_optimizer_updates=0,physical_admission=False)
 assert sum(g['records'] for g in groups)==len(eq)+len(iq)
 (O/'results.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(dict(groups=groups,linear_ramp_target_violations=len(raw),physics_ticks=0)),flush=True)
if __name__=='__main__':main()
