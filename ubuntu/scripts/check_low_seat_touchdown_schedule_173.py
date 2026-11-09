"""One bounded OFFLINE schedule design, unchanged164 pose/force polylines.

Reserve feedback rate only on its two ankle-pitch joints. Other moving joints
retain the existing applied.15rad/s ceiling. At12mm planned gap, hold0.5s in this
software fixture, then cap modeled normal approach at6mm/s. A future runtime
must require current measured settling, not trust this fixture. No simulation.
"""
from pathlib import Path
import json,datetime,hashlib
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel
from posture_episode_ppo import PosturePPO
from low_seat_gated_command_128 import GatedCommandPath
from low_seat_balance_feedback_160 import BalanceFeedback
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'touchdown_schedule_173'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Path173(GatedCommandPath):
 def __init__(self,plan,m):
  super().__init__(plan,m);self.cap=np.full(len(self.ids),np.float32(.003),np.float32)
  for j,n in enumerate(self.ids):
   if m.names[n] in ['left_ankle_pitch_joint','right_ankle_pitch_joint']:self.cap[j]=np.float32(.00225)
  self.gaps=np.array([x['pelvis_gap_m'] for x in self.path]);self.gaps[80:]=0.
  assert np.all(np.diff(self.gaps)<=1e-10)
  self.hover=float(np.interp(.012,self.gaps[:81][::-1],self.arc[:81][::-1]));self.hold_ticks=0;self.released=False
 def advance(self,allowed,load):
  prev=self.nominal.copy();start=self.position;limit=float(np.interp(float(np.clip(load,0,.1)),self.loads[80:],self.arc[80:]))
  if not self.released:
   if abs(start-self.hover)<1e-10:
    if self.hold_ticks<25:self.hold_ticks+=1;allowed=False
    else:self.released=True
   if not self.released:limit=min(limit,self.hover)
  if not allowed or start>=limit:
   self.last_target=prev.copy();self.last_info=dict(start=start,end=start,paused=True,hold=self.hold_ticks);return prev.copy()
  gap=float(np.interp(start,self.arc,self.gaps))
  def valid(s):
   goal=self.at(s);raw=prev[self.ids]+(goal[self.ids]-prev[self.ids])/self.alpha
   normal_ok=not self.released or gap-float(np.interp(s,self.arc,self.gaps))<=.006*.02+1e-14
   return bool(normal_ok and np.all(raw>=self.lo) and np.all(raw<=self.hi) and np.all(abs(self.alpha*(raw-prev[self.ids]))<=self.cap)),goal,raw
  edge=int(np.searchsorted(self.arc,start,side='right'));safe=start
  while edge<len(self.arc):
   p=min(float(self.arc[edge]),limit)
   if valid(p)[0]:
    safe=p
    if p==limit:break
    edge+=1
   else:
    lo,hi=safe,p
    for _ in range(45):
     mid=(lo+hi)/2
     if valid(mid)[0]:lo=mid
     else:hi=mid
    safe=lo;break
  assert safe>start+1e-12
  okay,goal,raw=valid(safe);assert okay
  target=prev.copy();target[self.ids]=raw;self.nominal[self.ids]=prev[self.ids]+np.clip(self.alpha*(np.clip(raw,self.lo,self.hi)-prev[self.ids]),-self.cap,self.cap)
  self.position=safe;self.last_target=target;self.last_info=dict(start=start,end=safe,paused=False,hold=self.hold_ticks);return target.copy()
def main():
 O.mkdir(exist_ok=False);torch.set_num_threads(2);m=BodyModel(D/'cooked_model_50/model.json');base=D/'normal_landing3s_06/candidate';mf=read(base/'manifest.json');cp=Path(mf['args']['checkpoint']);assert sha(cp)==mf['parent_checkpoint_sha256']
 fx=dict(np.load(base/'sensor_inputs.npz'));parent=PosturePPO(np.zeros(414),np.ones(414),0,'cuda:0');parent.load(cp)
 with torch.no_grad():act=parent.choose(torch.tensor(fx['parent_observation'][0],device='cuda:0'),False)[0]
 assert np.array_equal(act.cpu().numpy(),fx['parent_action'][0]);off=(.008*torch.tanh(act)).cpu().numpy()
 lim={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'};lower=torch.tensor([lim[n][0]+.03 for n in m.names[:23]],dtype=torch.float32);upper=torch.tensor([lim[n][1]-.03 for n in m.names[:23]],dtype=torch.float32)
 six=[m.names.index(s+'_'+j+'_joint') for j in ['hip_pitch','knee','ankle_pitch'] for s in ['left','right']];rows=[]
 for c in [0,1,6,7]:
  plan=read(D/f'balanced_force_paths_144/plan_{c}.json')['plan'];e=Path173(plan,m);f=BalanceFeedback(m,plan);Q=np.array([r['q'] for r in plan['path']]);angles=Rotation.from_quat(np.array([r['root'][3:] for r in plan['path']])).as_euler('xyz');previous=f.pitch[0];targets=[];positions=[];gaps=[];nominal=[]
  for tick in range(1000):
   nom=e.advance(True,.1);q=np.array([np.interp(e.position,e.arc,Q[:,j]) for j in range(37)]);abc=np.array([np.interp(e.position,e.arc,angles[:,j]) for j in range(3)]);rate=(abc[1]-previous)/.02;previous=abc[1];g=-Rotation.from_euler('xyz',abc).as_matrix().T[:,2]
   tar,rec=f.update(q,g,np.array([0.,rate,0.]),e.position,nom,np.array([[0.,50.,0.],[0.,50.,0.]]),[20.,20.],100.)
   targets.append(tar);positions.append(e.position);gaps.append(float(np.interp(e.position,e.arc,e.gaps)));nominal.append(e.nominal.copy())
  targets=np.array(targets);positions=np.array(positions);gaps=np.array(gaps);done=np.flatnonzero(positions==e.arc[-1]);duration=float((done[0]+1)*.02) if len(done) else None;arrays=[];clipped=[]
  for device in ['cpu','cuda:0']:
   cmd=torch.tensor(e.Q[0],device=device,dtype=torch.float32);out=[];nc=0
   for tick,target in enumerate(targets):
    ref=torch.tensor(target,device=device);parent_off=torch.zeros(37,device=device)
    if tick>=700:
     for j in range(3):parent_off[six[2*j:2*j+2]]=float(off[c,j])
    desired=ref[:23]+parent_off[:23];nc+=int(((desired<lower.to(device))|(desired>upper.to(device))).sum())
    cmd[:23]+=(.1*(desired.clamp(min=lower.to(device),max=upper.to(device))-cmd[:23])).clamp(-.003,.003);out.append(cmd.cpu().numpy().copy())
   arrays.append(np.array(out));clipped.append(nc)
  assert np.array_equal(*arrays) and not any(clipped);arr=arrays[0];before=np.vstack([e.Q[0].astype(np.float32),arr[:-1]]);assert np.all(arr<=before+np.float32(.003)) and np.all(arr>=before-np.float32(.003));assert np.array_equal(arr[:,23:],np.broadcast_to(e.Q[0,23:].astype(np.float32),arr[:,23:].shape))
  near=np.flatnonzero(positions[:-1]>=e.hover-1e-10);maxspeed=float(np.max(abs(np.diff(gaps)[near]))/.02)
  assert maxspeed<=.006+1e-10 and e.hold_ticks==25
  np.savez_compressed(O/f'schedule_{c}.npz',targets=targets,positions=positions,gaps=gaps,commands=arr,nominal=np.array(nominal))
  rows.append(dict(case=c,command_ticks=1000,duration_s=duration,time_gate_passed=duration is not None and duration<=13.5,planned_touchdown_speed_limit_m_s=.006,maximum_sampled_near_gap_speed_m_s=maxspeed,synthetic_hover_hold_s=.5,cpu_gpu_exact=True,clipped_targets=0,maximum_applied_step_rad=float(abs(arr-before).max()),unchanged_endpoint=True,physical_prediction=False));print(rows[-1],flush=True)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,rows=rows,all_time_gates_passed=all(r['time_gate_passed'] for r in rows),physics_ticks=0,optimizer_updates=0,runtime_installed=False,physical_admission=False,limitations='Same sampled model pose/force paths. No new between-knot geometry proof, actual force or dynamic validation. Settling and load permission are synthetic here; a future runtime must use current sensors and unchanged support guards. Two remaining geometry types still lack feasible paths. This is one chosen schedule, not a search or minimum-time proof.',source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),Path(__file__).with_name('low_seat_gated_command_128.py'),Path(__file__).with_name('low_seat_balance_feedback_160.py'),D/'balanced_force_paths_144/SHA256.json',cp]})
 (O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':main()
