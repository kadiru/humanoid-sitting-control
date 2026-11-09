"""Synthetic private storage/link ordering/launch capture checks, zero physics."""
from pathlib import Path
from types import SimpleNamespace as NS
import datetime,hashlib,json
import numpy as np
import torch
import warp as wp
from low_seat_drive_recorder_244 import DriveRecorder
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'drive_recorder_preparation_245'
def fixture(device,ordered,mode):
 q=torch.arange(12,dtype=torch.float32,device=device).reshape(3,4);qd=q/10;pubq=torch.full_like(q,-2);pubv=torch.full_like(q,-3);pose=torch.arange(63,dtype=torch.float32,device=device).reshape(3,3,7);vel=torch.arange(54,dtype=torch.float32,device=device).reshape(3,3,6)
 data=NS(has_joint_ordering=ordered,has_body_ordering=ordered,joint_ordering=NS(user_to_backend=wp.from_torch(torch.tensor([2,0,3,1],dtype=torch.int32,device=device))),body_ordering=NS(user_to_backend=wp.from_torch(torch.tensor([2,0,1],dtype=torch.int32,device=device))),_joint_pos=NS(data=wp.from_torch(pubq),timestamp=3.),_joint_vel=NS(data=wp.from_torch(pubv),timestamp=3.),_body=NS(data=wp.from_torch(pose.clone()),timestamp=2.))
 def desc(x):return NS(data_address=int(x.ptr),tensor=x)
 def get(x,mutate=False):
  def f(d):
   if mutate:wp.to_torch(data._body.data)[0,0,0]+=1
   wp.copy(d.tensor,wp.from_torch(x));return True
  return f
 back=NS(get_dof_positions=get(q),get_dof_velocities=get(qd),get_link_transforms=get(pose,mode=='mutate_body'),get_link_velocities=get(vel),get_dof_position_targets=get(q+.1),get_dof_velocity_targets=get(torch.zeros_like(q)))
 data._root_view=NS(count=3,max_dofs=4,max_links=3,_frontend=NS(get_tensor_desc=desc),_backend=back)
 inputs=[wp.from_torch(q+.1),wp.from_torch(torch.zeros_like(q)),wp.from_torch(torch.zeros_like(q)),data._joint_pos.data,data._joint_vel.data]+[wp.from_torch(torch.full_like(q,x)) for x in [160.,4.,20.]]
 collection=NS(_implicit_executor=NS(kernel_inputs=inputs),_applied_effort=wp.from_torch(torch.zeros_like(q)),_computed_effort=wp.from_torch(torch.zeros_like(q)))
 rec=DriveRecorder(data,collection);caught=False
 if mode=='bad_descriptor':
  arr,de=rec.extra['link_transforms'];de.tensor=data._body.data;de.data_address=int(data._body.data.ptr)
 try:rec.capture(1600,8.,torch.ones_like(q))
 except AssertionError:caught=True
 if mode!='good':assert caught;return dict(device=device,ordered=ordered,mode=mode,passed=True)
 assert not caught
 order=[2,0,1] if ordered else [0,1,2];assert np.array_equal(rec.records[0]['link_transforms'],pose.cpu().numpy()[:,order]);assert np.array_equal(rec.records[0]['link_velocities'],vel.cpu().numpy()[:,order])
 def launch(c):
  a=[wp.to_torch(x) for x in inputs];v=a[5]*(a[0]-a[3])+a[6]*(a[1]-a[4])+a[2];wp.to_torch(c._computed_effort).copy_(v);wp.to_torch(c._applied_effort).copy_(v.clamp(-20,20));return 'original_return'
 assert rec.launch(launch,collection,1600,8.)=='original_return'
 assert np.array_equal(rec.launches[0]['cached_q'],pubq.cpu().numpy())
 assert np.array_equal(rec.launches[0]['applied_effort'],wp.to_torch(collection._applied_effort).cpu().numpy())
 return dict(device=device,ordered=ordered,mode=mode,passed=True)
def main():
 O.mkdir(exist_ok=False);wp.init();checks=[fixture(d,o,m) for d in ['cpu','cuda:1'] for o in [False,True] for m in ['good','mutate_body','bad_descriptor']]
 files=[Path(__file__).resolve(),Path(__file__).with_name('low_seat_drive_recorder_244.py'),Path(__file__).with_name('low_seat_private_recorder_198.py')];out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,checks=checks,physics_ticks=0,optimizer_updates=0,physical_admission=False,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
 (O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({'results.json':hashlib.sha256((O/'results.json').read_bytes()).hexdigest()},indent=2));print(json.dumps(out))
if __name__=='__main__':main()
