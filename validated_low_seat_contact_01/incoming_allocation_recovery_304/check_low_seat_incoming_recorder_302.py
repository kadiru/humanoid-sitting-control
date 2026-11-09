"""CPU/GPU private wrench/property fixtures, zero simulation."""
from pathlib import Path
from types import SimpleNamespace as NS
import datetime,hashlib,json
import numpy as np
import torch,warp as wp
from low_seat_incoming_recorder_301 import IncomingRecorder
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'incoming_recorder_preparation_302'
def fixture(device,ordered,mode):
 q=torch.arange(12,dtype=torch.float32,device=device).reshape(3,4);qd=q/10;pubq=torch.full_like(q,-2);pubv=torch.full_like(q,-3);pose=torch.arange(63,dtype=torch.float32,device=device).reshape(3,3,7);vel=torch.arange(54,dtype=torch.float32,device=device).reshape(3,3,6);wrench=vel+1000
 data=NS(has_joint_ordering=ordered,has_body_ordering=ordered,joint_ordering=NS(user_to_backend=wp.from_torch(torch.tensor([2,0,3,1],dtype=torch.int32,device=device))),body_ordering=NS(user_to_backend=wp.from_torch(torch.tensor([2,0,1],dtype=torch.int32,device=device))),_joint_pos=NS(data=wp.from_torch(pubq),timestamp=3.),_joint_vel=NS(data=wp.from_torch(pubv),timestamp=3.),_body=NS(data=wp.from_torch(pose.clone()),timestamp=2.),_joint_stiffness=wp.from_torch(torch.full_like(q,160.)))
 def desc(x):return NS(data_address=int(x.ptr),tensor=x)
 def get(x,which=''):
  def f(d):
   if mode=='unavailable' and which=='wrench':return False
   if mode=='mutate_body' and which=='wrench':wp.to_torch(data._body.data)[0,0,0]+=1
   if mode=='mutate_property' and which=='dof_stiffnesses':wp.to_torch(data._joint_stiffness)[0,0]+=1
   wp.copy(d.tensor,wp.from_torch(x));return True
  return f
 back=NS(get_dof_positions=get(q),get_dof_velocities=get(qd),get_link_transforms=get(pose),get_link_velocities=get(vel),get_dof_position_targets=get(q+.1),get_dof_velocity_targets=get(torch.zeros_like(q)),get_link_incoming_joint_force=get(wrench,'wrench'))
 prop={n:q+(j+1)*100 for j,n in enumerate(['dof_stiffnesses','dof_dampings','dof_armatures','dof_max_forces'])};prop['dof_friction_properties']=torch.stack([q+1,q+2,q+3],dim=-1)
 for n,x in prop.items():setattr(back,'get_'+n,get(x,n))
 data._root_view=NS(count=3,max_dofs=4,max_links=3,_param_device='cpu',_frontend=NS(get_tensor_desc=desc),_backend=back,_existing_property=wp.from_torch(q.cpu().clone()))
 inp=[wp.from_torch(q+.1),wp.from_torch(torch.zeros_like(q)),wp.from_torch(torch.zeros_like(q)),data._joint_pos.data,data._joint_vel.data]+[wp.from_torch(torch.full_like(q,x)) for x in [160.,4.,20.]]
 col=NS(_implicit_executor=NS(kernel_inputs=inp),_applied_effort=wp.from_torch(torch.zeros_like(q)),_computed_effort=wp.from_torch(torch.zeros_like(q)));rec=IncomingRecorder(data,col)
 if mode=='bad_wrench_descriptor':
  a,d=rec.extra['link_incoming_joint_force'];d.data_address=int(data._body.data.ptr);d.tensor=data._body.data
 if mode=='bad_property_descriptor':
  a,d=rec.properties['dof_stiffnesses'];d.data_address=int(data._root_view._existing_property.ptr);d.tensor=data._root_view._existing_property
 caught=False
 try:rec.preflight();rec.capture(1600,8.,torch.ones_like(q));rec.capture(2600,13.,torch.ones_like(q))
 except AssertionError:caught=True
 if mode!='good':assert caught;return dict(device=device,ordered=ordered,mode=mode,passed=True,rejected=True)
 assert not caught;bo=[2,0,1] if ordered else [0,1,2];jo=[2,0,3,1] if ordered else [0,1,2,3]
 assert np.array_equal(rec.preflight_record,wrench.cpu().numpy()[:,bo]);assert len(rec.records)==2
 for r in rec.records:assert np.array_equal(r['link_incoming_joint_force'],wrench.cpu().numpy()[:,bo]) and np.array_equal(r['q'],q.cpu().numpy()[:,jo])
 assert [r['substep'] for r in rec.property_records]==[0,1600,2600]
 for r in rec.property_records:
  for n,x in prop.items():assert np.array_equal(r[n],x.cpu().numpy()[:,jo])
 assert rec.new_query_calls==18
 # Stored copies must not alias private backend destinations after later writes.
 wp.to_torch(rec.extra['link_incoming_joint_force'][0]).fill_(-123)
 assert np.array_equal(rec.records[0]['link_incoming_joint_force'],wrench.cpu().numpy()[:,bo])
 return dict(device=device,ordered=ordered,mode=mode,passed=True,metadata=rec.metadata())
def main():
 O.mkdir(exist_ok=False);wp.init();checks=[fixture(d,o,m) for d in ['cpu','cuda:1'] for o in [False,True] for m in ['good','mutate_body','mutate_property','bad_wrench_descriptor','bad_property_descriptor','unavailable']]
 files=[Path(__file__).resolve()]+[Path(__file__).with_name(n) for n in ['low_seat_incoming_recorder_301.py','low_seat_drive_recorder_244.py','low_seat_private_recorder_198.py']];out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,checks=checks,physics_ticks=0,optimizer_updates=0,physical_admission=False,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files});(O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({'results.json':hashlib.sha256((O/'results.json').read_bytes()).hexdigest()},indent=2));print(json.dumps(dict(passed=True,fixtures=len(checks),physics_ticks=0)),flush=True)
if __name__=='__main__':main()
