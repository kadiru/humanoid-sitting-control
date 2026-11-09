"""Private incoming joint wrench/property readback; never a controller input."""
import numpy as np
import warp as wp
from low_seat_drive_recorder_244 import DriveRecorder
from low_seat_private_recorder_198 import copy

class IncomingRecorder(DriveRecorder):
 def __init__(self,data,collection):
  super().__init__(data,collection);v=data._root_view
  a=wp.zeros((v.count,v.max_links,6),dtype=wp.float32,device=self.cache[0].data.device);self.extra['link_incoming_joint_force']=(a,v._frontend.get_tensor_desc(a))
  self.properties={};self.property_records=[];self.property_checks=[];self.new_query_calls=0
  for name,tail in [('dof_stiffnesses',()),('dof_dampings',()),('dof_armatures',()),('dof_max_forces',()),('dof_friction_properties',(3,))]:
   a=wp.zeros((v.count,v.max_dofs)+tail,dtype=wp.float32,device=('cpu' if v._param_device==-1 else f'cuda:{v._param_device}'));self.properties[name]=(a,v._frontend.get_tensor_desc(a))
  self.preflight_record=None
 def buffers(self):
  ans={}
  for owner,label in [(self.data,'data'),(self.data._root_view,'view')]:
   for name,v in vars(owner).items():
    if isinstance(v,wp.array):ans[label+':'+name]=(str(v.device),int(v.ptr),copy(v))
  return ans
 def buffers_unchanged(self,before):
  after=self.buffers();assert set(before)==set(after)
  for k,(device,ptr,value) in before.items():
   assert device==after[k][0] and ptr==after[k][1] and value.shape==after[k][2].shape and value.dtype==after[k][2].dtype and value.tobytes()==after[k][2].tobytes(),k
  return len(before)
 def private_check(self,a,desc):
  assert int(desc.data_address)==int(a.ptr),'Private descriptor mismatch'
  device=str(a.device);lo=int(a.ptr);hi=lo+a.numpy().nbytes
  for d,p,v in self.buffers().values():
   if d==device:assert hi<=p or p+v.nbytes<=lo,'Private destination aliases public property/view buffer'
 def read_properties(self,substep):
  before=self.snapshot();raw=self.buffers();r=dict(substep=int(substep))
  for name,(a,d) in self.properties.items():
   self.private_check(a,d);assert getattr(self.data._root_view._backend,'get_'+name)(d),name;self.new_query_calls+=1
   x=copy(a);assert np.isfinite(x).all();r[name]=x[:,self.order].copy()
  self.check_unchanged(before);self.property_checks.append(self.buffers_unchanged(raw));self.property_records.append(r)
 def preflight(self):
  before=self.snapshot();raw=self.buffers();a,d=self.extra['link_incoming_joint_force'];self.private_check(a,d)
  assert self.data._root_view._backend.get_link_incoming_joint_force(d);self.new_query_calls+=1
  value=copy(a)[:,self.link_order].copy();assert np.isfinite(value).all();self.check_unchanged(before);self.property_checks.append(self.buffers_unchanged(raw));self.preflight_record=value
  self.read_properties(0)
 def capture(self,substep,time_s,command):
  raw=self.buffers();super().capture(substep,time_s,command);self.new_query_calls+=1
  self.property_checks.append(self.buffers_unchanged(raw));assert np.isfinite(self.records[-1]['link_incoming_joint_force']).all()
  if substep in [1600,2600]:self.read_properties(substep)
 def metadata(self):
  r=super().metadata();r.update(incoming_frame='Child incoming joint frame, force xyz then torque xyz; transmitted wrench, not isolated motor effort.',new_query_calls=self.new_query_calls,property_snapshots=len(self.property_records),property_read_steps=[r['substep'] for r in self.property_records],property_queries=list(self.properties),property_destination_devices={n:str(a.device) for n,(a,d) in self.properties.items()},property_buffer_checks=len(self.property_checks),all_existing_property_and_view_buffers_byte_exact=True,preflight_completed=self.preflight_record is not None)
  return r
