"""Private link/drive recording, not feedback. No public state getter is used.

The198 private joint recorder is retained. Additional descriptors have their own
storage. Every query checks all allocated timestamped public caches byte-for-byte.
The implicit launch wrapper records existing arguments, never replaces them.
"""
import numpy as np
import warp as wp
from low_seat_private_recorder_198 import PassiveJointRecorder,copy,tensor

class DriveRecorder(PassiveJointRecorder):
 def __init__(self,data,collection):
  super().__init__(data,collection);self.link_order=copy(data.body_ordering.user_to_backend).astype(int) if data.has_body_ordering else np.arange(data._root_view.max_links)
  assert np.array_equal(np.sort(self.link_order),np.arange(data._root_view.max_links))
  view=data._root_view;self.extra={};self.launches=[];self.cache_checks=[]
  for name,shape in [('link_transforms',(view.count,view.max_links,7)),('link_velocities',(view.count,view.max_links,6)),('dof_position_targets',(view.count,view.max_dofs)),('dof_velocity_targets',(view.count,view.max_dofs))]:
   a=wp.zeros(shape,dtype=wp.float32,device=self.cache[0].data.device);desc=view._frontend.get_tensor_desc(a)
   assert int(desc.data_address)==int(a.ptr)
   self.extra[name]=(a,desc)
 def snapshot(self):
  ans={}
  for name,obj in vars(self.data).items():
   if hasattr(obj,'timestamp') and hasattr(obj,'data') and isinstance(obj.data,wp.array):
    ans[name]=(float(obj.timestamp),int(obj.data.ptr),copy(obj.data))
  return ans
 def check_unchanged(self,before):
  after=self.snapshot();assert set(before)==set(after)
  for name,(timestamp,ptr,value) in before.items():
   assert timestamp==after[name][0] and ptr==after[name][1] and np.array_equal(value,after[name][2]),name
  return len(before)
 def capture(self,substep,time_s,command):
  before=self.snapshot();super().capture(substep,time_s,command)
  for name,(a,desc) in self.extra.items():
   assert int(desc.data_address)==int(a.ptr),name
   lo=int(a.ptr);hi=lo+a.size*4
   for _,(_,p,v) in before.items():assert hi<=p or p+v.nbytes<=lo,'Private descriptor overlaps public cache'
   assert getattr(self.data._root_view._backend,'get_'+name)(desc),name
   order=self.link_order if name.startswith('link_') else self.order
   self.records[-1][name]=copy(a)[:,order].copy()
  self.cache_checks.append(self.check_unchanged(before))
 def launch(self,original,collection,substep,time_s):
  # Capture precisely the buffers consumed at this call, including nonzero
  # velocity/feedforward targets if present. Do not infer them from later state.
  before=self.snapshot();inputs=self.executor.kernel_inputs
  names=['target_q','target_qd','target_effort','cached_q','cached_qd','kp','kd','limits']
  row=dict(substep_before=int(substep),t_before=float(time_s),**{n:copy(x) for n,x in zip(names,inputs[:8])})
  assert self.check_unchanged(before)==len(before)
  result=original(collection)
  row['computed_effort']=copy(collection._computed_effort);row['applied_effort']=copy(collection._applied_effort)
  self.launches.append(row);return result
 def metadata(self):
  out=super().metadata();out.update(body_user_to_backend=self.link_order.tolist(),additional_private_queries=list(self.extra),extra_backend_pointers=[int(x[0].ptr) for x in self.extra.values()],all_allocated_timestamped_caches_unchanged=True,cache_count_min=min(self.cache_checks,default=0),cache_count_max=max(self.cache_checks,default=0),implicit_launch_records=len(self.launches),input_replacement=False)
  return out
