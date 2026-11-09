"""Isolate command-rate headroom on the preserved, rejected joint17 path.

Explicit pre-training control revision: selected approach8<=t<15 has0.20rad/s
applied command cap instead of0.15. All planner/feedback/filter/material/physical
labels are unchanged. No claim that a larger command cap repairs contact safety.
"""
from pathlib import Path
import json,shutil
import numpy as np
import torch
import low_seat_joint_path_controller_17 as previous

def install(harness,config):
 previous.install(harness,config);Parent=harness.FeedbackEnv
 class TrackingHeadroomEnv(Parent):
  def __init__(self,*args,**kwargs):
   self.headroom_ready=False;self.headroom_records=[]
   super().__init__(*args,**kwargs)
   self.headroom_ready=True
  def _physics_step(self,actions):
   if not self.headroom_ready:return super()._physics_step(actions)
   time=self.motion_time().clone();active=self.warming==0
   selected=(self.repair_duration>5)&(time>=8.)&(time<15.)&active
   old=self.profile_values['command_step'].clone();new=old.clone()
   for i in selected.nonzero().flatten().tolist():new[int(self.index[i])]=.004
   cap=new[self.index]
   assert torch.equal(cap!=old[self.index],selected),'Shared profile unexpectedly changes an unselected row'
   before=self.previous.clone();self.profile_values['command_step']=new
   try:result=super()._physics_step(actions)
   finally:self.profile_values['command_step']=old
   self.headroom_records.append(dict(t=time.cpu().numpy(),active=active.cpu().numpy(),selected=selected.cpu().numpy(),cap=cap.cpu().numpy(),before=before.cpu().numpy(),after=self.previous.cpu().numpy().copy()))
   return result
  def close(self):
   super().close()
   records=self.headroom_records
   np.savez_compressed(self.out/'command_headroom_ticks_31.npz',**{k:np.stack([v[k] for v in records]) for k in records[0]})
   (self.out/'tracking_headroom_contract_31.json').write_text(json.dumps(dict(scope='Isolated pre-training applied command-rate change; planner/feedback/filter/parent and physical criteria unchanged.',start_s=8.,end_s=15.,selected_rate_rad_s=.20,previous_rate_rad_s=.15,selected_command_step_rad=.004,unchanged_filter=.1,first8_exact_required=True,unselected_controller_arithmetic_unchanged=True,unselected_full20_trace_exactness_measured_not_assumed=True,prior_joint17_failed_exact_and_physical_gates_unchanged=True,optimizer_updates=0),indent=2))
   shutil.copy2(__file__,self.out/'source'/Path(__file__).name)
 harness.FeedbackEnv=TrackingHeadroomEnv
