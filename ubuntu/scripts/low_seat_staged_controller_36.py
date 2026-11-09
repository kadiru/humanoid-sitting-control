"""Pre-training staged approach with current-geometry descent pause.

Selected first1s remains exact. The new 1--15s measured/model path uses existing
feedback gains, correctly timestamped foot anchors, previous-.35rad arm offset,
.175rad/s nominal and .20rad/s applied bounds. Missing current geometry pauses
normal descent without advancing the reference or load-controller path clock.
Frozen parent timing and physical labels stay unchanged. No admission implied.
"""
from pathlib import Path
import inspect,textwrap,json,shutil
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import low_seat_normal_duration_36 as normal
from low_seat_staged_planner_36 import plan,eligible,measured,DURATION,START,LAND_START,LAND_END

def install(harness,config):
    normal.install(harness,config);bridge=harness.parent_harness.traction.bridge
    original=bridge.BridgeEnv.reference;src=textwrap.dedent(inspect.getsource(original))
    assert src.count('u=(elapsed/5.*24).clamp(0,24);')==1 and src.count('*24/5.*(u<24)')==1
    src=src.replace('u=(elapsed/5.*24).clamp(0,24);',"u=(elapsed/5.*24).clamp(0,24)\n    if hasattr(self,'repair_duration'):u=torch.where(self.repair_duration>5.,(elapsed/14.*24).clamp(0,24),u)\n    ")
    src=src.replace('*24/5.*(u<24)',"*24/5.*(u<24)\n    if hasattr(self,'repair_duration'):velocity=torch.where(self.repair_duration>5.,(self.target_paths[ids,lo+1,1]-self.target_paths[ids,lo,1])*24/14.*(u<24),velocity)")
    scope=dict(original.__globals__);exec(src,scope);bridge.BridgeEnv.reference=scope['reference']
    Parent=harness.FeedbackEnv
    class StagedEnv(Parent):
        def __init__(self,*a,**kw):
            self.staged_ready=False;self.staged_records=[]
            super().__init__(*a,**kw)
            self.repair_duration=torch.full((self.num_envs,),5.,device=self.device)
            self.staged_ready=True
        def _reset_rows(self,ids):
            if self.staged_ready:self.repair_duration[ids]=5.
            return super()._reset_rows(ids)
        def _observe(self,force=False):
            super()._observe(force)
            if not self.staged_ready or not self.planning_enabled:return
            pending=(self.near_start<0)&(self.warming==0)&(self.motion_time()>=START)&(self.motion_time()<START+.1)
            q=self.robot.data.joint_pos.torch.cpu().numpy();g=self.robot.data.projected_gravity_b.torch.cpu().numpy()
            for i in pending.nonzero().flatten().tolist():
                if not self.geometry[i,11] or self.geometry[i,12]>.5:continue
                origin=self.reset_library['root_pose'][self.index[i],int(self.blend[i])].copy();origin[0]=float(self.estimated_root_x[i])
                if not eligible(measured(self.model,q[i],g[i],self.geometry[i],origin,self.calibration)):continue
                c,qs,a=plan(self.model,q[i],g[i],self.geometry[i],origin,self.previous[i].cpu().numpy(),self.calibration,bridge.inertial,config)
                self.near_paths[i]=torch.tensor(c,device=self.device,dtype=torch.float32);self.near_q[i]=torch.tensor(qs,device=self.device,dtype=torch.float32)
                self.near_start[i]=self.episode_length_buf[i];self.repair_duration[i]=DURATION
                self.target_paths[i]=torch.tensor([[r[0],Rotation.from_quat(r[3:]).as_euler('xyz')[1]] for r in a['root_path']],device=self.device);self.target_ready[i]=True
                self.plan_audit.append(dict(case=i,attempt=int(self.attempt[i]),tick=int(self.episode_length_buf[i]),stage=int(self.blend[i]),**a))
            self._augment()
        def _physics_step(self,actions):
            if not self.staged_ready:return super()._physics_step(actions)
            time=self.motion_time().clone();active=self.warming==0
            selected=(self.repair_duration>5.)&active
            elapsed=(self.episode_length_buf-self.near_start).float()*.02
            geometry=self.geometry.copy()
            fresh=torch.tensor((geometry[:,11]>0)&(geometry[:,12]<=.5),device=self.device)
            paused=selected&(elapsed+.02>=LAND_START)&(elapsed<LAND_END)&~fresh
            start_before=self.near_start.clone();self.near_start[paused]+=1
            cap_selected=selected&(time>=START)&(time<15.)
            old=self.profile_values['command_step'].clone();new=old.clone()
            for i in cap_selected.nonzero().flatten().tolist():new[int(self.index[i])]=.004
            cap=new[self.index];assert torch.equal(cap!=old[self.index],cap_selected)
            before=self.previous.clone();self.profile_values['command_step']=new
            try:result=super()._physics_step(actions)
            finally:self.profile_values['command_step']=old
            self.staged_records.append(dict(t=time.cpu().numpy(),active=active.cpu().numpy(),selected=selected.cpu().numpy(),cap_selected=cap_selected.cpu().numpy(),paused=paused.cpu().numpy(),geometry=geometry,elapsed_before=elapsed.cpu().numpy(),start_before=start_before.cpu().numpy(),start_after=self.near_start.cpu().numpy().copy(),cap=cap.cpu().numpy(),before=before.cpu().numpy(),after=self.previous.cpu().numpy().copy()))
            return result
        def close(self):
            super().close();rr=self.staged_records
            np.savez_compressed(self.out/'staged_ticks_36.npz',**{k:np.stack([r[k] for r in rr]) for k in rr[0]})
            sources=['low_seat_staged_planner_36.py',Path(__file__).name,'low_seat_normal_duration_36.py','low_seat_load_duration_36.py','low_seat_contact_duration_36.py']
            for n in sources:shutil.copy2(Path(__file__).with_name(n),self.out/'source'/n)
            (self.out/'staged_contract_36.json').write_text(json.dumps(dict(first_exact_seconds=1.,unselected_controller_source_unchanged=True,unselected_full_trace_exactness_measured_not_assumed=True,nominal_duration_s=14.,nominal_normal_descent=[START+LAND_START,START+LAND_END],pause='Current geometry invalid or age>0.5s pauses the same reference/load/landing path clock before and during normal descent; physics and outcome clocks continue. No earlier fit is exposed as a fresh measurement.',foot_anchors='Actual encoder/IMU frame at planning timestamp, not later callback.',selected_cap_rad_s=.20,ordinary_cap_rad_s=.15,selected_cap_window=[1.,15.],parent_timing='Frozen59117 unchanged11/15s, absolute clock; no early61111 pulse in isolating probe.',optimizer_updates=0,bridge_reference_original=inspect.getsource(original),bridge_reference_replacement=src),indent=2))
    harness.FeedbackEnv=StagedEnv
