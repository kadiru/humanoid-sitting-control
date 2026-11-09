"""Time-reparameterized sensor-based transfer; first10s and physical rules unchanged."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import level_rotation
DURATION=20./3
def timed_reference(self):
    original=self._positive_original_split_reference()
    if not self.planning_enabled:return original
    elapsed=(self.episode_length_buf-self.near_start).float()*.02
    durations=torch.full((self.num_envs,),5.,device=self.device)
    for audit in self.plan_audit:
        if audit.get('positive_candidate_applied',False):durations[audit['case']]=DURATION
    u=(elapsed/durations*24).clamp(0,24);lo=u.floor().long().clamp_max(23);f=u-lo;ids=torch.arange(self.num_envs,device=self.device)
    path=(1-f[:,None])*self.near_paths[ids,lo]+f[:,None]*self.near_paths[ids,lo+1]
    nominal=torch.where((self.near_start>=0)[:,None],path,original);ref=nominal.clone()
    target=(1-f[:,None])*self.target_paths[ids,lo]+f[:,None]*self.target_paths[ids,lo+1]
    velocity=(self.target_paths[ids,lo+1,1]-self.target_paths[ids,lo,1])*24/durations*(u<24)
    g=self.robot.data.projected_gravity_b.torch;pitch=torch.atan2(g[:,0],-g[:,2])
    correction=.6*(pitch-target[:,1])+.15*(self.robot.data.root_ang_vel_b.torch[:,1]-velocity)
    if self.estimated_root_x is not None:correction+=self.estimated_root_x-target[:,0]
    correction=correction.clamp(-.25,.25)*(elapsed/.2).clamp(0,1)*self.target_ready*(self.near_start>=0)
    ref[:,self.ankle_ids]+=correction[:,None]
    if self.foot_cached_tick!=self.t:
        q=self.robot.data.joint_pos.torch.cpu().numpy();gravity=g.cpu().numpy();values=[]
        for i in range(self.num_envs):
            R=level_rotation(gravity[i]);fk=self.model.fk(q[i]);values.append([Rotation.from_matrix(R@fk[s+'_ankle_roll_link'][:3,:3]).as_euler('xyz')[1] for s in ['left','right']])
        self.foot_pitch=torch.tensor(values,device=self.device,dtype=torch.float32);self.foot_cached_tick=self.t
    ramp=((elapsed-.5)/1.).clamp(0,1);ramp=ramp.square()*(3-2*ramp)*self.target_ready*(self.near_start>=0)
    shifted=(ref[:,self.ankle_ids]-nominal[:,self.ankle_ids])*ramp[:,None]
    ref[:,self.ankle_ids]-=shifted;ref[:,self.hip_pitch_ids]+=shifted
    ref[:,self.ankle_ids]+=(-3.*self.foot_pitch).clamp(-.15,.15)*ramp[:,None]
    return ref
