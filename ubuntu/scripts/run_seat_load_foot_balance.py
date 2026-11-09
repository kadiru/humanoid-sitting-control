"""Sensor-only foot orientation feedback for supported weight transfer."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import level_rotation
import run_seat_load_slow as slow

class FootBalancedUnload(slow.SlowUnload):
    def __init__(self,*args,**kwargs):
        self.foot_pitch=None;self.foot_pitch_rate=None
        super().__init__(*args,**kwargs)
        shutil.copy2(__file__,self.out/'foot_balance_source.py')
        slow.original.base.write_json(self.out/'foot_balance_manifest.json',dict(sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs='Encoders and IMU only. Each foot pitch from world-level FK; finite-difference pitch rate. Applied after10s; target floor-parallel foot orientation.',cap_rad=.15))
    def _observe(self,force=False):
        super()._observe(force)
        q=self.robot.data.joint_pos.torch.cpu().numpy();gravity=self.robot.data.projected_gravity_b.torch.cpu().numpy()
        pitches=[]
        for qi,gi in zip(q,gravity):
            fk=self.model.fk(qi);R=level_rotation(gi)
            pitches.append([Rotation.from_matrix(R@fk[s+'_ankle_roll_link'][:3,:3]).as_euler('xyz')[1] for s in ['left','right']])
        value=torch.tensor(np.array(pitches),device=self.device,dtype=torch.float32)
        if self.foot_pitch_rate is None:self.foot_pitch_rate=torch.zeros_like(value)
        if self.foot_pitch is not None:
            rate=((value-self.foot_pitch)/.02).clamp(-2,2)
            self.foot_pitch_rate=.8*self.foot_pitch_rate+.2*rate
            self.foot_pitch_rate[self.episode_length_buf==0]=0
        self.foot_pitch=value
        self._augment()
    def reference(self):
        ref=super().reference()
        if self.foot_pitch is None:return ref
        profiles=[self.raw.cfg.geometry_bank[i]['control_profile'] for i in self.index]
        kp=torch.tensor([p.get('foot_kp',1.) for p in profiles],device=self.device)[:,None]
        kd=torch.tensor([p.get('foot_kd',.08) for p in profiles],device=self.device)[:,None]
        ramp=((self.episode_length_buf*.02-10)/1.).clamp(0,1)[:,None]
        ref[:,self.ankle_ids]+=ramp*(-kp*self.foot_pitch-kd*self.foot_pitch_rate).clamp(-.15,.15)
        return ref

slow.original.SplitEnv=FootBalancedUnload
if __name__=='__main__':slow.original.main()
