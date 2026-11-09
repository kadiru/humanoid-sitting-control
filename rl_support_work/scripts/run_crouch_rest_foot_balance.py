"""Separate seated hip balance from encoder/IMU foot-flatness feedback."""
import argparse,sys,json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import level_rotation

p=argparse.ArgumentParser(add_help=False);p.add_argument('--foot-gain',type=float,default=1.5);p.add_argument('--hip-balance-share',type=float,default=1.)
foot_args,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_crouch_rest_audited as audit
feedback=audit.feedback;static=feedback.static;probe=audit.probe

class FootBalanceEnv(audit.AuditedEnv):
    def __init__(self,raw,*args,**kwargs):
        self.foot_cached_tick=-1;self.foot_pitch=torch.zeros(raw.num_envs,2,device=raw.device);self.foot_records=[]
        self.hip_pitch_ids=[raw.scene['robot'].joint_names.index(s+'_hip_pitch_joint') for s in ['left','right']]
        super().__init__(raw,*args,**kwargs)
    def reference(self):
        ref=super().reference()
        if not self.planning_enabled:return ref
        if self.foot_cached_tick!=self.t:
            q=self.robot.data.joint_pos.torch.cpu().numpy();gravity=self.robot.data.projected_gravity_b.torch.cpu().numpy();values=[]
            for i in range(self.num_envs):
                R=level_rotation(gravity[i]);fk=self.model.fk(q[i]);values.append([Rotation.from_matrix(R@fk[s+'_ankle_roll_link'][:3,:3]).as_euler('xyz')[1] for s in ['left','right']])
            self.foot_pitch=torch.tensor(values,device=self.device,dtype=torch.float32);self.foot_cached_tick=self.t
            if self.t%5==0:self.foot_records.append(dict(t=self.t*.02,pitch=values))
        elapsed=(self.episode_length_buf-self.near_start).float()*.02
        ramp=((elapsed-static.static_args.settle_fraction*probe.probe_args.transfer_seconds)/1.).clamp(0,1);ramp=ramp.square()*(3-2*ramp)*self.target_ready*(self.near_start>=0)
        nominal=static.StaticEnv.reference(self)
        correction=(ref[:,self.ankle_ids]-nominal[:,self.ankle_ids])*ramp[:,None]*foot_args.hip_balance_share
        ref[:,self.ankle_ids]-=correction;ref[:,self.hip_pitch_ids]+=correction
        ref[:,self.ankle_ids]+=(-foot_args.foot_gain*self.foot_pitch).clamp(-.15,.15)*ramp[:,None]
        return ref
    def close(self):
        super().close();(self.out/'foot_feedback_config.json').write_text(json.dumps(vars(foot_args),indent=2));(self.out/'foot_pitch_observer.json').write_text(json.dumps(self.foot_records));(self.out/'source'/'run_crouch_rest_foot_balance.py').write_text(Path(__file__).read_text())

probe.progressive.ProgressiveEnv=FootBalanceEnv
if __name__=='__main__':probe.progressive.main()
