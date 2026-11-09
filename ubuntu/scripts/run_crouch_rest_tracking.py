"""Reference-only encoder integral compensation for changing support loads."""
import argparse,sys,json
from pathlib import Path
import torch

p=argparse.ArgumentParser(add_help=False);p.add_argument('--integral-gain',type=float,default=.5);p.add_argument('--integral-cap',type=float,default=.12)
track_args,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_crouch_rest_clearance as clear
probe=clear.probe

class TrackingEnv(clear.ClearanceEnv):
    def __init__(self,raw,*args,**kwargs):
        self.integral=torch.zeros(raw.num_envs,37,device=raw.device)
        self.integral_ids=[raw.scene['robot'].joint_names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
        self.integral_trace=[];super().__init__(raw,*args,**kwargs)
    def reference(self):
        return super().reference()+self.integral
    def _physics_step(self,actions):
        if self.planning_enabled:
            u=((self.episode_length_buf-self.near_start).float()*.02/probe.probe_args.transfer_seconds*24).clamp(0,24)
            lo=u.floor().long().clamp_max(23);f=u-lo;ids=torch.arange(self.num_envs,device=self.device)
            target=(1-f[:,None])*self.near_q[ids,lo]+f[:,None]*self.near_q[ids,lo+1]
            error=target-self.robot.data.joint_pos.torch
            active=(self.near_start>=0)&(self.warming==0)
            increment=track_args.integral_gain*.02*error[:,self.integral_ids].clamp(-.1,.1)*active[:,None]
            self.integral[:,self.integral_ids]=(self.integral[:,self.integral_ids]+increment).clamp(-track_args.integral_cap,track_args.integral_cap)
        result=super()._physics_step(actions)
        if self.t%5==0:self.integral_trace.append(self.integral.cpu().tolist())
        return result
    def close(self):
        super().close();(self.out/'tracking_config.json').write_text(json.dumps(vars(track_args),indent=2));(self.out/'integral_trace.json').write_text(json.dumps(self.integral_trace));(self.out/'source'/'run_crouch_rest_tracking.py').write_text(Path(__file__).read_text())

probe.progressive.ProgressiveEnv=TrackingEnv
if __name__=='__main__':probe.progressive.main()
