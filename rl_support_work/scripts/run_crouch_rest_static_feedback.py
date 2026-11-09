"""Add measured balance feedback to the staged static-load reference."""
import argparse,sys,json
from pathlib import Path
import torch
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser(add_help=False);p.add_argument('--feedback-x',type=float,default=1.5);p.add_argument('--feedback-pitch',type=float,default=.6);p.add_argument('--feedback-gyro',type=float,default=.15)
feedback_args,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_crouch_rest_statics as static
probe=static.probe

class FeedbackEnv(static.StaticEnv):
    def __init__(self,raw,*args,**kwargs):
        self.target_paths=torch.zeros(raw.num_envs,25,2,device=raw.device)
        self.target_ready=torch.zeros(raw.num_envs,dtype=torch.bool,device=raw.device);self.target_cursor=0
        super().__init__(raw,*args,**kwargs)
    def _observe(self,force=False):
        super()._observe(force)
        for a in self.plan_audit[self.target_cursor:]:
            path=a['root_path'];self.target_paths[a['case']]=torch.tensor([[r[0],Rotation.from_quat(r[3:]).as_euler('xyz')[1]] for r in path],device=self.device);self.target_ready[a['case']]=True
        self.target_cursor=len(self.plan_audit);self._augment()
    def reference(self):
        ref=super().reference()
        if not self.planning_enabled:return ref
        elapsed=(self.episode_length_buf-self.near_start).float()*.02
        u=(elapsed/probe.probe_args.transfer_seconds*24).clamp(0,24);lo=u.floor().long().clamp_max(23);f=u-lo;ids=torch.arange(self.num_envs,device=self.device)
        target=(1-f[:,None])*self.target_paths[ids,lo]+f[:,None]*self.target_paths[ids,lo+1]
        velocity=(self.target_paths[ids,lo+1,1]-self.target_paths[ids,lo,1])*24/probe.probe_args.transfer_seconds*(u<24)
        g=self.robot.data.projected_gravity_b.torch;pitch=torch.atan2(g[:,0],-g[:,2])
        correction=feedback_args.feedback_pitch*(pitch-target[:,1])+feedback_args.feedback_gyro*(self.robot.data.root_ang_vel_b.torch[:,1]-velocity)
        if self.estimated_root_x is not None:correction+=feedback_args.feedback_x*(self.estimated_root_x-target[:,0])
        correction=correction.clamp(-.25,.25)*(elapsed/.2).clamp(0,1)*self.target_ready*(self.near_start>=0)
        ref[:,self.ankle_ids]+=correction[:,None]
        return ref
    def close(self):
        super().close();(self.out/'feedback_config.json').write_text(json.dumps(vars(feedback_args),indent=2));(self.out/'source'/'run_crouch_rest_static_feedback.py').write_text(Path(__file__).read_text())

probe.progressive.ProgressiveEnv=FeedbackEnv
if __name__=='__main__':probe.progressive.main()
