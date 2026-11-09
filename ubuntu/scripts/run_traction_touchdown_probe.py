"""Remove the touchdown horizontal impulse by moving balance feedback to hips earlier."""
import argparse,sys,json
from pathlib import Path
import torch
p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--hip-ramp-seconds',type=float,default=.3)
cfg,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_traction_planner_probe_v2 as planned
shear=planned.shear
class TouchdownEnv(planned.PlannedEnv):
    def reference(self):
        ref=super().reference()
        if not self.planning_enabled:return ref
        elapsed=(self.episode_length_buf-self.near_start).float()*.02
        u=(elapsed/5*24).clamp(0,24);lo=u.floor().long().clamp_max(23);f=u-lo;ids=torch.arange(self.num_envs,device=self.device)
        target=(1-f[:,None])*self.target_paths[ids,lo]+f[:,None]*self.target_paths[ids,lo+1]
        velocity=(self.target_paths[ids,lo+1,1]-self.target_paths[ids,lo,1])*24/5*(u<24)
        g=self.robot.data.projected_gravity_b.torch;pitch=torch.atan2(g[:,0],-g[:,2])
        correction=.6*(pitch-target[:,1])+.15*(self.robot.data.root_ang_vel_b.torch[:,1]-velocity)
        if self.estimated_root_x is not None:correction+=self.estimated_root_x-target[:,0]
        correction=correction.clamp(-.25,.25)*(elapsed/.2).clamp(0,1)*self.target_ready*(self.near_start>=0)
        old=((elapsed-.5)/1).clamp(0,1);old=old.square()*(3-2*old)
        new=(elapsed/cfg.hip_ramp_seconds).clamp(0,1);new=new.square()*(3-2*new)
        delta=(new-old)*self.target_ready*(self.near_start>=0)
        ref[:,self.hip_pitch_ids]+=correction[:,None]*delta[:,None]
        ref[:,self.ankle_ids]-=correction[:,None]*delta[:,None]
        ref[:,self.ankle_ids]+=(-3*self.foot_pitch).clamp(-.15,.15)*delta[:,None]
        return ref
    def close(self):
        super().close();(self.out/'touchdown_config.json').write_text(json.dumps(vars(cfg),indent=2));(self.out/'source'/'run_traction_touchdown_probe.py').write_text(Path(__file__).read_text())
shear.progressive.ProgressiveEnv=TouchdownEnv
if __name__=='__main__':shear.progressive.main()
