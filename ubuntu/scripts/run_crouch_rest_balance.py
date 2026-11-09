"""Transfer probe retaining encoder/IMU balance around the moving reference."""
import argparse,sys,json
from pathlib import Path
import torch

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--balance-kp',type=float,default=.6);p.add_argument('--balance-kd',type=float,default=.15)
p.add_argument('--balance-x',type=float,default=1.);p.add_argument('--balance-cap',type=float,default=.25)
balance_args,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_crouch_rest_handoff as hand
probe=hand.probe

class BalancedEnv(hand.HandoffEnv):
    def __init__(self,raw,*args,**kwargs):
        self.balance_meta=torch.zeros(raw.num_envs,5,device=raw.device);self.audit_cursor=0
        super().__init__(raw,*args,**kwargs)
    def _observe(self,force=False):
        super()._observe(force)
        for a in self.plan_audit[self.audit_cursor:]:
            self.balance_meta[a['case']]=torch.tensor([a['pitch'],a['origin'][0],a['pivot'][0],a['pivot'][2],a['rear_shift']],device=self.device)
        self.audit_cursor=len(self.plan_audit)
        self._augment()
    def reference(self):
        ref=super().reference()
        if not self.planning_enabled:return ref
        elapsed=(self.episode_length_buf-self.near_start).float()*.02
        u=(elapsed/probe.probe_args.transfer_seconds).clamp(0,1);alpha=u.square()*(3-2*u)
        start,x0,px,pz,rear=self.balance_meta.unbind(1)
        pitch_goal=start+(probe.probe_args.final_pitch-start)*alpha
        velocity_goal=(probe.probe_args.final_pitch-start)*6*u*(1-u)/probe.probe_args.transfer_seconds
        xgoal=x0+px*(start.cos()-pitch_goal.cos())+pz*(start.sin()-pitch_goal.sin())+rear*alpha
        gravity=self.robot.data.projected_gravity_b.torch;pitch=torch.atan2(gravity[:,0],-gravity[:,2])
        delta=balance_args.balance_kp*(pitch-pitch_goal)+balance_args.balance_kd*(self.robot.data.root_ang_vel_b.torch[:,1]-velocity_goal)
        if self.estimated_root_x is not None:delta+=balance_args.balance_x*(self.estimated_root_x-xgoal)
        delta=delta.clamp(-balance_args.balance_cap,balance_args.balance_cap)
        ramp=(elapsed/.3).clamp(0,1);ramp=ramp.square()*(3-2*ramp)
        delta*=ramp*(self.near_start>=0)
        ref[:,self.ankle_ids]+=delta[:,None]
        return ref
    def close(self):
        super().close();(self.out/'balance_config.json').write_text(json.dumps(vars(balance_args),indent=2));(self.out/'source'/'run_crouch_rest_balance.py').write_text(Path(__file__).read_text())

probe.progressive.ProgressiveEnv=BalancedEnv
if __name__=='__main__':probe.progressive.main()
