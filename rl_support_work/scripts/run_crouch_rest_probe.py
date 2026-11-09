"""Isolated transfer probes; preserves the original approach and all old code."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
import run_progressive_seat_ppo as progressive
from crouch_rest_motion import continuous_plan

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--bias-delay',type=float,default=0.)
parser.add_argument('--rear-shift',type=float,default=0.)
parser.add_argument('--final-pitch',type=float,default=0.)
parser.add_argument('--gap-final',type=float,default=.001)
parser.add_argument('--transfer-seconds',type=float,default=5.)
parser.add_argument('--lock-transfer',action='store_true')
probe_args,remaining=parser.parse_known_args();sys.argv=[sys.argv[0]]+remaining

def plan(*args,**kwargs):
    c,q,a=continuous_plan(*args,**kwargs,bias_delay=probe_args.bias_delay,rear_shift=probe_args.rear_shift,final_pitch=probe_args.final_pitch,gap_final=probe_args.gap_final)
    if probe_args.lock_transfer:c[:]=c[0]
    return c,q,a

class ProbeEnv(progressive.ProgressiveEnv):
    def __init__(self,*args,**kwargs):
        self.contact_diagnostics=[];super().__init__(*args,**kwargs)
        (self.out/'probe_config.json').write_text(json.dumps(vars(probe_args),indent=2))
    def reference(self):
        # Preserve the exact original approach until the sensor-planned handoff.
        ref=progressive.original.SplitEnv.reference(self)
        if not self.planning_enabled:return ref
        near=self.near_start>=0
        u=((self.episode_length_buf-self.near_start).float()*progressive.DT/probe_args.transfer_seconds*24).clamp(0,24)
        lo=u.floor().long().clamp_max(23);f=u-lo;ids=torch.arange(self.num_envs,device=self.device)
        command=(1-f[:,None])*self.near_paths[ids,lo]+f[:,None]*self.near_paths[ids,lo+1]
        return torch.where(near[:,None],command,ref)
    def _physics_step(self,actions):
        # No untrained residual is allowed to modify the successful approach.
        actions=actions.clone();actions[self.motion_time()<10]=0
        result=super()._physics_step(actions)
        tick=int(self.episode_length_buf[0]);start=0 if self.args.curriculum_stage>=3 else 500
        if start<tick<=start+300 and tick%2==0:
            sensor=self.raw.scene['all_contacts'];forces=sensor.data.net_normal_forces_w_history.torch.norm(dim=-1).max(1).values
            m,d,flags=progressive.support.metrics(self.raw)
            self.contact_diagnostics.append(dict(tick=tick,body_names=sensor.body_names,forces=forces.cpu().tolist(),foot_force=m['force_w'][:,5:].cpu().tolist(),slip=d['max_slip_m_s'].cpu().tolist(),traction=d['traction_ok'].cpu().tolist(),wrong=d['wrong_force_N'].cpu().tolist(),pitch=torch.atan2(self.robot.data.projected_gravity_b.torch[:,0],-self.robot.data.projected_gravity_b.torch[:,2]).cpu().tolist()))
        return result
    def get_observations(self):
        obs=super().get_observations();obs['learn']*=((self.motion_time()>=10)&(self.near_start>=0)).float()[:,None];return obs
    def close(self):
        super().close();(self.out/'contact_diagnostics.json').write_text(json.dumps(self.contact_diagnostics))
        for name in ['run_crouch_rest_probe.py','crouch_rest_motion.py']:
            p=Path(__file__).with_name(name);(self.out/'source'/name).write_text(p.read_text())

progressive.rolling_plan=plan
progressive.ProgressiveEnv=ProbeEnv
if __name__=='__main__':progressive.main()
