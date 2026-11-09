"""Reference-only 30 s timing diagnostic, explicitly separate from the 20 s study.

The inherited driver executes its first1000 ticks. close completes500 more.
finish ignores the inherited timeout until1500; there are no resets or learning.
Window metrics are recomputed from every-tick values for the actual30 s horizon.
This harness must not be used for PPO.
"""
from pathlib import Path
import json
import numpy as np
import torch
import run_crouch_rest_audited as audit
probe=audit.probe

class LongProbe(audit.AuditedEnv):
    def __init__(self,raw,args,*rest,**kwargs):
        assert args.mode=='reference';self.window_ticks=[];super().__init__(raw,args,*rest,**kwargs)
    def finish(self,ids):
        ids=ids[self.episode_length_buf[ids]>=1500]
        if len(ids):super().finish(ids)
    def _physics_step(self,actions):
        result=super()._physics_step(actions);m,d,_=probe.progressive.support.metrics(self.raw)
        torque=self.robot.data.applied_torque.torch;total=torque.square().sum(1)
        normal=(m['force_w']*m['normal_w']).sum(-1);tangent=(m['force_w']-normal[...,None]*m['normal_w']).norm(dim=-1);loaded=m['force_w'].norm(dim=-1)>.01*m['bodyweight_N'][:,None]
        util=torch.where(loaded,tangent/(.6*normal.clamp_min(1.e-7)),torch.zeros_like(normal)).max(1).values
        self.window_ticks.append({k:v.cpu().numpy() for k,v in dict(total_squared_torque_Nm2=total,total_absolute_torque_Nm=torque.abs().sum(1),all_joint_rms_Nm=(total/37).sqrt(),leg_squared_torque_Nm2=torque[:,self.leg].square().sum(1),upper_squared_torque_Nm2=torque[:,self.upper].square().sum(1),support_load_bw=d['rest_normal_load_bw'],vertical_support_load_bw=d['rest_vertical_load_bw'],foot_load_bw=d['foot_load_bw'],max_friction_utilization=util).items()})
        return result
    def close(self):
        assert int(self.episode_length_buf.min())==1000 and len(self.episodes)==0
        with torch.inference_mode():
            for _ in range(100):self.step(torch.zeros(self.num_envs,23,device=self.device))
        assert int(self.episode_length_buf.min())==1500 and len(self.episodes)==self.num_envs
        arrays={k:np.array([x[k] for x in self.window_ticks]) for k in self.window_ticks[0]}
        for row in self.episodes:
            i=row['case']
            for label,start in [('metrics_whole_trial',0),('metrics_last10s',1000),('metrics_last5s',1250)]:row[label]={k:float(v[start:,i].mean()) for k,v in arrays.items()}
            row.update(late_ticks=500,last5_ticks=250)
        super().close()
        (self.out/'long_probe_contract.json').write_text(json.dumps(dict(reference_only=True,seconds=30,physics_ticks=1500,window_ticks=[1500,500,250],episodes_jsonl_metrics_superseded_by_recomputed_results=True),indent=2))
        np.savez_compressed(self.out/'window_metrics.npz',**arrays)
        (self.out/'source'/'run_crouch_rest_long_probe.py').write_text(Path(__file__).read_text())

probe.progressive.ProgressiveEnv=LongProbe
if __name__=='__main__':probe.progressive.main()
