"""Versioned measured-balance experiment on the unchanged150 geometry path."""
from pathlib import Path
import json,shutil
import numpy as np
import torch
import low_seat_balanced_runtime_v2_148 as previous
from low_seat_balance_feedback_160 import BalanceFeedback

def install(harness,landing_config,config):
    previous.install(harness,landing_config,config);Parent=harness.FeedbackEnv
    class BalancedEnv(Parent):
        def __init__(self,*a,**kw):
            self.balance_ready=False;self.balance_records=[]
            super().__init__(*a,**kw);self.balance_ready=True
        def reference(self):
            if self.balance_ready:
                for state in self.fixed_states.values():
                    if 'balance_feedback' not in state:
                        state['engine'].cap=np.float32(.00225)
                        state['balance_feedback']=BalanceFeedback(self.cooked_model,state['plan'])
                        state['balance_tick']=self.t;state['balance_target']=state['target'].copy()
            ref=super().reference()
            if not self.balance_ready or not self.planning_enabled:return ref
            data=self.robot.data;q=data.joint_pos.torch.cpu().numpy();gravity=data.projected_gravity_b.torch.cpu().numpy();gyro=data.root_ang_vel_b.torch.cpu().numpy()
            for i,state in self.fixed_states.items():
                if 'balance_feedback' not in state or self.warming[i]:continue
                if state['balance_tick']!=self.t and self.fixed_observed_tick==self.t:
                    state['balance_tick']=self.t;f=state['balance_feedback']
                    condition=self.shear_records[-1]['condition'][i] if self.shear_records else [np.inf,np.inf]
                    target,a=f.update(q[i],gravity[i],gyro[i],state['engine'].position,state['target'],self.filtered_wrench[i],condition,self.wrench_observer.weight)
                    state['balance_target']=target
                    self.balance_records.append(dict(tick=self.t,case=i,q=q[i].tolist(),gravity=gravity[i].tolist(),gyro=gyro[i].tolist(),
                        geometry=self.geometry[i].tolist(),foot_wrench=self.filtered_wrench[i].tolist(),conditions=np.asarray(condition).tolist(),
                        weight=self.wrench_observer.weight,position=state['engine'].position,nominal_target=state['target'].tolist(),target=target.tolist(),feedback=a))
                ref[i]=torch.tensor(state['balance_target'],device=self.device,dtype=ref.dtype)
            return ref
        def close(self):
            super().close();(self.out/'balance_feedback_161.json').write_text(json.dumps(self.balance_records))
            (self.out/'balance_contract_161.json').write_text(json.dumps(dict(scope=__doc__,selected_only=True,initial_prefix_s=1.,
                change='Current measured .6/.15/1 pitch/gyro/root-x feedback, .06rad offset ceiling with modeled ankle torque/parent reserve. Nominal rate.1125 plus feedback.0375 within unchanged applied.15. Active through nominal-path pause.',
                unchanged='150 poses, force allocation, current views, measured guards, physical criteria/materials/filter/fingers/frozen parent. No fresh holdout or optimizer.',physical_admission=False),indent=2))
            for n in ['low_seat_balance_feedback_160.py','low_seat_balance_runtime_161.py']:
                shutil.copy2(Path(__file__).with_name(n),self.out/'source'/n)
    harness.FeedbackEnv=BalancedEnv
