"""Standing-reset PPO around a frozen sensor-only traction-safe reference.

Reference choices are loaded from a preflight-approved configuration. Simulator
contact truth is recorded only for reward/evaluation, never supplied to control.
"""
import argparse,sys,json
from pathlib import Path
import numpy as np
import torch
from types import SimpleNamespace
p=argparse.ArgumentParser(add_help=False);p.add_argument('--traction-config',required=True)
option,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
config=json.loads(Path(option.traction_config).read_text());cfg=SimpleNamespace(**config['shear'])
import run_crouch_rest_audited_ppo as audited
from foot_wrench_observer import FootWrenchObserver
bridge=audited.bridge;progressive=bridge.progressive
from traction_clearance_planner_v2 import plan
from traction_feedback import hip_touchdown
progressive.rolling_plan=lambda *a,**kw:plan(*a,**kw,inertial=bridge.inertial,**config['planner'])

class TractionEnv(audited.AuditedBridge):
    def __init__(self,raw,*args,**kwargs):
        self.shear_ready=False;self.shear_tick=None;self.shear_records=[];self.contact_records=[]
        super().__init__(raw,*args,**kwargs)
        self.wrench_observer=FootWrenchObserver(self.model,bridge.inertial)
        self.shear_offset=np.zeros((self.num_envs,2));self.filtered_wrench=np.zeros((self.num_envs,2,3));self.shear_correction=torch.zeros(self.num_envs,37,device=self.device)
        self.sagittal_ids=np.array([[self.model.names.index(s+'_'+j+'_joint') for j in ['hip_pitch','knee','ankle_pitch']] for s in ['left','right']]);self.shear_ready=True
        self.shear_tick=np.full(self.num_envs,-1,dtype=int)
        reference=self.reference()
        obs=self.get_observations()
        np.savez_compressed(self.out/'traction_startup.npz',policy=obs['policy'].cpu().numpy(),reference=reference.cpu().numpy(),learn=obs['learn'].cpu().numpy())
    def reference(self):
        ref=super().reference()
        if not self.shear_ready or not self.planning_enabled:return ref
        if self.t%5==0 and np.any(self.shear_tick!=self.t):
            q=self.robot.data.joint_pos.torch.cpu().numpy();torque=self.robot.data.applied_torque.torch.cpu().numpy();gravity=self.robot.data.projected_gravity_b.torch.cpu().numpy();time=self.motion_time().cpu().numpy();correction=self.shear_correction.cpu().numpy().copy();conditions=[];est=[]
            for i in range(self.num_envs):
                w,condition,J=self.wrench_observer.estimate(q[i],torque[i],gravity[i],return_jacobian=True);est.append(w);conditions.append(condition)
                if self.shear_tick[i]==self.t:continue
                self.shear_tick[i]=self.t
                self.filtered_wrench[i]=.5*self.filtered_wrench[i]+.5*w
                seat_fraction=1-self.filtered_wrench[i,:,1].sum()/self.wrench_observer.weight
                if self.warming[i]==0 and time[i]>=cfg.shear_start and seat_fraction>.05 and max(condition)<200:
                    error=self.filtered_wrench[i,:,0]/self.wrench_observer.weight
                    self.shear_offset[i]=np.clip(self.shear_offset[i]+cfg.shear_gain*.1*np.clip(error,-.2,.2),-cfg.shear_cap,cfg.shear_cap)
                for side in range(2):
                    if condition[side]<200:correction[i,self.sagittal_ids[side]]=np.clip(np.linalg.solve(J[side].T,[self.shear_offset[i,side],0,0]),-.12,.12)
            self.shear_correction=torch.tensor(correction,device=self.device,dtype=torch.float32)
            self.shear_records.append(dict(t=self.t*.02,wrench=np.array(est),filtered=self.filtered_wrench.copy(),offset=self.shear_offset.copy(),condition=np.array(conditions),correction=correction.copy()))
        result=ref+self.shear_correction
        return hip_touchdown(self,result,config['hip_ramp_seconds']) if config.get('hip_ramp_seconds') else result
    def _physics_step(self,actions):
        result=super()._physics_step(actions)
        m,d,f=progressive.support.metrics(self.raw)
        self.contact_records.append(dict(t=float(self.episode_length_buf[0])*.02,force=m['force_w'].cpu().numpy(),slip=m['slip_m_s'].cpu().numpy(),normal=m['normal_w'].cpu().numpy(),weight=m['bodyweight_N'].cpu().numpy(),wrong=d['wrong_force_N'].cpu().numpy(),no_slip=d['no_slip'].cpu().numpy(),traction=d['traction_ok'].cpu().numpy(),failure=m['failure'].cpu().numpy()))
        return result
    def _reset_rows(self,ids):
        # Clear selected rows before inherited reset/observation callbacks.
        if self.shear_ready:
            ix=ids.cpu().numpy();self.shear_offset[ix]=0;self.filtered_wrench[ix]=0;self.shear_correction[ids]=0;self.shear_tick[ix]=-1
        super()._reset_rows(ids)
    def get_observations(self):
        obs=super().get_observations()
        if self.shear_ready:
            scale=np.array([self.wrench_observer.weight,self.wrench_observer.weight,self.wrench_observer.weight*.4])
            values=np.concatenate([(self.filtered_wrench/scale).reshape(self.num_envs,6),self.shear_offset/.03],axis=1)
            extra=torch.tensor(values,device=self.device,dtype=torch.float32).clamp(-5,5)
        else:extra=torch.zeros(self.num_envs,8,device=self.device)
        return progressive.base.TensorDict({'policy':torch.cat([obs['policy'],extra],1),'learn':obs['learn']},batch_size=[self.num_envs])
    def close(self):
        super().close()
        for label,records in [('shear_observer',self.shear_records),('every_tick',self.contact_records)]:
            if records:np.savez_compressed(self.out/(label+'.npz'),**{k:np.array([x[k] for x in records]) for k in records[0]})
        (self.out/'traction_config.json').write_text(json.dumps(config,indent=2))
        for name in ['run_traction_transfer_ppo.py','foot_wrench_observer.py','traction_clearance_planner_v2.py','traction_feedback.py']:(self.out/'source'/name).write_text(Path(__file__).with_name(name).read_text())
        for name in ['manifest.json','sensor_audit.json']:
            p=self.out/name;v=json.loads(p.read_text());v.update(observations=414,channels=414,traction_contract='406 inherited sensor/reference channels plus six estimated filtered sagittal foot-wrench channels and two bounded admittance offsets. No contact labels or live truth poses.');p.write_text(json.dumps(v,indent=2))
        (self.out/'traction_contract.json').write_text(json.dumps(dict(observations=414,actions=23,physics_ticks_per_decision=5,control_start_seconds=bridge.bridge_args.learning_start,all_surfaces_eligible=bridge.bridge_args.eligible_indices,approach='reference only through control-start time',configuration=config),indent=2))

progressive.ProgressiveEnv=TractionEnv
if __name__=='__main__':progressive.main()
