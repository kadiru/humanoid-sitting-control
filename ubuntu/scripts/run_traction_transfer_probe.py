"""Bounded foot-shear admittance from encoders/torques/IMU; diagnostic only.

The original approach is unchanged. Contact truth is recorded after stepping,
never read by the reference or observer. No PPO runs in this probe entrypoint.
"""
import argparse,sys,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser(add_help=False);p.add_argument('--shear-gain',type=float,default=.04);p.add_argument('--shear-cap',type=float,default=.03);p.add_argument('--shear-start',type=float,default=10.5)
cfg,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_crouch_rest_audited_ppo as audited
from foot_wrench_observer import FootWrenchObserver
bridge=audited.bridge;progressive=bridge.progressive

class TractionEnv(audited.AuditedBridge):
    def __init__(self,raw,*args,**kwargs):
        self.shear_ready=False;self.shear_tick=-1;self.shear_records=[];self.contact_records=[]
        super().__init__(raw,*args,**kwargs)
        assert self.args.mode=='reference','No learning through a diagnostic probe'
        self.wrench_observer=FootWrenchObserver(self.model,bridge.inertial)
        self.shear_offset=np.zeros((self.num_envs,2));self.filtered_wrench=np.zeros((self.num_envs,2,3));self.shear_correction=torch.zeros(self.num_envs,37,device=self.device)
        self.sagittal_ids=np.array([[self.model.names.index(s+'_'+j+'_joint') for j in ['hip_pitch','knee','ankle_pitch']] for s in ['left','right']]);self.shear_ready=True
    def reference(self):
        ref=super().reference()
        if not self.shear_ready or not self.planning_enabled:return ref
        if self.shear_tick!=self.t and self.t%5==0:
            self.shear_tick=self.t
            q=self.robot.data.joint_pos.torch.cpu().numpy();torque=self.robot.data.applied_torque.torch.cpu().numpy();gravity=self.robot.data.projected_gravity_b.torch.cpu().numpy();time=self.motion_time().cpu().numpy();correction=np.zeros_like(q);conditions=[];est=[]
            for i in range(self.num_envs):
                w,condition,J=self.wrench_observer.estimate(q[i],torque[i],gravity[i],return_jacobian=True);est.append(w);conditions.append(condition)
                self.filtered_wrench[i]=.5*self.filtered_wrench[i]+.5*w
                seat_fraction=1-self.filtered_wrench[i,:,1].sum()/self.wrench_observer.weight
                if time[i]>=cfg.shear_start and seat_fraction>.05 and max(condition)<200:
                    error=self.filtered_wrench[i,:,0]/self.wrench_observer.weight
                    self.shear_offset[i]=np.clip(self.shear_offset[i]+cfg.shear_gain*.1*np.clip(error,-.2,.2),-cfg.shear_cap,cfg.shear_cap)
                for side in range(2):
                    if condition[side]<200:correction[i,self.sagittal_ids[side]]=np.clip(np.linalg.solve(J[side].T,[self.shear_offset[i,side],0,0]),-.12,.12)
            self.shear_correction=torch.tensor(correction,device=self.device,dtype=torch.float32)
            self.shear_records.append(dict(t=self.t*.02,wrench=np.array(est),filtered=self.filtered_wrench.copy(),offset=self.shear_offset.copy(),condition=np.array(conditions),correction=correction.copy()))
        return ref+self.shear_correction
    def _physics_step(self,actions):
        result=super()._physics_step(actions)
        m,d,f=progressive.support.metrics(self.raw)
        self.contact_records.append(dict(t=float(self.episode_length_buf[0])*.02,force=m['force_w'].cpu().numpy(),slip=m['slip_m_s'].cpu().numpy(),normal=m['normal_w'].cpu().numpy(),weight=m['bodyweight_N'].cpu().numpy(),wrong=d['wrong_force_N'].cpu().numpy(),no_slip=d['no_slip'].cpu().numpy(),traction=d['traction_ok'].cpu().numpy(),failure=m['failure'].cpu().numpy()))
        return result
    def close(self):
        super().close()
        for label,records in [('shear_observer',self.shear_records),('every_tick',self.contact_records)]:
            if records:np.savez_compressed(self.out/(label+'.npz'),**{k:np.array([x[k] for x in records]) for k in records[0]})
        (self.out/'traction_config.json').write_text(json.dumps(vars(cfg),indent=2))
        for name in ['run_traction_transfer_probe.py','foot_wrench_observer.py']:(self.out/'source'/name).write_text(Path(__file__).with_name(name).read_text())

progressive.ProgressiveEnv=TractionEnv
if __name__=='__main__':progressive.main()
