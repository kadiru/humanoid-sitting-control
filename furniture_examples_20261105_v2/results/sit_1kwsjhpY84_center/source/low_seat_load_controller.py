"""Replace scheduled unloading with bounded sensor-estimated load transfer.

This is a new development candidate. Prior controllers remain unchanged.
"""
import json,shutil
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sagittal_static_bias import static_bias
from humanoid_sitting.shared_frame import apply,transform
import low_seat_contact_controller as landing

def plan(*args,inertial,config):
    commands,qs,a=landing.plan(*args,inertial=inertial,config=config)
    if not a.get('landing_repair_applied',False):return commands,qs,a
    model=args[0];n=np.asarray(a['normal']);d=a['offset'];zero=[];full=[];share=[]
    for q,root in zip(qs,a['root_path']):
        root=np.asarray(root);T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3])
        pelvis=apply(T,model.vertices['pelvis']);depth=pelvis@n+d
        x=float(pelvis[depth<depth.min()+.002,0].mean())
        b0,s0=static_bias(model,inertial,q,root,x,0.)
        b1,s1=static_bias(model,inertial,q,root,x,1.)
        zero.append(b0.tolist());full.append(b1.tolist());share.append(s1['predicted_seat_bw'])
    stiffness=np.array([200.,200.,160.]*2)
    a['load_feedback_basis']=dict(zero=zero,full=full,max_share=share,
        scheduled=[(np.asarray(s['torque'])/stiffness).tolist() for s in a['static_path']],
        scope='Static model basis; runtime blend uses torque/encoder/IMU foot-wrench estimate only')
    return commands,qs,a

def install(harness,config):
    landing.install(harness,config)
    bridge=harness.parent_harness.traction.bridge
    bridge.progressive.rolling_plan=lambda *a,**kw:plan(*a,**kw,inertial=bridge.inertial,config=config)
    Parent=harness.FeedbackEnv

    class LoadEnv(Parent):
        def __init__(self,*a,**kw):
            self.load_ready=False;self.load_records=[]
            super().__init__(*a,**kw)
            self.load_states={};self.load_cursor=0;self.load_tick=-1
            self.load_correction=torch.zeros(self.num_envs,37,device=self.device)
            self.load_ids=[self.model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
            self.load_ready=True

        def _reset_rows(self,ids):
            if self.load_ready:
                for i in ids.cpu().tolist():self.load_states.pop(i,None)
                self.load_correction[ids]=0;self.load_tick=-1
            return super()._reset_rows(ids)

        def reference(self):
            ref=super().reference()
            if not self.load_ready or not self.planning_enabled:return ref
            if self.t%5==0 and self.load_tick!=self.t:
                self.load_tick=self.t
                for a in self.plan_audit[self.load_cursor:]:
                    if 'load_feedback_basis' in a:
                        self.load_states[a['case']]=dict(start=a['tick'],fraction=0.,
                            **{k:np.asarray(v) for k,v in a['load_feedback_basis'].items() if k!='scope'})
                self.load_cursor=len(self.plan_audit)
                correction=np.zeros((self.num_envs,37));records=[]
                for i,state in self.load_states.items():
                    if self.warming[i] or self.near_start[i]<0:continue
                    elapsed=(int(self.episode_length_buf[i])-state['start'])*.02
                    u=np.clip(elapsed/5*24,0,24);lo=min(int(u),23);f=u-lo
                    interp=lambda key:(1-f)*state[key][lo]+f*state[key][lo+1]
                    max_share=float(interp('max_share'))
                    observed=float(1-self.filtered_wrench[i,:,1].sum()/self.wrench_observer.weight)
                    valid=bool(np.max(self.shear_records[-1]['condition'][i])<200)
                    desired=np.clip((observed-config['load_deadband'])/max(max_share,.05),0.,1.) if valid else 0.
                    state['fraction']+=float(np.clip(desired-state['fraction'],-config['load_fraction_step'],config['load_fraction_step']))
                    actual=interp('zero')+state['fraction']*(interp('full')-interp('zero'))
                    shift=np.clip(actual-interp('scheduled'),-config['load_command_cap'],config['load_command_cap'])
                    if elapsed<=0:shift[:]=0.
                    correction[i,self.load_ids]=shift
                    records.append(dict(case=i,observed_share=observed,observer_condition_valid=valid,model_max_share=max_share,
                        applied_fraction=state['fraction'],correction=shift.tolist()))
                self.load_correction=torch.tensor(correction,device=self.device,dtype=torch.float32)
                self.load_records.append(dict(tick=self.t,states=records))
            return ref+self.load_correction

        def close(self):
            super().close()
            (self.out/'load_feedback_audit.json').write_text(json.dumps(self.load_records))
            shutil.copy2(__file__,self.out/'source'/Path(__file__).name)

    harness.FeedbackEnv=LoadEnv
