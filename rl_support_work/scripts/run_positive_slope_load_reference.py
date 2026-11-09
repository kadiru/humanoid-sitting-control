"""Physical development probe with unchanged contact rules and extra labels."""
import argparse,sys,json,shutil
from pathlib import Path
import numpy as np
import torch
extra=argparse.ArgumentParser(add_help=False);extra.add_argument('--positive-config',required=True)
option,rest=extra.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_posture_episode_ppo as harness
from positive_slope_load_planner import plan
config=json.loads(Path(option.positive_config).read_text())
harness.traction.progressive.rolling_plan=lambda *a,**kw:plan(*a,**kw,inertial=harness.traction.bridge.inertial,**config)

class DiagnosticEnv(harness.EpisodeEnv):
    def __init__(self,raw,*args,**kwargs):
        self.load_ready=False;self.load_records=[];self.load_cursor=0
        self.load_paths=torch.zeros(raw.num_envs,25,3,6,device=raw.device)
        self.load_shares=torch.zeros(raw.num_envs,25,device=raw.device)
        self.load_enabled=torch.zeros(raw.num_envs,dtype=torch.bool,device=raw.device)
        super().__init__(raw,*args,**kwargs)
        self.load_ids=[self.model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
        self.load_ready=True
    def _observe(self,force=False):
        super()._observe(force)
        for a in self.plan_audit[self.load_cursor:]:
            i=a['case'];self.load_enabled[i]=a.get('positive_candidate_applied',False)
            if self.load_enabled[i]:
                bn=np.array(a['bias_noseat_path']);bf=np.array(a['bias_fullseat_path']);planned=np.array([x['torque'] for x in a['static_path']])/np.array([200,200,160]*2)
                self.load_paths[i]=torch.tensor(np.stack([bn,bf,planned],axis=1),device=self.device,dtype=torch.float32)
                self.load_shares[i]=torch.tensor(a['fullseat_share_path'],device=self.device,dtype=torch.float32)
        self.load_cursor=len(self.plan_audit)
    def _reset_rows(self,ids):
        self.load_enabled[ids]=False
        super()._reset_rows(ids)
    def reference(self):
        ref=super().reference()
        if not self.load_ready:return ref
        u=((self.episode_length_buf-self.near_start).float()*.02/5*24).clamp(0,24)
        lo=u.floor().long().clamp_max(23);f=u-lo;ids=torch.arange(self.num_envs,device=self.device)
        values=(1-f[:,None,None])*self.load_paths[ids,lo]+f[:,None,None]*self.load_paths[ids,lo+1]
        share=(1-f)*self.load_shares[ids,lo]+f*self.load_shares[ids,lo+1]
        actual=torch.tensor(1-self.filtered_wrench[:,:,1].sum(1)/self.wrench_observer.weight,device=self.device,dtype=torch.float32).clamp(0,.9)
        ratio=(actual/share.clamp_min(.05)).clamp(0,1)
        correction=(values[:,0]+(values[:,1]-values[:,0])*ratio[:,None]-values[:,2]).clamp(-.08,.08)
        enabled=self.load_enabled&(self.near_start>=0)&(self.warming==0)
        correction*=enabled[:,None]
        ref[:,self.load_ids]+=correction
        if self.t%5==0:self.load_records.append(dict(t=self.t*.02,estimated_seat_share=actual.cpu().numpy(),planned_max_share=share.cpu().numpy(),correction=correction.cpu().numpy()))
        return ref
    def _physics_step(self,actions):
        result=super()._physics_step(actions)
        m,d,f=harness.support.metrics(self.raw);r=self.raw.scene['robot'];positions=[]
        for sensor,body,rest in harness.support.CONTACTS:
            positions.append(self.raw.scene[sensor].data.contact_pos_w.torch[:,0,0].cpu().numpy())
        self.contact_records[-1].update(legal=d['contacts_legal'].cpu().numpy(),inside=m['inside_patch'].cpu().numpy(),
            contact_position=np.stack(positions,1)-self.raw.scene.env_origins.cpu().numpy()[:,None,:],
            unexpected_body_normal=self.raw.scene['all_contacts'].data.net_normal_forces_w_history.torch.norm(dim=-1).max(1).values.cpu().numpy())
        return result
    def close(self):
        names=list(self.raw.scene['all_contacts'].body_names)
        (self.out/'contact_body_names.json').write_text(json.dumps(names))
        super().close()
        np.savez_compressed(self.out/'load_feedback.npz',**{k:np.array([x[k] for x in self.load_records]) for k in self.load_records[0]})
        for name in ['run_positive_slope_load_reference.py','positive_slope_load_planner.py']:shutil.copy2(Path(__file__).with_name(name),self.out/'source'/name)
        (self.out/'positive_config.json').write_text(json.dumps(config,indent=2))
harness.EpisodeEnv=DiagnosticEnv
if __name__=='__main__':harness.main()
