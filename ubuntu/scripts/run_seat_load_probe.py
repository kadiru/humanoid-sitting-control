"""New seat-load study: remove standing motor preload only after contact.

This is a development probe, not a modification of any frozen study. All
runtime geometric planning uses the existing measured plane and encoder FK.
"""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import run_contact_effort_repair as original
from humanoid_sitting.contact_aware_reference import measured_level_plane,front_contact_motion

class SeatLoadProbe(original.SplitEnv):
    def __init__(self,raw,args,out,goals,resets):
        super().__init__(raw,args,out,goals,resets)
        ends=[];qends=[];pends=[];xends=[];audit=[]
        calibration=json.loads(Path(args.plane_calibration).read_text())
        leg=[i for i,n in enumerate(self.model.names) if any(s in n for s in ['hip_','knee','ankle'])]
        for i,j in enumerate(self.index):
            profile=raw.cfg.geometry_bank[j]['control_profile']
            if profile.get('unload',False):
                n,d,o=measured_level_plane(self.model,self.robot.data.joint_pos.torch[i].cpu().numpy(),self.robot.data.projected_gravity_b.torch[i].cpu().numpy(),self.geometry[i],resets['root_pose'][j,0])
                q,c,r,check=front_contact_motion(self.model,resets['q'][j],resets['command'][j],resets['root_pose'][j],n,d,calibration,profile['unload_pitch'],profile['unload_x'],profile['unload_gap'])
                assert check['max_sagittal_residual_m']<.003,check
                c[-1,leg]=q[-1,leg]+profile.get('preload_scale',1.)*(c[-1,leg]-q[-1,leg])
                ends.append(c[-1]);qends.append(q[-1]);pends.append([Rotation.from_quat(r[-1,3:]).as_euler('xyz')[1]]);xends.append([r[-1,0]]);audit.append(dict(case=i,preload_scale=profile.get('preload_scale',1.),**check))
            else:
                ends.append(self.adapted_paths[i,-1].cpu().numpy());qends.append(self.adapted_qpaths[i,-1].cpu().numpy());pends.append(self.adapted_pitch[i,-1].cpu().numpy());xends.append(self.adapted_root_x[i,-1].cpu().numpy())
        for key,values in [('adapted_paths',ends),('adapted_qpaths',qends),('adapted_pitch',pends),('adapted_root_x',xends)]:
            value=getattr(self,key);setattr(self,key,torch.cat((value,torch.tensor(np.array(values),device=self.device,dtype=value.dtype)[:,None]),1))
        np.savez_compressed(self.out/'unloading_paths.npz',command=self.adapted_paths.cpu().numpy(),q=self.adapted_qpaths.cpu().numpy(),pitch=self.adapted_pitch.cpu().numpy(),root_x=self.adapted_root_x.cpu().numpy())
        shutil.copy2(__file__,out/'seat_load_source.py')
        original.base.write_json(out/'seat_load_manifest.json',dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plans=audit,scope='New development probe. No changes to prior controller before12s or physical criteria. Leg preload scale changes final commands; no live truth in controller.'))
        self._augment()
    def phase(self):
        t=self.episode_length_buf*original.DT
        return torch.where(t<=12,t/2,6+(t-12)/2).clamp_max(7)

original.SplitEnv=SeatLoadProbe
if __name__=='__main__':original.main()
