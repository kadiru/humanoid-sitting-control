"""Independent post-contact reference probe, not PPO or a frozen-study change."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import run_contact_effort_repair as original
from humanoid_sitting.contact_aware_reference import measured_level_plane,front_contact_motion

class UnloadProbe(original.SplitEnv):
    def __init__(self,raw,args,out,goals,resets):
        assert args.mode=='reference' and args.surface=='sitting'
        super().__init__(raw,args,out,goals,resets)
        ends=[];qends=[];pends=[];xends=[];audit=[]
        calibration=json.loads(Path(args.plane_calibration).read_text())
        for i,j in enumerate(self.index):
            profile=raw.cfg.geometry_bank[j]['control_profile']
            if profile.get('unload',False):
                n,d,o=measured_level_plane(self.model,self.robot.data.joint_pos.torch[i].cpu().numpy(),self.robot.data.projected_gravity_b.torch[i].cpu().numpy(),self.geometry[i],resets['root_pose'][j,0])
                q,c,r,check=front_contact_motion(self.model,resets['q'][j],resets['command'][j],resets['root_pose'][j],n,d,calibration,profile['unload_pitch'],profile['unload_x'],profile['unload_gap'])
                assert check['max_sagittal_residual_m']<.003,check
                ends.append(c[-1]);qends.append(q[-1]);pends.append([Rotation.from_quat(r[-1,3:]).as_euler('xyz')[1]]);xends.append([r[-1,0]]);audit.append(dict(case=i,**check))
            else:
                ends.append(self.adapted_paths[i,-1].cpu().numpy());qends.append(self.adapted_qpaths[i,-1].cpu().numpy());pends.append(self.adapted_pitch[i,-1].cpu().numpy());xends.append(self.adapted_root_x[i,-1].cpu().numpy())
        for key,values in [('adapted_paths',ends),('adapted_qpaths',qends),('adapted_pitch',pends),('adapted_root_x',xends)]:
            value=getattr(self,key);setattr(self,key,torch.cat((value,torch.tensor(np.array(values),device=self.device,dtype=value.dtype)[:,None]),1))
        np.savez_compressed(self.out/'unloading_paths.npz',command=self.adapted_paths.cpu().numpy(),q=self.adapted_qpaths.cpu().numpy(),pitch=self.adapted_pitch.cpu().numpy(),root_x=self.adapted_root_x.cpu().numpy())
        source=Path(__file__);shutil.copy2(source,out/'unloading_source.py')
        original.base.write_json(out/'unloading_manifest.json',dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='Extra reference segment12–14s, same20s trials and physical criteria, reference-only, no PPO. Initial reference/kinematic feedback unchanged until12s. Static depth/encoder/IMU-derived target; no live simulator truth input.',plans=audit))
        self._augment()
    def phase(self):
        t=self.episode_length_buf*original.DT
        return torch.where(t<=12,t/2,6+(t-12)/2).clamp_max(7)

original.SplitEnv=UnloadProbe
if __name__=='__main__':original.main()
