"""Contact-clearance-aware unloading with dense foot-preserving IK knots.

The previous two-endpoint joint interpolation did not preserve the feet along
the path. This new probe reconstructs every unloading knot from the robot FK.
"""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import run_contact_effort_repair as original
from humanoid_sitting.contact_aware_reference import measured_level_plane,front_contact_motion
from humanoid_sitting.two_surface_path import sample_path

class SupportedUnload(original.SplitEnv):
    def __init__(self,raw,args,out,goals,resets):
        self.unloading_ready=False
        super().__init__(raw,args,out,goals,resets)
        calibration=json.loads(Path(args.plane_calibration).read_text())
        leg=[i for i,n in enumerate(self.model.names) if any(s in n for s in ['hip_','knee','ankle'])]
        allc=[];allq=[];allp=[];allx=[];audit=[]
        for i,j in enumerate(self.index):
            profile=raw.cfg.geometry_bank[j]['control_profile'];cs=[];qs=[];ps=[];xs=[]
            n,d,o=measured_level_plane(self.model,self.robot.data.joint_pos.torch[i].cpu().numpy(),self.robot.data.projected_gravity_b.torch[i].cpu().numpy(),self.geometry[i],resets['root_pose'][j,0])
            for u in np.linspace(0,1,9)[1:]:
                alpha=u*u*(3-2*u)
                pitch=profile['contact_pitch']*(1-alpha)
                x=profile['contact_x']*(1-alpha)+profile.get('rest_x',-.18)*alpha
                gap_alpha=min(1.,u*4) if profile.get('early_clearance',False) else alpha
                gap=(profile['contact_gap']+.008)*(1-gap_alpha)+.001*gap_alpha
                q,c,r,check=front_contact_motion(self.model,resets['q'][j],resets['command'][j],resets['root_pose'][j],n,d,calibration,pitch,x,gap)
                assert check['max_sagittal_residual_m']<.003,check
                scale=1-alpha*(1-profile.get('rest_preload',.15))
                c[-1,leg]=q[-1,leg]+scale*(c[-1,leg]-q[-1,leg])
                for side in ['left','right']:c[-1,self.model.names.index(side+'_ankle_pitch_joint')]+=profile.get('rest_ankle',0.)*alpha
                cs.append(c[-1]);qs.append(q[-1]);ps.append([pitch]);xs.append([x]);audit.append(dict(case=i,u=float(u),preload_scale=float(scale),**check))
            allc.append(cs);allq.append(qs);allp.append(ps);allx.append(xs)
        for key,values in [('adapted_paths',allc),('adapted_qpaths',allq),('adapted_pitch',allp),('adapted_root_x',allx)]:
            old=getattr(self,key);setattr(self,key,torch.cat((old[:,:6],torch.tensor(np.array(values),device=self.device,dtype=old.dtype)),1))
        self.unloading_ready=True
        self.unload_seconds=torch.tensor([raw.cfg.geometry_bank[i]['control_profile'].get('unload_seconds',4.) for i in self.index],device=self.device)
        np.savez_compressed(out/'cartesian_paths.npz',command=self.adapted_paths.cpu().numpy(),q=self.adapted_qpaths.cpu().numpy(),pitch=self.adapted_pitch.cpu().numpy(),root_x=self.adapted_root_x.cpu().numpy())
        shutil.copy2(__file__,out/'cartesian_source.py')
        original.base.write_json(out/'cartesian_manifest.json',dict(sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plans=audit,scope='Eight foot-preserving IK unload knots from10–14s. No clearance arch; retain a small final leg preload to keep foot friction margin. Plane measured at standing startup; no live truth. Physical validation still required.'))
        self._augment()

    def phase(self):
        if not self.unloading_ready:return super().phase()
        t=self.episode_length_buf*original.DT
        return torch.where(t<=10,t/2,5+8*(t-10)/self.unload_seconds).clamp_max(13)

    def reference(self):
        ref=super().reference()
        if not self.unloading_ready:return ref
        u=((self.phase()-5)/8).clamp(0,1);alpha=u.square()*(3-2*u)
        plan=sample_path(self.adapted_paths,self.phase())
        return (1-alpha[:,None])*ref+alpha[:,None]*plan

    def close(self):
        data=self.robot.data;root=data.root_pose_w.torch.clone();root[:,:3]-=self.raw.scene.env_origins
        np.savez_compressed(self.out/'settled_candidates.npz',root_pose=root.cpu().numpy(),root_velocity=data.root_vel_w.torch.cpu().numpy(),q=data.joint_pos.torch.cpu().numpy(),qd=data.joint_vel.torch.cpu().numpy(),command=self.previous.cpu().numpy(),valid=(self.alive&(self.held>=250)).cpu().numpy())
        super().close()

original.SplitEnv=SupportedUnload
if __name__=='__main__':original.main()
