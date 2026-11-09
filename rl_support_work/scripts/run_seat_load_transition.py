"""Development transition to the zero-leg-preload seated target."""
from pathlib import Path
import hashlib,shutil
import numpy as np
import torch
import run_seat_load_probe as probe
from humanoid_sitting.two_surface_path import sample_path

class SeatLoadTransition(probe.SeatLoadProbe):
    def reference(self):
        ref=super().reference()
        if self.adapted_paths is None or self.adapted_paths.shape[1]<8:return ref
        ramp=(self.phase()-6).clamp(0,1);ramp=ramp.square()*(3-2*ramp)
        enabled=torch.tensor([self.raw.cfg.geometry_bank[i]['control_profile'].get('fade_feedback',False) for i in self.index],device=self.device)
        ramp*=enabled
        plan=sample_path(self.adapted_paths,self.phase())
        return (1-ramp[:,None])*ref+ramp[:,None]*plan

    def close(self):
        data=self.robot.data;root=data.root_pose_w.torch.clone();root[:,:3]-=self.raw.scene.env_origins
        np.savez_compressed(self.out/'settled_candidates.npz',root_pose=root.cpu().numpy(),root_velocity=data.root_vel_w.torch.cpu().numpy(),q=data.joint_pos.torch.cpu().numpy(),qd=data.joint_vel.torch.cpu().numpy(),command=self.previous.cpu().numpy(),valid=(self.alive&(self.held>=250)).cpu().numpy())
        shutil.copy2(__file__,self.out/'transition_source.py')
        probe.original.base.write_json(self.out/'transition_manifest.json',dict(sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),feedback='IMU/FK ankle feedback fades12–14s only for explicitly flagged profiles; final command is constant, no force or pose truth used.'))
        super().close()

probe.original.SplitEnv=SeatLoadTransition
if __name__=='__main__':probe.original.main()
