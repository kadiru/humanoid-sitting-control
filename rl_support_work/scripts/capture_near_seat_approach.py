"""Log physical intermediate states without changing the current controller."""
from pathlib import Path
import numpy as np
import run_contact_effort_repair as original

class CaptureApproach(original.SplitEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.states={};self.capture_state(0)
    def capture_state(self,tick):
        d=self.robot.data;r=d.root_pose_w.torch.clone();r[:,:3]-=self.raw.scene.env_origins
        self.states[tick]=dict(root_pose=r.cpu().numpy().copy(),root_velocity=d.root_vel_w.torch.cpu().numpy().copy(),q=d.joint_pos.torch.cpu().numpy().copy(),qd=d.joint_vel.torch.cpu().numpy().copy(),command=self.previous.cpu().numpy().copy())
    def step(self,actions):
        result=super().step(actions);tick=int(self.episode_length_buf[0])
        if tick in [300,400,500]:self.capture_state(tick)
        return result
    def close(self):
        for tick,state in self.states.items():np.savez_compressed(self.out/f'state_{tick:04d}.npz',**state)
        super().close()
original.SplitEnv=CaptureApproach
if __name__=='__main__':original.main()
