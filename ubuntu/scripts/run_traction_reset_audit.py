"""Reset-isolation audit after sensor feedback and seat-plan memory are active."""
import json
from pathlib import Path
import numpy as np
import torch
import run_traction_transfer_ppo as traction
class ResetAudit(traction.TractionEnv):
    def reset_probe(self):
        super().reset_probe()
        (self.out/'reset_probe.json').rename(self.out/'reset_probe_initial.json')
        with torch.inference_mode():
            for _ in range(160):self.step(torch.zeros(self.num_envs,23,device=self.device))
        assert self.target_ready.all() and (self.near_start>=0).all()
        ids=torch.tensor([0],device=self.device);d=self.robot.data
        def snap():
            tensors=[d.joint_pos.torch,d.root_pose_w.torch,self.previous,self.obs['policy'],self.near_paths,self.target_paths,self.near_start,self.target_ready,self.episode_length_buf,self.alive,self.shear_correction]
            arrays=[self.shear_offset,self.filtered_wrench,self.shear_tick]
            return [x[1:].cpu().numpy().copy() for x in tensors]+[x[1:].copy() for x in arrays]
        before=snap();clock=self.raw.sim.get_physics_step_count()
        with torch.inference_mode():self._reset_rows(ids)
        exact=[bool(np.array_equal(a,b)) for a,b in zip(before,snap())]
        assert all(exact) and clock==self.raw.sim.get_physics_step_count()
        assert not self.target_ready[0] and self.near_start[0]<0 and self.warming[0]>0 and self.episode_length_buf[0]==0
        assert not self.shear_offset[0].any() and not self.shear_correction[0].any()
        with torch.inference_mode():
            for _ in range(10):self.step(torch.zeros(self.num_envs,23,device=self.device))
        assert not self.target_ready[0] and not self.get_observations()['learn'][0].any()
        assert not self.shear_offset[0].any() and not self.shear_correction[0].any()
        (self.out/'reset_probe.json').write_text(json.dumps(dict(passed=True,other_rows_exact=exact,clock_unchanged=True,reference_and_feedback_memory_cleared=True,selected_reset_teacher_only=True,selected_no_warm_failure=bool(self.warm_failure_ticks[0]==0)),indent=2))
    def close(self):
        super().close();(self.out/'source'/'run_traction_reset_audit.py').write_text(Path(__file__).read_text())
traction.progressive.ProgressiveEnv=ResetAudit
if __name__=='__main__':traction.progressive.main()
