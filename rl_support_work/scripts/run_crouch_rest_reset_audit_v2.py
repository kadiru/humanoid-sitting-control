"""Exercise indexed reset after the bridge controller has built its plan."""
from pathlib import Path
import json
import torch
import run_crouch_rest_safe_ppo as bridge

class ResetAudit(bridge.BridgeEnv):
    def reset_probe(self):
        super().reset_probe()
        (self.out/'reset_probe.json').rename(self.out/'reset_probe_initial.json')
        # Start clean after the initial isolation probe; every row independently
        # reaches its own contact-aware controller phase.
        with torch.inference_mode():
            for _ in range(160):self.step(torch.zeros(self.num_envs,23,device=self.device))
        assert self.target_ready.all() and (self.near_start>=0).all()
        ids=torch.tensor([0],device=self.device);data=self.robot.data
        def snap():
            return [x[1:].clone() for x in [data.joint_pos.torch,data.root_pose_w.torch,self.previous,self.obs['policy'],self.near_paths,self.target_paths,self.near_start,self.target_ready,self.episode_length_buf,self.alive]]
        before=snap();clock=self.raw.sim.get_physics_step_count();
        with torch.inference_mode():self._reset_rows(ids)
        after=snap()
        exact=[bool(torch.equal(a,b)) for a,b in zip(before,after)];assert all(exact) and clock==self.raw.sim.get_physics_step_count()
        assert not self.target_ready[0] and self.near_start[0]<0 and self.warming[0]>0 and self.episode_length_buf[0]==0
        with torch.inference_mode():
            for _ in range(10):self.step(torch.zeros(self.num_envs,23,device=self.device))
        assert not self.target_ready[0] and self.near_start[0]<0 and self.episode_length_buf[0]<100 and not self.get_observations()['learn'][0].any()
        (self.out/'reset_probe.json').write_text(json.dumps(dict(passed=True,other_rows_exact=exact,clock_unchanged=True,planned_controller_memory_cleared=True,selected_reset_teacher_only=True,selected_no_warm_failure=bool(self.warm_failure_ticks[0]==0)),indent=2))
    def close(self):
        super().close();(self.out/'source'/'run_crouch_rest_reset_audit.py').write_text(Path(__file__).read_text())

bridge.progressive.ProgressiveEnv=ResetAudit
if __name__=='__main__':bridge.progressive.main()
