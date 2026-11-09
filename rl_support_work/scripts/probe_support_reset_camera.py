"""Diagnostic only: compare reset camera estimates with evaluation labels."""
import json
import numpy as np
import train_support_rl_independent as trial

old_admit=trial.IndependentSupportVecEnv._admit
def diagnose(self,ids):
    old_admit(self,ids)
    actual=[self.cfg.geometry_bank[i]['height'] for i in trial.experiment.furniture.indices(self.raw).tolist()]
    record=dict(global_step=self.t,ids=ids.tolist(),estimated_height=self.sensor.geometry[:,3].tolist(),
        true_height=actual,ready=self.ready.tolist(),gravity=self.robot.data.projected_gravity_b.torch.tolist())
    print('RESET_CAMERA_DIAGNOSTIC',json.dumps(record),flush=True)
    if self.t==0:
        camera=self.raw.scene['startup_camera']
        np.savez_compressed(self.out/'startup_capture.npz',depth=camera.data.output['distance_to_image_plane'].torch.cpu().numpy(),
            K=camera.data.intrinsic_matrices.torch.cpu().numpy(),gravity=self.robot.data.projected_gravity_b.torch.cpu().numpy(),
            q=self.robot.data.joint_pos.torch.cpu().numpy(),mount=self.mount,quat=self.quat,true_height=actual)
trial.IndependentSupportVecEnv._admit=diagnose
trial.experiment.main()
