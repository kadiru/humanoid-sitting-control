"""Short, matched diagnostic replay, not a repair trial or training run.

84 control steps = 8.4s, 10080 active physics transitions over24 environments.
Initial25ticks*24=600 settling transitions separate. No optimizer or holdout.
"""
import argparse,sys,json,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
p=argparse.ArgumentParser();p.add_argument('--diagnostic-kind',choices=['baseline','changed'],required=True);option=p.parse_args()
OUT=D/'sensor_prefix_24'/option.diagnostic_kind
argv=json.loads((D/'joint_path_17/candidate.command.json').read_text())[3:];k=argv.index('--landing-config');del argv[k:k+2];argv[argv.index('--output')+1]=str(OUT);sys.argv=[__file__]+argv
import run_low_positive_feedback_v2 as harness
if option.diagnostic_kind=='changed':from low_seat_joint_path_controller_17 import install
else:from low_seat_normal_timing_controller import install
install(harness,json.loads((D/'normal_landing3s_guarded_09.json').read_text()))

def main():
 OUT.mkdir(exist_ok=False);(OUT/'source').mkdir();a=harness.parent_harness.arguments();cfg,goals,resets=harness.parent_harness.build(a,OUT);captures=[]
 with harness.base.launch_simulation(cfg,a):
  raw=harness.base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped;env=harness.FeedbackEnv(raw,a,OUT,goals,resets);clock=raw.sim.get_physics_step_count()
  with torch.inference_mode():
   for step in range(84):
    env.step(torch.zeros(24,23,device=env.device))
    if (step+1)%2==0 and step>=73:
     cam=raw.scene['rear_camera'];data=env.robot.data
     pos,quat=cam._view.get_world_poses() # read-only current scene transform; no extra exposure or step
     captures.append(dict(t=round((step+1)*.1,6),depth=cam.data.output['distance_to_image_plane'].torch.cpu().numpy().copy(),K=cam.data.intrinsic_matrices.torch.cpu().numpy().copy(),
      q=data.joint_pos.torch.cpu().numpy().copy(),gravity=data.projected_gravity_b.torch.cpu().numpy().copy(),body=data.body_pose_w.torch.cpu().numpy().copy(),
      geometry=env.geometry.copy(),camera_position=pos.torch.cpu().numpy().copy(),camera_orientation=quat.torch.cpu().numpy().copy()))
  assert env.active_transitions==10080 and raw.sim.get_physics_step_count()-clock==420 and not env.episodes
  np.savez_compressed(OUT/'raw_sensor_captures.npz',**{k:np.array([x[k] for x in captures]) for k in captures[0]})
  (OUT/'diagnostic_accounting.json').write_text(json.dumps(dict(kind=option.diagnostic_kind,elapsed_s=8.4,active_transitions=env.active_transitions,indexed_warmup_transitions=env.warmup_transitions,initial_settling_excluded=600,optimizer_updates=0,full20s_episodes=0,fresh_holdout=False,capture_times=[x['t'] for x in captures],scope='Short physical sensor diagnostic; no physical repair success or policy effort comparison is claimed.'),indent=2))
  env.close()
 shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
 print('DIAGNOSTIC_PREFIX_COMPLETE',option.diagnostic_kind,flush=True)
if __name__=='__main__':main()
