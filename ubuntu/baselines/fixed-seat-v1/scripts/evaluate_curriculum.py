"""Bounded curriculum evaluation with contact records and an explicit promotion gate."""
import argparse,json,sys
from datetime import datetime
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from isaaclab.app import add_launcher_args,launch_simulation
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,create_rsl_rl_runner,check_rsl_rl_version,handle_deprecated_rsl_rl_cfg
from humanoid_sitting.tasks import curriculum as c,mdp
import humanoid_sitting.tasks
p=argparse.ArgumentParser();p.add_argument('--stage',type=int,choices=range(4),default=0);p.add_argument('--checkpoint');p.add_argument('--num_envs',type=int,default=32);p.add_argument('--seconds',type=float,default=12);p.add_argument('--video',action='store_true');p.add_argument('--output',required=True)
add_launcher_args(p);args=p.parse_args();sys.argv=[sys.argv[0]]
out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
cfg=[c.SeatedStabilityEnvCfg,c.NearSeatEnvCfg,c.PartialDescentEnvCfg,c.StandingTransitionEnvCfg][args.stage]()
cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device
if args.video:
 args.visualizer=['kit'];args.enable_cameras=True
 cfg.video_recorders=[VideoRecorderCfg(source='visualizer:kit',output_dir=str(out),video_length=int(args.seconds/.02),video_interval=100000,output_filename_prefix='evaluation',frame_stride=2)]
agent=handle_deprecated_rsl_rl_cfg(c.CurriculumPPORunnerCfg(),check_rsl_rl_version());agent.device=args.device
with launch_simulation(cfg,args):
 env=RslRlVecEnvWrapper(gym.make(f'Humanoid-Sit-Curriculum-{args.stage}-G1',cfg=cfg));raw=env.unwrapped;robot=raw.scene['robot'];policy=None
 if args.checkpoint:
  runner=create_rsl_rl_runner(env,agent);runner.load(args.checkpoint);policy=runner.get_inference_policy(device=args.device)
 obs=env.get_observations();frames={k:[] for k in ['root_pose','lin_vel','ang_vel','gravity','support','body_forces','foot_seat','joint_pos','actions','done']};terms={};peakhold=0
 with torch.inference_mode():
  for step in range(int(args.seconds/raw.step_dt)):
   actions=policy(obs) if policy else torch.zeros((raw.num_envs,env.num_actions),device=raw.device)
   values=dict(root_pose=robot.data.root_pose_w.torch-torch.cat((raw.scene.env_origins,torch.zeros((raw.num_envs,4),device=raw.device)),dim=1),lin_vel=robot.data.root_lin_vel_w.torch,ang_vel=robot.data.root_ang_vel_w.torch,gravity=robot.data.projected_gravity_b.torch,support=mdp.support_fraction(raw),body_forces=raw.scene['all_contacts'].data.net_normal_forces_w.torch,foot_seat=torch.stack([raw.scene[f'{s}_foot_seat'].data.normal_force_matrix_w.torch.sum(dim=(1,2)) for s in ('left','right')],dim=1),joint_pos=robot.data.joint_pos.torch,actions=actions)
   for k,v in values.items():
    assert torch.isfinite(v).all(),(step,k)
    frames[k].append(v.cpu().numpy().copy())
   obs,reward,done,info=env.step(actions);frames['done'].append(done.cpu().numpy().copy())
   assert torch.isfinite(reward).all()
   peakhold=max(peakhold,float(raw.sit_peak_hold.max()))
   for name in raw.termination_manager.active_terms:terms[name]=terms.get(name,0)+int(raw.termination_manager.get_term(name).sum())
   if policy:policy.reset(done)
 episodes=raw.completed_sitting_episodes;success=sum(r['success'] for r in episodes);rate=success/len(episodes) if episodes else None
 sustained=sum(r['success'] and r['peak_hold_s']>=3.0 for r in episodes)
 maxhold=max([peakhold]+[r['peak_hold_s'] for r in episodes]);failures=sum(v for k,v in terms.items() if k!='time_out')
 result=dict(stage=args.stage,stage_name=c.STAGES[args.stage]['name'],checkpoint=args.checkpoint,num_envs=raw.num_envs,seconds=args.seconds,completed_episodes=len(episodes),successful_episodes=success,success_rate=rate,sustained_3s_episodes=sustained,sustained_3s_rate=sustained/len(episodes) if episodes else None,peak_hold_s=maxhold,termination_counts=terms,episodes=episodes,body_names=raw.scene['all_contacts'].body_names,robot_mass_kg=float(robot.data.body_mass.torch[0].sum()),promotion_gate=dict(min_completed_episodes=32,min_success_rate=.9,min_hold_s=3.,max_failures=0),promotion_passed=bool(len(episodes)>=32 and sustained/len(episodes)>=.9 and maxhold>=3. and failures==0),policy='checkpoint' if policy else 'zero residual scripted reference',gravity=list(cfg.sim.gravity),root_unconstrained=True)
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out/'rollouts.npz',**{k:np.stack(v) for k,v in frames.items()},dt=raw.step_dt)
 print('CURRICULUM_EVALUATION',json.dumps({k:v for k,v in result.items() if k not in ['episodes','body_names']}),flush=True);env.close()
