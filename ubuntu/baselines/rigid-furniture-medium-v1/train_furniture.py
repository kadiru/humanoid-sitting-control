"""Bounded furniture PPO warm start; preserves fixed-seat policy, adds yaw inputs."""
import argparse,json,sys
from pathlib import Path
import gymnasium as gym
import torch
from isaaclab.app import add_launcher_args,launch_simulation
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,create_rsl_rl_runner,check_rsl_rl_version,handle_deprecated_rsl_rl_cfg
from humanoid_sitting.tasks.furniture import FurnitureEnvCfg
from humanoid_sitting.tasks.full_transition import FullTransitionPPORunnerCfg
import humanoid_sitting.tasks
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
p.add_argument('--geometry-stage',type=int,default=1);p.add_argument('--geometry-seed',type=int,default=310)
p.add_argument('--seed',type=int,default=31);p.add_argument('--iterations',type=int,default=30)
p.add_argument('--families',action='store_true');p.add_argument('--arm-clearance',action='store_true')
p.add_argument('--reference-mode',choices=['normal','gentle_recovery'],default='normal');add_launcher_args(p)
a=p.parse_args();sys.argv=[sys.argv[0]];out=Path(a.output).resolve();out.mkdir(exist_ok=False)
cfg=FurnitureEnvCfg(geometry_stage=a.geometry_stage,geometry_seed=a.geometry_seed,include_families=a.families,arm_clearance=a.arm_clearance,reference_mode=a.reference_mode)
cfg.scene.num_envs=64;cfg.sim.device=a.device;cfg.seed=a.seed
agent=handle_deprecated_rsl_rl_cfg(FullTransitionPPORunnerCfg(),check_rsl_rl_version());agent.device=a.device;agent.seed=a.seed;agent.save_interval=10
with launch_simulation(cfg,a):
    env=RslRlVecEnvWrapper(gym.make('Humanoid-Sit-Rigid-Furniture-G1',cfg=cfg))
    runner=create_rsl_rl_runner(env,agent,log_dir=str(out))
    loaded=torch.load(a.checkpoint,weights_only=False,map_location=a.device)
    for group in ('actor_state_dict','critic_state_dict'):
        if loaded[group]['mlp.0.weight'].shape[1]==127:loaded[group]['mlp.0.weight']=torch.nn.functional.pad(loaded[group]['mlp.0.weight'],(0,2))
    runner.alg.load(loaded,{'actor':True,'critic':True},True)
    runner.save(str(out/'warm_start.pt'))
    (out/'experiment.json').write_text(json.dumps(dict(arguments=vars(a),geometry_bank=cfg.geometry_bank,
        parent_checkpoint=a.checkpoint,additional_yaw_weights_initialized_zero=True,optimizer_reinitialized=True,
        environment_transitions=64*24*a.iterations),indent=2)+'\n')
    runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=False)
    env.close()
