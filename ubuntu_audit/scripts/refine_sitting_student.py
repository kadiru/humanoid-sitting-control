"""Short full-action PPO refinement of BC; no scripted reference in action path."""
import argparse,json,itertools,hashlib,time
from pathlib import Path
import gymnasium as gym,torch
from isaaclab.app import add_launcher_args,launch_simulation
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,create_rsl_rl_runner,check_rsl_rl_version,handle_deprecated_rsl_rl_cfg
from humanoid_sitting.tasks.student import StudentEnvCfg
from humanoid_sitting.tasks.full_transition import FullTransitionPPORunnerCfg
from humanoid_sitting.student_network import Student
import humanoid_sitting.tasks
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--iterations',type=int,default=60);p.add_argument('--seed',type=int,default=111);p.add_argument('--geometry-seed',type=int,default=22003);p.add_argument('--actor-learning-rate',type=float,default=1e-5);p.add_argument('--critic-learning-rate',type=float,default=1e-5);p.add_argument('--anchor-dataset',nargs='*',default=[]);add_launcher_args(p);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
cfg=StudentEnvCfg(geometry_seed=a.geometry_seed,geometry_stage=2,geometry_count=16,include_families=True,arm_clearance=True);cfg.scene.num_envs=64;cfg.sim.device=a.device;cfg.seed=a.seed;cfg.episode_length_s=20
agent=handle_deprecated_rsl_rl_cfg(FullTransitionPPORunnerCfg(),check_rsl_rl_version());agent.device=a.device;agent.seed=a.seed;agent.save_interval=20
agent.actor.obs_normalization=False;agent.critic.obs_normalization=True;agent.actor.distribution_cfg.init_std=.003;agent.algorithm.learning_rate=1e-5;agent.algorithm.schedule='fixed';agent.algorithm.entropy_coef=0.;agent.algorithm.num_learning_epochs=3
with launch_simulation(cfg,a):
    env=RslRlVecEnvWrapper(gym.make('Humanoid-Sit-Learned-G1',cfg=cfg));runner=create_rsl_rl_runner(env,agent,log_dir=str(out))
    model,metadata=Student.load(a.checkpoint,a.device);model.train();runner.alg.actor.mlp=model
    runner.alg.optimizer=torch.optim.Adam([{'params':list(runner.alg.actor.parameters()),'lr':a.actor_learning_rate},{'params':list(runner.alg.critic.parameters()),'lr':a.critic_learning_rate}])
    if a.anchor_dataset:
        import numpy as np
        X=[];Y=[]
        for directory in a.anchor_dataset:
            data=np.load(Path(directory)/'rollouts.npz');result=json.loads((Path(directory)/'results.json').read_text());good=np.array([o['success'] for o in result['outcomes']]);X.append(data['student_observations'][:,good].reshape(-1,99));Y.append(data['targets'][:,good].reshape(-1,37))
        X=torch.tensor(np.concatenate(X),device=a.device);Y=torch.tensor(np.concatenate(Y),device=a.device)
        anchor_optimizer=torch.optim.Adam(model.parameters(),lr=a.actor_learning_rate)
        original_update=runner.alg.update
        def anchored_update():
            losses=original_update()
            for _ in range(2):
                ids=torch.randint(len(X),(2048,),device=a.device)
                loss=((model(X[ids])-Y[ids])/model.target_std).square().mean()
                anchor_optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);anchor_optimizer.step()
            losses['imitation_anchor']=loss.item()
            return losses
        runner.alg.update=anchored_update
    before={k:v.detach().clone() for k,v in model.state_dict().items()};runner.save(str(out/'warm_start.pt'))
    (out/'command.json').write_text(json.dumps(vars(a),indent=2)+'\n')
    (out/'geometry_bank.json').write_text(json.dumps(cfg.geometry_bank,indent=2)+'\n')
    start=time.time();runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=False)
    exported=metadata|dict(state_dict=model.state_dict(),rl_refinement=dict(seed=a.seed,iterations=a.iterations,transitions=64*24*a.iterations,geometry_seed=a.geometry_seed,teacher_reference_used=False))
    torch.save(exported,out/'student_final.pt')
    verification=dict(finite=all(torch.isfinite(v).all().item() for v in model.state_dict().values()),maximum_parameter_change=max((v-before[k]).abs().max().item() for k,v in model.state_dict().items()),actor_optimizer_includes_student_parameters=all(any(p is q for group in runner.alg.optimizer.param_groups for q in group['params']) for p in model.parameters()),transitions=64*24*a.iterations,elapsed_s=time.time()-start,teacher_reference_in_action_path=False,unchanged_actuator_limits=True)
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n');print('STUDENT_RL_VERIFICATION',verification,flush=True);env.close()
