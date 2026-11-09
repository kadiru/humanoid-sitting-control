"""Bounded sitting evaluation; optionally probe an authored seated configuration."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import gymnasium as gym
import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_rl.rsl_rl import (RslRlVecEnvWrapper, create_rsl_rl_runner,
                              check_rsl_rl_version, handle_deprecated_rsl_rl_cfg)
from humanoid_sitting.tasks.config import SittingEnvCfg, SittingPPORunnerCfg
import humanoid_sitting.tasks
from humanoid_sitting.tasks import mdp

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint")
parser.add_argument("--probe_seated", action="store_true")
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--output", default="sitting-evaluation")
add_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
cfg = SittingEnvCfg()
cfg.scene.num_envs = 16 if args.checkpoint else 4
cfg.sim.device = args.device
if args.probe_seated:
    cfg.scene.robot.init_state.pos = (-0.28, 0.0, 0.48)
    cfg.scene.robot.init_state.joint_pos[".*_hip_pitch_joint"] = -1.50
    cfg.scene.robot.init_state.joint_pos[".*_knee_joint"] = 1.70
    cfg.scene.robot.init_state.joint_pos[".*_ankle_pitch_joint"] = -0.20
agent = handle_deprecated_rsl_rl_cfg(SittingPPORunnerCfg(), check_rsl_rl_version())
agent.device = args.device
with launch_simulation(cfg, args):
    env = RslRlVecEnvWrapper(gym.make("Humanoid-Sit-Fixed-G1", cfg=cfg))
    raw = env.unwrapped
    initial_body_positions = dict(zip(raw.scene["robot"].body_names,
        (raw.scene["robot"].data.body_pos_w.torch[0] - raw.scene["robot"].data.root_pos_w.torch[0]).tolist()))
    policy = None
    if args.checkpoint:
        runner = create_rsl_rl_runner(env, agent)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=args.device)
    obs = env.get_observations()
    frames = {"root_pose": [], "joint_pos": [], "actions": [], "seat_support_fraction": [],
              "seat_contact_forces": [], "done": []}
    ever_success = torch.zeros(raw.num_envs, device=raw.device, dtype=torch.bool)
    peak_hold = 0.0
    peak_support = 0.0
    peak_body_up = torch.zeros(len(raw.scene["all_contacts"].body_names), device=raw.device)
    term_counts = {}
    with torch.inference_mode():
        for _ in range(args.steps):
            actions = policy(obs) if policy else torch.zeros((raw.num_envs, env.num_actions), device=raw.device)
            assert torch.isfinite(actions).all()
            # Capture pre-action state/contact and the action at this timestamp.
            frames["root_pose"].append(raw.scene["robot"].data.root_pose_w.torch.cpu().numpy().copy())
            frames["joint_pos"].append(raw.scene["robot"].data.joint_pos.torch.cpu().numpy().copy())
            frames["actions"].append(actions.cpu().numpy().copy())
            fraction = mdp.support_fraction(raw)
            frames["seat_support_fraction"].append(fraction.cpu().numpy().copy())
            frames["seat_contact_forces"].append(torch.stack([
                raw.scene[name].data.normal_force_matrix_w.torch.sum(dim=(1, 2))
                for name in mdp.SUPPORT_SENSORS], dim=1).cpu().numpy().copy())
            peak_support = max(peak_support, float(fraction.max()))
            peak_body_up = torch.maximum(peak_body_up, raw.scene["all_contacts"].data.net_normal_forces_w.torch[..., 2].max(dim=0).values)
            obs, reward, done, info = env.step(actions)
            assert torch.isfinite(reward).all()
            frames["done"].append(done.cpu().numpy().copy())
            ever_success |= raw.sit_hit
            peak_hold = max(peak_hold, float(raw.sit_peak_hold.max()))
            for name in raw.termination_manager.active_terms:
                term_counts[name] = term_counts.get(name, 0) + int(raw.termination_manager.get_term(name).sum())
            if policy:
                policy.reset(done)
    episodes = raw.completed_sitting_episodes
    successes = sum(x["success"] for x in episodes)
    result = {"checkpoint": args.checkpoint, "probe_seated": args.probe_seated,
              "steps": args.steps, "num_envs": raw.num_envs,
              "completed_episodes": len(episodes), "successful_episodes": successes,
              "success_rate": successes / len(episodes) if episodes else None,
              "environments_with_success": int(ever_success.sum()),
              "peak_hold_s": max([peak_hold] + [x["peak_hold_s"] for x in episodes]),
              "peak_support_fraction": peak_support, "termination_counts": term_counts,
              "active_episodes_censored": raw.num_envs,
              "seat_top_height_m": 0.30,
              "robot_mass_kg": float(raw.scene["robot"].data.body_mass.torch[0].sum()),
              "initial_body_positions_relative_root": initial_body_positions,
              "peak_body_upward_forces": dict(zip(raw.scene["all_contacts"].body_names, peak_body_up.tolist())),
              "final_body_contact_forces": dict(zip(raw.scene["all_contacts"].body_names,
                  raw.scene["all_contacts"].data.net_normal_forces_w.torch[0].tolist())),
              "success_definition": "1 s continuous pelvis/thigh seat support >25% weight, pelvis over seat, seated height, upright, low linear/angular speed; descent never faster than 0.8 m/s"}
    Path(f"../logs/{args.output}.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(f"../logs/{args.output}-rollouts.npz", **{k: np.stack(v) for k,v in frames.items()},
                        dt=raw.step_dt, seat_pose=raw.scene["seat"].data.root_pose_w.torch.cpu().numpy(),
                        support_sensor_names=np.array(mdp.SUPPORT_SENSORS),
                        joint_names=np.array(raw.scene["robot"].joint_names))
    print("SITTING_EVALUATION", json.dumps(result), flush=True)
    env.close()
