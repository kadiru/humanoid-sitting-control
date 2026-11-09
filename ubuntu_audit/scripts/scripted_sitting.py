"""Gravity-driven scripted hold diagnostics; no root constraints or external support."""
import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sensors import ContactSensorCfg
from humanoid_sitting.tasks.config import SittingEnvCfg
import humanoid_sitting.tasks
from humanoid_sitting.tasks import mdp

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["sweep", "hold", "standing", "transition", "descent"], default="sweep")
parser.add_argument("--poses", help="JSON list of diagnostic poses")
parser.add_argument("--pose", help="JSON pose selected by an earlier sweep")
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--video", action="store_true")
parser.add_argument("--output", default=None)
add_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
out = Path(args.output or f"../runs/{datetime.now():%Y%m%d_%H%M%S}_{args.mode}").resolve()
out.mkdir(parents=True, exist_ok=False)
cfg = SittingEnvCfg()
if args.mode == "sweep":
    poses = [dict(hip=h, knee=k, ankle=-h-k, x=x, z=0.48, pitch=0.0)
             for h, k, x in itertools.product([-1.0, -1.25, -1.5, -1.7], [1.4, 1.7, 2.0, 2.2], [-0.42, -0.34, -0.26, -0.18])]
elif args.mode in ("standing", "transition", "descent"):
    poses = [dict(hip=-0.20, knee=0.42, ankle=-0.23, x=0.0, z=0.74, pitch=0.0)]
else:
    poses = [json.loads(Path(args.pose).read_text())]
for side in ("left", "right"):
    for label,body in [("foot","ankle_roll_link"),("knee","knee_link")]:
        setattr(cfg.scene, f"{side}_{label}_seat", ContactSensorCfg(prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_{body}", filter_prim_paths_expr=["{ENV_REGEX_NS}/Seat"]))
if args.poses:
    poses = json.loads(Path(args.poses).read_text())
cfg.scene.num_envs = len(poses)
cfg.sim.device = args.device
cfg.episode_length_s = args.seconds + 10.0
# Diagnostic failures are measured, never hidden by automatic resets.
cfg.terminations.fallen = None
cfg.terminations.pelvis_floor = None
cfg.terminations.time_out = None
if args.video:
    args.visualizer = ["kit"]
    args.enable_cameras = True
    cfg.video_recorders = [VideoRecorderCfg(source="visualizer:kit", output_dir=str(out),
        video_length=int(args.seconds / 0.02), video_interval=100000,
        output_filename_prefix="scripted", frame_stride=2)]
with launch_simulation(cfg, args):
    env = gym.make("Humanoid-Sit-Fixed-G1", cfg=cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    q = robot.data.default_joint_pos.torch.clone()
    root = robot.data.default_root_pose.torch.clone()
    for i, pose in enumerate(poses):
        root[i, :3] = env.scene.env_origins[i] + torch.tensor([pose["x"], 0.0, pose["z"]], device=env.device)
        root[i, 3:] = torch.tensor([0.0, np.sin(pose["pitch"] / 2), 0.0, np.cos(pose["pitch"] / 2)], device=env.device)
        for j, name in enumerate(robot.joint_names):
            if name.endswith("hip_pitch_joint"):
                q[i, j] = pose["hip"]
            elif name.endswith("knee_joint"):
                q[i, j] = pose["knee"]
            elif name.endswith("ankle_pitch_joint"):
                q[i, j] = pose["ankle"]
    limits = robot.data.joint_pos_limits.torch
    q = torch.maximum(torch.minimum(q, limits[..., 1]), limits[..., 0])
    robot.write_joint_position_to_sim_index(position=q)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
    robot.write_root_pose_to_sim_index(root_pose=root)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((env.num_envs, 6), device=env.device))
    robot.reset()
    # Root/joint states are set only here. All subsequent motion is unconstrained physics.
    target_actions = (q - robot.data.default_joint_pos.torch) / cfg.actions.joints.scale
    frames = {k: [] for k in ["root_pose", "lin_vel", "ang_vel", "gravity", "support", "body_forces", "foot_seat", "knee_seat", "com", "foot_positions", "actions", "joint_pos"]}
    hip_ids=[j for j,n in enumerate(robot.joint_names) if n.endswith("hip_pitch_joint")]
    ankle_ids=[j for j,n in enumerate(robot.joint_names) if n.endswith("ankle_pitch_joint")]
    foot_ids=[robot.body_names.index(f"{side}_ankle_roll_link") for side in ("left","right")]
    goal=q.clone()
    for j,name in enumerate(robot.joint_names):
        if name.endswith("hip_pitch_joint"): goal[:,j]=-1.25
        elif name.endswith("knee_joint"): goal[:,j]=1.4
        elif name.endswith("ankle_pitch_joint"): goal[:,j]=-0.15
    def params(name,default):
        return torch.tensor([p.get(name,default) for p in poses],device=env.device)
    duration=params("duration",3.);kp=params("balance_kp",0.);kd=params("balance_kd",0.)
    com_offset=params("com_offset",-.2);switch=params("seat_switch",.25)
    torso_kp=params("torso_kp",0.);torso_kd=params("torso_kd",.2);torso_lean=params("torso_lean",.15)
    for step in range(int(args.seconds / env.step_dt)):
        with torch.inference_mode():
            step_actions=target_actions.clone()
            phase=((step*env.step_dt-(0. if args.mode=="descent" else 2.))/duration).clamp(0,1)
            transitioning=args.mode in ("transition","descent")
            if transitioning:
                reference=q+phase[:,None]*(goal-q)
                step_actions=(reference-robot.data.default_joint_pos.torch)/cfg.actions.joints.scale
            com=(robot.data.body_com_pos_w.torch*robot.data.body_mass.torch[...,None]).sum(dim=1)/robot.data.body_mass.torch.sum(dim=1,keepdim=True)
            feet=robot.data.body_pos_w.torch[:,foot_ids]
            blend=(mdp.support_fraction(env)<=switch).float() if transitioning else torch.ones_like(kp)
            error=com[:,0]-feet[:,:,0].mean(dim=1)-(com_offset*phase if transitioning else 0.)
            correction=(kp*error+kd*robot.data.root_lin_vel_w.torch[:,0]).clamp(-.5,.5)*(kp!=0)*blend
            step_actions[:,ankle_ids]+=correction[:,None]/cfg.actions.joints.scale
            pitch=torch.atan2(robot.data.projected_gravity_b.torch[:,0],-robot.data.projected_gravity_b.torch[:,2])
            desired_pitch=-.06+torso_lean*torch.sin(torch.pi*phase)
            hip_correction=(torso_kp*(pitch-desired_pitch)+torso_kd*robot.data.root_ang_vel_w.torch[:,1]).clamp(-.4,.4)*(torso_kp!=0)*blend
            step_actions[:,hip_ids]+=hip_correction[:,None]/cfg.actions.joints.scale
            env.step(step_actions)
            values = dict(root_pose=robot.data.root_pose_w.torch - torch.cat((env.scene.env_origins, torch.zeros((env.num_envs, 4), device=env.device)), dim=1),
                lin_vel=robot.data.root_lin_vel_w.torch, ang_vel=robot.data.root_ang_vel_w.torch,
                gravity=robot.data.projected_gravity_b.torch, support=mdp.support_fraction(env),
                body_forces=env.scene["all_contacts"].data.net_normal_forces_w.torch,
                foot_seat=torch.stack([env.scene[f"{side}_foot_seat"].data.normal_force_matrix_w.torch.sum(dim=(1,2)) for side in ("left", "right")], dim=1),
                knee_seat=torch.stack([env.scene[f"{side}_knee_seat"].data.normal_force_matrix_w.torch.sum(dim=(1,2)) for side in ("left", "right")], dim=1),
                com=com-env.scene.env_origins, foot_positions=feet-env.scene.env_origins[:,None,:],
                actions=step_actions, joint_pos=robot.data.joint_pos.torch)
            for name, value in values.items():
                assert torch.isfinite(value).all(), (step, name)
                frames[name].append(value.cpu().numpy().copy())
    arrays = {k: np.stack(v) for k,v in frames.items()}
    # Require a stable late hold, not an isolated contact spike.
    late = slice(int(2.0 / env.step_dt), None)
    support = arrays["support"][late]
    speed = np.linalg.norm(arrays["lin_vel"][late], axis=-1)
    angular = np.linalg.norm(arrays["ang_vel"][late], axis=-1)
    upright = -arrays["gravity"][late, :, 2]
    xyz = arrays["root_pose"][late, :, :3]
    inside = (abs(xyz[..., 0] + 0.40) < 0.22) & (abs(xyz[..., 1]) < 0.40)
    height = (xyz[..., 2] > 0.42) & (xyz[..., 2] < 0.55)
    valid = (support > 0.25) & (upright > 0.90) & (speed < 0.20) & (angular < 0.35) & inside & height
    if args.mode == "standing":
        valid=(upright>.96)&(speed<.05)&(angular<.20)&(xyz[...,2]>.65)&(support<.1)
    summaries = []
    for i,p in enumerate(poses):
        summaries.append(dict(index=i, pose=p, valid_hold_fraction=float(valid[:, i].mean()),
            support_mean=float(support[:,i].mean()), support_min=float(support[:,i].min()),
            upright_min=float(upright[:,i].min()), speed_max=float(speed[:,i].max()),
            angular_speed_max=float(angular[:,i].max()), final_xyz=xyz[-1,i].tolist(),
            stable_seated_hold=bool(valid[:,i].all()) if args.mode!="standing" else False,
            stable_standing_hold=bool(valid[:,i].all()) if args.mode=="standing" else None,
            max_descent_speed=float(-arrays["lin_vel"][:,i,2].min())))
    ranked = sorted(summaries, key=lambda s:(s["valid_hold_fraction"], s["support_mean"] if s["upright_min"]>0.9 else 0), reverse=True)
    result = dict(mode=args.mode, gravity=list(cfg.sim.gravity), seconds=args.seconds,
        settle_seconds=2.0, num_envs=env.num_envs, root_unconstrained=True,
        configured_actuators_only=True, outcomes=ranked, body_names=env.scene["all_contacts"].body_names, articulation_body_names=robot.body_names,
        actual_joint_targets=q.cpu().tolist(), joint_names=robot.joint_names)
    (out/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    (out/"best_pose.json").write_text(json.dumps(ranked[0]["pose"], indent=2)+"\n")
    np.savez_compressed(out/"rollouts.npz", **arrays, dt=env.step_dt, initial_actions=target_actions.cpu().numpy())
    print("SCRIPTED_RESULTS", str(out), json.dumps(ranked[:4]), flush=True)
    env.close()
