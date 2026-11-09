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
cfg.scene.lazy_sensor_update = False
cfg.scene.seat.spawn.activate_contact_sensors = True
cfg.scene.seat_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Seat",history_length=4)
for name in ("pelvis_seat","left_thigh_seat","right_thigh_seat","left_foot_seat","right_foot_seat","left_knee_seat","right_knee_seat","all_contacts"):
    getattr(cfg.scene,name).history_length = 4
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
        root[i, :3] = env.scene.env_origins[i] + torch.tensor([pose["x"], pose.get("y",0.0), pose["z"]], device=env.device)
        root[i, 3:] = torch.tensor([0.0, np.sin(pose["pitch"] / 2), 0.0, np.cos(pose["pitch"] / 2)], device=env.device)
        for j, name in enumerate(robot.joint_names):
            if name.endswith("hip_pitch_joint"):
                q[i, j] = pose["hip"]
            elif name.endswith("knee_joint"):
                q[i, j] = pose["knee"]
            elif name.endswith("ankle_pitch_joint"):
                q[i, j] = pose["ankle"]
    for i,pose in enumerate(poses):
        if pose.get("joint_noise",0):
            rng=np.random.default_rng(pose.get("seed",42)+pose.get("initial_config",0)*1000)
            q[i]+=torch.tensor(rng.uniform(-pose["joint_noise"],pose["joint_noise"],robot.num_joints),device=env.device,dtype=q.dtype)
    limits = robot.data.joint_pos_limits.torch
    q = torch.maximum(torch.minimum(q, limits[..., 1]), limits[..., 0])
    robot.write_joint_position_to_sim_index(position=q)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
    robot.write_root_pose_to_sim_index(root_pose=root)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((env.num_envs, 6), device=env.device))
    robot.reset()
    seat_pose=env.scene["seat"].data.root_pose_w.torch.clone()
    for i,pose in enumerate(poses):
        seat_pose[i,:3]=env.scene.env_origins[i]+torch.tensor([pose.get("seat_x",-.4),0.,pose.get("seat_top",.3)-.05],device=env.device)
    env.scene["seat"].write_root_pose_to_sim_index(root_pose=seat_pose)
    env.scene["seat"].reset()
    # Root/joint states are set only here. All subsequent motion is unconstrained physics.
    target_actions = (q - robot.data.default_joint_pos.torch) / cfg.actions.joints.scale
    frames = {k: [] for k in ["root_pose", "lin_vel", "ang_vel", "gravity", "support", "body_forces", "foot_seat", "knee_seat", "com", "foot_positions", "actions", "body_pose", "seat_force_peak", "unintended_peak", "joint_pos"]}
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
    all_ankle_ids=[j for j,n in enumerate(robot.joint_names) if "ankle_" in n]
    robot.write_joint_stiffness_to_sim_index(stiffness=params("ankle_stiffness",20.)[:,None].expand(-1,len(all_ankle_ids)).contiguous(),joint_ids=all_ankle_ids)
    robot.write_joint_damping_to_sim_index(damping=params("ankle_damping",2.)[:,None].expand(-1,len(all_ankle_ids)).contiguous(),joint_ids=all_ankle_ids)
    duration=params("duration",4.);kp=params("balance_kp",8.);kd=params("balance_kd",1.5)
    hip_deep=params("hip_deep",-1.5);knee_deep=params("knee_deep",1.4);lean=params("lean",.30)
    hip_exp=params("hip_exp",1.);knee_exp=params("knee_exp",1.)
    knee_ids=[j for j,n in enumerate(robot.joint_names) if n.endswith("knee_joint")]
    arm_ids=[i for i,n in enumerate(env.scene["all_contacts"].body_names) if any(part in n for part in ("shoulder","elbow","wrist","palm","zero_link","one_link","two_link","three_link","four_link","five_link","six_link"))]
    for step in range(int(args.seconds / env.step_dt)):
        with torch.inference_mode():
            t=step*env.step_dt
            pre=min(max(t-2.,0.),1.)
            phase=((t-3.)/duration).clamp(0,1)
            recover=((t-3.-duration)/params("recover_duration",2.)).clamp(0,1)
            # Lean and flex hips before the final knee bend. Feet remain on the floor.
            hips=(-.2+(-.55+.2)*pre)+(hip_deep+.55)*phase.pow(hip_exp)
            knees=(.42+(.55-.42)*pre)+(knee_deep-.55)*phase.pow(knee_exp)
            lean_ref=.10*pre+(lean-.10)*phase
            hips=hips*(1-recover)+(-1.25)*recover
            knees=knees*(1-recover)+1.4*recover
            lean_ref*=1-recover
            reference=q.clone();reference[:,hip_ids]=hips[:,None];reference[:,knee_ids]=knees[:,None]
            reference[:,ankle_ids]=(-hips-knees-lean_ref)[:,None]
            com=(robot.data.body_com_pos_w.torch*robot.data.body_mass.torch[...,None]).sum(dim=1)/robot.data.body_mass.torch.sum(dim=1,keepdim=True)
            feet=robot.data.body_pos_w.torch[:,foot_ids]
            support=mdp.support_fraction(env)
            blend=(support<.25).float()
            error=com[:,0]-feet[:,:,0].mean(dim=1)-params("com_offset",0.)*phase
            correction=(kp*error+kd*robot.data.root_lin_vel_w.torch[:,0]).clamp(-.35,.35)*blend
            reference[:,ankle_ids]+=correction[:,None]
            reference=torch.maximum(torch.minimum(reference,limits[...,1]),limits[...,0])
            step_actions=(reference-robot.data.default_joint_pos.torch)/cfg.actions.joints.scale
            env.step(step_actions)
            seat_history=torch.stack([env.scene[name].data.normal_force_matrix_w_history.torch.sum(dim=(2,3)) for name in mdp.SUPPORT_SENSORS],dim=2)
            seat_peak=env.scene["seat_contacts"].data.net_normal_forces_w_history.torch.sum(dim=2).norm(dim=-1).max(dim=1).values
            wrong_history=torch.stack([env.scene[f"{side}_{body}_seat"].data.normal_force_matrix_w_history.torch.sum(dim=(2,3)).norm(dim=-1) for side in ("left","right") for body in ("foot","knee")],dim=2).sum(dim=2)
            all_history=env.scene["all_contacts"].data.net_normal_forces_w_history.torch
            wrong_peak=torch.maximum(wrong_history.max(dim=1).values,all_history[:,:,arm_ids].norm(dim=-1).flatten(1).max(dim=1).values)
            values = dict(body_pose=robot.data.body_pose_w.torch-torch.cat((env.scene.env_origins[:,None,:],torch.zeros((env.num_envs,1,4),device=env.device)),dim=-1), seat_force_peak=seat_peak,unintended_peak=wrong_peak,root_pose=robot.data.root_pose_w.torch - torch.cat((env.scene.env_origins, torch.zeros((env.num_envs, 4), device=env.device)), dim=1),
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
    seat_xyz=(seat_pose[:,:3]-env.scene.env_origins).cpu().numpy()
    inside = (abs(xyz[..., 0] -seat_xyz[None,:,0]) < 0.22) & (abs(xyz[..., 1]) < 0.40)
    height = (xyz[..., 2]-seat_xyz[None,:,2] > 0.17) & (xyz[..., 2]-seat_xyz[None,:,2] < 0.30)
    valid = (support > 0.25) & (upright > 0.90) & (speed < 0.20) & (angular < 0.35) & inside & height
    if args.mode == "standing":
        valid=(upright>.96)&(speed<.05)&(angular<.20)&(xyz[...,2]>.65)&(support<.1)
    valid &= (arrays["unintended_peak"][late]<10)
    summaries = []
    for i,p in enumerate(poses):
        summaries.append(dict(index=i, pose=p, valid_hold_fraction=float(valid[:, i].mean()),
            support_mean=float(support[:,i].mean()), support_min=float(support[:,i].min()),
            upright_min=float(upright[:,i].min()), speed_max=float(speed[:,i].max()),
            angular_speed_max=float(angular[:,i].max()), final_xyz=xyz[-1,i].tolist(),
            stable_seated_hold=bool(valid[:,i].all()) if args.mode!="standing" else False,
            stable_standing_hold=bool(valid[:,i].all()) if args.mode=="standing" else None,
            max_descent_speed=float(-arrays["lin_vel"][:,i,2].min()),
            final_3s_hold=bool(valid[-150:,i].all()),
            max_unintended_contact_N=float(arrays["unintended_peak"][:,i].max()),
            max_seat_impact_N=float(arrays["seat_force_peak"][:,i].max()),
            fell=bool(((arrays["root_pose"][:,i,2]<.28)|(arrays["gravity"][:,i,2]>-.65)).any()),
            passed=bool(not ((arrays["root_pose"][:,i,2]<.28)|(arrays["gravity"][:,i,2]>-.65)).any() and valid[-150:,i].all() and arrays["unintended_peak"][:,i].max()<=10 and -arrays["lin_vel"][:,i,2].min()<.8)))
    ranked = sorted(summaries, key=lambda s:(not s["passed"],s["max_unintended_contact_N"],s["max_seat_impact_N"]))
    result = dict(mode=args.mode, gravity=list(cfg.sim.gravity), seconds=args.seconds,
        settle_seconds=2.0, num_envs=env.num_envs, root_unconstrained=True,
        configured_actuators_only=True, outcomes=ranked, seat_poses=seat_xyz.tolist(),force_sample_dt=cfg.sim.dt,body_names=env.scene["all_contacts"].body_names, articulation_body_names=robot.body_names,
        actual_joint_targets=q.cpu().tolist(), joint_names=robot.joint_names)
    (out/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    (out/"best_pose.json").write_text(json.dumps(ranked[0]["pose"], indent=2)+"\n")
    np.savez_compressed(out/"rollouts.npz", **arrays, dt=env.step_dt, initial_actions=target_actions.cpu().numpy())
    print("SCRIPTED_RESULTS", str(out), json.dumps(ranked[:4]), flush=True)
    env.close()
