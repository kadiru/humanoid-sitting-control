"""Bounded checkpoint inference and G1 collision inspection without rendering."""
import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_physx.physics import PhysxCfg
from isaaclab_rl.rsl_rl import (RslRlVecEnvWrapper, create_rsl_rl_runner,
                              check_rsl_rl_version, handle_deprecated_rsl_rl_cfg)
import isaaclab_tasks
from isaaclab_tasks.utils import resolve_task_config

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--full_asset", action="store_true")
add_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point", play_mode=True)
cfg.sim.physics = PhysxCfg()
cfg.sim.device = args.device
cfg.scene.num_envs = 4
if args.full_asset:
    from isaaclab_assets import G1_CFG
    cfg.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
cfg.seed = 42
agent_cfg.device = args.device
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
with launch_simulation(cfg, args):
    from pxr import Usd, UsdGeom, UsdPhysics
    import numpy as np
    import omni.usd

    env = RslRlVecEnvWrapper(gym.make("Isaac-Velocity-Flat-G1", cfg=cfg))
    runner = create_rsl_rl_runner(env, agent_cfg)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=args.device)
    obs = env.get_observations()
    resets = 0
    with torch.inference_mode():
        for _ in range(120):
            actions = policy(obs)
            assert torch.isfinite(actions).all()
            obs, reward, done, info = env.step(actions)
            assert torch.isfinite(reward).all()
            resets += int(done.sum().item())
            policy.reset(done)
    stage = Usd.Stage.Open(cfg.scene.robot.spawn.usd_path)
    collisions = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"]).ComputeWorldBound(prim).ComputeAlignedRange()
            authored_points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            points = np.asarray(authored_points) if authored_points is not None else np.empty((0, 3))
            collisions.append({"path": path, "type": prim.GetTypeName(),
                "enabled": UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
                "purpose": str(UsdGeom.Imageable(prim).GetPurposeAttr().Get()),
                "approximation": str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()),
                "local_mesh_bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()] if points.size else None,
                "world_bounds": [list(bound.GetMin()), list(bound.GetMax())] if not bound.IsEmpty() else None})
    asset_label = "full" if args.full_asset else "minimal"
    Path(f"../logs/g1-{asset_label}-collision-geometry.json").write_text(json.dumps(collisions, indent=2) + "\n")
    result = {"checkpoint": args.checkpoint, "steps": 120, "environments": 4,
              "resets": resets, "finite_actions_and_rewards": True,
              "robot_body_names": env.unwrapped.scene["robot"].body_names,
              "collision_shapes": len(collisions)}
    Path("../logs/g1-evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("G1_CHECKPOINT_EVALUATION_PASSED", json.dumps(result), flush=True)
    env.close()
