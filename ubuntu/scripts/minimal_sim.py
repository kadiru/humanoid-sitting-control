"""Bounded Isaac Sim PhysX infrastructure check, using the selected Lab API."""
import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_physx.physics import PhysxCfg

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args = parser.parse_args()
cfg = SimulationCfg(dt=0.01, device=args.device, physics=PhysxCfg())
with launch_simulation(cfg, args):
    import carb
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg

    sim = SimulationContext(cfg)
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)
    cube = RigidObject(RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    ))
    sim.reset()
    for _ in range(120):
        cube.write_data_to_sim()
        sim.step()
        cube.update(cfg.dt)
    settings = carb.settings.get_settings()
    result = {
        "steps": 120,
        "device": cfg.device,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "physics_cuda_device": settings.get("/physics/cudaDevice"),
        "renderer_active_cuda_gpus": settings.get("/renderer/multiGpu/activeCudaGpus"),
        "cube_final_height": cube.data.root_pos_w.torch[0, 2].item(),
    }
    assert result["cuda_device_name"] == "NVIDIA GeForce RTX 4090", result
    assert 0.08 < result["cube_final_height"] < 0.12, result
    Path("../logs/minimal-sim-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("MINIMAL_SIM_PASSED", json.dumps(result), flush=True)
