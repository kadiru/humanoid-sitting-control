"""First fixed-seat experiment: known geometry, aligned floating-base G1."""
import isaaclab.sim as sim
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp as base
from isaaclab.managers import (ObservationGroupCfg, ObservationTermCfg, RewardTermCfg,
                               TerminationTermCfg, EventTermCfg, SceneEntityCfg)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets import G1_CFG
from isaaclab_physx.physics import PhysxCfg
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
from . import mdp


@configclass
class SittingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim.GroundPlaneCfg(), collision_group=-1)
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim.DomeLightCfg(intensity=1500.0))
    robot: ArticulationCfg = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    seat = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Seat",
        spawn=sim.CuboidCfg(size=(0.50, 0.90, 0.10),
            rigid_props=sim.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            collision_props=sim.UsdPhysicsCollisionCfg(),
            mass_props=sim.MassCfg(mass=50.0),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.70)),
            physics_material=sim.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.40, 0.0, 0.25)),
    )
    pelvis_seat = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Seat"], history_length=3)
    left_thigh_seat = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/left_hip_yaw_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Seat"], history_length=3)
    right_thigh_seat = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/right_hip_yaw_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Seat"], history_length=3)
    all_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/[^/]*", history_length=3)


@configclass
class ActionsCfg:
    joints = base.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        lin_vel = ObservationTermCfg(func=base.base_lin_vel)
        ang_vel = ObservationTermCfg(func=base.base_ang_vel)
        gravity = ObservationTermCfg(func=base.projected_gravity)
        joints = ObservationTermCfg(func=base.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=base.joint_vel_rel)
        actions = ObservationTermCfg(func=base.last_action)
        seat = ObservationTermCfg(func=mdp.seat_information)
        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False
    policy = PolicyCfg()


@configclass
class EventsCfg:
    reset_root = EventTermCfg(func=base.reset_root_state_uniform, mode="reset",
        params={"pose_range": {}, "velocity_range": {}})
    reset_joints = EventTermCfg(func=base.reset_joints_by_scale, mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)})


@configclass
class RewardsCfg:
    target = RewardTermCfg(func=mdp.target_pose, weight=2.0)
    support = RewardTermCfg(func=mdp.seated_support, weight=4.0)
    hold = RewardTermCfg(func=mdp.stable_hold, weight=6.0)
    descent = RewardTermCfg(func=mdp.fast_descent, weight=-2.0)
    orientation = RewardTermCfg(func=base.flat_orientation_l2, weight=-1.0)
    action_rate = RewardTermCfg(func=base.action_rate_l2, weight=-0.02)
    joint_limits = RewardTermCfg(func=base.joint_pos_limits, weight=-1.0)
    torque = RewardTermCfg(func=base.joint_torques_l2, weight=-2.0e-6)
    termination = RewardTermCfg(func=base.is_terminated, weight=-10.0)


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=base.time_out, time_out=True)
    fallen = TerminationTermCfg(func=mdp.fallen)
    pelvis_floor = TerminationTermCfg(func=mdp.pelvis_floor_contact)


@configclass
class SittingEnvCfg(ManagerBasedRLEnvCfg):
    scene = SittingSceneCfg(num_envs=128, env_spacing=3.0, replicate_physics=True)
    sim = sim.SimulationCfg(dt=0.005, render_interval=4, physics=PhysxCfg())
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    decimation = 4
    episode_length_s = 8.0
    seed = 42
    def __post_init__(self):
        # Keep the hands clear of the seat during the first transition experiment.
        self.scene.robot.init_state.joint_pos["left_shoulder_roll_joint"] = 1.20
        self.scene.robot.init_state.joint_pos["right_shoulder_roll_joint"] = -1.20
        self.sim.default_visualizer_cfg = KitVisualizerCfg(
            eye=(1.6, 1.6, 1.25), lookat=(-0.15, 0.0, 0.45),
            origin_type="env", origin_env_index=0,
        )


@configclass
class SittingPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_sitting"
        self.max_iterations = 100
        self.save_interval = 25
        self.actor.distribution_cfg.init_std = 0.5
