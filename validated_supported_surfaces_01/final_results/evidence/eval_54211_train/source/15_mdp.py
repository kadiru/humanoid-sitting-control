"""Seat-specific observations, support measurement, rewards and failures."""
import torch

SUPPORT_SENSORS = ("pelvis_seat", "left_thigh_seat", "right_thigh_seat")


def seat_information(env):
    robot = env.scene["robot"].data
    seat = env.scene["seat"].data.root_pos_w.torch
    return torch.cat((seat - robot.root_pos_w.torch,
                      torch.tensor([0.50, 0.90, 0.10], device=env.device).expand(env.num_envs, -1)), dim=-1)


def support_fraction(env):
    # Positive world-Z normal force on the robot from the seat only.
    upward = torch.zeros(env.num_envs, device=env.device)
    for name in SUPPORT_SENSORS:
        forces = env.scene[name].data.normal_force_matrix_w.torch
        upward += forces[..., 2].sum(dim=(1, 2)).clamp_min(0)
    weight = env.scene["robot"].data.body_mass.torch.sum(dim=1) * 9.81
    return upward / weight.clamp_min(1.0)


def update_state(env):
    if not hasattr(env, "sit_hold"):
        env.sit_hold = torch.zeros(env.num_envs, device=env.device)
        env.sit_peak_hold = env.sit_hold.clone()
        env.sit_peak_support = env.sit_hold.clone()
        env.sit_hit = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env.sit_descent_ok = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        env.sit_state_step = -1
    if env.sit_state_step == env.common_step_counter:
        return
    env.sit_state_step = env.common_step_counter
    robot = env.scene["robot"].data
    delta = robot.root_pos_w.torch - env.scene["seat"].data.root_pos_w.torch
    fraction = support_fraction(env)
    inside = (delta[:, 0].abs() < 0.22) & (delta[:, 1].abs() < 0.40)
    height = (delta[:, 2] > 0.17) & (delta[:, 2] < 0.30)
    stable = (robot.root_lin_vel_w.torch.norm(dim=1) < 0.20) & (robot.root_ang_vel_w.torch.norm(dim=1) < 0.35)
    upright = robot.projected_gravity_b.torch[:, 2] < -0.90
    env.sit_descent_ok &= robot.root_lin_vel_w.torch[:, 2] > -0.80
    valid = (fraction > 0.25) & inside & height & stable & upright & env.sit_descent_ok
    if getattr(env.cfg, "require_clean_seat_contacts", False):
        from .curriculum import invalid_foot_seat
        valid &= ~invalid_foot_seat(env)
    env.sit_hold = torch.where(valid, env.sit_hold + env.step_dt, torch.zeros_like(env.sit_hold))
    env.sit_peak_hold = torch.maximum(env.sit_peak_hold, env.sit_hold)
    env.sit_peak_support = torch.maximum(env.sit_peak_support, fraction)
    env.sit_hit |= env.sit_hold >= 1.0


def seated_support(env):
    update_state(env)
    return support_fraction(env).clamp(0.0, 1.0) * (env.scene["robot"].data.projected_gravity_b.torch[:, 2] < -0.90)


def stable_hold(env):
    update_state(env)
    return env.sit_hold.clamp(max=1.0)


def target_pose(env):
    robot = env.scene["robot"].data
    delta = robot.root_pos_w.torch - env.scene["seat"].data.root_pos_w.torch
    # Source USD pelvis hull extends 0.152 m below its body origin.
    target = torch.tensor([0.12, 0.0, 0.21], device=env.device)
    return torch.exp(-((delta - target).square().sum(dim=1)) / 0.05)


def fast_descent(env):
    return (-env.scene["robot"].data.root_lin_vel_w.torch[:, 2] - 0.35).clamp_min(0).square()


def fallen(env):
    robot = env.scene["robot"].data
    z = robot.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return (z < 0.28) | (robot.projected_gravity_b.torch[:, 2] > -0.65)


def pelvis_floor_contact(env):
    # Seat contact is expressly allowed; distinguish it from floor contact.
    pelvis = env.scene["pelvis_seat"].data
    net = pelvis.net_normal_forces_w.torch[:, 0]
    seat = pelvis.normal_force_matrix_w.torch[:, 0].sum(dim=1)
    return (net - seat).norm(dim=1) > 30.0
