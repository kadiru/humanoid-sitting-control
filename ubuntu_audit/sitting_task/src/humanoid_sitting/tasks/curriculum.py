"""Residual-control curriculum, with explicit promotion gates and unchanged legacy task."""
import math
import torch
from isaaclab.envs import mdp as base
from isaaclab.managers import ObservationTermCfg, EventTermCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from .config import SittingEnvCfg, SittingPPORunnerCfg

# All stages share policy/action dimensions. Promotion is external and measured.
STAGES = {
    0: dict(name='seated_stability', hip=-1.25, knee=1.4, ankle=-0.15, x=-0.3769, z=0.453, duration=0.0),
    1: dict(name='near_seat_descent', hip=-1.10, knee=1.35, ankle=-0.25, x=-0.34, z=0.48, duration=1.0),
    2: dict(name='partial_descent', hip=-0.90, knee=1.40, ankle=-0.50, x=-0.30, z=0.52, duration=1.5),
    3: dict(name='standing_to_sitting', hip=-0.20, knee=0.42, ankle=-0.23, x=0.0, z=0.74, duration=5.0),
}

def goal_joints(env):
    q=env.scene['robot'].data.default_joint_pos.torch.clone()
    for j,name in enumerate(env.scene['robot'].joint_names):
        if name.endswith('hip_pitch_joint'): q[:,j]=-1.25
        elif name.endswith('knee_joint'): q[:,j]=1.4
        elif name.endswith('ankle_pitch_joint'): q[:,j]=-0.15
    return q

def progress(env):
    duration=STAGES[env.cfg.curriculum_stage]['duration']
    elapsed=env.episode_length_buf.float()*env.step_dt
    if env.cfg.curriculum_stage==3: elapsed=(elapsed-2.0).clamp_min(0)
    return (elapsed/max(duration,env.step_dt)).clamp(0,1).unsqueeze(1) if duration else torch.ones((env.num_envs,1),device=env.device)

def target_pose(env):
    delta=env.scene['robot'].data.root_pos_w.torch-env.scene['seat'].data.root_pos_w.torch
    target=torch.tensor([0.0231,0.,0.1993],device=env.device)
    return torch.exp(-(delta-target).square().sum(dim=1)/.02)

def goal_error(env):
    return env.scene['robot'].data.joint_pos.torch-goal_joints(env)

class ResidualSittingAction(base.JointPositionAction):
    def process_actions(self,actions):
        # Bounded residuals cannot request extreme finger or leg positions.
        super().process_actions(actions.clamp(-1,1))
        phase=progress(self._env)
        start=self._asset.data.default_joint_pos.torch
        reference=start+(goal_joints(self._env)-start)*phase
        if self._env.cfg.curriculum_stage==3:
            from .mdp import support_fraction
            data=self._asset.data
            mass=data.body_mass.torch
            com=(data.body_com_pos_w.torch*mass[...,None]).sum(dim=1)/mass.sum(dim=1,keepdim=True)
            ids=[self._asset.body_names.index(f'{side}_ankle_roll_link') for side in ('left','right')]
            feet=data.body_pos_w.torch[:,ids,0].mean(dim=1)
            desired=-.30*phase[:,0]
            correction=(8.0*(com[:,0]-feet-desired)+1.5*data.root_lin_vel_w.torch[:,0]).clamp(-.5,.5)
            correction*=support_fraction(self._env)<.25
            for j,n in enumerate(self._asset.joint_names):
                if n.endswith('ankle_pitch_joint'): reference[:,j]+=correction
        self._processed_actions += reference[:,self._joint_ids]
        limits=self._asset.data.joint_pos_limits.torch[:,self._joint_ids]
        self._processed_actions=torch.maximum(torch.minimum(self._processed_actions,limits[...,1]),limits[...,0])

@configclass
class ResidualSittingActionCfg(base.JointPositionActionCfg):
    class_type=ResidualSittingAction
    use_default_offset=False
    offset=0.0
    scale=0.08

def invalid_foot_seat(env):
    # Feet may support on the floor, but cannot wedge against the seat.
    force=torch.zeros(env.num_envs,device=env.device)
    for side in ('left','right'):
        f=env.scene[f'{side}_foot_seat'].data.normal_force_matrix_w.torch.sum(dim=(1,2))
        force+=f.norm(dim=-1)
        knee=env.scene[f'{side}_knee_seat'].data.normal_force_matrix_w.torch.sum(dim=(1,2))
        force+=knee.norm(dim=-1)
    sensor=env.scene['all_contacts']
    net=sensor.data.net_normal_forces_w.torch
    downward=(net[...,2] < -10.0).any(dim=1)
    hand_ids=[i for i,n in enumerate(sensor.body_names) if any(part in n for part in ('shoulder','elbow','wrist','hand','zero_link','one_link','two_link','three_link','four_link','five_link','six_link'))]
    arm_contact=(net[:,hand_ids].norm(dim=-1)>10.0).any(dim=1)
    return (force>10.0) | downward | arm_contact

@configclass
class SeatedStabilityEnvCfg(SittingEnvCfg):
    curriculum_stage: int=0
    require_clean_seat_contacts: bool=True
    def __post_init__(self):
        super().__post_init__()
        stage=STAGES[self.curriculum_stage]
        self.scene.robot.init_state.pos=(stage['x'],0.0,stage['z'])
        # The full hand asset has asymmetric, nonzero thumb limits.
        self.scene.robot.init_state.joint_pos.update({'left_one_joint':1.0,'right_one_joint':-1.0,'left_two_joint':0.52,'right_two_joint':-0.52})
        for suffix,value in [('hip_pitch_joint',stage['hip']),('knee_joint',stage['knee']),('ankle_pitch_joint',stage['ankle'])]:
            self.scene.robot.init_state.joint_pos[f'.*_{suffix}']=value
        for side in ('left','right'):
            for body,label in [('ankle_roll_link','foot'),('knee_link','knee')]:
                setattr(self.scene,f'{side}_{label}_seat',ContactSensorCfg(prim_path=f'{{ENV_REGEX_NS}}/Robot/{side}_{body}',filter_prim_paths_expr=['{ENV_REGEX_NS}/Seat']))
        self.actions.joints=ResidualSittingActionCfg(asset_name='robot',joint_names=['.*'])
        self.observations.policy.joints=ObservationTermCfg(func=goal_error)
        self.observations.policy.phase=ObservationTermCfg(func=progress)
        self.events.reset_root.params['pose_range']={'x':(-0.003,0.003),'y':(-0.003,0.003)}
        self.events.reset_joints=EventTermCfg(func=base.reset_joints_by_offset,mode='reset',params={'position_range':(-0.01,0.01),'velocity_range':(0.,0.)})
        self.terminations.foot_seat=TerminationTermCfg(func=invalid_foot_seat)
        self.rewards.target.func=target_pose
        self.episode_length_s=12.0 if self.curriculum_stage==3 else 6.0
        if self.curriculum_stage==3:
            self.scene.robot.init_state.rot=(0.,math.sin(-.02/2),0.,math.cos(-.02/2))

@configclass
class NearSeatEnvCfg(SeatedStabilityEnvCfg):
    curriculum_stage: int=1

@configclass
class PartialDescentEnvCfg(SeatedStabilityEnvCfg):
    curriculum_stage: int=2

@configclass
class StandingTransitionEnvCfg(SeatedStabilityEnvCfg):
    curriculum_stage: int=3

@configclass
class CurriculumPPORunnerCfg(SittingPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name='g1_seated_curriculum'
        self.max_iterations=20
        self.save_interval=5
        self.actor.distribution_cfg.init_std=0.10
