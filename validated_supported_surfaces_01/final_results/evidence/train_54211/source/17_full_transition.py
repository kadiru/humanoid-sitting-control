"""Validated forward-leaning stand-to-sit reference and bounded residual PPO task.

Legacy curriculum tasks and checkpoints are retained. Only ankle PD gains change;
all original actuator effort limits, contact checks and descent limits remain.
"""
import math
import torch
from isaaclab.envs import mdp as base
from isaaclab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from . import mdp
from .curriculum import SeatedStabilityEnvCfg, CurriculumPPORunnerCfg, goal_error

SUPPORT_NAMES = mdp.SUPPORT_SENSORS
ARM_PARTS = ('shoulder','elbow','wrist','palm','hand','zero_link','one_link','two_link',
             'three_link','four_link','five_link','six_link')


def elapsed(env):
    return getattr(env, 'transition_time', torch.zeros(env.num_envs,device=env.device))


def phase_observation(env):
    return (elapsed(env)/9.).clamp(0,1).unsqueeze(1)


def reference_targets(env, time=None):
    """All support arises from contacts; this computes joint targets only."""
    robot=env.scene['robot'];time=elapsed(env) if time is None else time
    pre=(time-2.).clamp(0,1)
    descent=((time-3.)/4.).clamp(0,1)
    recover=((time-7.)/2.).clamp(0,1)
    hips=-.2-.35*pre-1.05*descent
    knees=.42+.13*pre+.85*descent
    lean=.10*pre+.05*descent
    hips=hips*(1-recover)-1.25*recover
    knees=knees*(1-recover)+1.4*recover
    lean*=1-recover
    q=robot.data.default_joint_pos.torch.clone()
    mass=robot.data.body_mass.torch
    com=(robot.data.body_com_pos_w.torch*mass[...,None]).sum(dim=1)/mass.sum(dim=1,keepdim=True)
    feet_ids=[robot.body_names.index(f'{s}_ankle_roll_link') for s in ('left','right')]
    feet=robot.data.body_pos_w.torch[:,feet_ids,0].mean(dim=1)
    correction=(8.*(com[:,0]-feet)+1.5*robot.data.root_lin_vel_w.torch[:,0]).clamp(-.35,.35)
    correction*=mdp.support_fraction(env)<.25
    for j,name in enumerate(robot.joint_names):
        if name.endswith('hip_pitch_joint'):q[:,j]=hips
        elif name.endswith('knee_joint'):q[:,j]=knees
        elif name.endswith('ankle_pitch_joint'):q[:,j]=-hips-knees-lean+correction
    limits=robot.data.joint_pos_limits.torch
    return torch.maximum(torch.minimum(q,limits[...,1]),limits[...,0])


def seat_peak_force(env):
    # Four 5 ms physics samples, rather than only the last 20 ms control sample.
    history=env.scene['seat_contacts'].data.net_normal_forces_w_history.torch
    return history.sum(dim=2).norm(dim=-1).max(dim=1).values


def unintended_peak_force(env):
    histories=[]
    for side in ('left','right'):
        for body in ('foot','knee'):
            history=env.scene[f'{side}_{body}_seat'].data.normal_force_matrix_w_history.torch
            histories.append(history.sum(dim=(2,3)).norm(dim=-1))
    contact=env.scene['all_contacts']
    all_history=contact.data.net_normal_forces_w_history.torch
    arm_ids=[i for i,n in enumerate(contact.body_names) if any(p in n for p in ARM_PARTS)]
    arm=all_history[:,:,arm_ids].norm(dim=-1).flatten(1).max(dim=1).values
    downward=(-all_history[...,2]).clamp_min(0).flatten(1).max(dim=1).values
    other=torch.stack(histories,dim=2).sum(dim=2).max(dim=1).values
    return torch.maximum(torch.maximum(arm,other),downward)


def unintended_contact(env):
    return unintended_peak_force(env)>10.


def excessive_impact(env):
    weight=env.scene['robot'].data.body_mass.torch.sum(dim=1)*9.81
    return seat_peak_force(env)>2.*weight


def impact_penalty(env):
    weight=env.scene['robot'].data.body_mass.torch.sum(dim=1)*9.81
    return (seat_peak_force(env)/weight-1.).clamp_min(0).square()


def target_pose(env):
    delta=env.scene['robot'].data.root_pos_w.torch-env.scene['seat'].data.root_pos_w.torch
    goal=torch.tensor([.116,0.,.1984],device=env.device)
    return torch.exp(-(delta-goal).square().sum(dim=1)/.02)


class FullTransitionAction(base.JointPositionAction):
    def __init__(self,cfg,env):
        env.transition_time=torch.zeros(env.num_envs,device=env.device)
        super().__init__(cfg,env)

    def process_actions(self,actions):
        super().process_actions(actions.clamp(-1,1))
        self._processed_actions+=reference_targets(self._env)[:,self._joint_ids]
        limits=self._asset.data.joint_pos_limits.torch[:,self._joint_ids]
        self._processed_actions=torch.maximum(torch.minimum(self._processed_actions,limits[...,1]),limits[...,0])
        # Independent of episode_length_buf, which PPO randomizes at startup.
        self._env.transition_time+=self._env.step_dt

    def reset(self,env_ids=None):
        super().reset(env_ids)
        self._env.transition_time[env_ids if env_ids is not None else slice(None)]=0.


@configclass
class FullTransitionActionCfg(base.JointPositionActionCfg):
    class_type=FullTransitionAction
    use_default_offset=False
    offset=0.
    scale=.08


@configclass
class FullTransitionEnvCfg(SeatedStabilityEnvCfg):
    curriculum_stage:int=3
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.pos=(0.,0.,.72)
        self.scene.robot.init_state.rot=(0.,math.sin(-.02/2),0.,math.cos(-.02/2))
        self.scene.robot.actuators['feet'].stiffness=160.
        self.scene.robot.actuators['feet'].damping=4.
        self.scene.seat.spawn.activate_contact_sensors=True
        self.scene.seat_contacts=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Seat',history_length=4)
        self.scene.lazy_sensor_update=False
        for name in (*SUPPORT_NAMES,'left_foot_seat','right_foot_seat','left_knee_seat','right_knee_seat','all_contacts'):
            getattr(self.scene,name).history_length=4
        self.actions.joints=FullTransitionActionCfg(asset_name='robot',joint_names=['.*'])
        self.observations.policy.phase=ObservationTermCfg(func=phase_observation)
        self.events.reset_root.params['pose_range']={
            'x':(-.015,.015),'y':(-.008,.008),'z':(-.005,.005),
            'pitch':(-.015,.015),'roll':(-.005,.005),'yaw':(-.005,.005)}
        self.events.reset_joints.params['position_range']=(-.01,.01)
        self.terminations.foot_seat=TerminationTermCfg(func=unintended_contact)
        self.terminations.seat_impact=TerminationTermCfg(func=excessive_impact)
        self.rewards.target.func=target_pose
        self.rewards.impact=RewardTermCfg(func=impact_penalty,weight=-2.)
        self.episode_length_s=16.


@configclass
class FullTransitionPPORunnerCfg(CurriculumPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name='g1_full_transition'
        self.max_iterations=40
        self.save_interval=10
