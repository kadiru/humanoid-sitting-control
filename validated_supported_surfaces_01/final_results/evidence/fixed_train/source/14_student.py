"""Teacher-free joint-target actions and explicit 99-dimensional student state.

No inverse kinematics, teacher targets, furniture-family identifiers, or reference
trajectory are used. Numerical geometry and model-derived balance state are still
privileged inputs; RGB-D substitution is a separate experiment.
"""
import torch
from isaaclab.envs import mdp as base
from isaaclab.managers import ObservationTermCfg
from isaaclab.utils import configclass
from . import furniture,mdp
from .geometry_curriculum import GeometryV12TrainingEnvCfg

def student_observation(env):
    robot=env.scene['robot'];d=robot.data;t=env.transition_time
    root=d.root_pos_w.torch-env.scene.env_origins
    if not hasattr(env,'student_anchor'):
        env.student_anchor=root[:,:2].clone()
        env.student_first_support=torch.full_like(t,float('nan'))
    invalid=torch.isnan(env.student_anchor[:,0])
    env.student_anchor[invalid]=root[invalid,:2]
    support=mdp.support_fraction(env)
    env.student_first_support=torch.where((support>.25)&torch.isnan(env.student_first_support),t,env.student_first_support)
    has_contact=torch.isfinite(env.student_first_support)
    age=torch.where(has_contact,t-env.student_first_support,torch.zeros_like(t))
    geom=torch.tensor([[g[k] for k in ('height','depth','width','x','y','yaw')] for g in env.cfg.geometry_bank],device=env.device)[furniture.indices(env)]
    anchor=torch.stack((geom[:,3]+geom[:,5].cos()*geom[:,1]/2-env.student_anchor[:,0],geom[:,4]-env.student_anchor[:,1]),dim=1)
    mass=d.body_mass.torch;com=(d.body_com_pos_w.torch*mass[...,None]).sum(1)/mass.sum(1,keepdim=True)
    feet=d.body_pos_w.torch[:,[robot.body_names.index(f'{s}_ankle_roll_link') for s in ('left','right')],0].mean(1)
    return torch.cat((d.root_lin_vel_b.torch,d.root_ang_vel_b.torch,d.projected_gravity_b.torch,
        d.joint_pos.torch-d.default_joint_pos.torch,d.joint_vel.torch,
        furniture.seat_observation(env),furniture.orientation_observation(env),root[:,2:3],
        (t/20).unsqueeze(1),anchor,(com[:,0]-feet).unsqueeze(1),support.unsqueeze(1),
        (age/20).unsqueeze(1),has_contact.float().unsqueeze(1)),dim=1)

class StudentJointAction(base.JointPositionAction):
    def __init__(self,cfg,env):
        env.transition_time=torch.zeros(env.num_envs,device=env.device)
        super().__init__(cfg,env)
    def process_actions(self,actions):
        assert torch.isfinite(actions).all()
        self._raw_actions[:]=actions
        limits=self._asset.data.joint_pos_limits.torch[:,self._joint_ids]
        self._processed_actions=actions.clamp(min=limits[...,0],max=limits[...,1])
        self._env.transition_time+=self._env.step_dt
    def reset(self,env_ids=None):
        super().reset(env_ids)
        ids=env_ids if env_ids is not None else slice(None)
        self._env.transition_time[ids]=0
        if hasattr(self._env,'student_anchor'):
            self._env.student_anchor[ids]=float('nan');self._env.student_first_support[ids]=float('nan')

@configclass
class StudentJointActionCfg(base.JointPositionActionCfg):
    class_type=StudentJointAction
    use_default_offset=False
    offset=0.
    scale=1.

@configclass
class StudentEnvCfg(GeometryV12TrainingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joints=StudentJointActionCfg(asset_name='robot',joint_names=['.*'])
        for name in ('lin_vel','ang_vel','gravity','joints','joint_vel','actions','seat','phase','seat_orientation'):
            setattr(self.observations.policy,name,None)
        self.observations.policy.student=ObservationTermCfg(func=student_observation)

class FrozenDefaultResidualAction(StudentJointAction):
    """Ablation: old residual actor with only a fixed standing joint offset."""
    def process_actions(self,actions):
        self._raw_actions[:]=actions.clamp(-1,1)
        limits=self._asset.data.joint_pos_limits.torch[:,self._joint_ids]
        self._processed_actions=(self._asset.data.default_joint_pos.torch[:,self._joint_ids]+.08*self._raw_actions).clamp(min=limits[...,0],max=limits[...,1])
        self._env.transition_time+=self._env.step_dt

@configclass
class FrozenDefaultResidualActionCfg(StudentJointActionCfg):
    class_type=FrozenDefaultResidualAction

# Only a dummy encoder observation is exposed through the environment wrapper.
# The deployment actor receives the independent typed SensorPacket, not this wrapper.
def deployment_encoders(env):
    from humanoid_sitting.deployment import BODY_JOINT_NAMES
    r=env.scene['robot'];return r.data.joint_pos.torch[:,[r.joint_names.index(n) for n in BODY_JOINT_NAMES]]

@configclass
class DeploymentEnvCfg(StudentEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.student=None
        self.observations.policy.encoders=ObservationTermCfg(func=deployment_encoders)
