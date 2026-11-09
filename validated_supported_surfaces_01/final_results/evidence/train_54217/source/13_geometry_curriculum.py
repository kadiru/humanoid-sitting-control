"""Warm-start training adapter for the frozen geometry reference.

Partial episode resets refresh only the reset environments' geometry anchors.
"""
import torch
from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass
from .geometry_contact_v11 import GeometryAction, GeometryActionCfg, GeometryEnvCfg
from . import furniture

class TrainingGeometryAction(GeometryAction):
    def reset(self,env_ids=None):
        super().reset(env_ids)
        device=self._env.device
        ids=torch.arange(self._env.num_envs,device=device) if env_ids is None else (torch.arange(self._env.num_envs,device=device)[env_ids] if isinstance(env_ids,slice) else env_ids)
        pending=getattr(self,'pending_reset',torch.zeros(self._env.num_envs,dtype=torch.bool,device=device))
        pending[ids]=True;self.pending_reset=pending
    def initialize_geometry(self):
        old={k:getattr(self,k).clone() for k in ('deep','seated','geometry','alignment') if hasattr(self,k)}
        super().initialize_geometry()
        if old:
            keep=~self.pending_reset
            for k,v in old.items():getattr(self,k)[keep]=v[keep]
        self.pending_reset=torch.zeros(self._env.num_envs,dtype=torch.bool,device=self._env.device)

@configclass
class TrainingGeometryActionCfg(GeometryActionCfg):
    class_type=TrainingGeometryAction

def geometry_pose_reward(env):
    delta=env.scene['robot'].data.root_pos_w.torch-env.scene['seat'].data.root_pos_w.torch
    g=furniture.geometry_tensor(env);c=g[:,3].cos();s=g[:,3].sin()
    local=torch.stack((c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1],delta[:,2]),dim=1)
    goal=torch.stack((g[:,0]/2-.12,torch.zeros_like(c),torch.full_like(c,.1984)),dim=1)
    return torch.exp(-(local-goal).square().sum(1)/.02)

@configclass
class GeometryTrainingEnvCfg(GeometryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joints=TrainingGeometryActionCfg(asset_name='robot',joint_names=['.*'])
        self.episode_length_s=20.
        self.rewards.target.func=geometry_pose_reward

# Initialization/IK and reset state are identical between v11 and v12; only the
# common height-to-lean rule changes. Keep v11 selectable to reproduce its pilot.
from .geometry_contact_v12 import GeometryAction as V12GeometryAction

class V12TrainingGeometryAction(TrainingGeometryAction):
    process_actions=V12GeometryAction.process_actions

@configclass
class V12TrainingGeometryActionCfg(TrainingGeometryActionCfg):
    class_type=V12TrainingGeometryAction

@configclass
class GeometryV12TrainingEnvCfg(GeometryTrainingEnvCfg):
    geometry_stage:int=2
    geometry_count:int=16
    include_families:bool=True
    arm_clearance:bool=True
    def __post_init__(self):
        super().__post_init__()
        self.actions.joints=V12TrainingGeometryActionCfg(asset_name='robot',joint_names=['.*'])
