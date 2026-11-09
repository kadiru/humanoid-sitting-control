"""Small, promotion-gated rigid furniture banks using the fixed-seat controller.

No cushion compliance. Banks are immutable per experiment; dimensions and poses
are regenerated from recorded seeds, with independent held-out geometry banks.
"""
import math
import numpy as np
import torch
import isaaclab.sim as sim
from isaaclab.managers import EventTermCfg, ObservationTermCfg
from isaaclab.utils import configclass
from .full_transition import FullTransitionEnvCfg, FullTransitionActionCfg, FullTransitionAction

RANGES = {
    0: dict(height=.005,depth=.025,width=.025,x=.005,y=.005,yaw=math.radians(1)),
    1: dict(height=.015,depth=.075,width=.10,x=.015,y=.015,yaw=math.radians(3)),
    2: dict(height=.025,depth=.10,width=.15,x=.025,y=.025,yaw=math.radians(5)),
}


def bank(stage,seed,count=12,families=False):
    rng=np.random.default_rng(seed);r=RANGES[stage];items=[]
    for i in range(count):
        family=['chair','bench','box','rigid_couch'][i%4] if families else 'slab'
        items.append(dict(family=family,depth=.5+rng.uniform(-r['depth'],r['depth']),
            width=.9+rng.uniform(-r['width'],r['width']),height=.3+rng.uniform(-r['height'],r['height']),
            x=-.4+rng.uniform(-r['x'],r['x']),y=rng.uniform(-r['y'],r['y']),yaw=rng.uniform(-r['yaw'],r['yaw'])))
    return items


def spawn_furniture(path,cfg,translation=None,orientation=None,**kwargs):
    """One kinematic rigid body with collision cuboids for seat/back/arms/box."""
    from pxr import Usd
    seat_cfg=sim.CuboidCfg(size=cfg.size,rigid_props=cfg.rigid_props,
        collision_props=cfg.collision_props,mass_props=cfg.mass_props,
        physics_material=cfg.physics_material,visual_material=cfg.visual_material,
        activate_contact_sensors=True)
    prim=sim.spawn_cuboid(path,seat_cfg,translation=translation,orientation=orientation,**kwargs)
    # Additional collision shapes belong to the same parent rigid body, not independent props.
    d,w,_=cfg.size
    pieces=[]
    if cfg.family in ('chair','rigid_couch'):
        pieces.append(('Back',(0.06,w,.35),(-d/2+.03,0.,.225)))
    if cfg.family=='rigid_couch':
        pieces += [(('Arm_Left' if s<0 else 'Arm_Right'),(d,.06,.20),(0.,s*(w/2+.03),.15)) for s in (-1,1)]
    if cfg.family=='box':pieces.append(('BoxBase',(d,w,cfg.top_height-.1),(0.,0.,-(cfg.top_height)/2)))
    if cfg.family in ('chair','bench','rigid_couch'):
        for x in (-d/2+.05,d/2-.05):
            for y in (-w/2+.05,w/2-.05):
                pieces.append((f'Leg_{len(pieces)}',(.05,.05,cfg.top_height-.1),(x,y,-cfg.top_height/2)))
    for name,size,pos in pieces:
        sim.spawn_cuboid(path+'/'+name,sim.CuboidCfg(size=size,collision_props=cfg.collision_props,
            visual_material=cfg.visual_material,physics_material=cfg.physics_material),translation=pos)
    sim.activate_contact_sensors(path)
    return prim


@configclass
class FurnitureSpawnCfg(sim.CuboidCfg):
    func=spawn_furniture
    family:str='slab'
    top_height:float=.3


def indices(env):
    # CloneCfg sequential groups equal-weight prototypes contiguously.
    return ((torch.arange(env.num_envs,device=env.device)+.5)*len(env.cfg.geometry_bank)/env.num_envs).long()


def geometry_tensor(env):
    return torch.tensor([[g['depth'],g['width'],.1,g['yaw']] for g in env.cfg.geometry_bank],device=env.device)[indices(env)]


def seat_observation(env):
    delta=env.scene['seat'].data.root_pos_w.torch-env.scene['robot'].data.root_pos_w.torch
    geometry=geometry_tensor(env);yaw=geometry[:,3];c=yaw.cos();s=yaw.sin()
    # Known seat-local numerical position and actual dimensions. Keep checkpoint width.
    relative=torch.stack((c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1],delta[:,2]),dim=1)
    return torch.cat((relative,geometry[:,:3]),dim=1)


def orientation_observation(env):
    yaw=geometry_tensor(env)[:,3]
    return torch.stack((yaw.sin(),yaw.cos()-1.),dim=1)


def reset_seats(env,env_ids):
    if env_ids is None or isinstance(env_ids,slice):env_ids=torch.arange(env.num_envs,device=env.device)[env_ids if env_ids is not None else slice(None)]
    g=torch.tensor([[v['x'],v['y'],v['height']-.05,v['yaw']] for v in env.cfg.geometry_bank],device=env.device)[indices(env)[env_ids]]
    pose=env.scene['seat'].data.default_root_pose.torch[env_ids].clone()
    pose[:,:3]=env.scene.env_origins[env_ids]+g[:,:3]
    pose[:,3:]=0.;pose[:,5]=(g[:,3]/2).sin();pose[:,6]=(g[:,3]/2).cos()
    env.scene['seat'].write_root_pose_to_sim_index(root_pose=pose,env_ids=env_ids)
    env.scene['seat'].write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(env_ids),6),device=env.device),env_ids=env_ids)


class FurnitureAction(FullTransitionAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        env.arm_lift_time=torch.full((env.num_envs,),float('nan'),device=env.device)

    def reset(self,env_ids=None):
        super().reset(env_ids)
        self._env.arm_lift_time[env_ids if env_ids is not None else slice(None)]=float('nan')

    def process_actions(self,actions):
        super().process_actions(actions)
        mode=self._env.cfg.reference_mode
        if mode=='frozen_standing':
            from .full_transition import reference_targets
            residual=self._processed_actions-reference_targets(self._env,time=self._env.transition_time-self._env.step_dt)[:,self._joint_ids]
            self._processed_actions=reference_targets(self._env,time=torch.zeros_like(self._env.transition_time))[:,self._joint_ids]+residual
        elif mode in ('phase_lag','gentle_recovery'):
            from .full_transition import reference_targets
            t=self._env.transition_time-self._env.step_dt
            residual=self._processed_actions-reference_targets(self._env,time=t)[:,self._joint_ids]
            mapped=(t-1.).clamp_min(0) if mode=='phase_lag' else torch.where(t<=7.,t,7.+(t-7.)*.5)
            self._processed_actions=reference_targets(self._env,time=mapped)[:,self._joint_ids]+residual
        limits=self._asset.data.joint_pos_limits.torch[:,self._joint_ids]
        self._processed_actions=torch.maximum(torch.minimum(self._processed_actions,limits[...,1]),limits[...,0])


@configclass
class FurnitureActionCfg(FullTransitionActionCfg):
    class_type=FurnitureAction


@configclass
class FurnitureEnvCfg(FullTransitionEnvCfg):
    geometry_stage:int=0
    geometry_seed:int=310
    geometry_count:int=12
    include_families:bool=False
    arm_clearance:bool=False
    reference_mode:str='normal'
    geometry_bank:list=None
    def __post_init__(self):
        super().__post_init__()
        if self.arm_clearance:
            self.scene.robot.init_state.joint_pos['.*_elbow_pitch_joint']=2.2
            self.scene.robot.init_state.joint_pos['left_shoulder_pitch_joint']=.8
            self.scene.robot.init_state.joint_pos['right_shoulder_pitch_joint']=.8
            self.scene.robot.init_state.joint_pos['left_shoulder_roll_joint']=1.6
            self.scene.robot.init_state.joint_pos['right_shoulder_roll_joint']=-1.6
        if self.geometry_bank is None:self.geometry_bank=bank(self.geometry_stage,self.geometry_seed,self.geometry_count,self.include_families)
        original=self.scene.seat.spawn
        assets=[FurnitureSpawnCfg(size=(g['depth'],g['width'],.1),family=g['family'],top_height=g['height'],
            rigid_props=original.rigid_props,collision_props=original.collision_props,mass_props=original.mass_props,
            visual_material=original.visual_material,physics_material=original.physics_material,
            activate_contact_sensors=True) for g in self.geometry_bank]
        self.scene.seat.spawn=sim.MultiAssetSpawnerCfg(assets_cfg=assets,activate_contact_sensors=True,random_choice=False)
        self.scene.replicate_physics=False
        self.events.reset_seats=EventTermCfg(func=reset_seats,mode='reset')
        self.observations.policy.seat=ObservationTermCfg(func=seat_observation)
        self.observations.policy.seat_orientation=ObservationTermCfg(func=orientation_observation)
        self.actions.joints=FurnitureActionCfg(asset_name='robot',joint_names=['.*'])
