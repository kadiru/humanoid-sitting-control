"""One seat-affordance-conditioned leg generator; no furniture-type branches.

Planar inverse kinematics uses joint frames exported from the actual loaded G1.
Only joint targets are applied. Initial root poses, effort caps and safety stay fixed.
"""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from . import furniture, full_transition, mdp
from isaaclab.utils import configclass

class LegKinematics:
    def __init__(self):
        path=Path(__file__).resolve().parents[4]/'runs/g1_kinematics_geometry_04/model.json'
        model=json.loads(path.read_text())
        self.chain=[next(j for j in model['joints'] if j['name']==name) for name in
            ['left_hip_pitch_joint','left_hip_roll_joint','left_hip_yaw_joint','left_knee_joint','left_ankle_pitch_joint','left_ankle_roll_joint']]
        self.frames=[(self.frame(j['p0'],j['q0']),np.linalg.inv(self.frame(j['p1'],j['q1']))) for j in self.chain]
    @staticmethod
    def frame(p,q):
        t=np.eye(4);t[:3,:3]=Rotation.from_quat(q).as_matrix();t[:3,3]=p;return t
    def foot(self,v,lean):
        t=np.eye(4)
        for j,(a,b),angle in zip(self.chain,self.frames,[v[0],0,0,v[1],-v.sum()-lean,0]):
            r=np.eye(4);axis=np.zeros(3);axis['XYZ'.index(j['axis'])]=angle;r[:3,:3]=Rotation.from_rotvec(axis).as_matrix();t=t@a@r@b
        return t[:3,3][[0,2]]
    def solve(self,base,lean,delta):
        goal=self.foot(np.array(base),lean)-delta
        solution=least_squares(lambda v:self.foot(v,lean)-goal,base,bounds=([-2.,.2],[-.4,2.3]))
        if np.linalg.norm(solution.fun)>1e-4:raise ValueError('Unreachable seat affordance')
        return solution.x

class GeometryAction(full_transition.FullTransitionAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.kinematics=LegKinematics();self.initialized=False
    def reset(self,env_ids=None):
        super().reset(env_ids)
        self.initialized=False
    def initialize_geometry(self):
        env=self._env;robot=env.scene['robot'];device=env.device
        g=np.array([[v[k] for k in ('height','depth','width','x','y','yaw')] for v in env.cfg.geometry_bank])[furniture.indices(env).cpu().numpy()]
        initial=(robot.data.root_pos_w.torch-env.scene.env_origins).cpu().numpy()
        if env.cfg.reference_mode=='canonical_geometry':g[:]=[.3,.5,.9,-.4,0,0];initial[:,:2]=0
        # Support target stays 12 cm behind the front edge, rather than a fixed world X.
        front=g[:,3]+np.cos(g[:,5])*g[:,1]/2
        dx=np.clip(.15*(front-(-.15)-initial[:,0]),-.008,0.)
        dz=g[:,0]-.3
        self.deep=torch.tensor(np.array([self.kinematics.solve([-1.6,1.4],.15,np.array([0.,z])) for x,z in zip(dx,dz)]),device=device,dtype=torch.float32)
        self.seated=torch.tensor(np.array([self.kinematics.solve([-1.25,1.4],0,np.array([x,z])) for x,z in zip(dx,dz)]),device=device,dtype=torch.float32)
        self.geometry=torch.tensor(g,device=device,dtype=torch.float32)
        self.alignment=torch.tensor(g[:,4]-initial[:,1],device=device,dtype=torch.float32)
        self.initialized=True
    def process_actions(self,actions):
        if not self.initialized:self.initialize_geometry()
        env=self._env;robot=env.scene['robot'];t=env.transition_time
        # Six-second descent and four-second recovery are common to all objects.
        pre=(t-2).clamp(0,1);down=((t-3)/8).clamp(0,1);up=((t-11)/4).clamp(0,1)
        hips=(-.2-.35*pre)*(1-down)+self.deep[:,0]*down
        knees=(.42+.13*pre)*(1-down)+self.deep[:,1]*down
        hips=hips*(1-up)+self.seated[:,0]*up;knees=knees*(1-up)+self.seated[:,1]*up
        lean=(.1*pre+.05*down+60*(.3-self.geometry[:,0]).clamp_min(0).square()*down)*(1-up)
        brake=(-robot.data.root_lin_vel_w.torch[:,0]-.04).clamp(0,.20)*.4
        brake*=mdp.support_fraction(env)<.25
        q=full_transition.reference_targets(env,time=torch.zeros_like(t))
        mass=robot.data.body_mass.torch;com=(robot.data.body_com_pos_w.torch*mass[...,None]).sum(1)/mass.sum(1,keepdim=True)
        ids=[robot.body_names.index(f'{s}_ankle_roll_link') for s in ('left','right')]
        feet=robot.data.body_pos_w.torch[:,ids,0].mean(1)
        correction=(8*(com[:,0]-feet)+1.5*robot.data.root_lin_vel_w.torch[:,0]).clamp(-.35,.35)*(mdp.support_fraction(env)<.25)
        for j,n in enumerate(robot.joint_names):
            if n.endswith('hip_pitch_joint'):q[:,j]=hips-brake
            elif n.endswith('knee_joint'):q[:,j]=knees
            elif n.endswith('ankle_pitch_joint'):q[:,j]=-hips-knees-lean+correction
            elif n.endswith('hip_yaw_joint'):q[:,j]=robot.data.default_joint_pos.torch[:,j]+self.geometry[:,5]*down*.5
            elif n.endswith('hip_roll_joint'):q[:,j]=robot.data.default_joint_pos.torch[:,j]-self.alignment.clamp(-.03,.03)*down
            elif n.endswith('shoulder_pitch_joint'):q[:,j]=.8+4*(.3-self.geometry[:,0])*down
        self._raw_actions[:]=actions.clamp(-1,1)
        self._processed_actions=q[:,self._joint_ids]+.08*self._raw_actions
        limits=robot.data.joint_pos_limits.torch[:,self._joint_ids]
        self._processed_actions=self._processed_actions.clamp(min=limits[...,0],max=limits[...,1])
        env.transition_time+=env.step_dt

@configclass
class GeometryActionCfg(full_transition.FullTransitionActionCfg):
    class_type=GeometryAction

@configclass
class GeometryEnvCfg(furniture.FurnitureEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joints=GeometryActionCfg(asset_name='robot',joint_names=['.*'])
