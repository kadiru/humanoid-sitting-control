"""Seat-conditioned quasi-static IK with actual G1 mass and joint frames.

Foot placement is measured after standing settles. A shared pelvis path maintains
COM above the feet during descent; seat contact then permits upright recovery.
There are no branches or parameters keyed by furniture identity.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch
from isaaclab.utils import configclass
from . import furniture, full_transition, mdp
from .geometry_reference import LegKinematics

class BalancedKinematics(LegKinematics):
    def __init__(self):
        super().__init__()
        path=Path(__file__).resolve().parents[4]/'runs/g1_kinematics_geometry_com_01/model.json'
        model=json.loads(path.read_text());self.model=model
        names=model['body_names'];masses=np.array(model['body_mass']);local=np.array(model['body_com_local'])
        q=dict(zip(model['joint_names'],model['default_q']))
        q.update(left_shoulder_pitch_joint=.8,right_shoulder_pitch_joint=.8,left_shoulder_roll_joint=1.6,right_shoulder_roll_joint=-1.6,left_elbow_pitch_joint=2.2,right_elbow_pitch_joint=2.2)
        transforms={'pelvis':np.eye(4)};pending=model['joints'].copy()
        while pending:
            rest=[]
            for j in pending:
                if not j['body0'] or j['body0'][0] not in transforms:rest.append(j);continue
                a=self.frame(j['p0'],j['q0']);b=np.linalg.inv(self.frame(j['p1'],j['q1']));r=np.eye(4)
                if j['name'] in q:
                    axis=np.zeros(3);axis['XYZ'.index(j['axis'])]=q[j['name']];r[:3,:3]=Rotation.from_rotvec(axis).as_matrix()
                transforms[j['body1'][0]]=transforms[j['body0'][0]]@a@r@b
            if len(rest)==len(pending):raise ValueError('Incomplete G1 frame graph')
            pending=rest
        legnames=[j['body1'][0] for j in self.chain]
        self.leg_mass=np.array([masses[names.index(n)] for n in legnames]);self.leg_com=np.array([local[names.index(n)] for n in legnames])
        excluded=set(legnames+[n.replace('left_','right_') for n in legnames]);self.upper_sum=np.zeros(3)
        for i,n in enumerate(names):
            if n not in excluded:self.upper_sum+=masses[i]*(transforms[n][:3,:3]@local[i]+transforms[n][:3,3])
        self.total_mass=masses.sum()
    def evaluate(self,v):
        hip,knee,pitch=v;t=np.eye(4);weighted=self.upper_sum.copy()
        for j,(a,b),angle,mass,com in zip(self.chain,self.frames,[hip,0,0,knee,-hip-knee-pitch,0],self.leg_mass,self.leg_com):
            r=np.eye(4);axis=np.zeros(3);axis['XYZ'.index(j['axis'])]=angle;r[:3,:3]=Rotation.from_rotvec(axis).as_matrix();t=t@a@r@b
            weighted+=2*mass*(t[:3,:3]@com+t[:3,3])
        rot=Rotation.from_euler('y',pitch).as_matrix()
        return (rot@t[:3,3])[[0,2]],(rot@(weighted/self.total_mass))[0]
    def solve_balanced(self,pelvis,foot,initial):
        def residual(v):
            f,com=self.evaluate(v)
            return [f[0]+pelvis[0]-foot[0],f[1]+pelvis[1]-foot[1],com+pelvis[0]-foot[0]]
        sol=least_squares(residual,initial,bounds=([-2.3,.05,-.2],[-.05,2.4,.8]),xtol=1e-9,ftol=1e-9,gtol=1e-9)
        return sol.x,float(np.linalg.norm(sol.fun))
    def solve_upright(self,pelvis,foot,initial):
        sol=least_squares(lambda v:self.evaluate([*v,0])[0]+pelvis-foot,initial[:2],bounds=([-2.3,.05],[-.05,2.4]))
        return np.array([*sol.x,0]),float(np.linalg.norm(sol.fun))

class BalancedAction(full_transition.FullTransitionAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env);self.model=BalancedKinematics()
        self.ready=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        self.plans=torch.zeros((env.num_envs,33,3),device=env.device)
        self.hold_targets=torch.zeros((env.num_envs,3),device=env.device)
        self.contact_time=torch.full((env.num_envs,),float('nan'),device=env.device)
        self.plan_audit=[]
    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'ready'):
            ix=env_ids if env_ids is not None else slice(None);self.ready[ix]=False;self.contact_time[ix]=float('nan')
    def compile(self,ids):
        env=self._env;r=env.scene['robot'];indices=furniture.indices(env).cpu().numpy()
        feetids=[r.body_names.index(f'{s}_ankle_roll_link') for s in ('left','right')]
        feet=(r.data.body_pos_w.torch[:,feetids].mean(1)-env.scene.env_origins).cpu().numpy()
        roots=(r.data.root_pos_w.torch-env.scene.env_origins).cpu().numpy()
        for i in ids.cpu().tolist():
            g=env.cfg.geometry_bank[indices[i]].copy()
            if env.cfg.reference_mode=='canonical_geometry':g.update(height=.3,depth=.5,width=.9,x=-.4,y=0,yaw=0)
            front=g['x']+np.cos(g['yaw'])*g['depth']/2
            goal=np.array([front-.12,g['height']+.1484]);start=roots[i,[0,2]];foot=feet[i,[0,2]]
            previous=np.array([-.2,.42,0.]);plan=[];errors=[]
            for s in np.linspace(0,1,33):
                smooth=10*s**3-15*s**4+6*s**5
                pelvis=start*(1-smooth)+goal*smooth
                previous,error=self.model.solve_balanced(pelvis,foot,previous);plan.append(previous.copy());errors.append(error)
            if max(errors)>1.e-3:
                raise ValueError(f'Quasi-static seat path is infeasible under the reference pitch bound: {max(errors):.4f} m residual')
            final,error=self.model.solve_upright(goal,foot,previous)
            if error>1.e-3:raise ValueError(f'Upright seated endpoint is unreachable: {error:.4f} m')
            self.plans[i]=torch.tensor(np.array(plan),device=env.device,dtype=torch.float32)
            self.hold_targets[i]=torch.tensor(final,device=env.device,dtype=torch.float32);self.ready[i]=True
            self.plan_audit.append(dict(env_index=i,geometry=g,foot_anchor_xz=foot.tolist(),pelvis_start_xz=start.tolist(),pelvis_goal_xz=goal.tolist(),max_ik_residual_m=max(errors),upright_ik_residual_m=error,maximum_reference_pitch_rad=max(x[2] for x in plan)))
    def process_actions(self,actions):
        env=self._env;r=env.scene['robot'];t=env.transition_time
        ids=torch.nonzero(~self.ready&(t>=2.),as_tuple=False).flatten()
        if len(ids):self.compile(ids)
        phase=((t-2.)/8.).clamp(0,1);scaled=phase*32;lo=scaled.long().clamp_max(31);fraction=(scaled-lo).unsqueeze(1)
        ix=torch.arange(env.num_envs,device=env.device)
        angles=self.plans[ix,lo]*(1-fraction)+self.plans[ix,lo+1]*fraction
        contact=(mdp.support_fraction(env)>.25)&(t>5.)
        self.contact_time=torch.where(contact&torch.isnan(self.contact_time),t,self.contact_time)
        # Complete the path before upright recovery; contacts determine its start.
        start=torch.maximum(torch.nan_to_num(self.contact_time,nan=1.e6),torch.full_like(t,10.))
        recovery=((t-start)/4).clamp(0,1).unsqueeze(1)
        angles=angles*(1-recovery)+self.hold_targets*recovery
        q=full_transition.reference_targets(env,time=torch.zeros_like(t))
        geometry=furniture.geometry_tensor(env)
        for j,n in enumerate(r.joint_names):
            if n.endswith('hip_pitch_joint'):q[:,j]=torch.where(self.ready,angles[:,0],q[:,j])
            elif n.endswith('knee_joint'):q[:,j]=torch.where(self.ready,angles[:,1],q[:,j])
            elif n.endswith('ankle_pitch_joint'):q[:,j]=torch.where(self.ready,-angles.sum(1),q[:,j])
            elif n.endswith('hip_yaw_joint'):q[:,j]=-geometry[:,3]*phase
        self._raw_actions[:]=actions.clamp(-1,1)
        self._processed_actions=q[:,self._joint_ids]+.08*self._raw_actions
        limits=r.data.joint_pos_limits.torch[:,self._joint_ids];self._processed_actions=self._processed_actions.clamp(min=limits[...,0],max=limits[...,1])
        env.transition_time+=env.step_dt

@configclass
class BalancedActionCfg(full_transition.FullTransitionActionCfg):
    class_type=BalancedAction

@configclass
class BalancedEnvCfg(furniture.FurnitureEnvCfg):
    def __post_init__(self):
        super().__post_init__();self.actions.joints=BalancedActionCfg(asset_name='robot',joint_names=['.*']);self.episode_length_s=20.
