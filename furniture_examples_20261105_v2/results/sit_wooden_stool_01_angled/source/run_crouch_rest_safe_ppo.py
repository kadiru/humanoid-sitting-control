"""Small PPO pilot on physically validated surfaces; all resets start standing.

The actor-control window moves earlier only after validation. Training geometry
membership is a sample-selection mask, never a runtime actor/critic input.
Evaluation applies the policy to every condition, including the failed surfaces.
"""
import argparse,sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import level_rotation
import run_progressive_seat_ppo as progressive
from crouch_rest_validated_controller import plan,static_args

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--learning-start',type=float,choices=[10.,11.,13.,15.],default=15.)
p.add_argument('--eligible-indices',default='0,1,2,4,5,8')
p.add_argument('--bridge-inertial',default='/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
bridge_args,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
inertial=json.loads(Path(bridge_args.bridge_inertial).read_text())
progressive.rolling_plan=lambda *args,**kwargs:plan(*args,**kwargs,inertial=inertial)

class BridgeEnv(progressive.ProgressiveEnv):
    def __init__(self,raw,args,*rest,**kwargs):
        assert args.curriculum_stage==0,'Every reset starts standing; no frozen moving crouch resets'
        self.target_paths=torch.zeros(raw.num_envs,25,2,device=raw.device)
        self.target_ready=torch.zeros(raw.num_envs,dtype=torch.bool,device=raw.device);self.target_cursor=0
        self.foot_cached_tick=-1;self.foot_pitch=torch.zeros(raw.num_envs,2,device=raw.device)
        self.hip_pitch_ids=[raw.scene['robot'].joint_names.index(s+'_hip_pitch_joint') for s in ['left','right']]
        self.decision_enabled=torch.zeros(raw.num_envs,dtype=torch.bool,device=raw.device)
        self.control_ticks=0;self.teacher_ticks=0;self.excluded_ticks=0;self.control_by_case=torch.zeros(raw.num_envs,dtype=torch.long,device=raw.device)
        self.control_start_min=torch.full((raw.num_envs,),1.e9,device=raw.device)
        super().__init__(raw,args,*rest,**kwargs)
        self.span=torch.tensor([.008 if any(s in n for s in ['hip_roll','hip_yaw','ankle_roll']) else .012 if any(s in n for s in ['hip_pitch','knee','ankle_pitch']) else .015 if 'torso' in n else .08 for n in progressive.base.BODY_JOINT_NAMES],device=self.device)
        progressive.base.write_json(self.out/'bridge_contract.json',dict(args=vars(bridge_args),observations=406,actions=23,span=self.span.cpu().tolist(),initial_state='standing every episode',control_window_start_s=bridge_args.learning_start,approach_locked_until_s=10.,geometry_mask_training_only=True,evaluation_all_conditions=True,reward='unchanged run_progressive_seat_ppo physics reward; effort over all37joints, target65percent seat load is a soft reward, not a restored old-study60percent advancement gate',admission='whole20s success and zero all-tick slip, traction or illegal-contact violations on six development surfaces; three failed surfaces excluded from learning and reported as stress tests'))
        self._augment()
    def eligible(self):
        if self.args.mode!='train':return torch.ones(self.num_envs,dtype=torch.bool,device=self.device)
        indices=[int(x) for x in bridge_args.eligible_indices.split(',')]
        return torch.tensor(np.isin(self.index,indices),device=self.device)
    def _reset_rows(self,ids):
        super()._reset_rows(ids)
        self.target_ready[ids]=False;self.foot_cached_tick=-1;self.decision_enabled[ids]=False
        self._augment()
    def _observe(self,force=False):
        super()._observe(force)
        for a in self.plan_audit[self.target_cursor:]:
            self.target_paths[a['case']]=torch.tensor([[r[0],Rotation.from_quat(r[3:]).as_euler('xyz')[1]] for r in a['root_path']],device=self.device);self.target_ready[a['case']]=True
        self.target_cursor=len(self.plan_audit);self._augment()
    def reference(self):
        original=progressive.original.SplitEnv.reference(self)
        if not self.planning_enabled:return original
        elapsed=(self.episode_length_buf-self.near_start).float()*.02
        u=(elapsed/5.*24).clamp(0,24);lo=u.floor().long().clamp_max(23);f=u-lo;ids=torch.arange(self.num_envs,device=self.device)
        path=(1-f[:,None])*self.near_paths[ids,lo]+f[:,None]*self.near_paths[ids,lo+1]
        nominal=torch.where((self.near_start>=0)[:,None],path,original);ref=nominal.clone()
        target=(1-f[:,None])*self.target_paths[ids,lo]+f[:,None]*self.target_paths[ids,lo+1]
        velocity=(self.target_paths[ids,lo+1,1]-self.target_paths[ids,lo,1])*24/5.*(u<24)
        g=self.robot.data.projected_gravity_b.torch;pitch=torch.atan2(g[:,0],-g[:,2])
        correction=.6*(pitch-target[:,1])+.15*(self.robot.data.root_ang_vel_b.torch[:,1]-velocity)
        if self.estimated_root_x is not None:correction+=self.estimated_root_x-target[:,0]
        correction=correction.clamp(-.25,.25)*(elapsed/.2).clamp(0,1)*self.target_ready*(self.near_start>=0)
        ref[:,self.ankle_ids]+=correction[:,None]
        if self.foot_cached_tick!=self.t:
            q=self.robot.data.joint_pos.torch.cpu().numpy();gravity=g.cpu().numpy();values=[]
            for i in range(self.num_envs):
                R=level_rotation(gravity[i]);fk=self.model.fk(q[i]);values.append([Rotation.from_matrix(R@fk[s+'_ankle_roll_link'][:3,:3]).as_euler('xyz')[1] for s in ['left','right']])
            self.foot_pitch=torch.tensor(values,device=self.device,dtype=torch.float32);self.foot_cached_tick=self.t
        ramp=((elapsed-.5)/1.).clamp(0,1);ramp=ramp.square()*(3-2*ramp)*self.target_ready*(self.near_start>=0)
        shifted=(ref[:,self.ankle_ids]-nominal[:,self.ankle_ids])*ramp[:,None]
        ref[:,self.ankle_ids]-=shifted;ref[:,self.hip_pitch_ids]+=shifted
        ref[:,self.ankle_ids]+=(-3.*self.foot_pitch).clamp(-.15,.15)*ramp[:,None]
        return ref
    def get_observations(self):
        obs=super().get_observations()
        obs['learn']*=((self.motion_time()>=bridge_args.learning_start)&self.eligible()).float()[:,None]
        return obs
    def step(self,actions):
        self.decision_enabled=(self.warming==0)&(self.motion_time()>=bridge_args.learning_start)&self.eligible()
        return super().step(actions)
    def _physics_step(self,actions):
        enabled=self.decision_enabled&(self.warming==0)&(self.motion_time()>=bridge_args.learning_start)&self.eligible()
        if not self.policy_enabled:enabled[:]=False
        actions=actions.clone();actions[~enabled]=0
        if self.args.mode in ['train','evaluate']:
            active=self.warming==0;self.control_ticks+=int(enabled.sum());self.teacher_ticks+=int((active&~enabled&self.eligible()).sum());self.excluded_ticks+=int((active&~self.eligible()).sum());self.control_by_case+=enabled
            self.control_start_min=torch.where(enabled,torch.minimum(self.control_start_min,self.motion_time()),self.control_start_min)
        return super()._physics_step(actions)
    def close(self):
        super().close()
        progressive.base.write_json(self.out/'control_accounting.json',dict(control_ticks=self.control_ticks,teacher_ticks=self.teacher_ticks,excluded_ticks=self.excluded_ticks,control_by_case=self.control_by_case.cpu().tolist(),earliest_control_s=self.control_start_min.cpu().tolist(),active_physics=self.active_transitions,indexed_warmup=self.warmup_transitions))
        for name in ['run_crouch_rest_safe_ppo.py','crouch_rest_validated_controller.py','crouch_rest_motion.py','sagittal_static_bias.py']:
            f=Path(__file__).with_name(name);(self.out/'source'/name).write_text(f.read_text())
        progressive.base.write_json(self.out/'bridge_sources.json',{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(__file__).with_name('crouch_rest_validated_controller.py'),Path(__file__).with_name('crouch_rest_motion.py'),Path(__file__).with_name('sagittal_static_bias.py'),Path(bridge_args.bridge_inertial)]})

progressive.ProgressiveEnv=BridgeEnv
if __name__=='__main__':progressive.main()
