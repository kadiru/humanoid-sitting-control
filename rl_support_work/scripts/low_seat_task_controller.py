"""Coordinate bounded sagittal pelvis translation and angle through leg IK.

Selected normal-descent rows replace the inherited ankle/hip balance heuristic
and translation-only damping with one fixed-foot kinematic correction. The
measured plane, load feedforward, shear observer and frozen parent are retained.
The Jacobian constrains foot x/z/pitch; roll/yaw are not independently fixed.
No live simulator pose or contact label enters this controller.
"""
import json,shutil
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import level_rotation
from low_seat_normal_controller import install as install_normal

def task_rhs(relative, delta, angle):
    return np.array([-delta[0]-relative[2]*angle,
                     -delta[2]+relative[0]*angle, -angle])

def install(harness,config):
    install_normal(harness,config)
    Parent=harness.FeedbackEnv
    class TaskEnv(Parent):
        def __init__(self,*a,**kw):
            self.task_ready=False;self.task_records=[]
            super().__init__(*a,**kw)
            self.task_tick=-1;self.task_correction=torch.zeros(self.num_envs,37,device=self.device)
            self.task_ready=True

        def _reset_rows(self,ids):
            if self.task_ready:self.task_correction[ids]=0;self.task_tick=-1
            return super()._reset_rows(ids)

        def reference(self):
            ref=super().reference()
            if not self.task_ready or not self.planning_enabled:return ref
            elapsed=(self.episode_length_buf-self.near_start).float()*.02
            u=(elapsed/5*24).clamp(0,24);lo=u.floor().long().clamp_max(23);f=u-lo;ids=torch.arange(self.num_envs,device=self.device)
            target=(1-f[:,None])*self.target_paths[ids,lo]+f[:,None]*self.target_paths[ids,lo+1]
            velocity=(self.target_paths[ids,lo+1,1]-self.target_paths[ids,lo,1])*24/5*(u<24)
            g=self.robot.data.projected_gravity_b.torch;pitch=torch.atan2(g[:,0],-g[:,2]);gyro=self.robot.data.root_ang_vel_b.torch[:,1]
            old_c=.6*(pitch-target[:,1])+.15*(gyro-velocity)
            if self.estimated_root_x is not None:old_c+=self.estimated_root_x-target[:,0]
            old_c=old_c.clamp(-.25,.25)*(elapsed/.2).clamp(0,1)*self.target_ready*(self.near_start>=0)
            old_blend=((elapsed-.5)/1.).clamp(0,1);old_blend=old_blend.square()*(3-2*old_blend)
            selected=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
            for i in self.landing_states:selected[i]=True
            selected&=(self.near_start>=0)&(self.warming==0)
            # Remove the inherited bridge heuristic and translation-only loop.
            ref[:,self.hip_pitch_ids]-=(old_c*old_blend*selected)[:,None]
            ref[:,self.ankle_ids]-=(old_c*(1-old_blend)*selected)[:,None]
            ref[:,self.ankle_ids]-=(-3*self.foot_pitch).clamp(-.15,.15)*(old_blend*selected)[:,None]
            ref-=self.landing_correction*selected[:,None]
            if self.t%5==0 and self.task_tick!=self.t:
                self.task_tick=self.t;q=self.robot.data.joint_pos.torch.cpu().numpy();gravity=g.cpu().numpy()
                correction=np.zeros((self.num_envs,37));records=[]
                for i,state in self.landing_states.items():
                    if not selected[i]:continue
                    R=level_rotation(gravity[i]);fk=self.model.fk(q[i]);position=state['previous'];v=state['velocity']
                    ell=float(elapsed[i]);ui=np.clip(ell/5*24,0,24);l=min(int(ui),23);fr=ui-l
                    target_xyz=(1-fr)*state['roots'][l,:3]+fr*state['roots'][l+1,:3]
                    target_v=(state['roots'][l+1,:3]-state['roots'][l,:3])*24/5*(ui<24)
                    delta=np.array([config['kp_x'],0,config['kp_z']])*(target_xyz-position)
                    delta+=np.array([config['kd_x'],0,config['kd_z']])*(target_v-v)
                    delta=np.clip(delta,-config['translation_cap'],config['translation_cap'])
                    angle=np.clip(config['task_pitch_kp']*float(target[i,1]-pitch[i])+config['task_pitch_kd']*float(velocity[i]-gyro[i]),-config['task_pitch_cap'],config['task_pitch_cap'])
                    ramp=np.clip(ell/.5,0,1);ramp=ramp*ramp*(3-2*ramp)
                    for side in ['left','right']:
                        indices=[self.model.names.index(side+'_'+j+'_joint') for j in ['hip_pitch','knee','ankle_pitch']]
                        J=[];eps=1.e-5
                        for j in indices:
                            values=[]
                            for sign in [-1,1]:
                                trial=q[i].astype(float).copy();trial[j]+=sign*eps;foot=self.model.fk(trial)[side+'_ankle_roll_link']
                                values.append(np.r_[(R@foot[:3,3])[[0,2]],Rotation.from_matrix(R@foot[:3,:3]).as_euler('xyz')[1]])
                            J.append((values[1]-values[0])/(2*eps))
                        relative=R@fk[side+'_ankle_roll_link'][:3,3]
                        shift=np.linalg.solve(np.asarray(J).T,task_rhs(relative,delta,angle))
                        correction[i,indices]=np.clip(shift,-config['task_joint_cap'],config['task_joint_cap'])*ramp
                    records.append(dict(case=i,position=position.tolist(),target=target_xyz.tolist(),translation=delta.tolist(),angle_correction=float(angle),pitch=float(pitch[i]),target_pitch=float(target[i,1]),gyro=float(gyro[i])))
                self.task_correction=torch.tensor(correction,device=self.device,dtype=torch.float32)
                self.task_records.append(dict(tick=self.t,states=records,correction=correction.tolist()))
            return ref+self.task_correction

        def close(self):
            super().close();(self.out/'task_feedback_audit.json').write_text(json.dumps(self.task_records))
            shutil.copy2(__file__,self.out/'source'/Path(__file__).name)

    harness.FeedbackEnv=TaskEnv
