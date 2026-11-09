"""PPO pilot 3: independent attempts and a wider rigid-seat curriculum.

Uses the unchanged reward, frozen scaffold, sensor-only 144-channel contract,
and bounded 23-joint actions. Training resets each failed or completed attempt;
evaluation always records the full horizon with failures latched. A reset robot
holds for one normal control tick before taking its first camera measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from tensordict import TensorDict
import train_support_rl as experiment
from humanoid_sitting.support_readiness import startup_measurement, admitted_targets, admitted_corrections
from humanoid_sitting.support_sampling import (ResettableSensorInterface,
    reset_controller_rows, prime_contact_history, episode_boundaries)


class UnavailableDepth:
    def read(self, *_):
        values=np.zeros(10,dtype=np.float32); values[-1]=1.
        return values,dict(usable=False,reason='Startup floor/seat geometry unavailable')


class IndependentSupportVecEnv(experiment.SupportVecEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if options.reset_probe:
            self.reset_probe()

    def reset(self):
        with torch.inference_mode():
            self.raw.reset()
            self.t=0
            n=self.num_envs
            self.episode_length_buf.zero_()
            self.attempt=torch.zeros(n,dtype=torch.long,device=self.device)
            self.refusal_reasons=[None]*n
            self.observers=[UnavailableDepth() for _ in range(n)]
            self.live=torch.zeros(n,10,device=self.device); self.live[:,-1]=1.
            self.ready=torch.zeros(n,dtype=torch.bool,device=self.device)
            self.warming=torch.ones_like(self.ready)
            self.alive=self.ready.clone()
            self.flags={k:self.ready.clone() for k in ('fall','wrong_contact','pelvis_floor','impact','fast_descent','torque')}
            for name in ('hold','peak_hold','peak_force','total_reward','economy_sum','hold_samples','foot_sum','depth_count'):
                setattr(self,name,torch.zeros(n,device=self.device))
            self.initial_standing=self.ready.clone()
            self.standing_checked=self.ready.clone()
            self.previous=self.robot.data.joint_pos.torch.clone()
            self.controller=experiment.SoftLandingController(self.policy,self.contact,mode='actions')
            self.state=None
            self.sensor=ResettableSensorInterface(self.robot,[dict(valid=False)]*n,self.device,delay_steps=1,seed=self.args.seed+901)
            self.pending=torch.ones(n,device=self.device,dtype=torch.bool)
            self._observe(update_depth=True)
        return self.obs

    def _admit(self,ids):
        """Read new camera measurements only for freshly reset robots."""
        camera=self.raw.scene['startup_camera']; camera.update(0.,force_recompute=True)
        depth=camera.data.output['distance_to_image_plane'].torch.cpu().numpy()
        K=camera.data.intrinsic_matrices.torch.cpu().numpy()
        gravity=self.robot.data.projected_gravity_b.torch.cpu().numpy()
        estimates=[startup_measurement(experiment.estimate_body_frame,depth[i],K[i],self.mount,self.quat,gravity[i]) for i in ids.tolist()]
        self.sensor.reset_rows(ids,estimates)
        self.ready[ids]=self.sensor.ready[ids]
        self.alive[ids]=self.ready[ids]
        self.warming[ids]=False
        self.pending[ids]=True
        reset_controller_rows(self.controller,self.state,ids)
        live=self.raw.scene['live_camera']; live.update(0.,force_recompute=True)
        intrinsics=live.data.intrinsic_matrices.torch.cpu().numpy()
        for i,e in zip(ids.tolist(),estimates):
            self.refusal_reasons[i]=e.get('reason')
            self.observers[i]=(experiment.CurrentSupportDepth(experiment.BODY,intrinsics[i],self.live_T,e['surface_center'][2])
                               if e['valid'] else UnavailableDepth())
        with (self.out/'startup_admissions.jsonl').open('a') as f:
            f.write(json.dumps(dict(global_step=self.t,env_ids=ids.tolist(),attempt=self.attempt[ids].tolist(),
                ready=self.ready[ids].tolist(),refusal_reasons=[self.refusal_reasons[i] for i in ids.tolist()]))+'\n')

    def _reset_rows(self,ids):
        # IsaacLab's indexed reset writes selected physics state, forwards
        # kinematics and renders without any physics step or shared-clock tick.
        self.raw.reset(env_ids=ids)
        self.previous[ids]=self.robot.data.joint_pos.torch[ids]
        reset_controller_rows(self.controller,self.state,ids)
        self.pending[ids]=True
        self.live[ids]=0.; self.live[ids,-1]=1.
        self.episode_length_buf[ids]=0
        self.attempt[ids]+=1
        self.initial_standing[ids]=False; self.standing_checked[ids]=False
        for value in self.flags.values(): value[ids]=False
        for name in ('hold','peak_hold','peak_force','total_reward','economy_sum','hold_samples','foot_sum','depth_count'):
            getattr(self,name)[ids]=0
        # PhysX/RTX can still expose pre-reset geometry until the next ordinary
        # physics tick. Wait locally; never step the whole batch inside reset.
        self.warming[ids]=True
        self.ready[ids]=False; self.alive[ids]=False
        self.sensor.reset_rows(ids,[dict(valid=False)]*len(ids))
        for i in ids.tolist():
            self.observers[i]=UnavailableDepth(); self.refusal_reasons[i]=None

    def _observe(self,update_depth):
        # Reset rows get fresh geometry immediately. Other robots keep the
        # original 10 Hz depth schedule and their own controller/filter memory.
        selected=(torch.ones_like(self.pending) if update_depth else self.pending.clone()) & ~self.warming
        self.live[~selected,-1]=(self.live[~selected,-1]+.02).clamp(max=1.)
        if selected.any():
            started=time.perf_counter()
            camera=self.raw.scene['live_camera']; camera.update(0.,force_recompute=True)
            depth=camera.data.output['distance_to_image_plane'].torch.cpu().numpy()
            q=self.robot.data.joint_pos.torch.cpu().numpy()
            gravity=self.robot.data.projected_gravity_b.torch.cpu().numpy()
            for i in selected.nonzero().flatten().tolist():
                self.live[i]=torch.as_tensor(self.observers[i].read(depth[i],q[i],gravity[i])[0],device=self.device)
            self.depth_times.append(time.perf_counter()-started)
        packet=self.sensor.read(self.previous)
        prime_contact_history(self.controller,packet,self.pending)
        nominal,self.state=self.controller(packet[:,None,:],self.state)
        self.nominal=admitted_targets(nominal[:,0],self.previous[:,self.ids],self.ready).clone()
        reset_controller_rows(self.controller,self.state,self.warming.nonzero().flatten())
        vector=torch.cat((packet,self.nominal,self.live,self.controller.last_contact),-1)
        assert vector.shape==(self.num_envs,144) and torch.isfinite(vector).all()
        self.obs=TensorDict({'policy':vector},batch_size=[self.num_envs])
        self.pending.zero_()

    def finish_episode(self,ids=None):
        if ids is None: ids=torch.arange(self.num_envs,device=self.device)
        rows=super().finish_episode()
        result=[]
        for i in ids.tolist():
            row=rows[i]
            row.update(episode=int(self.attempt[i]),duration_s=float(self.episode_length_buf[i])*.02,
                       steps=int(self.episode_length_buf[i]),standing_checked=bool(self.standing_checked[i]),
                       startup_ready=bool(self.ready[i]),startup_refusal_reason=self.refusal_reasons[i],
                       depth_availability=float(self.depth_count[i]/max(1,int(self.episode_length_buf[i]))))
            row['failures']['perception_unavailable']=not row['startup_ready']
            row['success']=row['success'] and row['startup_ready'] and row['standing_checked']
            row['end_reason']=('horizon' if row['steps']>=self.max_episode_length else 'failure')
            result.append(row)
        return result

    def step(self,actions):
        start=time.perf_counter()
        with torch.inference_mode():
            assert torch.isfinite(actions).all()
            residual=admitted_corrections(actions,self.ready).clamp(-1,1)
            command=self.previous.clone()
            target=self.nominal+.08*residual
            target=target.clamp(min=self.previous[:,self.ids]-.05,max=self.previous[:,self.ids]+.05)
            limits=self.robot.data.joint_pos_limits.torch[:,self.ids]
            command[:,self.ids]=target.clamp(min=limits[...,0],max=limits[...,1])
            delta=command[:,self.ids]-self.previous[:,self.ids]
            self.raw.step(command)
            assert not self.raw.reset_buf.any(), 'Unexpected automatic simulator reset'
            if self.warming.any():self._admit(self.warming.nonzero().flatten())
            m=experiment.physical_metrics(self.raw)
            reward,good,terms=experiment.support_objective(m,delta,residual,self.alive)
            self.alive &= ~m['failure']
            for key in self.flags:self.flags[key] |= m['flags'][key]
            self.hold=torch.where(good,self.hold+.02,torch.zeros_like(self.hold))
            self.peak_hold=torch.maximum(self.peak_hold,self.hold)
            self.peak_force=torch.maximum(self.peak_force,m['peak_force'])
            self.total_reward+=reward
            self.economy_sum+=good*m['leg_rms_nm']; self.foot_sum+=good*m['foot_fraction']; self.hold_samples+=good
            self.depth_count+=self.live[:,8]
            self.previous=command.clone(); self.t+=1; self.episode_length_buf+=1
            standing_due=self.episode_length_buf==76
            self.initial_standing[standing_due]=m['standing'][standing_due]
            self.standing_checked[standing_due]=True
            done=episode_boundaries(self.alive,self.ready,self.episode_length_buf,self.max_episode_length,self.args.mode=='train')
            extras={'log':{f'Reward/{k}':v.mean().item() for k,v in terms.items()}}
            # Trace full physical evaluation BEFORE any reset; no labels enter observations.
            if self.record and self.t%5==0:
                self._observe(update_depth=True)
                self.trace.append(dict(t=self.t*.02,body=self.robot.data.body_pose_w.torch.cpu().numpy().copy(),
                    root=self.robot.data.root_pose_w.torch.cpu().numpy().copy(),support=m['support'].cpu().numpy(),
                    foot=m['foot_fraction'].cpu().numpy(),torque=m['leg_rms_nm'].cpu().numpy(),
                    force=m['peak_force'].cpu().numpy(),tilt=m['tilt2'].cpu().numpy(),
                    live=self.live.cpu().numpy().copy(),true_clearance=m['true_clearance'].cpu().numpy(),
                    correction=residual.cpu().numpy().copy(),alive=self.alive.cpu().numpy().copy()))
            ids=done.nonzero().flatten()
            if len(ids):
                records=self.finish_episode(ids); self.episodes.extend(records)
                with (self.out/'completed_attempts.jsonl').open('a') as f:
                    for row in records:f.write(json.dumps(row)+'\n')
                extras['log'].update({'Trial/success':np.mean([r['success'] for r in records]),
                    'Trial/peak_hold_s':np.mean([r['hold_s'] for r in records])})
                self._reset_rows(ids)
            # Exactly one observation/controller advance per physical tick for
            # non-reset robots. Evaluation tracing already made that observation.
            if not (self.record and self.t%5==0):
                self._observe(update_depth=self.t%5==0)
            elif len(ids):
                assert done.all() and self.args.mode!='train'
                # Evaluation's final full-batch reset has no subsequent action.
                self._observe(update_depth=True)
            if self.args.mode!='train' and done.all():self.t=0
            self.step_times.append(time.perf_counter()-start)
        return self.obs,reward,done.long(),extras

    def reset_probe(self):
        """Simulator integration test; separate from any research evaluation."""
        with torch.inference_mode():
            for _ in range(7):self.step(torch.zeros(self.num_envs,23,device=self.device))
            ids=torch.tensor([0],device=self.device); other=torch.arange(1,self.num_envs,device=self.device)
            def capture():
                d=self.robot.data
                values=dict(root=d.root_pose_w.torch[other],joint=d.joint_pos.torch[other],
                    velocity=d.joint_vel.torch[other],root_velocity=d.root_vel_w.torch[other],
                    controller=self.state[:,other],contact=self.controller.last_contact[other],
                    torque_filter=self.sensor.torque[other],geometry=self.sensor.geometry[other],
                    previous=self.previous[other],lengths=self.episode_length_buf[other],
                    phase=self.raw.transition_time[other],live=self.live[other],obs=self.obs['policy'][other])
                if self.controller.contact_state is not None:values['contact_state']=self.controller.contact_state[:,other]
                for j,frame in enumerate(self.sensor.buffer):
                    for k,v in enumerate(frame):values[f'delay_{j}_{k}']=v[other]
                return {k:v.clone() for k,v in values.items()}
            before=capture(); clock=self.raw._sim_step_counter
            self._reset_rows(ids)
            after=capture()
            errors={k:float((after[k].float()-v.float()).abs().max()) for k,v in before.items()}
            assert max(errors.values())<1.e-6,errors
            assert self.raw._sim_step_counter==clock
            assert not self.state[:,0].any()
            self._observe(update_depth=False)
            assert (self.obs['policy'][0,:23]-self.robot.data.joint_pos.torch[0,self.ids]).abs().max()<.01
            assert self.warming[0] and not self.ready[0] and not self.obs['policy'][0,98:109].any()
            assert not self.obs['policy'][0,132:141].any() and self.obs['policy'][0,141]==1
            # The age channel of unavailable live geometry is intentionally 1.
            # Sensor admission occurs after the next ordinary tick for all robots.
            old_lengths=self.episode_length_buf.clone()
            self.step(torch.zeros(self.num_envs,23,device=self.device))
            assert not self.warming[0]
            assert torch.equal(self.episode_length_buf[1:],old_lengths[1:]+1)
            cold=experiment.SoftLandingController(self.policy,self.contact,mode='actions')
            # Same batch shape avoids conflating CUDA kernel/batch rounding with
            # reset correctness; only robot 0 is expected to match a cold start.
            expected,cold_state=cold(self.obs['policy'][:,None,:109],None)
            torch.testing.assert_close(self.state[:,0:1],cold_state[:,0:1],atol=2.e-6,rtol=1.e-5)
            torch.testing.assert_close(self.controller.last_contact[0:1],cold.last_contact[0:1],atol=2.e-6,rtol=1.e-5)
            if self.ready[0]:torch.testing.assert_close(self.nominal[0],expected[0,0],atol=2.e-6,rtol=1.e-5)
            m=experiment.physical_metrics(self.raw)
            accepted=self.live[:,8]>0
            error=(self.live[:,0]-m['true_clearance']).abs()
            assert not (accepted & (error>.02)).any(),error
            actual=torch.tensor([self.cfg.geometry_bank[i]['height'] for i in experiment.furniture.indices(self.raw).tolist()],device=self.device)
            height_error=(self.sensor.geometry[:,3]-actual).abs()
            assert not (self.ready & (height_error>.015)).any(),height_error
            # Real step and horizon boundary must reset only the selected robot.
            self.episode_length_buf[0]=self.max_episode_length-1
            old_lengths=self.episode_length_buf.clone()
            _,_,done,_=self.step(torch.zeros(self.num_envs,23,device=self.device))
            assert done[0] and not done[1:].any(),done
            assert self.episode_length_buf[0]==0
            assert torch.equal(self.episode_length_buf[1:],old_lengths[1:]+1)
            result=dict(passed=True,other_robot_max_deltas=errors,no_physics_tick_during_reset=True,cold_controller_equivalent=True,
                        single_horizon_done=done.tolist(),accepted_clearance_error_m=error[accepted].tolist(),
                        startup_height_error_m=height_error[self.ready].tolist(),camera_admission_after_one_normal_tick=True)
            (self.out/'reset_probe.json').write_text(json.dumps(result,indent=2)+'\n')
            print('RESET_PROBE',json.dumps(result),flush=True)
            # Probe trials are separate and do not enter the subsequent smoke run.
            (self.out/'probe_attempts.json').write_text(json.dumps(self.episodes,indent=2)+'\n')
            self.episodes.clear(); self.reset()

    def close(self):
        active=self.finish_episode()
        for row in active:
            row['complete']=False; row['success']=False; row['end_reason']='training_budget'
        accounting=dict(complete_attempts=len(self.episodes),
            completed_transitions=sum(x['steps'] for x in self.episodes),
            unfinished_transitions=int(self.episode_length_buf.sum()),
            unfinished_attempts=active if self.args.mode=='train' else [],
            evaluation_horizon_s=self.args.seconds,
            training_ends_at_first_failure=True,finite_horizon_terminal=True)
        (self.out/'attempt_accounting.json').write_text(json.dumps(accounting,indent=2)+'\n')
        super().close()


def curriculum_bank(count,seed,balanced):
    rng=np.random.default_rng(seed)
    bank=experiment.furniture.bank(2,seed,count,True)
    ranges=dict(depth=(.40,.64),width=(.78,1.14),height=(.26,.35),yaw=(-8.,8.),x=(-.44,-.36),y=(-.04,.04))
    for key,(lo,hi) in ranges.items():
        values=rng.permutation(np.tile(np.linspace(lo,hi,4),count//4)) if balanced else rng.uniform(lo,hi,count)
        if key=='yaw':values=np.deg2rad(values)
        for g,v in zip(bank,values):g[key]=float(v)
    return bank


parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--bank-profile',choices=['broad','legacy'],default='broad')
parser.add_argument('--reset-probe',action='store_true')
options,remaining=parser.parse_known_args()
sys.argv=[sys.argv[0],*remaining]
meta=argparse.ArgumentParser(add_help=False)
meta.add_argument('--output',required=True); meta.add_argument('--mode',required=True)
meta.add_argument('--geometry-count',type=int,required=True);meta.add_argument('--geometry-seed',type=int,required=True)
args,_=meta.parse_known_args()
original_cfg=experiment.DeploymentEnvCfg


def independent_cfg(**kwargs):
    out=Path(args.output)
    sources=[Path(__file__),experiment.ROOT/'sitting_task/src/humanoid_sitting/support_sampling.py',
             experiment.ROOT/'sitting_task/src/humanoid_sitting/support_readiness.py',
             experiment.ROOT/'scripts/test_support_sampling.py']
    manifest=json.loads((out/'manifest.json').read_text())
    for source in sources:
        shutil.copy2(source,out/'source'/source.name)
        manifest['hashes'][str(source.relative_to(experiment.ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    manifest['sampling']=dict(independent_resets=True,training_early_failure_terminal=True,
        evaluation_full_horizon=True,reset_physics_steps=0,startup_hold_control_ticks=1,reset_probe=options.reset_probe,
        finite_horizon_terminal=True,unfinished_attempts_recorded=True)
    manifest['bank_profile']=options.bank_profile
    if options.bank_profile=='broad':
        assert args.geometry_count%4==0
        kwargs['geometry_bank']=curriculum_bank(args.geometry_count,args.geometry_seed,args.mode=='train')
        manifest['curriculum']='Fixed wider rigid prototypes with randomized robot starts at every independent attempt; no holdout copied.'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return original_cfg(**kwargs)


experiment.SupportVecEnv=IndependentSupportVecEnv
experiment.DeploymentEnvCfg=independent_cfg
if __name__=='__main__':experiment.main()
