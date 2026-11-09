"""Correctly timed 50-Hz geometry with an independently valid distance input.

An isolated entry point; original experiment scripts and checkpoints stay intact.
Runtime estimation uses only encoders, IMU and calibrated depth. True physics is
recorded separately in step() and used by the inherited training objective.
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
from rsl_rl.algorithms import PPO
from humanoid_sitting.support_camera import acquire_current_camera
from humanoid_sitting.support_landing_geometry import LandingGeometryTracker
from humanoid_sitting.support_continuous_features import ContinuousPatchFeatures
from humanoid_sitting.support_split_observation import (
    distance_features, expand_parent_state, install_split_linear)
from humanoid_sitting.support_retention import RetentionPPO

p = argparse.ArgumentParser(add_help=False)
p.add_argument('--parent', required=True)
p.add_argument('--bank-from')
p.add_argument('--interface', choices=['legacy','split'], default='split')
options, rest = p.parse_known_args(); sys.argv = [sys.argv[0], *rest]
import train_support_rl_independent as independent
experiment = independent.experiment


class ImageRecorder:
    def __init__(self, observer, env, case):
        self.observer, self.env, self.case = observer, env, case

    def read(self, depth, q, gravity):
        self.env.images[self.case] = np.asarray(depth).copy()
        return self.observer.read(depth, q, gravity)


class SplitObservationEnv(independent.IndependentSupportVecEnv):
    def __init__(self, *args, **kwargs):
        self.trackers = {}; self.patches = {}; self.cached = {}; self.images = {}
        self.control_records = []; self.label_records = []; self.feature_records = []
        self.sensor_records = []; self.depth_records = []; self.startup_records = []
        self.calibration = {}; self.camera_audit = []; self.reset_checks = []
        super().__init__(*args, **kwargs)

    def reset(self):
        self.trackers.clear(); self.patches.clear(); self.cached.clear()
        return super().reset()

    def _reset_rows(self, ids):
        unaffected = {i: (id(v), v.stamp) for i,v in self.trackers.items() if i not in ids.tolist()}
        super()._reset_rows(ids)
        for i in ids.tolist():
            self.trackers.pop(i, None); self.patches.pop(i, None); self.cached.pop(i, None)
        assert unaffected == {i: (id(self.trackers[i]),self.trackers[i].stamp) for i in unaffected}
        self.reset_checks.append(dict(t=self.t*.02,ids=ids.tolist(),other_trackers_unchanged=True,
                                      selected_trackers_cleared=True))

    def _admit(self, ids):
        acquire_current_camera(self.raw.scene['startup_camera'])
        acquire_current_camera(self.raw.scene['live_camera'])
        super()._admit(ids)
        d = self.robot.data; startup = self.raw.scene['startup_camera']
        for i in ids.tolist():
            if not self.ready[i]: continue
            observer = self.observers[i]
            depth = startup.data.output['distance_to_image_plane'].torch[i].cpu().numpy().copy()
            K = startup.data.intrinsic_matrices.torch[i].cpu().numpy().copy()
            q = d.joint_pos.torch[i].cpu().numpy().copy()
            gravity = d.projected_gravity_b.torch[i].cpu().numpy().copy()
            measured = experiment.estimate_body_frame(depth,K,self.mount,self.quat,gravity)
            tracker = LandingGeometryTracker(experiment.BODY,observer.K,observer.T_P_C,observer.seat_height)
            initialized = tracker.initialize_from_measured_floor((self.t+1)*.02,q,gravity,measured['floor_height_body_m'])
            self.trackers[i] = tracker; self.patches[i] = ContinuousPatchFeatures(tracker)
            self.cached.pop(i,None)
            self.observers[i] = ImageRecorder(observer,self,i)
            self.calibration[str(i)] = dict(K=observer.K.tolist(),T_P_C=observer.T_P_C.tolist(),
                estimated_seat_height_m=float(observer.seat_height))
            self.startup_records.append(dict(case=i,t=(self.t+1)*.02,attempt=int(self.attempt[i]),
                depth=depth,K=K,q=q,gravity=gravity,initialized=initialized))

    def _observe(self, update_depth):
        self.images = {}
        if self.t>0 and (update_depth or self.pending.any()) and (~self.warming).any():
            sim = self.raw.sim; step = sim.get_physics_step_count()
            q_before = self.robot.data.joint_pos.torch.clone()
            sim.render_context.update_scene_state(step)
            camera = self.raw.scene['live_camera']; acquire_current_camera(camera)
            assert step == sim.get_physics_step_count()
            assert torch.equal(q_before,self.robot.data.joint_pos.torch)
            age = camera._timestamp.numpy()-camera._timestamp_last_update.numpy()
            assert abs(age).max()<1.e-6
            self.camera_audit.append(dict(t=self.t*.02,max_capture_age_s=float(abs(age).max()),
                                         physics_unchanged=True))
        # Exactly one call advances the nominal recurrent controller.
        super()._observe(update_depth)
        stamp = self.t*.02
        d = self.robot.data
        q = d.joint_pos.torch.cpu().numpy().copy()
        gravity = d.projected_gravity_b.torch.cpu().numpy().copy()
        gyro = d.root_ang_vel_b.torch.cpu().numpy().copy()
        extras = np.tile(distance_features({},stamp),(self.num_envs,1))
        for i in range(self.num_envs):
            if i not in self.trackers or self.warming[i]: continue
            if self.cached.get(i,{}).get('stamp') == stamp:
                v, extra = self.cached[i]['patch'], self.cached[i]['distance']
            else:
                start=time.perf_counter()
                r = self.trackers[i].update(stamp,q[i],gravity[i],gyro[i],self.images.get(i))
                v, detail = self.patches[i].update(stamp,r,q[i],gravity[i],gyro[i],self.images.get(i))
                extra = distance_features(r,stamp)
                self.cached[i] = dict(stamp=stamp,patch=v,distance=extra)
                self.feature_records.append(dict(t=stamp,case=i,attempt=int(self.attempt[i]),
                    tracker=r,patch_valid=bool(v[8]),patch_reason=detail['reason'],distance=extra.tolist(),
                    processing_s=time.perf_counter()-start))
            self.live[i] = torch.as_tensor(v,device=self.device)
            extras[i] = extra
        self.obs['policy'][:,132:142] = self.live
        if options.interface == 'split':
            self.obs['policy'] = torch.cat((self.obs['policy'],torch.as_tensor(extras,device=self.device)),1)
        assert self.obs['policy'].shape[1] == (151 if options.interface=='split' else 144)
        assert torch.isfinite(self.obs['policy']).all()
        if self.args.mode!='train' and stamp>0 and (~self.warming).any():
            self.sensor_records.append(dict(t=stamp,q=q,gravity=gravity,gyro=gyro))
            for i,depth in self.images.items():
                self.depth_records.append(dict(t=stamp,case=i,depth=depth.squeeze().astype(np.float32)))

    def step(self, actions):
        # These label values are archived after the observation has been built;
        # they never enter the estimator, actor or critic.
        m = experiment.physical_metrics(self.raw)
        self.label_records.append(dict(t=self.t*.02,true_clearance=m['true_clearance'].cpu().numpy().copy(),
            failure=m['failure'].cpu().numpy().copy(),alive=self.alive.cpu().numpy().copy(),
            support=m['support'].cpu().numpy().copy(),force=m['peak_force'].cpu().numpy().copy()))
        self.control_records.append(dict(t=self.t*.02,policy=self.obs['policy'].cpu().numpy().copy(),
            actions=actions.cpu().numpy().copy(),ready=self.ready.cpu().numpy().copy(),
            attempt=self.attempt.cpu().numpy().copy()))
        result = super().step(actions)
        rate = self.controller.last_rate.reshape(-1)
        assert torch.all((rate==0.) | (rate==1.))
        assert torch.all(rate[self.ready & ~self.warming] == 1.)
        return result

    def close(self):
        def archive(name,rows):
            if rows:
                np.savez_compressed(self.out/name,**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
        archive('controller.npz',self.control_records); archive('evaluation_labels.npz',self.label_records)
        if self.args.mode!='train':
            archive('proprioception.npz',self.sensor_records); archive('depth.npz',self.depth_records)
            archive('startup.npz',self.startup_records)
        with (self.out/'split_feature_audit.jsonl').open('w') as f:
            for r in self.feature_records: f.write(json.dumps(r)+'\n')
        (self.out/'interface_audit.json').write_text(json.dumps(dict(
            channels=151 if options.interface=='split' else 144,control_hz=50,depth_hz=10,
            camera=self.camera_audit,resets=self.reset_checks,
            privileged_runtime_inputs=False,normalization_frozen=True),indent=2)+'\n')
        (self.out/'calibration.json').write_text(json.dumps(dict(cases=self.calibration,
            startup_T_P_C=dict(mount=self.mount.tolist(),quaternion_xyzw=self.quat.tolist()),
            model='model.json'),indent=2)+'\n')
        shutil.copy2(experiment.BODY,self.out/'model.json')
        super().close()


original_cfg = experiment.DeploymentEnvCfg
def split_cfg(**kwargs):
    cfg = original_cfg(**kwargs); out = Path(independent.args.output)
    manifest = json.loads((out/'manifest.json').read_text())
    sources = [Path(__file__).resolve()]
    for name in ('support_camera','support_landing_geometry','support_continuous_features',
                 'support_split_observation','support_retention','deployment','sim_deployment_sensor','soft_landing'):
        sources.append(experiment.ROOT/f'sitting_task/src/humanoid_sitting/{name}.py')
    for source in sources:
        shutil.copy2(source,out/'source'/source.name)
        manifest['hashes'][str(source.relative_to(experiment.ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest['split_interface'] = dict(mode=options.interface,parent=options.parent,
        parent_sha256=hashlib.sha256(Path(options.parent).read_bytes()).hexdigest(),
        appended='clearance_m, normal_P[3], distance_valid, measurement_age_s, conditional_budget_m',
        legacy='first 144 channels unchanged; channel140 remains complete-patch validity',
        initialized_new_weights='zero; old first-layer GEMM retained',
        normalizers='all frozen; new feature scales .3,1,1,1,1,.1,.02',
        optimizer='fresh PPO, no KL penalty; existing reward and action bounds unchanged',
        bank_from=options.bank_from)
    manifest['actor_inputs'] = f'{151 if options.interface=="split" else 144} sensor-derived channels'
    manifest['critic_inputs'] = manifest['actor_inputs']
    manifest['synchronization'] = 'current depth at100ms; FK+IMU tracking at20ms; original109 packet delay20ms'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return cfg

original_bank = independent.curriculum_bank
def bank(count,seed,balanced):
    if options.bank_from:
        result=json.loads(Path(options.bank_from).read_text()); assert len(result)==count
        return result
    return original_bank(count,seed,balanced)

original_runner = experiment.create_rsl_rl_runner
def split_runner(env,agent,log_dir):
    runner=original_runner(env,agent,log_dir=log_dir)
    assert type(runner.alg) is PPO
    if options.interface=='split':
        install_split_linear(runner.alg.actor); install_split_linear(runner.alg.critic)
    parent=torch.load(options.parent,map_location=env.device,weights_only=False)
    actor=parent['actor_state_dict']; critic=parent['critic_state_dict']
    if options.interface=='split':
        actor=expand_parent_state(actor); critic=expand_parent_state(critic)
    runner.alg.__class__=RetentionPPO
    runner.alg.configure_retention(actor,0.,freeze_normalization=True)
    if env.args.mode=='train' or not env.args.checkpoint:
        assert not env.args.checkpoint,'Training must begin from the declared parent'
        runner.alg.actor.load_state_dict(actor); runner.alg.critic.load_state_dict(critic)
        prepared=Path(log_dir)/'prepared.pt'; runner.save(str(prepared)); env.args.checkpoint=str(prepared)
    old_update=runner.alg.update
    def audited_update():
        result=old_update()
        result['new_distance_weight_max']=float(runner.alg.actor.mlp[0].weight[:,144:].abs().max()) if options.interface=='split' else 0.
        assert all(np.isfinite(v) for v in result.values())
        with (Path(log_dir)/'ppo_updates.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        return result
    runner.alg.update=audited_update
    return runner

independent.curriculum_bank=bank
experiment.DeploymentEnvCfg=split_cfg
experiment.SupportVecEnv=SplitObservationEnv
experiment.create_rsl_rl_runner=split_runner
if __name__=='__main__':experiment.main()
