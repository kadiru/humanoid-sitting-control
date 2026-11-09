"""Second pilot with sensor-derived startup refusal and complete accounting.

The previous balanced run stopped on an unobservable reset. This entry point
keeps such attempts: it holds their initial posture, packs invalid geometry and
sit-command=0, and counts them as refused failures. It never resamples or inserts
simulator geometry. Admitted attempts retain the canonical policy/reward path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
import train_support_rl as experiment
from humanoid_sitting.support_readiness import (startup_measurement,
    AdmissionSensorInterface, admitted_targets, admitted_corrections)

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--output',required=True); p.add_argument('--mode',required=True)
p.add_argument('--geometry-count',type=int,required=True); p.add_argument('--geometry-seed',type=int,required=True)
p.add_argument('--bank-from')
a,_=p.parse_known_args()
if a.bank_from:
    index=sys.argv.index('--bank-from'); del sys.argv[index:index+2]


class UnavailableDepth:
    def read(self, *_):
        values=np.zeros(10,dtype=np.float32); values[-1]=1.
        return values,dict(usable=False,reason='Startup floor/seat geometry unavailable')


class GuardedSupportVecEnv(experiment.SupportVecEnv):
    def reset(self):
        with torch.inference_mode():
            self.raw.reset(); self.t=0; self.episode_length_buf.zero_()
            q=self.robot.data.joint_pos.torch.clone()
            for _ in range(5):
                self.robot.set_joint_position_target(q)
                self.raw.scene.write_data_to_sim(); self.raw.sim.step(); self.raw.scene.update(.005)
            startup=self.raw.scene['startup_camera']; startup.update(.005,force_recompute=True)
            depth=startup.data.output['distance_to_image_plane'].torch.cpu().numpy()
            K=startup.data.intrinsic_matrices.torch.cpu().numpy()
            gravity=self.robot.data.projected_gravity_b.torch.cpu().numpy()
            estimates=[startup_measurement(experiment.estimate_body_frame,depth[i],K[i],self.mount,self.quat,gravity[i]) for i in range(self.num_envs)]
            self.refusal_reasons=[e.get('reason') for e in estimates]
            self.sensor=AdmissionSensorInterface(self.robot,estimates,self.device,delay_steps=1,seed=self.args.seed+901+self.episode_number)
            self.ready=self.sensor.ready.clone()
            self.controller=experiment.SoftLandingController(self.policy,self.contact,mode='actions'); self.state=None
            self.previous=self.robot.data.joint_pos.torch.clone()
            live=self.raw.scene['live_camera']; live.update(.005,force_recompute=True)
            K=live.data.intrinsic_matrices.torch.cpu().numpy()
            self.observers=[experiment.CurrentSupportDepth(experiment.BODY,K[i],self.live_T,e['surface_center'][2]) if e['valid'] else UnavailableDepth() for i,e in enumerate(estimates)]
            self.live=torch.zeros(self.num_envs,10,device=self.device); self.live[:,-1]=1.
            # Refused attempts receive no task reward and cannot become success.
            self.alive=self.ready.clone()
            self.flags={k:torch.zeros_like(self.alive) for k in ('fall','wrong_contact','pelvis_floor','impact','fast_descent','torque')}
            self.hold=torch.zeros(self.num_envs,device=self.device); self.peak_hold=self.hold.clone()
            self.peak_force=self.hold.clone(); self.total_reward=self.hold.clone()
            self.economy_sum=self.hold.clone(); self.hold_samples=self.hold.clone(); self.foot_sum=self.hold.clone()
            self.depth_count=self.hold.clone(); self.initial_standing=experiment.physical_metrics(self.raw)['standing']
            self._observe(update_depth=True)
            record=dict(episode=self.episode_number,ready=self.ready.tolist(),refusal_reasons=self.refusal_reasons)
            with (self.out/'startup_admissions.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
        return self.obs

    def _observe(self, update_depth):
        super()._observe(update_depth)
        self.nominal=admitted_targets(self.nominal,self.previous[:,self.ids],self.ready)
        self.obs['policy'][:,109:132]=self.nominal

    def step(self, actions):
        return super().step(admitted_corrections(actions,self.ready))

    def finish_episode(self):
        rows=super().finish_episode()
        for i,row in enumerate(rows):
            row['startup_ready']=bool(self.ready[i])
            row['startup_refusal_reason']=self.refusal_reasons[i]
            row['failures']['perception_unavailable']=not row['startup_ready']
            row['success']=row['success'] and row['startup_ready']
        return rows


original_cfg=experiment.DeploymentEnvCfg


def guarded_cfg(**kwargs):
    out=Path(a.output)
    sources=[Path(__file__),experiment.ROOT/'sitting_task/src/humanoid_sitting/support_readiness.py',experiment.ROOT/'scripts/test_support_readiness.py']
    manifest=json.loads((out/'manifest.json').read_text())
    for source in sources:
        shutil.copy2(source,out/'source'/source.name)
        manifest['hashes'][str(source.relative_to(experiment.ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    manifest['startup_admission']='Unobservable startup is held and counted as a refused trial; no resampling or geometry fallback.'
    if a.mode=='train':
        assert a.geometry_count==16 and not a.bank_from
        rng=np.random.default_rng(a.geometry_seed)
        bank=experiment.furniture.bank(2,a.geometry_seed,16,True)
        ranges=dict(depth=[.41,.47,.53,.59],width=[.77,.86,.95,1.04],height=[.278,.294,.310,.324],yaw=np.deg2rad([-5.,-1.7,1.7,5.]).tolist())
        for key,levels in ranges.items():
            for g,value in zip(bank,rng.permutation(np.tile(levels,4))):g[key]=float(value)
        manifest['training_bank_design']='16 balanced geometries; no held-out examples copied.'
        kwargs['geometry_bank']=bank
    elif a.bank_from:
        path=Path(a.bank_from); bank=json.loads(path.read_text()); assert len(bank)==a.geometry_count
        kwargs['geometry_bank']=bank
        manifest['saved_geometry_bank']=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return original_cfg(**kwargs)


experiment.SupportVecEnv=GuardedSupportVecEnv
experiment.DeploymentEnvCfg=guarded_cfg
experiment.main()
