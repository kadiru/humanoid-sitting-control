"""Evaluation-only synchronized landing capture; original policy inputs unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
p=argparse.ArgumentParser(add_help=False);p.add_argument('--bank-from')
options,remaining=p.parse_known_args();sys.argv=[sys.argv[0],*remaining]
import train_support_rl_independent as independent
experiment=independent.experiment
assert independent.args.mode=='evaluate','Capture cannot train'

class RecordedObserver:
    def __init__(self,observer,env,index):self.observer=observer;self.env=env;self.index=index
    def read(self,depth,q,gravity):
        values,diagnostic=self.observer.read(depth,q,gravity)
        self.env.camera_frames.append(dict(t=self.env.t*.02,case=self.index,
            depth=np.asarray(depth,dtype=np.float32).squeeze().copy(),legacy=values.copy(),
            diagnostic=diagnostic))
        return values,diagnostic

class CaptureEnv(independent.IndependentSupportVecEnv):
    def __init__(self,*args,**kwargs):
        self.camera_frames=[];self.sensor_frames=[];self.label_frames=[]
        self.last_saved_tick=-1;self.calibration={};self.startup_frames=[]
        super().__init__(*args,**kwargs)

    def _admit(self,ids):
        super()._admit(ids)
        camera=self.raw.scene['startup_camera']
        for i in ids.tolist():
            if not self.ready[i]:continue
            observer=self.observers[i]
            self.calibration[str(i)]=dict(K=observer.K.tolist(),T_P_C=observer.T_P_C.tolist(),
                estimated_seat_height_m=float(observer.seat_height))
            self.startup_frames.append(dict(case=i,depth=camera.data.output['distance_to_image_plane'].torch[i].cpu().numpy().copy(),
                K=camera.data.intrinsic_matrices.torch[i].cpu().numpy().copy(),
                q=self.robot.data.joint_pos.torch[i].cpu().numpy().copy(),
                gravity=self.robot.data.projected_gravity_b.torch[i].cpu().numpy().copy()))
            self.observers[i]=RecordedObserver(observer,self,i)

    def _observe(self,update_depth):
        super()._observe(update_depth)
        if self.t<=0 or self.t==self.last_saved_tick:return
        self.last_saved_tick=self.t;d=self.robot.data
        # Robot-equivalent encoder/IMU/estimated motor-torque channels only.
        self.sensor_frames.append(dict(t=self.t*.02,q=d.joint_pos.torch.cpu().numpy().copy(),
            dq=d.joint_vel.torch.cpu().numpy().copy(),gravity=d.projected_gravity_b.torch.cpu().numpy().copy(),
            gyro=d.root_ang_vel_b.torch.cpu().numpy().copy(),torque=d.applied_torque.torch.cpu().numpy().copy()))
        # Separate label archive: never passed into RecordedObserver or policy.
        m=experiment.physical_metrics(self.raw)
        self.label_frames.append(dict(t=self.t*.02,true_clearance=m['true_clearance'].cpu().numpy().copy(),
            support=m['support'].cpu().numpy().copy(),failure=m['failure'].cpu().numpy().copy(),
            body_pose=d.body_pose_w.torch.cpu().numpy().copy(),root_pose=d.root_pose_w.torch.cpu().numpy().copy(),
            foot_fraction=m['foot_fraction'].cpu().numpy().copy()))

    def close(self):
        np.savez_compressed(self.out/'proprioception.npz',**{k:np.stack([x[k] for x in self.sensor_frames]) for k in self.sensor_frames[0]})
        np.savez_compressed(self.out/'evaluation_labels.npz',**{k:np.stack([x[k] for x in self.label_frames]) for k in self.label_frames[0]})
        np.savez_compressed(self.out/'depth.npz',**{k:np.stack([x[k] for x in self.camera_frames]) for k in ['t','case','depth','legacy']})
        np.savez_compressed(self.out/'startup.npz',**{k:np.stack([x[k] for x in self.startup_frames]) for k in self.startup_frames[0]})
        (self.out/'calibration.json').write_text(json.dumps(dict(cases=self.calibration,
            startup_T_P_C=dict(mount=self.mount.tolist(),quaternion_xyzw=self.quat.tolist()),
            joint_names=self.robot.joint_names,body_names=self.robot.body_names,
            model='model.json',sensor_samples=len(self.sensor_frames),depth_frames=len(self.camera_frames),
            original_controller_inputs_unchanged=True),indent=2)+'\n')
        with (self.out/'depth_diagnostics.jsonl').open('w') as f:
            for x in self.camera_frames:f.write(json.dumps(dict(t=x['t'],case=x['case'],**x['diagnostic']))+'\n')
        shutil.copy2(experiment.BODY,self.out/'model.json')
        super().close()

original_cfg=experiment.DeploymentEnvCfg
def capture_cfg(**kwargs):
    cfg=original_cfg(**kwargs);out=Path(independent.args.output);m=json.loads((out/'manifest.json').read_text())
    source=Path(__file__).resolve();shutil.copy2(source,out/'source'/source.name)
    m['hashes'][str(source.relative_to(experiment.ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    m['landing_capture']=dict(sensor_archive='proprioception.npz + depth.npz + calibration.json + model.json',
        labels_separate='evaluation_labels.npz; never observer or policy inputs',training=False,bank_from=options.bank_from)
    if options.bank_from:
        shutil.copy2(options.bank_from,out/'source/capture_geometry_bank.json')
        m['landing_capture']['bank_sha256']=hashlib.sha256(Path(options.bank_from).read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');return cfg
original_bank=independent.curriculum_bank
def bank(count,seed,balanced):
    if options.bank_from:
        result=json.loads(Path(options.bank_from).read_text());assert len(result)==count;return result
    return original_bank(count,seed,balanced)
independent.curriculum_bank=bank
experiment.SupportVecEnv=CaptureEnv
experiment.DeploymentEnvCfg=capture_cfg
if __name__=='__main__':experiment.main()
