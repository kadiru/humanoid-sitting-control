"""Frozen policy with current geometry at every 20 ms actor observation."""
import capture_support_tracking_control as measured
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
from humanoid_sitting.support_continuous_features import ContinuousPatchFeatures
timing=measured.timing;experiment=measured.experiment;capture=measured.capture


class ContinuousControlEnv(measured.TrackingControlEnv):
    def __init__(self,*args,**kwargs):
        self.patch_trackers={}
        super().__init__(*args,**kwargs)

    def _admit(self,ids):
        super()._admit(ids)
        for i in ids.tolist():
            if i in self.trackers:self.patch_trackers[i]=ContinuousPatchFeatures(self.trackers[i])

    def _observe(self,update_depth):
        # Advance the original scaffold exactly once, then replace only the
        # actor's explicit geometric feedback; no second controller call.
        timing.TimingAuditEnv._observe(self,update_depth)
        if self.t<=0 or self.t==self.tracking_tick:return
        self.tracking_tick=self.t;f=self.sensor_frames[-1]
        assert abs(f['t']-self.t*.02)<1e-7
        images={x['case']:x['depth'] for x in self.camera_frames[-self.num_envs:] if abs(x['t']-f['t'])<1e-7}
        for i in range(self.num_envs):
            if i not in self.trackers:
                r=dict(stamp=f['t'],valid=False,clearance_m=None,source='unavailable',reason='startup_refused')
            else:
                start=time.perf_counter();tracker=self.trackers[i]
                r=tracker.update(f['t'],f['q'][i],f['gravity'][i],f['gyro'][i],images.get(i))
                values,detail=self.patch_trackers[i].update(f['t'],r,f['q'][i],f['gravity'][i],f['gyro'][i],images.get(i))
                legacy=self.live[i].cpu().numpy().copy()
                self.live[i]=torch.as_tensor(values,device=self.device)
                self.feature_audit.append(dict(t=f['t'],case=i,source=r['source'],legacy=legacy.tolist(),fed=values.tolist(),**detail))
                self.shadow_times.append(time.perf_counter()-start)
            self.shadow_records.append(dict(case=i,**r))
        self.obs['policy'][:,132:142]=self.live
        labels=experiment.physical_metrics(self.raw)
        for name in ('peak_force','vz','leg_rms_nm','tilt2'):self.label_frames[-1][name]=labels[name].cpu().numpy().copy()

    def close(self):
        super().close()
        for filename,key,value in [('shadow_runtime.json','note','50 Hz tracked geometric feedback; 10 Hz new depth'),
                                   ('calibration.json','tracking_controller_inputs','50 Hz observed-patch transport with age and bounded uncertainty')]:
            p=self.out/filename;m=json.loads(p.read_text());m[key]=value;p.write_text(json.dumps(m,indent=2)+'\n')


original_cfg=experiment.DeploymentEnvCfg
def continuous_cfg(**kwargs):
    cfg=original_cfg(**kwargs);out=Path(capture.independent.args.output);m=json.loads((out/'manifest.json').read_text())
    for source in [Path(__file__).resolve(),experiment.ROOT/'sitting_task/src/humanoid_sitting/support_continuous_features.py']:
        shutil.copy2(source,out/'source'/source.name)
        m['hashes'][str(source.relative_to(experiment.ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    m['tracking_control'].update(actor_geometry_hz=50,actor_measured_patch_hz=10,
        unobserved_bounds='last observed point cloud transported with encoders/IMU for <=100 ms and <2 cm engineering budget')
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');return cfg
experiment.SupportVecEnv=ContinuousControlEnv
experiment.DeploymentEnvCfg=continuous_cfg
if __name__=='__main__':experiment.main()
