"""Run the new sensor-only tracker online beside the unchanged frozen policy."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from humanoid_sitting.support_landing_geometry import LandingGeometryTracker
import capture_support_landing_sync as sync
capture=sync.capture
experiment=capture.experiment

class ShadowCaptureEnv(sync.SynchronizedCaptureEnv):
    def __init__(self,*args,**kwargs):
        self.shadows={};self.shadow_records=[];self.shadow_times=[];self.shadow_last_tick=-1
        super().__init__(*args,**kwargs)

    def _admit(self,ids):
        super()._admit(ids)
        for i in ids.tolist():
            if not self.ready[i]:continue
            original=self.observers[i].observer
            tracker=LandingGeometryTracker(experiment.BODY,original.K,original.T_P_C,original.seat_height)
            f=next(f for f in reversed(self.startup_frames) if f['case']==i)
            measured=experiment.estimate_body_frame(f['depth'],f['K'],self.mount,self.quat,f['gravity'])
            tracker.initialize_from_measured_floor((self.t+1)*.02,f['q'],f['gravity'],measured['floor_height_body_m'])
            self.shadows[i]=tracker

    def _observe(self,update_depth):
        super()._observe(update_depth)
        if self.t<=0 or self.t==self.shadow_last_tick:return
        self.shadow_last_tick=self.t;f=self.sensor_frames[-1]
        assert abs(f['t']-self.t*.02)<1e-7
        images={x['case']:x['depth'] for x in self.camera_frames[-self.num_envs:] if abs(x['t']-f['t'])<1e-7}
        for i in range(self.num_envs):
            if i not in self.shadows:
                r=dict(stamp=f['t'],valid=False,clearance_m=None,source='unavailable',reason='startup_refused')
            else:
                start=time.perf_counter()
                r=self.shadows[i].update(f['t'],f['q'][i],f['gravity'][i],f['gyro'][i],images.get(i))
                self.shadow_times.append(time.perf_counter()-start)
            self.shadow_records.append(dict(case=i,**r))

    def close(self):
        with (self.out/'shadow_records.jsonl').open('w') as f:
            for r in self.shadow_records:f.write(json.dumps(r)+'\n')
        (self.out/'shadow_runtime.json').write_text(json.dumps(dict(samples=len(self.shadow_records),
            median_ms=float(np.median(self.shadow_times)*1000),p95_ms=float(np.quantile(self.shadow_times,.95)*1000),
            total_processing_s=float(sum(self.shadow_times)),fed_to_controller=False),indent=2)+'\n')
        super().close()

original_cfg=experiment.DeploymentEnvCfg
def shadow_cfg(**kwargs):
    cfg=original_cfg(**kwargs);out=Path(capture.independent.args.output);m=json.loads((out/'manifest.json').read_text())
    for source in [Path(__file__).resolve(),experiment.ROOT/'sitting_task/src/humanoid_sitting/support_landing_geometry.py',
                   experiment.ROOT/'scripts/test_support_landing_geometry.py']:
        shutil.copy2(source,out/'source'/source.name)
        m['hashes'][str(source.relative_to(experiment.ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    m['landing_capture']['shadow_tracking']=True
    m['landing_capture']['policy_receives_shadow_output']=False
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');return cfg
experiment.SupportVecEnv=ShadowCaptureEnv
experiment.DeploymentEnvCfg=shadow_cfg
if __name__=='__main__':experiment.main()
